"""Automatic full-body X-ray anatomy suggestion for MediSense AI Goal 1.

Uses the 33-class ONNX anatomy model and classes.txt label order.

This module suggests anatomy only; it does not diagnose fractures or other
pathology. Confidence is a workflow signal, not a clinically calibrated
probability. A high score must not be described as independently verified
anatomy.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = "convnext_tiny_onnx_33class"
DEFAULT_MIN_CONFIDENCE = 0.60
IMAGE_SIZE = 224
CLASS_COUNT = 33
PROBABILITY_SUM_TOLERANCE = 0.03

# Goal 1 default follows the documented model transform: resize the shorter edge,
# then center-crop to IMAGE_SIZE. This is supported by the repository's model-transform
# test, but the original training/export pipeline and label mapping still need audit.
# Do not treat this choice as proof of clinical accuracy; verify against the original
# training/export code and evaluate on independently labelled radiographs.
DEFAULT_PREPROCESSING_MODE = "center_crop"
SUPPORTED_PREPROCESSING_MODES = {
    "direct_resize",
    "center_crop",
    "letterbox",
    "dual_ensemble",
}
# Annotation cleanup has not been validated against this model's training pipeline.
# Keep it disabled by default until a labelled A/B evaluation demonstrates benefit.
# It can still be enabled explicitly for controlled evaluation via environment variable.
DEFAULT_CLEAN_COLORED_OVERLAYS = False
MAX_COLORED_OVERLAY_FRACTION = 0.06
MIN_COLORED_OVERLAY_FRACTION = 0.0001
CLEANED_SPECIFIC_MIN_MARGIN = 0.10
# Provisional ambiguity guard, not a validated clinical threshold. This prevents
# near-ties from being marked completed; it does not fix high-confidence wrong labels.
DEFAULT_MIN_TOP_TWO_MARGIN = 0.10

# These labels describe broad regions or study groupings rather than a specific
# anatomical site. They must not be silently converted into a specific site.
BROAD_ANATOMY_LABELS = {
    # These labels do not identify a sufficiently specific study site for
    # selecting a regional abnormality model without user confirmation.
    "EXTREMITY",
    "LOWER_EXTREMITY",
    "UPPER_EXTREMITY",
    "SPINE",
    "ARM",
    "LEG",
    "HEAD",
}


def _is_specific_anatomy_label(label: str) -> bool:
    return str(label).strip().upper() not in BROAD_ANATOMY_LABELS


def _default_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "models"
        / "xray"
        / "bodypart_full"
        / "model.onnx"
    )


def _default_labels_path(model_path: Path) -> Path:
    return model_path.parent / "classes.txt"


def _read_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"X-ray anatomy class labels were not found at '{path}'. "
            "The 33 labels must be kept beside model.onnx."
        )

    labels = [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if len(labels) != CLASS_COUNT:
        raise ValueError(
            f"Expected {CLASS_COUNT} anatomy labels in '{path}', "
            f"found {len(labels)}."
        )

    if len(set(labels)) != len(labels):
        raise ValueError(
            f"Anatomy label file contains duplicate labels: '{path}'."
        )

    return labels


class XRayAnatomyClassifier:
    """Load the 33-class ONNX anatomy classifier."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        min_confidence: float | None = None,
        labels_path: str | Path | None = None,
        preprocessing_mode: str | None = None,
    ) -> None:
        explicit_or_env_model = (
            checkpoint_path or os.getenv("XRAY_BODYPART_ONNX_PATH")
        )

        if not explicit_or_env_model:
            legacy_env_path = os.getenv(
                "XRAY_BODYPART_CHECKPOINT", ""
            ).strip()

            if legacy_env_path and Path(legacy_env_path).suffix.lower() == ".onnx":
                explicit_or_env_model = legacy_env_path

        self.model_path = Path(
            explicit_or_env_model or _default_model_path()
        ).expanduser()

        self.checkpoint_path = self.model_path

        labels_override = (
            labels_path or os.getenv("XRAY_BODYPART_CLASSES_PATH")
        )

        self.labels_path = (
            Path(labels_override).expanduser()
            if labels_override
            else _default_labels_path(self.model_path)
        )

        confidence_text = os.getenv(
            "XRAY_ANATOMY_MIN_CONFIDENCE",
            str(DEFAULT_MIN_CONFIDENCE),
        )

        threshold = (
            min_confidence
            if min_confidence is not None
            else float(confidence_text)
        )

        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")

        self.min_confidence = float(threshold)
        margin_text = os.getenv(
            "XRAY_ANATOMY_MIN_TOP_TWO_MARGIN",
            str(DEFAULT_MIN_TOP_TWO_MARGIN),
        )
        self.min_top_two_margin = float(margin_text)
        if not 0.0 <= self.min_top_two_margin <= 1.0:
            raise ValueError("XRAY_ANATOMY_MIN_TOP_TWO_MARGIN must be between 0 and 1.")

        if preprocessing_mode:
            configured_preprocessing = str(preprocessing_mode).strip().lower()
        else:
            settings_mode = None
            try:
                from app.core.config import get_settings
                settings_mode = getattr(get_settings(), "xray_anatomy_preprocessing", None)
            except Exception:
                settings_mode = None
            configured_preprocessing = str(
                settings_mode
                or os.getenv(
                    "XRAY_ANATOMY_PREPROCESSING",
                    DEFAULT_PREPROCESSING_MODE,
                )
            ).strip().lower()

        if configured_preprocessing not in SUPPORTED_PREPROCESSING_MODES:
            allowed = ", ".join(sorted(SUPPORTED_PREPROCESSING_MODES))
            raise ValueError(
                "Unsupported XRAY_ANATOMY_PREPROCESSING value "
                f"'{configured_preprocessing}'. Allowed values: {allowed}."
            )
        self.preprocessing_mode = configured_preprocessing
        cleanup_text = os.getenv(
            "XRAY_ANATOMY_CLEAN_COLORED_OVERLAYS",
            "true" if DEFAULT_CLEAN_COLORED_OVERLAYS else "false",
        ).strip().lower()
        self.clean_colored_overlays = cleanup_text in {"1", "true", "yes", "on"}
        self.model_sha256: str | None = None
        self.labels_sha256: str | None = None

        self.requested_device = (device or "cpu").strip().lower()
        self.device = "cpu"

        self.session = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.classes: list[str] = []
        self.idx_to_class: dict[int, str] = {}
        self.model_name = MODEL_NAME
        self.image_size = IMAGE_SIZE
        self._ort = None

    @property
    def is_loaded(self) -> bool:
        return self.session is not None and bool(self.classes)

    def load(self) -> None:
        """Load the ONNX model and validate its interface."""
        if self.is_loaded:
            return

        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(
                "The full-body anatomy classifier requires model.onnx. "
                f"Received '{self.model_path}'."
            )

        if (
            not self.model_path.is_file()
            or self.model_path.stat().st_size <= 0
        ):
            raise FileNotFoundError(
                "Full-body X-ray anatomy model was not found or is empty: "
                f"'{self.model_path}'."
            )

        import onnxruntime as ort

        labels = _read_labels(self.labels_path)

        available_providers = set(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        selected_device = "cpu"

        if (
            self.requested_device.startswith("cuda")
            and "CUDAExecutionProvider" in available_providers
        ):
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            selected_device = "cuda"

        session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        if len(inputs) != 1 or len(outputs) < 1:
            raise ValueError(
                "Unexpected ONNX interface: expected one image input "
                "and at least one output."
            )

        input_meta = inputs[0]
        input_shape = input_meta.shape

        if len(input_shape) != 4:
            raise ValueError(
                "Expected NCHW input [N,3,224,224], "
                f"got {input_shape}."
            )

        expected_dims = {
            1: 3,
            2: IMAGE_SIZE,
            3: IMAGE_SIZE,
        }

        for axis, expected in expected_dims.items():
            actual = input_shape[axis]

            if isinstance(actual, int) and actual != expected:
                raise ValueError(
                    f"Input dimension {axis} must be {expected}; "
                    f"model reports {actual}."
                )

        output_meta = outputs[0]
        output_shape = output_meta.shape

        if len(output_shape) != 2:
            raise ValueError(
                f"Expected output [N,{CLASS_COUNT}], got {output_shape}."
            )

        output_classes = output_shape[1]

        if (
            isinstance(output_classes, int)
            and output_classes != len(labels)
        ):
            raise ValueError(
                f"ONNX output has {output_classes} classes, but "
                f"classes.txt has {len(labels)} labels."
            )

        self._ort = ort
        self.session = session
        self.input_name = input_meta.name
        self.output_name = output_meta.name
        self.classes = labels
        self.idx_to_class = dict(enumerate(labels))
        self.device = selected_device
        self.model_sha256 = self._sha256_file(self.model_path)
        self.labels_sha256 = self._sha256_file(self.labels_path)

        logger.info(
            "Loaded anatomy classifier: model=%s classes=%d device=%s "
            "preprocessing=%s model_sha256=%s labels_sha256=%s",
            self.model_name,
            len(self.classes),
            self.device,
            self.preprocessing_mode,
            self.model_sha256,
            self.labels_sha256,
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Return a stable artifact fingerprint for reproducible audits."""
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_image_content(self, image: Any) -> None:
        """Reject obvious blank placeholder cards."""
        import numpy as np
        from PIL import Image, ImageOps

        checked = ImageOps.exif_transpose(image).convert("L")
        checked.thumbnail(
            (256, 256),
            resample=Image.Resampling.BILINEAR,
        )

        pixels = np.asarray(checked, dtype=np.uint8)

        if pixels.size == 0 or pixels.ndim != 2:
            raise ValueError(
                "The provided X-ray image is empty or invalid."
            )

        bright_fraction = float(np.mean(pixels >= 245))
        dark_fraction = float(np.mean(pixels <= 20))
        std_value = float(pixels.std())

        if (
            bright_fraction >= 0.90
            and dark_fraction <= 0.04
            and std_value < 35.0
        ):
            raise ValueError(
                "Image rejected: this appears to be a blank/placeholder "
                "card rather than a radiograph. Upload the actual X-ray."
            )

    def _preprocess_tensor(self, pil_image: Any, mode: str):
        """Prepare a 224x224 RGB tensor using the specified transform."""
        if not self.is_loaded:
            self.load()

        import numpy as np
        from PIL import Image, ImageOps

        if hasattr(pil_image, "convert"):
            image = pil_image
        else:
            image = Image.open(pil_image)

        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                "The provided X-ray image has invalid dimensions."
            )

        self._validate_image_content(image)

        if mode == "direct_resize":
            # Keep all visible anatomy in the tensor instead of cropping.
            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                resample=Image.Resampling.BILINEAR,
            )
        elif mode == "letterbox":
            # Retain true aspect ratio with black border padding.
            scale = IMAGE_SIZE / max(width, height)
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            im_resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
            pad_im = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
            pad_im.paste(im_resized, ((IMAGE_SIZE - new_w) // 2, (IMAGE_SIZE - new_h) // 2))
            image = pad_im
        else:
            # center_crop: Legacy aspect-ratio resize/crop behavior.
            scale = IMAGE_SIZE / min(width, height)
            resized_width = max(
                IMAGE_SIZE,
                int(round(width * scale)),
            )
            resized_height = max(
                IMAGE_SIZE,
                int(round(height * scale)),
            )
            image = image.resize(
                (resized_width, resized_height),
                resample=Image.Resampling.BILINEAR,
            )
            left = (resized_width - IMAGE_SIZE) // 2
            top = (resized_height - IMAGE_SIZE) // 2
            image = image.crop(
                (
                    left,
                    top,
                    left + IMAGE_SIZE,
                    top + IMAGE_SIZE,
                )
            )

        array = np.asarray(image, dtype=np.float32) / np.float32(255.0)

        mean = np.asarray(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )
        std = np.asarray(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        array = (array - mean) / std

        tensor = np.transpose(array, (2, 0, 1))[None, ...].astype(
            np.float32,
            copy=False,
        )

        expected_shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)

        if tensor.shape != expected_shape:
            raise ValueError(
                f"Preprocessed tensor has unexpected shape: {tensor.shape}."
            )

        return tensor

    def _preprocess(self, pil_image: Any):
        """Prepare a 224x224 RGB tensor using the configured transform."""
        mode = "center_crop" if self.preprocessing_mode == "dual_ensemble" else self.preprocessing_mode
        return self._preprocess_tensor(pil_image, mode=mode)

    def _remove_colored_annotations(self, pil_image: Any):
        """Return a cleaned inference copy and bounded cleanup metadata.

        The source model card says heavily processed inputs are out of domain.
        Some project test images contain yellow/blue/red annotation arrows. Small
        saturated-color overlays are inpainted only on a copy used for inference;
        the stored/uploaded image is never changed. Large color areas are left
        untouched to avoid destroying genuinely colorized studies.
        """
        from PIL import Image, ImageOps

        if hasattr(pil_image, "convert"):
            image = pil_image
        else:
            image = Image.open(pil_image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        meta = {
            "applied": False,
            "method": "none",
            "masked_pixel_fraction": 0.0,
            "reason": "disabled" if not self.clean_colored_overlays else "no_small_color_overlay_detected",
        }
        if not self.clean_colored_overlays:
            return image.copy(), meta

        try:
            import cv2
            import numpy as np
        except Exception as exc:
            meta["reason"] = f"opencv_unavailable:{type(exc).__name__}"
            logger.warning("Colored X-ray annotation cleanup skipped: %s", meta["reason"])
            return image.copy(), meta

        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
            meta["reason"] = "invalid_rgb_image"
            return image.copy(), meta

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Plain radiographs are usually grayscale. Saturated pixels are therefore
        # a useful conservative proxy for colored teaching arrows/points/labels.
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        mask = np.where((saturation >= 70) & (value >= 45), 255, 0).astype(np.uint8)
        fraction = float(np.count_nonzero(mask) / mask.size)
        meta["masked_pixel_fraction"] = round(fraction, 6)

        if fraction < MIN_COLORED_OVERLAY_FRACTION:
            return image.copy(), meta
        if fraction > MAX_COLORED_OVERLAY_FRACTION:
            meta["reason"] = "color_coverage_too_large_to_clean_safely"
            return image.copy(), meta

        # Dilate slightly so the coloured line edges are removed too. If dilation
        # expands the mask beyond the safe ceiling, do not inpaint the image.
        kernel = np.ones((3, 3), dtype=np.uint8)
        expanded = cv2.dilate(mask, kernel, iterations=1)
        expanded_fraction = float(np.count_nonzero(expanded) / expanded.size)
        if expanded_fraction > MAX_COLORED_OVERLAY_FRACTION:
            meta["reason"] = "expanded_color_mask_too_large_to_clean_safely"
            return image.copy(), meta

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cleaned_bgr = cv2.inpaint(bgr, expanded, 3, cv2.INPAINT_TELEA)
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        meta.update({
            "applied": True,
            "method": "opencv_telea_inpaint_saturated_overlay_mask_v1",
            "masked_pixel_fraction": round(expanded_fraction, 6),
            "reason": "small_saturated_color_overlay_removed_from_inference_copy",
        })
        logger.info(
            "X-ray anatomy annotation cleanup applied: masked_fraction=%.6f",
            expanded_fraction,
        )
        return Image.fromarray(cleaned_rgb, mode="RGB"), meta

    def _run_inference_probabilities(self, image: Any):
        """Run the configured ONNX artifact on one image and validate its scores."""
        import numpy as np

        if self.preprocessing_mode == "dual_ensemble":
            tensor_crop = self._preprocess_tensor(image, mode="center_crop")
            out_crop = self.session.run(
                [self.output_name],
                {self.input_name: tensor_crop},
            )
            raw_crop = np.asarray(out_crop[0], dtype=np.float32)
            probs_crop = self._interpret_output_scores(raw_crop[0])

            tensor_letterbox = self._preprocess_tensor(image, mode="letterbox")
            out_letterbox = self.session.run(
                [self.output_name],
                {self.input_name: tensor_letterbox},
            )
            raw_letterbox = np.asarray(out_letterbox[0], dtype=np.float32)
            probs_letterbox = self._interpret_output_scores(raw_letterbox[0])

            combined = (probs_crop + probs_letterbox) / 2.0
            return combined

        tensor = self._preprocess(image)
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )
        if not outputs:
            raise ValueError("ONNX anatomy model returned no outputs.")

        raw_scores = np.asarray(outputs[0], dtype=np.float32)
        if raw_scores.ndim != 2 or raw_scores.shape != (1, len(self.classes)):
            raise ValueError(
                f"ONNX anatomy output has shape {raw_scores.shape}; "
                f"expected (1, {len(self.classes)})."
            )
        return self._interpret_output_scores(raw_scores[0])

    def _rank_probabilities(self, probabilities):
        """Return top candidates and decision facts for a probability vector."""
        import numpy as np

        ranked_indices = np.argsort(probabilities)[::-1]
        top = int(ranked_indices[0])
        second = int(ranked_indices[1])
        confidence = float(probabilities[top])
        second_confidence = float(probabilities[second])
        margin = max(0.0, confidence - second_confidence)
        top_predictions = [
            {
                "label": self.classes[int(index)],
                "probability": round(float(probabilities[int(index)]), 4),
            }
            for index in ranked_indices[:5]
        ]
        return {
            "candidate": self.classes[top],
            "confidence": confidence,
            "second_candidate": self.classes[second],
            "second_confidence": second_confidence,
            "margin": margin,
            "top_predictions": top_predictions,
        }

    def _interpret_output_scores(self, raw_scores):
        """Validate scores without silently converting arbitrary values."""
        import numpy as np

        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)

        if not np.isfinite(scores).all():
            raise ValueError(
                "ONNX anatomy model returned non-finite scores."
            )

        output_name = (self.output_name or "").strip().lower()

        if "logit" in output_name:
            shifted = scores - float(np.max(scores))
            exp_scores = np.exp(shifted).astype(
                np.float32,
                copy=False,
            )

            denominator = float(exp_scores.sum())

            if (
                not np.isfinite(denominator)
                or denominator <= 0.0
            ):
                raise ValueError(
                    "ONNX anatomy logits could not be converted "
                    "to probabilities."
                )

            probabilities = exp_scores / denominator

        else:
            if (
                (scores < 0.0).any()
                or (scores > 1.0).any()
                or abs(float(scores.sum()) - 1.0)
                > PROBABILITY_SUM_TOLERANCE
            ):
                raise ValueError(
                    f"ONNX output '{self.output_name}' is not a valid "
                    "probability distribution. Refusing to reinterpret "
                    "it as logits; verify the exported model contract."
                )

            total = float(scores.sum())

            if total <= 0.0:
                raise ValueError(
                    "ONNX anatomy probability sum must be positive."
                )

            probabilities = scores / total

        probability_sum = float(probabilities.sum())

        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
            or probability_sum <= 0.0
            or abs(probability_sum - 1.0) > 1e-4
        ):
            raise ValueError(
                "ONNX anatomy probabilities are invalid; refusing to route."
            )

        return probabilities

    def predict_image(self, pil_image: Any) -> dict[str, Any]:
        """Suggest anatomy; annotation cleanup is opt-in until independently evaluated."""
        if not self.is_loaded:
            self.load()

        from PIL import Image, ImageOps

        if hasattr(pil_image, "convert"):
            original_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        else:
            with Image.open(pil_image) as opened:
                original_image = ImageOps.exif_transpose(opened).convert("RGB")

        # Always make the original inference run first. If small coloured overlays
        # are present, run a second pass on an inpainted copy. Select the cleaned
        # result only when it becomes a specific label and passes the same 60%
        # gate plus a conservative 10-point margin; never force WRIST from filename.
        original_probabilities = self._run_inference_probabilities(original_image)
        original_rank = self._rank_probabilities(original_probabilities)
        cleaned_image, cleanup_meta = self._remove_colored_annotations(original_image)
        selected_probabilities = original_probabilities
        selected_rank = original_rank
        selected_preprocessing = self.preprocessing_mode
        cleanup_result_rank = None
        selected_cleaned_view = False

        if cleanup_meta.get("applied"):
            cleaned_probabilities = self._run_inference_probabilities(cleaned_image)
            cleanup_result_rank = self._rank_probabilities(cleaned_probabilities)
            cleaned_candidate_is_specific = _is_specific_anatomy_label(
                cleanup_result_rank["candidate"]
            )
            cleaned_candidate_passes_gate = (
                cleaned_candidate_is_specific
                and cleanup_result_rank["confidence"] >= self.min_confidence
                and cleanup_result_rank["margin"] >= CLEANED_SPECIFIC_MIN_MARGIN
            )
            original_candidate_is_specific = _is_specific_anatomy_label(
                original_rank["candidate"]
            )
            original_candidate_passes_gate = (
                original_candidate_is_specific
                and original_rank["confidence"] >= self.min_confidence
                and original_rank["margin"] >= CLEANED_SPECIFIC_MIN_MARGIN
            )
            if cleaned_candidate_passes_gate and (
                not original_candidate_passes_gate
                or cleanup_result_rank["confidence"] >= original_rank["confidence"]
            ):
                selected_probabilities = cleaned_probabilities
                selected_rank = cleanup_result_rank
                selected_cleaned_view = True
                selected_preprocessing = f"{self.preprocessing_mode}+colored_overlay_cleanup"      

        candidate = selected_rank["candidate"]
        confidence = selected_rank["confidence"]
        second_candidate = selected_rank["second_candidate"]
        second_confidence = selected_rank["second_confidence"]
        top_two_margin = selected_rank["margin"]
        confidence_passes = confidence >= self.min_confidence
        margin_passes = top_two_margin >= self.min_top_two_margin
        candidate_is_specific = _is_specific_anatomy_label(candidate)
        accepted = (
            confidence_passes
            and margin_passes
            and candidate_is_specific
        )

        if not candidate_is_specific:
            message = (
                f"The anatomy classifier's best candidate is the broad region "
                f"{candidate.replace('_', ' ').title()} ({confidence:.1%}). "
                "This is not specific enough to select a regional abnormality model. "
                "Ask the user to confirm the exact X-ray body part; do not infer a "
                "specific site from this broad label."
            )
        elif accepted:
            message = (
                f"The anatomy classifier's leading suggestion is "
                f"{candidate.replace('_', ' ').title()} ({confidence:.1%}). "
                "It passed the provisional confidence and top-two margin thresholds, "
                "but this is not independent anatomy verification. The application "
                "must separately require anatomy confirmation and check whether a "
                "compatible abnormality model exists."
            )
        else:
            message = (
                f"The anatomy classifier's best candidate is "
                f"{candidate.replace('_', ' ').title()} ({confidence:.1%}). "
                f"Verification requires score >= {self.min_confidence:.1%} and top-two "
                f"margin >= {self.min_top_two_margin:.1%}; observed margin is "
                f"{top_two_margin:.1%}. Anatomy remains unverified; no regional "
                "abnormality model should be selected automatically."
            )
        warnings: list[str] = []
        if not confidence_passes:
            warnings.append(
                f"Prediction confidence ({confidence:.1%}) is below the review threshold ({self.min_confidence:.1%})."
            )
        if not margin_passes:
            warnings.append(
                f"Top-two candidate score margin ({top_two_margin:.1%}) is below the ambiguity threshold ({self.min_top_two_margin:.1%})."
            )
        if not candidate_is_specific:
            warnings.append(
                f"Predicted anatomy '{candidate}' is a broad anatomical region; specific site confirmation is required."
            )
        warnings.append(
            "Anatomy prediction is an automated assistive suggestion and must be confirmed by the user before regional analysis."
        )

        return {
            "success": True,
            "class_count": len(self.classes),
            "predicted_anatomy": candidate,
            "confidence": round(confidence, 4),
            "prediction_status": "completed" if accepted else "needs_confirmation",
            "requires_user_confirmation": True,
            "preprocessing_verified": False,
            "warnings": warnings,
            "status": "completed" if accepted else "unverified",
            "body_part_label": candidate if accepted else None,
            "candidate_body_part": candidate,
            "body_part_confidence": round(confidence, 4),
            "second_candidate_body_part": second_candidate,
            "second_candidate_confidence": round(second_confidence, 4),
            "top_two_margin": round(top_two_margin, 4),
            "confidence_threshold": self.min_confidence,
            "confidence_threshold_validated": False,
            "top_two_margin_threshold": self.min_top_two_margin,
            "top_two_margin_threshold_validated": False,
            "preprocessing_mode": selected_preprocessing,
            "preprocessing_id": selected_preprocessing,
            "preprocessing_verified_against_training": False,
            "colored_annotation_cleanup": cleanup_meta,
            "colored_cleanup_candidate": (
                {
                    "label": cleanup_result_rank["candidate"],
                    "confidence": round(cleanup_result_rank["confidence"], 4),
                    "top_two_margin": round(cleanup_result_rank["margin"], 4),
                    "selected_for_routing": selected_cleaned_view,
                    "top_predictions": cleanup_result_rank["top_predictions"],
                }
                if cleanup_result_rank is not None else None
            ),
            "inference_view_selected": "annotation_cleaned_copy" if selected_cleaned_view else "original_upload_copy",
            "model_sha256": self.model_sha256,
            "labels_sha256": self.labels_sha256,
            "anatomy_independently_verified": False,
            "candidate_is_specific_anatomy": candidate_is_specific,
            "routing_requires_independent_verification": True,
            "model_name": self.model_name,
            "model_checkpoint": self.model_path.name,
            "model_license": "Apache-2.0; see upstream model card",
            "device": self.device,
            "classifier_labels": list(self.classes),
            "supported_body_parts": list(self.classes),
            "top_predictions": selected_rank["top_predictions"],
            "message": message,
        }

    def predict_file(self, image_path: str | Path) -> dict[str, Any]:
        """Open and classify an X-ray image file."""
        from PIL import Image, ImageOps

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"X-ray image file was not found: '{path}'."
            )

        with Image.open(path) as image:
            image.load()
            result = self.predict_image(
                ImageOps.exif_transpose(image)
            )

        result["image_name"] = path.name
        return result
