"""Automatic full-body X-ray anatomy suggestion for MediSense AI Goal 1.

Uses the 33-class ONNX anatomy model and classes.txt label order.

This module suggests anatomy only; it does not diagnose fractures or other
pathology. Confidence is a workflow signal, not a clinically calibrated
probability. A high score must not be described as independently verified
anatomy.
"""
from __future__ import annotations

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

# These labels describe broad regions or study groupings rather than a specific
# anatomical site. They must not be silently converted into a specific site.
BROAD_ANATOMY_LABELS = {
    "EXTREMITY",
    "LOWER_EXTREMITY",
    "UPPER_EXTREMITY",
    "SPINE",
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

        logger.info(
            "Loaded anatomy classifier: model=%s classes=%d device=%s",
            self.model_name,
            len(self.classes),
            self.device,
        )

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

    def _preprocess(self, pil_image: Any):
        """Current RGB, center-crop and ImageNet-normalization pipeline.

        Training-pipeline parity remains unverified. Do not represent this
        transform as confirmed training parity until authoritative model
        preprocessing details have been established.
        """
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
        """Return the leading anatomy suggestion and alternatives."""
        if not self.is_loaded:
            self.load()

        import numpy as np

        tensor = self._preprocess(pil_image)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )

        if not outputs:
            raise ValueError(
                "ONNX anatomy model returned no outputs."
            )

        raw_scores = np.asarray(outputs[0], dtype=np.float32)

        if raw_scores.ndim != 2 or raw_scores.shape != (
            1,
            len(self.classes),
        ):
            raise ValueError(
                f"ONNX anatomy output has shape {raw_scores.shape}; "
                f"expected (1, {len(self.classes)})."
            )

        probabilities = self._interpret_output_scores(raw_scores[0])

        ranked_indices = np.argsort(probabilities)[::-1]

        best_idx = int(ranked_indices[0])
        candidate = self.classes[best_idx]
        confidence = float(probabilities[best_idx])

        second_idx = int(ranked_indices[1])
        second_candidate = self.classes[second_idx]
        second_confidence = float(probabilities[second_idx])

        top_two_margin = max(
            0.0,
            confidence - second_confidence,
        )

        accepted = confidence >= self.min_confidence

        top_predictions = [
            {
                "label": self.classes[int(index)],
                "probability": round(
                    float(probabilities[int(index)]),
                    4,
                ),
            }
            for index in ranked_indices[:5]
        ]

        if accepted:
            message = (
                f"The anatomy classifier's leading suggestion is "
                f"{candidate.replace('_', ' ').title()} "
                f"({confidence:.1%}). This is an unverified anatomy "
                "suggestion, not a diagnosis. The application must "
                "separately check whether a compatible abnormality "
                "model exists."
            )
        else:
            message = (
                f"The anatomy classifier's best candidate is "
                f"{candidate.replace('_', ' ').title()} "
                f"({confidence:.1%}), below the provisional review "
                f"threshold of {self.min_confidence:.1%}. Anatomy "
                "remains unverified; no regional abnormality model "
                "should be selected automatically."
            )

        return {
            "status": "completed" if accepted else "unverified",
            "body_part_label": candidate if accepted else None,
            "candidate_body_part": candidate,
            "body_part_confidence": round(confidence, 4),
            "second_candidate_body_part": second_candidate,
            "second_candidate_confidence": round(
                second_confidence,
                4,
            ),
            "top_two_margin": round(top_two_margin, 4),
            "confidence_threshold": self.min_confidence,
            "confidence_threshold_validated": False,
            "preprocessing_verified_against_training": False,
            "anatomy_independently_verified": False,
            "candidate_is_specific_anatomy": _is_specific_anatomy_label(candidate),
            "routing_requires_independent_verification": True,
            "model_name": self.model_name,
            "model_checkpoint": self.model_path.name,
            "model_license": "Apache-2.0; see upstream model card",
            "device": self.device,
            "classifier_labels": list(self.classes),
            "supported_body_parts": list(self.classes),
            "top_predictions": top_predictions,
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