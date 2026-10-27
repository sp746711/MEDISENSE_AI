"""Automatic full-body X-ray anatomy suggestion for MediSense AI Goal 1.

Uses the 33-class ConvNeXt-Tiny ONNX model stored at
backend/models/xray/bodypart_full/model.onnx, with class order from classes.txt.

This module identifies likely anatomy only. It does not detect fractures or
other pathology, and its confidence threshold is a workflow gate, not a
clinically calibrated probability. Unsupported/uncertain results must not be
routed to an unrelated regional abnormality model.
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

# These are anatomy classifier labels, not a list of regions with supported
# disease/abnormality models. The router independently checks model availability.

def _default_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "models" / "xray" / "bodypart_full" / "model.onnx"
    )


def _default_labels_path(model_path: Path) -> Path:
    return model_path.parent / "classes.txt"


def _read_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"X-ray anatomy class labels were not found at '{path}'. "
            "The 33 labels must be kept beside model.onnx."
        )
    labels = [line.strip().upper() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(labels) != CLASS_COUNT:
        raise ValueError(
            f"Expected {CLASS_COUNT} anatomy labels in '{path}', found {len(labels)}."
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"Anatomy label file contains duplicate labels: '{path}'.")
    return labels


class XRayAnatomyClassifier:
    """Load the broader 33-class ONNX anatomy classifier.

    Constructor compatibility:
      - ``checkpoint_path`` may be passed positionally/by keyword as in the
        earlier class. It now points to an ONNX model, not a PyTorch .pth file.
      - ``XRAY_BODYPART_ONNX_PATH`` optionally overrides the default model path.
      - ``XRAY_BODYPART_CLASSES_PATH`` optionally overrides classes.txt.
      - ``XRAY_ANATOMY_MIN_CONFIDENCE`` configures the provisional routing gate.

    The existing MURA regional abnormality checkpoints remain separate and are
    still selected by xray_router only if its model registry says they are
    compatible and available.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        min_confidence: float | None = None,
        labels_path: str | Path | None = None,
    ) -> None:
        explicit_or_env_model = checkpoint_path or os.getenv("XRAY_BODYPART_ONNX_PATH")
        # Continue accepting the historical env var only if it explicitly names
        # an ONNX file. A former .pth path must not be mistaken for this model.
        if not explicit_or_env_model:
            legacy_env_path = os.getenv("XRAY_BODYPART_CHECKPOINT", "").strip()
            if legacy_env_path and Path(legacy_env_path).suffix.lower() == ".onnx":
                explicit_or_env_model = legacy_env_path
        self.model_path = Path(explicit_or_env_model or _default_model_path()).expanduser()
        self.checkpoint_path = self.model_path  # backwards-compatible attribute
        labels_override = labels_path or os.getenv("XRAY_BODYPART_CLASSES_PATH")
        self.labels_path = Path(labels_override).expanduser() if labels_override else _default_labels_path(self.model_path)

        confidence_text = os.getenv("XRAY_ANATOMY_MIN_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE))
        threshold = min_confidence if min_confidence is not None else float(confidence_text)
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")
        self.min_confidence = float(threshold)

        # The published ONNX model is CPU-friendly. Honor an explicit CUDA request
        # only when ONNX Runtime actually exposes the CUDA provider; otherwise use CPU.
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
        """Load ONNX model and validate its input/output contract and labels."""
        if self.is_loaded:
            return
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(
                "The full-body anatomy classifier requires model.onnx. "
                f"Received '{self.model_path}'. Keep the older MURA .pth checkpoint "
                "in its existing bodypart folder; do not use it as this model."
            )
        if not self.model_path.is_file() or self.model_path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Full-body X-ray anatomy model was not found or is empty: '{self.model_path}'."
            )

        import onnxruntime as ort

        labels = _read_labels(self.labels_path)
        available_providers = set(ort.get_available_providers())
        providers: list[str] = ["CPUExecutionProvider"]
        selected_device = "cpu"
        if self.requested_device.startswith("cuda") and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            selected_device = "cuda"

        session = ort.InferenceSession(str(self.model_path), providers=providers)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise ValueError(
                "Unexpected ONNX model interface: expected one image input and at least one output."
            )

        input_meta = inputs[0]
        input_shape = input_meta.shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected NCHW image input shape [N,3,224,224], got {input_shape}.")
        expected_dims = {1: 3, 2: IMAGE_SIZE, 3: IMAGE_SIZE}
        for axis, expected in expected_dims.items():
            actual = input_shape[axis]
            if isinstance(actual, int) and actual != expected:
                raise ValueError(
                    f"ONNX input dimension {axis} must be {expected}; model reports {actual}."
                )

        output_meta = outputs[0]
        output_shape = output_meta.shape
        if len(output_shape) != 2:
            raise ValueError(f"Expected class output shape [N,{CLASS_COUNT}], got {output_shape}.")
        output_classes = output_shape[1]
        if isinstance(output_classes, int) and output_classes != len(labels):
            raise ValueError(
                f"ONNX output has {output_classes} classes but classes.txt has {len(labels)} labels."
            )

        self._ort = ort
        self.session = session
        self.input_name = input_meta.name
        self.output_name = output_meta.name
        self.classes = labels
        self.idx_to_class = dict(enumerate(labels))
        self.device = selected_device
        logger.info(
            "Loaded full-body X-ray anatomy classifier: model=%s classes=%d device=%s model_path=%s labels_path=%s",
            self.model_name, len(self.classes), self.device, self.model_path, self.labels_path,
        )

    def _preprocess(self, pil_image: Any):
        """Match the published model card: RGB, resize-short-edge, center crop, ImageNet norm."""
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
            raise ValueError("The provided X-ray image has invalid dimensions.")

        scale = IMAGE_SIZE / min(width, height)
        resized_width = max(IMAGE_SIZE, int(round(width * scale)))
        resized_height = max(IMAGE_SIZE, int(round(height * scale)))
        image = image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
        left = (resized_width - IMAGE_SIZE) // 2
        top = (resized_height - IMAGE_SIZE) // 2
        image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

        array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        tensor = np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
        if tensor.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Preprocessed tensor has unexpected shape: {tensor.shape}.")
        return tensor

    def predict_image(self, pil_image: Any) -> dict[str, Any]:
        """Return the top anatomy candidate, confidence, and top-5 suggestions."""
        if not self.is_loaded:
            self.load()
        import numpy as np

        tensor = self._preprocess(pil_image)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        if not outputs:
            raise ValueError("ONNX anatomy model returned no outputs.")
        raw_scores = np.asarray(outputs[0], dtype=np.float32)
        if raw_scores.ndim != 2 or raw_scores.shape != (1, len(self.classes)):
            raise ValueError(
                f"ONNX anatomy output has shape {raw_scores.shape}; expected (1, {len(self.classes)})."
            )
        scores = raw_scores[0]
        if not np.isfinite(scores).all():
            raise ValueError("ONNX anatomy model returned non-finite scores.")

        # ONNX exports may expose post-softmax probabilities or raw logits.
        # Accept a plausible probability distribution; otherwise convert logits
        # with numerically stable softmax instead of rejecting them outright.
        score_sum = float(scores.sum())
        looks_like_probabilities = (
            bool((scores >= 0.0).all())
            and bool((scores <= 1.0).all())
            and score_sum > 0.0
            and abs(score_sum - 1.0) <= 0.03
        )
        if looks_like_probabilities:
            probabilities = scores
        else:
            shifted = scores - float(np.max(scores))
            exp_scores = np.exp(shifted).astype(np.float32, copy=False)
            exp_sum = float(exp_scores.sum())
            if not np.isfinite(exp_sum) or exp_sum <= 0.0:
                raise ValueError("ONNX anatomy logits could not be converted to probabilities.")
            probabilities = exp_scores / exp_sum

        probability_sum = float(probabilities.sum())
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities > 1.0).any()
            or probability_sum <= 0.0
            or abs(probability_sum - 1.0) > 1e-4
        ):
            raise ValueError(
                f"ONNX anatomy probabilities are invalid (sum={probability_sum:.6f}); refusing to route."
            )

        ranked_indices = np.argsort(probabilities)[::-1]
        best_idx = int(ranked_indices[0])
        candidate = self.classes[best_idx]
        confidence = float(probabilities[best_idx])
        accepted = confidence >= self.min_confidence
        top_predictions = [
            {"label": self.classes[int(i)], "probability": round(float(probabilities[int(i)]), 4)}
            for i in ranked_indices[:5]
        ]
        return {
            # Existing router relies on status == "completed" and body_part_label.
            # This means the provisional confidence gate passed, not clinical verification.
            "status": "completed" if accepted else "unverified",
            "body_part_label": candidate if accepted else None,
            "candidate_body_part": candidate,
            "body_part_confidence": round(confidence, 4),
            "confidence_threshold": self.min_confidence,
            "confidence_threshold_validated": False,
            "model_name": self.model_name,
            "model_checkpoint": self.model_path.name,
            "model_license": "Apache-2.0; see upstream model card",
            "device": self.device,
            "classifier_labels": list(self.classes),
            # Compatibility name retained; these are anatomy classifier labels, not
            # a guarantee that every region has an abnormality model in the registry.
            "supported_body_parts": list(self.classes),
            "top_predictions": top_predictions,
            "message": (
                f"The anatomy classifier's leading suggestion is {candidate.replace('_', ' ').title()} "
                f"({confidence:.1%}). This is image-type classification, not a diagnosis. "
                "The application must separately check whether a compatible abnormality model exists."
                if accepted else
                f"The anatomy classifier's best candidate is {candidate.replace('_', ' ').title()} "
                f"({confidence:.1%}), below the provisional review threshold of {self.min_confidence:.1%}. "
                "Anatomy remains unverified; no regional abnormality model should be selected automatically."
            ),
        }

    def predict_file(self, image_path: str | Path) -> dict[str, Any]:
        """Open and classify an X-ray file path."""
        from PIL import Image, ImageOps

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"X-ray image file was not found: '{path}'.")
        with Image.open(path) as image:
            image.load()
            result = self.predict_image(ImageOps.exif_transpose(image))
        result["image_name"] = path.name
        return result
