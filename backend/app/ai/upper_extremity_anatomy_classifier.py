"""Optional seven-class upper-extremity anatomy classifier for MediSense AI.

This is a separate anatomy model (elbow/finger/forearm/hand/humerus/shoulder/wrist),
not a regional abnormality model. It is used only to refine a broad
UPPER_EXTREMITY/ARM result from the primary 33-class classifier. The model is
optional; without its trained ONNX artifact the primary classifier result is
preserved and no specific region is guessed.

A compatible checkpoint can be produced by backend/train_upper_extremity_anatomy_classifier.py.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UPPER_EXTREMITY_CLASSES = (
    "ELBOW",
    "FINGER",
    "FOREARM",
    "HAND",
    "HUMERUS",
    "SHOULDER",
    "WRIST",
)
IMAGE_SIZE = 224
DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_MIN_MARGIN = 0.10


def _default_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "models"
        / "xray"
        / "upper_extremity"
        / "bodypart7_densenet121.onnx"
    )


def _default_labels_path(model_path: Path) -> Path:
    return model_path.parent / "classes.txt"


def _load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Upper-extremity anatomy labels were not found: {path}")
    labels = [line.strip().upper() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if tuple(labels) != UPPER_EXTREMITY_CLASSES:
        raise ValueError(
            "Upper-extremity model label order mismatch. Expected exactly: "
            + ", ".join(UPPER_EXTREMITY_CLASSES)
            + f"; found: {labels}"
        )
    return labels


def rank_upper_extremity_scores(
    scores: Any,
    labels: tuple[str, ...] = UPPER_EXTREMITY_CLASSES,
) -> dict[str, Any]:
    """Validate already-normalized scores and return ranked labels.

    This pure function is intentionally easy to test. It does not claim clinical
    calibration; the gate is a provisional workflow policy only.
    """
    import numpy as np

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(labels):
        raise ValueError(f"Expected {len(labels)} scores, received {len(values)}")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Upper-extremity classifier returned invalid scores")
    total = float(values.sum())
    if total <= 0 or abs(total - 1.0) > 0.03:
        raise ValueError("Upper-extremity output is not a probability distribution")
    values = values / total
    order = np.argsort(values)[::-1]
    best_i, second_i = int(order[0]), int(order[1])
    confidence = float(values[best_i])
    second_confidence = float(values[second_i])
    margin = max(0.0, confidence - second_confidence)
    return {
        "candidate_body_part": labels[best_i],
        "confidence": confidence,
        "second_candidate_body_part": labels[second_i],
        "second_confidence": second_confidence,
        "top_two_margin": margin,
        "top_predictions": [
            {"label": labels[int(i)], "probability": round(float(values[int(i)]), 4)}
            for i in order[: len(labels)]
        ],
    }


class UpperExtremityAnatomyClassifier:
    """Load and run a dedicated seven-class ONNX anatomy model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        min_confidence: float | None = None,
        min_margin: float | None = None,
    ) -> None:
        self.model_path = Path(
            model_path or os.getenv("XRAY_UPPER_EXTREMITY_CLASSIFIER_ONNX_PATH") or _default_model_path()
        ).expanduser()
        self.labels_path = Path(
            labels_path or os.getenv("XRAY_UPPER_EXTREMITY_CLASSIFIER_CLASSES_PATH") or _default_labels_path(self.model_path)
        ).expanduser()
        self.min_confidence = float(
            min_confidence if min_confidence is not None else os.getenv(
                "XRAY_UPPER_EXTREMITY_MIN_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE)
            )
        )
        self.min_margin = float(
            min_margin if min_margin is not None else os.getenv(
                "XRAY_UPPER_EXTREMITY_MIN_MARGIN", str(DEFAULT_MIN_MARGIN)
            )
        )
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("Upper-extremity min_confidence must be between 0 and 1")
        if not 0.0 <= self.min_margin <= 1.0:
            raise ValueError("Upper-extremity min_margin must be between 0 and 1")
        self.session = None
        self.pytorch_model = None
        self.backend_type: str = "none"
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.classes: list[str] = []
        self.model_sha256: str | None = None

    def load(self) -> None:
        if self.session is not None or self.pytorch_model is not None:
            return

        import hashlib

        # 1. Try ONNX model path if present
        if self.model_path.is_file() and self.model_path.stat().st_size > 0:
            import onnxruntime as ort

            labels = _load_labels(self.labels_path)
            session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if len(inputs) != 1 or len(outputs) < 1:
                raise ValueError("Upper-extremity ONNX model must have one input and at least one output")
            input_shape = inputs[0].shape
            if len(input_shape) != 4:
                raise ValueError(f"Expected upper-extremity input [N,3,224,224], got {input_shape}")
            for axis, expected in ((1, 3), (2, IMAGE_SIZE), (3, IMAGE_SIZE)):
                actual = input_shape[axis]
                if isinstance(actual, int) and actual != expected:
                    raise ValueError(f"Upper-extremity input axis {axis} must be {expected}, got {actual}")
            output_shape = outputs[0].shape
            if len(output_shape) != 2 or (isinstance(output_shape[1], int) and output_shape[1] != len(labels)):
                raise ValueError(f"Expected upper-extremity output [N,{len(labels)}], got {output_shape}")
            self.session = session
            self.input_name = inputs[0].name
            self.output_name = outputs[0].name
            self.classes = labels
            self.backend_type = "onnx"
            self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
            logger.info("Loaded 7-class upper-extremity ONNX anatomy classifier sha256=%s", self.model_sha256)
            return

        # 2. Check for PyTorch Swin-Tiny checkpoint (models/xray/bodypart/best_model.pth)
        pytorch_ckpt = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "xray"
            / "bodypart"
            / "best_model.pth"
        )
        if pytorch_ckpt.is_file() and pytorch_ckpt.stat().st_size > 0:
            import torch
            import timm

            ckpt = torch.load(str(pytorch_ckpt), map_location="cpu")
            if not isinstance(ckpt, dict) or "model_state" not in ckpt:
                raise ValueError(f"Invalid PyTorch checkpoint format in {pytorch_ckpt}")
            model = timm.create_model("swin_tiny_patch4_window7_224", num_classes=7, pretrained=False)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self.pytorch_model = model
            self.backend_type = "pytorch_swin"
            self.classes = list(UPPER_EXTREMITY_CLASSES)
            self.model_sha256 = hashlib.sha256(pytorch_ckpt.read_bytes()).hexdigest()
            logger.info("Loaded 7-class upper-extremity PyTorch Swin-Tiny anatomy classifier sha256=%s", self.model_sha256)
            return

        raise FileNotFoundError(
            "Dedicated seven-class upper-extremity anatomy model is not installed. "
            f"Expected ONNX at: {self.model_path} or PyTorch checkpoint at: {pytorch_ckpt}."
        )

    @staticmethod
    def _preprocess(image: Any):
        import numpy as np
        from PIL import Image, ImageOps

        if hasattr(image, "convert"):
            pil = ImageOps.exif_transpose(image).convert("RGB")
        else:
            with Image.open(image) as opened:
                pil = ImageOps.exif_transpose(opened).convert("RGB")
        if pil.width <= 0 or pil.height <= 0:
            raise ValueError("Upper-extremity image has invalid dimensions")
        # This exact direct resize is also used by the training/export script.
        pil = pil.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
        array = np.asarray(pil, dtype=np.float32) / np.float32(255.0)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        return np.transpose(array, (2, 0, 1))[None].astype(np.float32, copy=False)

    def predict_image(self, image: Any) -> dict[str, Any]:
        """Classify an already-open PIL image (including a cleaned inference copy)."""
        self.load()
        import numpy as np

        tensor = self._preprocess(image)

        if self.backend_type == "pytorch_swin":
            import torch

            with torch.no_grad():
                t = torch.from_numpy(tensor)
                logits = self.pytorch_model(t)
                scores = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float64)
            model_name = "swin_tiny_mura_upper_extremity_7class"
        else:
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            if not outputs:
                raise ValueError("Upper-extremity model returned no output")
            raw = np.asarray(outputs[0], dtype=np.float32)
            if raw.shape != (1, len(self.classes)) or not np.isfinite(raw).all():
                raise ValueError(f"Invalid upper-extremity output shape/values: {raw.shape}")
            scores = raw[0].astype(np.float64)
            # The training script exports raw logits named `logits`.
            if "logit" in (self.output_name or "").lower():
                scores = scores - scores.max()
                exp = np.exp(scores)
                scores = exp / exp.sum()
            else:
                if (scores < 0).any() or (scores > 1).any() or abs(float(scores.sum()) - 1.0) > 0.03:
                    raise ValueError("Upper-extremity output is not a valid probability distribution")
                scores = scores / scores.sum()
            model_name = "densenet121_mura_upper_extremity_7class"

        ranked = rank_upper_extremity_scores(scores, tuple(self.classes))
        accepted = (
            ranked["confidence"] >= self.min_confidence
            and ranked["top_two_margin"] >= self.min_margin
        )
        ranked.update({
            "status": "accepted" if accepted else "unverified",
            "accepted": accepted,
            "confidence_threshold": self.min_confidence,
            "min_confidence_margin": self.min_margin,
            "model_name": model_name,
            "model_sha256": self.model_sha256,
            "preprocessing_id": "rgb_resize224_imagenet_norm_v1",
            "confidence_threshold_validated": False,
            "message": (
                "The dedicated upper-extremity classifier passed its provisional confidence and margin gates."
                if accepted else
                "The dedicated upper-extremity classifier did not meet its provisional confidence or ambiguity gate."
            ),
        })
        return ranked

    def predict_file(self, image_path: str | Path) -> dict[str, Any]:
        from PIL import Image, ImageOps

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Upper-extremity X-ray image was not found: {path}")
        with Image.open(path) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
            result = self.predict_image(image)
        result["image_name"] = path.name
        return result
