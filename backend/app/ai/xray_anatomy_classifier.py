"""Automatic X-ray body-part classification for MediSense AI Goal 1.

This component is deliberately separate from the regional abnormality/fracture
models. It predicts which of the seven MURA-supported upper-extremity regions
an image most resembles. It does not diagnose a disease and does not support
body parts outside the classifier's trained label set.

Expected checkpoint: a trusted MURA WebApp body-part checkpoint containing
`model_state`, `config.model_name`, and `class_to_idx`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

SUPPORTED_BODY_PARTS = {
    "ELBOW",
    "FINGER",
    "FOREARM",
    "HAND",
    "HUMERUS",
    "SHOULDER",
    "WRIST",
}


def _default_checkpoint_path() -> Path:
    """Resolve the checkpoint stored inside the MediSense backend by default."""
    return (
        Path(__file__).resolve().parents[2]
        / "models"
        / "xray"
        / "bodypart"
        / "best_model.pth"
    )


def _normalize_class_mapping(raw_mapping: Any) -> dict[int, str]:
    """Validate checkpoint label mapping and return index -> uppercase label."""
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise ValueError("Checkpoint has no valid class_to_idx mapping.")

    mapping: dict[int, str] = {}
    for raw_label, raw_index in raw_mapping.items():
        if not isinstance(raw_label, str):
            raise ValueError("Body-part labels must be strings.")
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid class index for label {raw_label!r}.") from exc
        label = raw_label.strip().upper()
        if not label or index < 0 or index in mapping:
            raise ValueError("Checkpoint contains duplicate or invalid class indices.")
        mapping[index] = label

    if set(mapping.values()) != SUPPORTED_BODY_PARTS:
        raise ValueError(
            "The body-part checkpoint labels do not match the seven supported "
            f"MURA classes. Found: {sorted(mapping.values())}"
        )
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("Checkpoint class indices must be contiguous and start at zero.")
    return mapping


class XRayAnatomyClassifier:
    """Load the pretrained Swin body-part classifier and classify an X-ray image.

    The default 0.60 confidence gate is a conservative *workflow gate*, not a
    clinically calibrated threshold. Set XRAY_ANATOMY_MIN_CONFIDENCE after
    validation against a representative holdout dataset. Low-confidence output
    remains unverified and must not be routed to a regional abnormality model.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        min_confidence: float | None = None,
    ) -> None:
        self.checkpoint_path = Path(
            checkpoint_path
            or os.getenv("XRAY_BODYPART_CHECKPOINT", str(_default_checkpoint_path()))
        ).expanduser()
        self.device_name = device or (
            "cuda" if self._cuda_available() else "cpu"
        )
        threshold_value = (
            min_confidence
            if min_confidence is not None
            else float(os.getenv("XRAY_ANATOMY_MIN_CONFIDENCE", "0.60"))
        )
        if not 0.0 <= float(threshold_value) <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")
        self.min_confidence = float(threshold_value)
        self.device = None
        self.model = None
        self.idx_to_class: dict[int, str] = {}
        self.image_size = 224
        self.model_name: str | None = None
        self._torch = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.device is not None

    def load(self) -> None:
        """Load the exact model architecture and state specified by the checkpoint."""
        if self.is_loaded:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "Automatic X-ray body-part checkpoint was not found at "
                f"'{self.checkpoint_path}'. Copy MURA_WebApp/backend/models/bodypart/"
                "best_model.pth to backend/models/xray/bodypart/best_model.pth, "
                "or set XRAY_BODYPART_CHECKPOINT to its full path."
            )

        import torch
        import timm

        # This checkpoint is a tensor/state-dictionary package. Keep safe loading
        # enabled instead of unpickling arbitrary Python objects.
        checkpoint = torch.load(
            str(self.checkpoint_path), map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Unsupported body-part checkpoint format.")

        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError("Body-part checkpoint is missing its model config.")
        model_name = config.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Body-part checkpoint does not specify config.model_name.")

        raw_state = checkpoint.get("model_state")
        if not isinstance(raw_state, dict) or not raw_state:
            raise ValueError("Body-part checkpoint does not contain model_state weights.")
        raw_mapping = checkpoint.get("class_to_idx")
        idx_to_class = _normalize_class_mapping(raw_mapping)
        image_size = config.get("img_size", 224)
        try:
            image_size = int(image_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("Checkpoint config.img_size is invalid.") from exc
        if image_size < 32 or image_size > 1024:
            raise ValueError(f"Unreasonable checkpoint image size: {image_size}.")

        device = torch.device(self.device_name)
        model = timm.create_model(
            model_name.strip(), pretrained=False, num_classes=len(idx_to_class)
        )
        # Strict loading prevents silent architecture/key mismatches.
        model.load_state_dict(raw_state, strict=True)
        model.to(device)
        model.eval()

        self._torch = torch
        self.device = device
        self.model = model
        self.idx_to_class = idx_to_class
        self.image_size = image_size
        self.model_name = model_name.strip()
        logger.info(
            "Loaded X-ray anatomy classifier: model=%s classes=%d device=%s checkpoint=%s",
            self.model_name,
            len(self.idx_to_class),
            self.device,
            self.checkpoint_path,
        )

    def _preprocess(self, pil_image: Any):
        """Apply checkpoint-matching CLAHE, resize and ImageNet normalization."""
        if not self.is_loaded:
            self.load()

        import cv2
        import numpy as np
        from PIL import Image
        from torchvision import transforms

        image = pil_image.convert("RGB") if hasattr(pil_image, "convert") else Image.open(pil_image).convert("RGB")
        rgb = np.asarray(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        enhanced_image = Image.fromarray(enhanced_rgb).resize(
            (self.image_size, self.image_size), resample=Image.Resampling.BILINEAR
        )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return transform(enhanced_image).unsqueeze(0)

    def predict_image(self, pil_image: Any) -> dict[str, Any]:
        """Return automatic body-part candidate plus transparent verification status."""
        if not self.is_loaded:
            self.load()

        torch = self._torch
        tensor = self._preprocess(pil_image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise ValueError("Body-part model returned an unexpected output shape.")
            if logits.shape[0] != 1 or logits.shape[1] != len(self.idx_to_class):
                raise ValueError(
                    f"Body-part model output shape {tuple(logits.shape)} does not match "
                    f"one image and {len(self.idx_to_class)} classes."
                )
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError("Body-part model returned non-finite scores.")
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence_tensor, index_tensor = torch.max(probabilities, dim=0)
            index = int(index_tensor.item())
            confidence = float(confidence_tensor.item())

        candidate = self.idx_to_class[index]
        verified = confidence >= self.min_confidence
        return {
            "status": "completed" if verified else "unverified",
            "body_part_label": candidate if verified else None,
            "candidate_body_part": candidate,
            "body_part_confidence": round(confidence, 4),
            "confidence_threshold": self.min_confidence,
            "confidence_threshold_validated": False,
            "model_name": self.model_name,
            "model_checkpoint": self.checkpoint_path.name,
            "supported_body_parts": sorted(SUPPORTED_BODY_PARTS),
            "message": (
                f"Image-type classifier identified {candidate.title()} with "
                f"confidence {confidence:.1%}. This is automated image classification, "
                "not a medical diagnosis."
                if verified
                else (
                    f"The image-type classifier's best candidate was {candidate.title()} "
                    f"({confidence:.1%}), below the configured review threshold. "
                    "The body part is unverified; do not route this image to a regional "
                    "abnormality model automatically."
                )
            ),
        }

    def predict_file(self, image_path: str | Path) -> dict[str, Any]:
        """Open an image path and run body-part classification."""
        from PIL import Image

        path = Path(image_path)
        with Image.open(path) as image:
            image.load()
            result = self.predict_image(image)
        result["image_name"] = path.name
        return result
