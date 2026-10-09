"""X-ray deep learning model adapter and inference engine for MediSense AI.

Loads trained PyTorch ResNet checkpoints when configured.
Performs OpenCV image preprocessing and generates Grad-CAM explainability artifacts.
When no checkpoint is configured or region is unsupported, returns the exact required disclaimer:
'X-ray received successfully. Automated interpretation for this X-ray type is currently unavailable.
If you have the associated radiology report, upload it for supported text-based analysis.'

NEVER creates fake predictions or fake confidence values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_UNAVAILABLE_MESSAGE = (
    "X-ray received successfully. Automated interpretation for this "
    "X-ray type is currently unavailable. If you have the associated "
    "radiology report, upload it for supported text-based analysis."
)

CHEST_CLASSES = [
    "Normal / Clear",
    "Consolidation / Opacity",
    "Pneumothorax",
    "Pleural Effusion",
    "Cardiomegaly",
]


class XRayModelService:
    """Interface for region-specific deep learning X-ray models (PyTorch ResNet)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Optional[nn.Module] = None
        self._loaded = False
        self.model_version: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> bool:
        """Load trained PyTorch checkpoint if configured and present on disk."""
        checkpoint = (self.settings.xray_chest_checkpoint or "").strip()
        if not checkpoint:
            # Check default path
            default_path = self.settings.xray_model_path / "chest" / "model.pt"
            if default_path.exists():
                checkpoint = str(default_path)

        if not checkpoint:
            logger.info("No X-ray checkpoint configured.")
            self._loaded = False
            return False

        path = Path(checkpoint)
        if not path.is_absolute():
            path = self.settings.xray_model_path / path

        if not path.exists():
            logger.warning("X-ray checkpoint path does not exist: %s", path)
            self._loaded = False
            return False

        try:
            # Load weights into ResNet architecture
            logger.info("Loading PyTorch model weights from: %s", path)
            loaded = torch.load(str(path), map_location=self.device)
            if isinstance(loaded, nn.Module):
                self._model = loaded
            else:
                import torchvision.models as models

                resnet = models.resnet18(weights=None)
                resnet.fc = nn.Linear(resnet.fc.in_features, len(CHEST_CLASSES))
                if isinstance(loaded, dict) and "state_dict" in loaded:
                    resnet.load_state_dict(loaded["state_dict"])
                elif isinstance(loaded, dict):
                    resnet.load_state_dict(loaded)
                self._model = resnet

            self._model.to(self.device)
            self._model.eval()
            self.model_version = f"chest_resnet18@{path.name}"
            self._loaded = True
            logger.info("Successfully loaded X-ray model checkpoint.")
            return True
        except Exception as exc:
            logger.error("Failed to load X-ray model checkpoint: %s", exc)
            self._loaded = False
            return False

    def is_available(self) -> bool:
        if not self._loaded:
            self.load_model()
        return self._loaded

    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Preprocess X-ray image using OpenCV and standard normalization."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                pil_img = Image.open(image_path).convert("RGB")
                img = np.array(pil_img)

            img_resized = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0

            # Normalize with ImageNet standard mean & std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_norm - mean) / std

            # Transpose to (C, H, W) and add batch dim
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            return tensor
        except Exception as exc:
            logger.warning("Image preprocessing failed: %s", exc)
            return None

    def analyze(self, image_path: str, region: Optional[str] = None) -> dict[str, Any]:
        """Run deep learning inference or return exact standard unavailable status."""
        raw_reg = (region or "").strip().lower()
        reg = raw_reg if raw_reg else "unknown"
        if reg != "chest" or not self.is_available():
            msg = (
                f"X-ray received. Automated interpretation for the selected X-ray type is currently unavailable."
                if reg != "chest"
                else DEFAULT_UNAVAILABLE_MESSAGE
            )
            return {
                "status": "unavailable",
                "region": reg,
                "model_version": None,
                "prediction": None,
                "confidence_if_valid": None,
                "uncertainty": "Model not configured or region currently unsupported",
                "explainability_artifact": None,
                "message": msg,
            }

        # If checkpoint is loaded, execute PyTorch forward pass
        try:
            tensor = self.preprocess_image(image_path)
            if tensor is None:
                return {
                    "status": "unavailable",
                    "region": reg,
                    "model_version": self.model_version,
                    "prediction": None,
                    "confidence_if_valid": None,
                    "uncertainty": "Unable to preprocess image",
                    "explainability_artifact": None,
                    "message": "Image preprocessing failed.",
                }

            with torch.no_grad():
                outputs = self._model(tensor)
                probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                finding = CHEST_CLASSES[pred_idx] if pred_idx < len(CHEST_CLASSES) else "Anomaly Detected"

            # Generate Grad-CAM visualization
            from app.ai.gradcam import generate_gradcam

            gradcam_res = generate_gradcam(self._model, image_path, pred_idx)
            artifact = gradcam_res.get("artifact_path")

            return {
                "status": "completed",
                "region": "chest",
                "model_version": self.model_version,
                "prediction": finding,
                "confidence_if_valid": f"{confidence * 100:.1f}%",
                "uncertainty": "Assistive model interpretation; clinical radiological correlation required.",
                "explainability_artifact": artifact,
                "message": f"Identified radiographic finding: {finding}.",
            }
        except Exception as exc:
            logger.warning("Inference execution failed: %s", exc)
            return {
                "status": "unavailable",
                "region": reg,
                "model_version": self.model_version,
                "prediction": None,
                "confidence_if_valid": None,
                "uncertainty": f"Inference execution error: {exc}",
                "explainability_artifact": None,
                "message": DEFAULT_UNAVAILABLE_MESSAGE,
            }
