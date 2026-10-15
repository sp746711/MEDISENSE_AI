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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block: type[BasicBlock], layers: list[int], num_classes: int = 5) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_resnet18(num_classes: int = len(CHEST_CLASSES)) -> nn.Module:
    """Instantiate a standard ResNet-18 architecture in pure PyTorch."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


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
                try:
                    import torchvision.models as models
                    resnet = models.resnet18(weights=None)
                    resnet.fc = nn.Linear(resnet.fc.in_features, len(CHEST_CLASSES))
                except Exception:
                    resnet = build_resnet18(num_classes=len(CHEST_CLASSES))

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
            from PIL import ImageOps

            # Robust loading: try OpenCV first
            img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                # Fallback to PIL for non-standard paths or formats
                pil_img = Image.open(image_path)
                pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
                img_rgb = np.array(pil_img)

            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            img_norm = img_resized.astype(np.float32) / 255.0

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

        # 1. Check image file existence and readability
        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            logger.warning("X-ray image file missing or empty: %s", image_path)
            return {
                "status": "unavailable",
                "reason_code": "invalid_image",
                "region": reg,
                "model_version": None,
                "prediction": None,
                "confidence_if_valid": None,
                "uncertainty": "Image file is missing or empty",
                "explainability_artifact": None,
                "message": "The uploaded X-ray file could not be read or is unavailable. Please upload a valid PNG or JPEG image.",
            }

        # 2. Check anatomical region support
        if reg != "chest":
            logger.info("Unsupported X-ray anatomical region requested: %s", reg)
            return {
                "status": "unavailable",
                "reason_code": "unsupported_region",
                "region": reg,
                "model_version": None,
                "prediction": None,
                "confidence_if_valid": None,
                "uncertainty": f"Automated interpretation for {reg} radiography is currently unsupported (vision pipeline is specialized for chest radiography)",
                "explainability_artifact": None,
                "message": (
                    f"X-ray received. Automated interpretation for {reg} X-ray is currently unavailable "
                    f"as the diagnostic vision model is specialized for chest radiography. "
                    f"If you have the associated radiology report, upload it for supported text-based analysis."
                ),
            }

        # 3. Check model checkpoint availability
        if not self.is_available():
            logger.info("Chest X-ray model checkpoint not loaded or not configured.")
            return {
                "status": "unavailable",
                "reason_code": "missing_checkpoint",
                "region": "chest",
                "model_version": None,
                "prediction": None,
                "confidence_if_valid": None,
                "uncertainty": "Model checkpoint not configured on disk; place PyTorch weights in backend/models/xray/chest/model.pt or set XRAY_CHEST_CHECKPOINT",
                "explainability_artifact": None,
                "message": DEFAULT_UNAVAILABLE_MESSAGE,
            }

        # 4. If checkpoint is loaded, execute PyTorch forward pass
        try:
            tensor = self.preprocess_image(image_path)
            if tensor is None:
                logger.warning("Image preprocessing failed for path: %s", image_path)
                return {
                    "status": "unavailable",
                    "reason_code": "preprocessing_failed",
                    "region": reg,
                    "model_version": self.model_version,
                    "prediction": None,
                    "confidence_if_valid": None,
                    "uncertainty": "Unable to decode and preprocess image pixels",
                    "explainability_artifact": None,
                    "message": "Image preprocessing failed. Please verify the image file format.",
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
