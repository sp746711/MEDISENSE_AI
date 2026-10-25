"""X-ray deep learning model adapter and inference engine for MediSense AI.

Supports:
- Existing, explicitly configured 5-class ResNet-18 state-dict checkpoints.
- A dedicated MURA DenseNet121 binary wrist-X-ray checkpoint.
- Optional TorchXRayVision DenseNet121 multi-label chest X-ray inference.

Safety:
- Never invent predictions when a model is unavailable.
- DenseNet121 scores are model outputs, NOT calibrated probabilities or diagnoses.
- Multi-label outputs are preserved; no softmax is applied across independent findings.
- Chest X-ray models must not be used for other body regions.
- The MURA wrist model must not be used for other body regions.
- The wrist output is a research-model score, not a clinical diagnosis.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.ai.xray_router import (
    CHEST_CLASSES,
    get_model_registry_entry,
    normalize_region_name,
    verify_checkpoint_integrity,
)
from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_UNAVAILABLE_MESSAGE = (
    "X-ray received successfully. Automated interpretation for this "
    "X-ray type is currently unavailable. If you have the associated "
    "radiology report, upload it for supported text-based analysis."
)

# TorchXRayVision's published DenseNet121 label order for densenet121-res224-all.
TORCHXRAYVISION_LABELS = (
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
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
    def __init__(
        self, block: type[BasicBlock], layers: list[int], num_classes: int = 5
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
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
    """Instantiate the existing ResNet-18 architecture."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


class XRayModelService:
    """Region-aware X-ray model service with strict model/region compatibility guards.

    Configure XRAY_DEFAULT_MODEL=chest_densenet121 to use TorchXRayVision's
    pretrained multi-label model. The default remains chest_resnet18 to avoid
    silently changing existing deployments; that option requires a trained
    compatible local checkpoint.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Optional[nn.Module] = None
        self._loaded = False
        self.model_version: Optional[str] = None
        self._provider: Optional[str] = None
        self._load_failure_code: Optional[str] = None
        self._load_failure_reason: Optional[str] = None
        self._checkpoint_path: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._xrv = None
        self._active_region = "chest"
        self._active_classes = list(CHEST_CLASSES)

    @property
    def configured_model(self) -> str:
        value = getattr(self.settings, "xray_default_model", None) or "chest_resnet18"
        return str(value).strip().lower()

    def load_model(self, region: str = "chest") -> bool:
        """Load a model explicitly configured for the requested anatomical region.

        TorchXRayVision is chest-only. Other regions may use a region-specific
        ResNet18 state_dict only when the registry declares labels, preprocessing,
        architecture and a checkpoint. No model is silently reused across regions.
        """
        canonical_region, recognized = normalize_region_name(region)
        entry = get_model_registry_entry(canonical_region) if recognized else None
        self._load_failure_code = None
        self._load_failure_reason = None
        self._model = None
        self._loaded = False
        self._provider = None
        self._xrv = None
        self.model_version = None
        self._active_region = canonical_region
        self._active_classes = list(entry.class_label_mapping) if entry else []

        if not entry or not entry.is_supported:
            self._load_failure_code = "unsupported_region"
            self._load_failure_reason = (
                f"No compatible trained image model is configured for region '{canonical_region}'."
            )
            return False

        if canonical_region == "chest" and self.configured_model in {
            "chest_densenet121", "densenet121", "torchxrayvision",
            "torch_xray_vision", "xrv_densenet121",
        }:
            return self._load_torchxrayvision()

        architecture = (entry.model_architecture or "").strip().lower()
        if canonical_region == "wrist" and architecture in {
            "mura_densenet121", "densenet121", "mura-densenet121"
        }:
            return self._load_mura_wrist_checkpoint(entry)
        if "resnet18" not in architecture:
            self._load_failure_code = "unsupported_model_architecture"
            self._load_failure_reason = (
                f"Model architecture '{entry.model_architecture}' is not implemented for {canonical_region}."
            )
            return False
        if not self._active_classes:
            self._load_failure_code = "class_labels_not_configured"
            self._load_failure_reason = f"No class labels are configured for {canonical_region}."
            return False
        return self._load_resnet_checkpoint(entry)

    def _load_mura_wrist_checkpoint(self, entry=None) -> bool:
        """Load the MURA wrist-specific DenseNet121 binary classifier.

        This checkpoint is structurally different from the existing ResNet
        checkpoints: its state dictionary is stored under ``model_state`` and
        its classifier produces one logit. It must only be used for wrist X-rays.
        """
        if self._active_region != "wrist":
            self._load_failure_code = "provider_region_mismatch"
            self._load_failure_reason = "The MURA checkpoint is wrist-only."
            return False

        project_root = Path(__file__).resolve().parents[2]
        configured = (os.getenv("XRAY_WRIST_CHECKPOINT", "") or "").strip()
        if configured:
            candidate = Path(configured).expanduser()
            if not candidate.is_absolute():
                # Relative paths are first resolved from the backend project root.
                candidate = project_root / candidate
            path = candidate
        else:
            path = project_root / "models" / "xray" / "wrist" / "XR_WRIST_FINAL_best.pt"

        self._checkpoint_path = str(path.resolve(strict=False))
        if not path.is_file() or path.stat().st_size <= 0:
            self._load_failure_code = "checkpoint_path_not_found"
            self._load_failure_reason = f"Wrist checkpoint not found or empty: {self._checkpoint_path}"
            logger.warning("%s", self._load_failure_reason)
            return False

        try:
            # weights_only=True avoids arbitrary pickle object loading.
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint must be a dictionary.")
            if checkpoint.get("area") != "XR_WRIST":
                raise ValueError("Checkpoint metadata area is not XR_WRIST.")
            if int(checkpoint.get("img_size", 0)) != 224:
                raise ValueError("Checkpoint metadata does not specify 224-pixel inputs.")
            state_dict = checkpoint.get("model_state")
            if not isinstance(state_dict, dict) or not state_dict:
                raise ValueError("Checkpoint is missing its model_state dictionary.")
            classifier_weight = state_dict.get("classifier.weight")
            classifier_bias = state_dict.get("classifier.bias")
            if (not isinstance(classifier_weight, torch.Tensor)
                    or tuple(classifier_weight.shape) != (1, 1024)
                    or not isinstance(classifier_bias, torch.Tensor)
                    or tuple(classifier_bias.shape) != (1,)):
                raise ValueError("Checkpoint classifier is not the expected one-logit DenseNet121 head.")

            from torchvision import models
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, 1)
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device)
            model.eval()

            self._model = model
            self._provider = "mura_densenet121"
            self.model_version = f"mura-densenet121-wrist@{path.name}"
            self._active_classes = ["normal", "abnormal"]
            self._loaded = True
            self._load_failure_code = None
            self._load_failure_reason = None
            logger.info("Loaded MURA DenseNet121 wrist checkpoint: %s", self._checkpoint_path)
            return True
        except Exception as exc:
            self._model = None
            self._loaded = False
            self.model_version = None
            self._provider = None
            self._load_failure_code = "mura_wrist_checkpoint_load_failed"
            self._load_failure_reason = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to load MURA wrist checkpoint: %s", exc)
            return False

    def _load_torchxrayvision(self) -> bool:
        """Load pretrained TorchXRayVision DenseNet121 and its published labels."""
        try:
            import torchxrayvision as xrv

            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            labels = list(getattr(model, "pathologies", []))
            if not labels:
                raise ValueError("TorchXRayVision model did not expose pathology labels.")
            missing = [label for label in TORCHXRAYVISION_LABELS if label not in labels]
            if missing:
                raise ValueError(
                    "Loaded TorchXRayVision model has an unexpected label set; "
                    f"missing expected labels: {missing}"
                )
            model = model.to(self.device)
            model.eval()
            self._model = model
            self._xrv = xrv
            self._provider = "torchxrayvision"
            self.model_version = "torchxrayvision-densenet121-res224-all"
            self._loaded = True
            self._load_failure_code = None
            self._load_failure_reason = None
            logger.info("Loaded TorchXRayVision DenseNet121 with %d labels.", len(labels))
            return True
        except Exception as exc:
            self._model = None
            self._loaded = False
            self.model_version = None
            self._provider = None
            self._load_failure_code = "densenet_load_failed"
            self._load_failure_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("TorchXRayVision DenseNet121 unavailable: %s", exc)
            return False

    def _load_resnet_checkpoint(self, entry=None) -> bool:
        """Load a region-specific ResNet18 state_dict after structural validation."""
        region = self._active_region
        classes = self._active_classes
        if entry is None:
            entry = get_model_registry_entry(region)
        if not entry or not classes:
            self._load_failure_code = "model_configuration_incomplete"
            self._load_failure_reason = f"Missing registry entry or class labels for {region}."
            return False

        # Region-specific environment configuration takes priority over cached settings.
        configured_checkpoint = (os.getenv(f"XRAY_{region.upper()}_CHECKPOINT", "") or "").strip()
        if region == "chest" and not configured_checkpoint:
            configured_checkpoint = (os.getenv("XRAY_CHEST_CHECKPOINT", "") or "").strip()
        if region == "chest" and not configured_checkpoint:
            configured_checkpoint = (getattr(self.settings, "xray_chest_checkpoint", "") or "").strip()
        model_root = Path(self.settings.xray_model_path).expanduser()
        if configured_checkpoint:
            configured_path = Path(configured_checkpoint).expanduser()
            if configured_path.is_absolute() or configured_path.exists():
                path = configured_path
            else:
                path = model_root / configured_path
        elif entry.checkpoint_location:
            configured_path = Path(entry.checkpoint_location).expanduser()
            path = configured_path if configured_path.is_absolute() else model_root / configured_path
        else:
            self._load_failure_code = "checkpoint_path_not_configured"
            self._load_failure_reason = f"No checkpoint path is configured for region '{region}'."
            return False

        self._checkpoint_path = str(path.resolve(strict=False))
        if not path.exists():
            self._load_failure_code = "checkpoint_path_not_found"
            self._load_failure_reason = f"Configured {region} X-ray checkpoint does not exist: {self._checkpoint_path}"
            logger.info("%s", self._load_failure_reason)
            return False
        valid, reason = verify_checkpoint_integrity(path, expected_classes=len(classes))
        if not valid:
            self._load_failure_code = "checkpoint_integrity_failed"
            self._load_failure_reason = str(reason)
            logger.warning("Checkpoint integrity verification failed for %s: %s", region, reason)
            return False

        try:
            loaded = torch.load(str(path), map_location=self.device, weights_only=True)
            if isinstance(loaded, nn.Module):
                raise ValueError("Full-model pickle checkpoints are not accepted; use a state_dict.")
            try:
                import torchvision.models as models
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, len(classes))
            except Exception:
                model = build_resnet18(num_classes=len(classes))
            if not isinstance(loaded, dict):
                raise ValueError("Checkpoint must be a state_dict or contain one.")
            state_dict = loaded.get("state_dict", loaded.get("model_state_dict", loaded))
            if not isinstance(state_dict, dict) or not state_dict:
                raise ValueError("Checkpoint does not contain a valid state dictionary.")
            if all(isinstance(key, str) for key in state_dict):
                state_dict = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
            self._model = model.to(self.device)
            self._model.eval()
            self._provider = "resnet18"
            self.model_version = f"{region}_resnet18@{path.name}"
            self._loaded = True
            self._load_failure_code = None
            self._load_failure_reason = None
            return True
        except Exception as exc:
            self._load_failure_code = "checkpoint_load_failed"
            self._load_failure_reason = f"{type(exc).__name__}: {exc}"
            logger.error("Failed to load %s ResNet checkpoint: %s", region, exc)
            self._model = None
            self._loaded = False
            self.model_version = None
            self._provider = None
            return False

    def is_available(self, region: str = "chest") -> bool:
        canonical, recognized = normalize_region_name(region)
        if not recognized:
            return False
        if self._loaded and self._active_region == canonical:
            return True
        return self.load_model(canonical)

    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Preprocess an image for the existing RGB/ImageNet ResNet model."""
        try:
            from PIL import ImageOps

            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                img_rgb = np.asarray(image)
            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            img_norm = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_norm - mean) / std
            return torch.from_numpy(img_norm.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        except Exception as exc:
            logger.warning("ResNet image preprocessing failed: %s", exc)
            return None

    def preprocess_mura_wrist(self, image_path: str) -> Optional[torch.Tensor]:
        """Preprocess wrist images with 224px RGB and ImageNet normalization."""
        try:
            from PIL import ImageOps
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            array = (array - mean) / std
            if array.shape != (224, 224, 3) or not np.isfinite(array).all():
                raise ValueError("Unexpected or non-finite wrist image tensor.")
            return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        except Exception as exc:
            logger.warning("MURA wrist image preprocessing failed: %s", exc)
            return None

    def preprocess_torchxrayvision(self, image_path: str) -> Optional[torch.Tensor]:
        """Use TorchXRayVision's own grayscale loading, crop, resize, and normalization."""
        try:
            if self._xrv is None:
                import torchxrayvision as xrv
                self._xrv = xrv
            image = self._xrv.utils.load_image(image_path)
            image = self._xrv.datasets.XRayCenterCrop()(image)
            image = self._xrv.datasets.XRayResizer(224)(image)
            array = np.asarray(image, dtype=np.float32)
            if array.ndim == 2:
                array = array[None, :, :]
            elif array.ndim == 3 and array.shape[0] != 1:
                # XRayVision's loader should yield one grayscale channel; reject
                # unexpected layouts rather than guessing channel order.
                raise ValueError(f"Expected one grayscale channel, got shape {array.shape}.")
            if array.shape != (1, 224, 224):
                raise ValueError(f"Unexpected preprocessed image shape: {array.shape}.")
            if not np.isfinite(array).all():
                raise ValueError("Preprocessed image contains non-finite pixel values.")
            return torch.from_numpy(array.copy()).unsqueeze(0).to(self.device)
        except Exception as exc:
            logger.warning("TorchXRayVision preprocessing failed: %s", exc)
            return None

    def _unavailable(
        self,
        canonical_region: str,
        reason_code: str,
        uncertainty: str,
        model_version: Optional[str] = None,
        message: str = DEFAULT_UNAVAILABLE_MESSAGE,
    ) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "reason_code": reason_code,
            "region": canonical_region,
            "model_version": model_version,
            "prediction": None,
            "findings": [],
            "confidence_if_valid": None,
            "uncertainty": uncertainty,
            "explainability_artifact": None,
            "message": message,
        }

    def analyze(self, image_path: str, region: Optional[str] = None) -> dict[str, Any]:
        """Run the selected model or return an explicit unavailable result."""
        raw_region = (region or "").strip()
        canonical_region, recognized = normalize_region_name(raw_region)
        registry_entry = get_model_registry_entry(canonical_region) if recognized else None
        display_region = (
            registry_entry.display_name
            if registry_entry
            else (canonical_region if canonical_region != "unknown" else "the selected anatomical region")
        )

        if not recognized or not registry_entry:
            return self._unavailable(
                canonical_region,
                "unrecognized_region",
                "The selected X-ray body region is missing or could not be recognized.",
                message=(
                    f"X-ray received. The selected body region '{raw_region or 'unknown'}' could not be matched "
                    "to a configured region. Please select a specific body region."
                ),
            )

        if not registry_entry.is_supported:
            return self._unavailable(
                canonical_region,
                "unsupported_region",
                f"Automated interpretation for {display_region} radiography is not configured.",
                message=(
                    f"X-ray received. Automated image interpretation for {display_region} is currently unavailable "
                    "because no compatible trained model is configured for this region. The image is not being "
                    "sent through the chest model. Upload the associated radiology report for text-based analysis."
                ),
            )

        if canonical_region != "chest" and self.configured_model in {
            "chest_densenet121", "densenet121", "torchxrayvision", "torch_xray_vision", "xrv_densenet121"
        }:
            # TorchXRayVision's configured DenseNet is chest-specific; it must never
            # be reused for another anatomy even if configuration is accidentally changed.
            pass

        try:
            image_file = Path(image_path)
            readable_candidate = image_file.is_file() and image_file.stat().st_size > 0
        except (OSError, TypeError, ValueError):
            readable_candidate = False

        if not readable_candidate:
            return self._unavailable(
                canonical_region,
                "invalid_image",
                "Image file is missing or empty.",
                message="The uploaded X-ray file could not be read. Please upload a valid PNG or JPEG image.",
            )

        if not self.is_available(canonical_region):
            failure_code = self._load_failure_code or "model_unavailable"
            failure_reason = self._load_failure_reason or "Model could not be loaded."
            is_development = str(getattr(self.settings, "app_env", "development")).lower() in {
                "development", "dev", "local"
            }
            message = (
                f"X-ray received. The {display_region} image model is not available. "
                "No prediction was generated; verify the region-specific trained checkpoint and model configuration."
            )
            if is_development:
                message += f" Diagnostic: {failure_code}: {failure_reason}"
            result = self._unavailable(
                canonical_region, failure_code, failure_reason, message=message
            )
            result["checkpoint_path"] = self._checkpoint_path
            return result

        if self._provider == "torchxrayvision":
            # This provider is only valid for chest radiographs.
            if canonical_region != "chest":
                return self._unavailable(canonical_region, "provider_region_mismatch", "TorchXRayVision provider is chest-only.")
            return self._analyze_torchxrayvision(image_path)
        if self._provider == "mura_densenet121":
            if canonical_region != "wrist":
                return self._unavailable(canonical_region, "provider_region_mismatch", "MURA DenseNet121 provider is wrist-only.")
            return self._analyze_mura_wrist(image_path)
        return self._analyze_resnet(image_path, canonical_region, list(self._active_classes))

    def _analyze_mura_wrist(self, image_path: str) -> dict[str, Any]:
        """Run wrist-only binary inference and preserve its limited research scope."""
        tensor = self.preprocess_mura_wrist(image_path)
        if tensor is None:
            return self._unavailable(
                "wrist", "preprocessing_failed", "Unable to decode and preprocess wrist image.",
                self.model_version,
                "Image preprocessing failed. Please verify the image file format.",
            )
        try:
            with torch.inference_mode():
                outputs = self._model(tensor)
            if not isinstance(outputs, torch.Tensor) or tuple(outputs.shape) != (1, 1):
                raise ValueError(f"Expected one wrist logit, received {getattr(outputs, 'shape', None)}")
            logit = outputs[0, 0].float()
            if not torch.isfinite(logit):
                raise ValueError("Wrist model produced a non-finite logit.")
            score = float(torch.sigmoid(logit).item())
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("Wrist model produced an invalid sigmoid score.")
            predicted_label = "abnormal" if score >= 0.5 else "normal"
            return {
                "status": "completed",
                "region": "wrist",
                "model_version": self.model_version,
                "prediction": predicted_label,
                "findings": [{
                    "label": predicted_label,
                    "model_score": round(score if predicted_label == "abnormal" else 1.0 - score, 5),
                    "abnormality_score": round(score, 5),
                    "score_type": "sigmoid_model_score_not_clinically_calibrated",
                    "is_actionable_finding": False,
                }],
                "abnormality_score": round(score, 5),
                "confidence_if_valid": None,
                "uncertainty": (
                    "Research-only MURA wrist abnormality-classifier output. The sigmoid score is not a clinically calibrated "
                    "probability and does not establish or exclude a fracture. The default 0.5 threshold is a "
                    "technical demo threshold, not a clinically validated decision threshold. Radiologist review is required."
                ),
                "explainability_artifact": None,
                "message": (
                    f"The research model's wrist output was '{predicted_label}' (normal/abnormal classification). This is not a diagnosis; "
                    "a qualified clinician/radiologist must interpret the X-ray."
                ),
            }
        except Exception as exc:
            logger.warning("MURA wrist inference failed: %s", exc)
            return self._unavailable(
                "wrist", "inference_failed", "Wrist inference could not be completed.", self.model_version,
                "Wrist X-ray analysis failed. No result is available; please consult a qualified radiologist.",
            )

    def _analyze_torchxrayvision(self, image_path: str) -> dict[str, Any]:
        tensor = self.preprocess_torchxrayvision(image_path)
        if tensor is None:
            return self._unavailable(
                "chest", "preprocessing_failed",
                "Unable to decode and preprocess image pixels.", self.model_version,
                "Image preprocessing failed. Please verify the image file format.",
            )
        try:
            with torch.inference_mode():
                outputs = self._model(tensor)
            if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2 or outputs.shape[0] != 1:
                raise ValueError(f"Unexpected DenseNet output shape: {getattr(outputs, 'shape', None)}")
            model_labels = list(getattr(self._model, "pathologies", []))
            if outputs.shape[1] != len(model_labels):
                raise ValueError("Model output count does not match its pathology labels.")
            scores = outputs[0].detach().float().cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("Model produced non-finite scores.")

            # These are raw model scores, not confirmed positive findings.
            # Do not apply softmax: these labels are multi-label, not mutually
            # exclusive. No validated local threshold is configured, so each
            # record is explicitly marked non-actionable until a separate,
            # validated decision policy exists.
            scores_by_label = []
            for label in TORCHXRAYVISION_LABELS:
                if label in model_labels:
                    score = float(scores[model_labels.index(label)])
                    scores_by_label.append({
                        "label": label,
                        "model_score": round(score, 5),
                        "detected": None,
                        "score_type": "unthresholded_model_score",
                        "is_actionable_finding": False,
                    })
            if len(scores_by_label) != len(TORCHXRAYVISION_LABELS):
                raise ValueError("Expected DenseNet121 label mapping could not be completed.")

            ranked_scores = sorted(
                scores_by_label,
                key=lambda item: item["model_score"],
                reverse=True,
            )
            return {
                "status": "completed",
                "region": "chest",
                "model_version": self.model_version,
                "prediction": None,
                # Keep the existing field for response compatibility, but make
                # clear these are ranked scores, not confirmed findings.
                "findings": ranked_scores,
                "scores": ranked_scores,
                "confidence_if_valid": None,
                "uncertainty": (
                    "Research-model scores only; they are not calibrated probabilities, "
                    "confirmed findings, or a diagnosis. No validated local decision threshold "
                    "is configured, so no score is marked as a positive/actionable finding. "
                    "A qualified clinician/radiologist must interpret the image."
                ),
                "explainability_artifact": None,
                "message": (
                    "Chest X-ray model scores were generated for multiple labels. "
                    "They are unthresholded research-model scores, not positive findings, "
                    "and require clinical/radiological review."
                ),
            }
        except Exception as exc:
            logger.warning("TorchXRayVision inference failed: %s", exc)
            return self._unavailable(
                "chest", "inference_failed",
                "Inference could not be completed. No prediction is available.",
                self.model_version,
            )

    def _analyze_resnet(self, image_path: str, region: str = "chest", classes: Optional[list[str]] = None) -> dict[str, Any]:
        tensor = self.preprocess_image(image_path)
        classes = list(classes or self._active_classes or CHEST_CLASSES)
        if tensor is None:
            return self._unavailable(
                region, "preprocessing_failed",
                "Unable to decode and preprocess image pixels.", self.model_version,
                "Image preprocessing failed. Please verify the image file format.",
            )
        try:
            with torch.inference_mode():
                outputs = self._model(tensor)
                if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
                    raise ValueError("Model output must be shaped [batch, classes].")
                if outputs.shape[0] != 1 or outputs.shape[1] != len(classes):
                    raise ValueError(
                        f"Model output shape {tuple(outputs.shape)} does not match "
                        f"one image and {len(classes)} configured classes for {region}."
                    )
                probs_tensor = torch.softmax(outputs, dim=1)[0]
                if not torch.isfinite(probs_tensor).all():
                    raise ValueError("Model produced non-finite class scores.")
                probs = probs_tensor.cpu().numpy()
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise ValueError("Model produced an invalid confidence score.")
                finding = classes[pred_idx]

            artifact = None
            try:
                from app.ai.gradcam import generate_gradcam
                gradcam_result = generate_gradcam(self._model, image_path, pred_idx)
                if isinstance(gradcam_result, dict):
                    artifact = gradcam_result.get("artifact_path")
            except Exception as exc:
                logger.warning("Grad-CAM generation failed; retaining inference result: %s", exc)

            return {
                "status": "completed",
                "region": region,
                "model_version": self.model_version,
                "prediction": finding,
                "findings": [{"label": finding, "model_score": round(confidence, 5)}],
                "confidence_if_valid": f"{confidence * 100:.1f}%",
                "uncertainty": (
                    "Assistive model output only. This score is not evidence of clinical accuracy; "
                    "clinical/radiological correlation is required."
                ),
                "explainability_artifact": artifact,
                "message": f"ResNet model output: {finding}. Qualified clinical review is required.",
            }
        except Exception as exc:
            logger.warning("ResNet inference failed: %s", exc)
            return self._unavailable(
                region, "inference_failed",
                "Inference could not be completed. No prediction or confidence score is available.",
                self.model_version,
            )


_xray_model_service: Optional[XRayModelService] = None


def get_xray_model_service() -> XRayModelService:
    """Return singleton instance of XRayModelService."""
    global _xray_model_service
    if _xray_model_service is None:
        _xray_model_service = XRayModelService()
    return _xray_model_service


def reset_xray_model_service() -> None:
    """Reset singleton instance for test isolation."""
    global _xray_model_service
    _xray_model_service = None
