"""X-ray body-region router and region-aware model registry for MediSense AI.

Defines canonical anatomical regions, alias normalizations, model architecture specifications,
expected preprocessing, class mappings, checkpoint paths, and availability disclosures.
Goal 1 update: broad classifier labels are recognized without being routed to narrower regional models.
Enforces strict medical-safety boundaries: only anatomical regions with verified, compatible
trained models are marked operational.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import math
import os
import re
import threading
from typing import Any, Optional

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Supported regions for computer-vision model evaluation
SUPPORTED_REGIONS: set[str] = {"chest"}  # Updated after registry configuration

MURA_WRIST_CLASSES = ["No abnormality detected", "Abnormality detected"]
MURA_WRIST_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "models" / "xray" / "wrist" / "XR_WRIST_FINAL_best.pt"
)
MURA_REGION_CHECKPOINTS = {
    "elbow": ("XR_ELBOW", "XR_ELBOW_FINAL_best.pt"),
    "finger": ("XR_FINGER", "XR_FINGER_FINAL_best.pt"),
    "forearm": ("XR_FOREARM", "XR_FOREARM_FINAL_best.pt"),
    "hand": ("XR_HAND", "XR_HAND_FINAL_best.pt"),
    "humerus": ("XR_HUMERUS", "XR_HUMERUS_FINAL_best.pt"),
    "shoulder": ("XR_SHOULDER", "XR_SHOULDER_FINAL_best.pt"),
}
MURA_BINARY_CLASSES = ["Normal", "Abnormal"]
MURA_PREPROCESSING = {
    "target_size": [224, 224],
    "color_mode": "RGB",
    "pixel_scaling": "1/255.0",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "interpolation": "bilinear",
    "exif_orientation_correction": True,
}

def _mura_checkpoint(region: str) -> str:
    """Return the region-specific MURA checkpoint path used by the inference service."""
    filename = MURA_REGION_CHECKPOINTS[region][1]
    return str(Path(__file__).resolve().parents[2] / "models" / "xray" / "mura_regions" / filename)


CHEST_CLASSES = [
    "Normal / Clear",
    "Consolidation / Opacity",
    "Pneumothorax",
    "Pleural Effusion",
    "Cardiomegaly",
]


# TorchXRayVision DenseNet121 label set. These are research-model outputs, not
# calibrated probabilities or a clinically validated diagnosis.
TORCHXRAYVISION_CLASSES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture",
    "Lung Opacity", "Enlarged Cardiomediastinum",
]


def _configured_model_name() -> str:
    """Return the model selected by environment/configuration, conservatively."""
    try:
        settings = get_settings()
        value = getattr(settings, "xray_default_model", None)
    except Exception:
        value = None
    return str(value or os.getenv("XRAY_DEFAULT_MODEL", "chest_resnet18")).strip().lower()


def _uses_torchxrayvision() -> bool:
    return _configured_model_name() in {
        "chest_densenet121", "densenet121", "torchxrayvision",
        "torch_xray_vision", "xrv_densenet121",
    }


def verify_checkpoint_integrity(
    checkpoint_path: Path, expected_classes: int = len(CHEST_CLASSES)
) -> tuple[bool, str]:
    """Safely validate checkpoint structure and classifier size.

    Only weights-only deserialization is allowed. Checkpoint validation does not
    establish model accuracy, training provenance, or clinical suitability.
    """
    try:
        if not checkpoint_path.exists():
            return False, f"Checkpoint file missing: {checkpoint_path}"
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            return False, f"Checkpoint is not a non-empty file: {checkpoint_path}"

        import torch
        import torch.nn as nn

        try:
            loaded = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        except Exception as load_exc:
            return False, (
                "Safe checkpoint loading failed. Use a compatible weights-only state_dict; "
                f"details: {load_exc}"
            )

        if isinstance(loaded, nn.Module):
            classifier = getattr(loaded, "fc", None)
            out_features = getattr(classifier, "out_features", None)
            if out_features is not None and int(out_features) != expected_classes:
                return False, (
                    f"Checkpoint output dimension mismatch: model has {out_features} "
                    f"classes, expected {expected_classes}."
                )
            return True, f"Checkpoint module structurally verified: {checkpoint_path}"

        if not isinstance(loaded, dict) or not loaded:
            return False, (
                f"Unsupported checkpoint object type: {type(loaded).__name__}; "
                "expected a state_dict or checkpoint dictionary."
            )

        # MURA checkpoints use {"model_state": ..., "area": "XR_*", ...} and
        # a one-logit sigmoid head. Structural validation applies to each region.
        mura_areas = {"XR_WRIST", *(area for area, _ in MURA_REGION_CHECKPOINTS.values())}
        if loaded.get("area") in mura_areas and isinstance(loaded.get("model_state"), dict):
            state = loaded["model_state"]
            weight = state.get("classifier.weight")
            bias = state.get("classifier.bias")
            if getattr(weight, "shape", None) != (1, 1024):
                return False, f"MURA {loaded.get('area')} checkpoint classifier.weight must have shape (1, 1024)."
            if getattr(bias, "shape", None) != (1,):
                return False, f"MURA {loaded.get('area')} checkpoint classifier.bias must have shape (1,)."
            if not any(k.startswith("features.") for k in state if isinstance(k, str)):
                return False, f"MURA {loaded.get('area')} checkpoint is missing DenseNet121 feature weights."
            return True, f"MURA {loaded.get('area')} DenseNet121 checkpoint structurally verified: {checkpoint_path}"

        state_dict = loaded.get("state_dict", loaded.get("model_state_dict", loaded))
        if not isinstance(state_dict, dict) or not state_dict:
            return False, "Checkpoint does not contain a non-empty state_dict."

        normalized = {
            (key[7:] if isinstance(key, str) and key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        if not all(isinstance(key, str) for key in normalized):
            return False, "Checkpoint state_dict contains invalid parameter keys."

        for key in ("fc.weight", "fc.bias"):
            if key in normalized:
                value = normalized[key]
                shape = getattr(value, "shape", None)
                if shape is None or len(shape) < 1:
                    return False, f"Checkpoint parameter {key} is not tensor-like."
                if int(shape[0]) != expected_classes:
                    return False, (
                        f"Checkpoint output dimension mismatch: {key} has {shape[0]} "
                        f"classes, expected {expected_classes}."
                    )
        has_tensor_weights = any(
            hasattr(value, "shape") and hasattr(value, "dtype")
            for value in normalized.values()
        )
        if not has_tensor_weights:
            return False, "Checkpoint state_dict contains no tensor weights."

        return True, f"Checkpoint state_dict structurally verified: {checkpoint_path}"
    except Exception as exc:
        return False, f"Checkpoint integrity check failed: {type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class XRayModelRegistryEntry:
    canonical_region: str
    display_name: str
    supported_modality: str
    checkpoint_location: Optional[str]
    model_architecture: Optional[str]
    class_label_mapping: list[str]
    expected_preprocessing: Optional[dict[str, Any]]
    model_version: Optional[str]
    source_provenance: Optional[str]
    supported_prediction_task: str
    is_supported: bool
    validation_status: str

    def check_availability(self) -> tuple[bool, str]:
        """Check that this region has a configured compatible model checkpoint.

        A registry flag by itself is never enough to mark a model available. Chest
        may use TorchXRayVision or XRAY_CHEST_CHECKPOINT; other regions require an
        explicit checkpoint path, architecture and non-empty class-label mapping.
        """
        if not self.is_supported:
            return False, (
                f"Automated interpretation for {self.canonical_region} radiography is currently unsupported "
                "because no compatible trained model is configured for this region."
            )

        settings = get_settings()
        architecture = (self.model_architecture or "").strip().lower()
        labels = self.class_label_mapping or []
        if not architecture:
            return False, "model_architecture_not_configured: No model architecture is configured."
        if not labels:
            return False, "class_labels_not_configured: No output class labels are configured."

        if self.canonical_region == "chest" and _uses_torchxrayvision():
            try:
                import torchxrayvision  # noqa: F401
            except Exception as exc:
                return False, (
                    "torchxrayvision_unavailable: TorchXRayVision could not be imported. "
                    f"Install/repair the package before selecting DenseNet121: {type(exc).__name__}: {exc}"
                )
            return True, (
                "TorchXRayVision DenseNet121 package is importable. Runtime model loading and inference "
                "must still succeed; outputs are research scores, not a clinical diagnosis."
            )

        if self.canonical_region == "wrist" and architecture in {
            "mura_densenet121", "densenet121", "mura-densenet121"
        }:
            pass
        elif self.canonical_region in MURA_REGION_CHECKPOINTS and architecture in {
            "mura_densenet121", "densenet121", "mura-densenet121"
        }:
            pass
        elif self.canonical_region != "chest" and "resnet18" not in architecture:
            return False, (
                "unsupported_model_architecture: This inference service supports configured ResNet18 "
                "state-dict models and region-specific MURA DenseNet121 checkpoints for wrist, elbow, "
                "finger, forearm, hand, humerus and shoulder."
            )

        # Per-region environment overrides take precedence over cached settings.
        # A checkpoint alone is insufficient: architecture and labels must also be
        # configured and must match the checkpoint's training output order.
        configured_path = (os.getenv(f"XRAY_{self.canonical_region.upper()}_CHECKPOINT", "") or "").strip()
        if self.canonical_region == "chest" and not configured_path:
            configured_path = (os.getenv("XRAY_CHEST_CHECKPOINT", "") or "").strip()
        if self.canonical_region == "chest" and not configured_path:
            configured_path = (getattr(settings, "xray_chest_checkpoint", "") or "").strip()
        if configured_path:
            p = Path(configured_path).expanduser()
            if not p.is_absolute() and not p.exists():
                p = Path(settings.xray_model_path) / p
        elif self.checkpoint_location:
            p = Path(self.checkpoint_location).expanduser()
            if not p.is_absolute():
                p = Path(settings.xray_model_path) / p
        else:
            return False, "checkpoint_path_not_configured: No checkpoint path is configured for this region."

        try:
            p = p.resolve()
        except (OSError, RuntimeError):
            p = p.absolute()
        if not p.exists():
            return False, f"checkpoint_path_not_found: Trained checkpoint was not found at {p}."
        if not p.is_file() or p.stat().st_size == 0:
            return False, f"checkpoint_path_invalid: Checkpoint is not a non-empty file: {p}"

        valid, reason = verify_checkpoint_integrity(p, expected_classes=len(labels))
        if not valid:
            return False, f"checkpoint_integrity_failed: {reason}"
        return True, f"Checkpoint structurally verified: {p}"

    def to_dict(self) -> dict[str, Any]:
        available, avail_msg = self.check_availability()
        d = asdict(self)
        if self.canonical_region == "chest" and _uses_torchxrayvision():
            d["model_architecture"] = "TorchXRayVision DenseNet121"
            d["class_label_mapping"] = list(TORCHXRAYVISION_CLASSES)
            d["expected_preprocessing"] = {
                "target_size": [224, 224],
                "color_mode": "grayscale",
                "input_shape": [1, 224, 224],
                "preprocessing": "TorchXRayVision load_image + XRayCenterCrop + XRayResizer",
                "normalization": "TorchXRayVision-specific; not ImageNet RGB normalization",
            }
            d["model_version"] = "torchxrayvision_densenet121_res224_all"
            d["source_provenance"] = (
                "TorchXRayVision published pretrained research model; local clinical validation "
                "and suitability have not been established"
            )
            d["supported_prediction_task"] = (
                "Multi-label chest radiograph research scoring across 18 labels; not a diagnosis"
            )
            d["validation_status"] = "RESEARCH_MODEL_NOT_CLINICALLY_VALIDATED"
        d["is_available"] = available
        d["availability_message"] = avail_msg
        if available:
            d["availability_status"] = "AVAILABLE"
        elif not self.is_supported:
            d["availability_status"] = "UNSUPPORTED_REGION"
        else:
            message = str(avail_msg).lower()
            if "torchxrayvision_unavailable" in message:
                d["availability_status"] = "UNAVAILABLE_TORCHXRAYVISION"
            elif "checkpoint_path_not_found" in message:
                d["availability_status"] = "UNAVAILABLE_MISSING_CHECKPOINT"
            elif "checkpoint_path_invalid" in message or "checkpoint_path_unreadable" in message:
                d["availability_status"] = "UNAVAILABLE_INVALID_CHECKPOINT_PATH"
            elif "checkpoint_integrity_failed" in message:
                d["availability_status"] = "UNAVAILABLE_CHECKPOINT_INTEGRITY_FAILED"
            else:
                d["availability_status"] = "UNAVAILABLE_CHECKPOINT_VERIFICATION_FAILED"
        return d


# Preprocessing expected by 224x224 convolutional neural networks
STANDARD_PREPROCESSING = {
    "target_size": [224, 224],
    "color_mode": "RGB",
    "pixel_scaling": "1/255.0",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "interpolation": "INTER_LINEAR",
    "exif_orientation_correction": True,
}

# Master Model Registry
XRAY_MODEL_REGISTRY: dict[str, XRayModelRegistryEntry] = {
    "chest": XRayModelRegistryEntry(
        canonical_region="chest",
        display_name="Chest (Thorax)",
        supported_modality="radiograph_image",
        checkpoint_location="chest/model.pt",
        model_architecture="ResNet18",
        class_label_mapping=CHEST_CLASSES,
        expected_preprocessing=STANDARD_PREPROCESSING,
        model_version="chest_resnet18@v1.0",
        source_provenance="Benchmark references only; actual checkpoint provenance requires independent verification",
        supported_prediction_task="Thoracic pathology classification (Normal vs Consolidation/Pneumothorax/Effusion/Cardiomegaly)",
        is_supported=True,
        validation_status="PENDING_VERIFIED_CHECKPOINT",
    ),
    "wrist": XRayModelRegistryEntry(
        canonical_region="wrist",
        display_name="Wrist",
        supported_modality="radiograph_image",
        checkpoint_location=str(MURA_WRIST_CHECKPOINT),
        model_architecture="mura_densenet121",
        class_label_mapping=MURA_WRIST_CLASSES,
        expected_preprocessing={
            "target_size": [224, 224],
            "color_mode": "RGB",
            "pixel_scaling": "1/255.0",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bilinear",
            "exif_orientation_correction": True,
        },
        model_version="MURA_XR_WRIST_DenseNet121_research_checkpoint",
        source_provenance="MURA wrist research checkpoint; provenance/performance not independently validated by MediSense AI",
        supported_prediction_task="Binary wrist radiograph abnormality screening score; not fracture-specific and not a diagnosis",
        is_supported=True,
        validation_status="RESEARCH_MODEL_NOT_CLINICALLY_VALIDATED",
    ),
    "hand": XRayModelRegistryEntry(
        canonical_region="hand",
        display_name="Hand",
        supported_modality="radiograph_image",
        checkpoint_location=_mura_checkpoint("hand"),
        model_architecture="mura_densenet121",
        class_label_mapping=MURA_BINARY_CLASSES,
        expected_preprocessing=MURA_PREPROCESSING,
        model_version="MURA_XR_HAND_DenseNet121_research_checkpoint",
        source_provenance="MURA region-specific research checkpoint; performance not independently validated by MediSense AI",
        supported_prediction_task="Hand radiograph binary normal/abnormal research screening score; not a diagnosis",
        is_supported=True,
        validation_status="RESEARCH_MODEL_NOT_CLINICALLY_VALIDATED",
    ),
    "forearm": XRayModelRegistryEntry(
        canonical_region="forearm",
        display_name="Forearm",
        supported_modality="radiograph_image",
        checkpoint_location=_mura_checkpoint("forearm"),
        model_architecture="mura_densenet121",
        class_label_mapping=MURA_BINARY_CLASSES,
        expected_preprocessing=MURA_PREPROCESSING,
        model_version="MURA_XR_FOREARM_DenseNet121_research_checkpoint",
        source_provenance="MURA region-specific research checkpoint; performance not independently validated by MediSense AI",
        supported_prediction_task="Forearm radiograph binary normal/abnormal research screening score; not a diagnosis",
        is_supported=True,
        validation_status="RESEARCH_MODEL_NOT_CLINICALLY_VALIDATED",
    ),
    "elbow": XRayModelRegistryEntry(
        canonical_region="elbow",
        display_name="Elbow",
        supported_modality="radiograph_image",
        checkpoint_location=_mura_checkpoint("elbow"),
        model_architecture="mura_densenet121",
        class_label_mapping=MURA_BINARY_CLASSES,
        expected_preprocessing=MURA_PREPROCESSING,
        model_version="MURA_XR_ELBOW_DenseNet121_research_checkpoint",
        source_provenance="MURA region-specific research checkpoint; performance not independently validated by MediSense AI",
        supported_prediction_task="Elbow radiograph binary normal/abnormal research screening score; not a diagnosis",
        is_supported=True,
        validation_status="RESEARCH_MODEL_NOT_CLINICALLY_VALIDATED",
    ),
    "shoulder": XRayModelRegistryEntry(
        canonical_region="shoulder",
        display_name="Shoulder",
        supported_modality="radiograph_image",
        checkpoint_location=_mura_checkpoint("shoulder"),
        model_architecture="mura_densenet121",
        class_label_mapping=MURA_BINARY_CLASSES,
        expected_preprocessing=MURA_PREPROCESSING,
        model_version="MURA_XR_SHOULDER_DenseNet121_research_checkpoint",
        source_provenance="MURA region-specific research checkpoint; performance not independently validated by MediSense AI",
        supported_prediction_task="Shoulder radiograph binary normal/abnormal research screening score; not a diagnosis",
        is_supported=True,
        validation_status="RESEARCH_MODEL_NOT_CLINICALLY_VALIDATED",
    ),
    "humerus": XRayModelRegistryEntry(
        canonical_region="humerus",
        display_name="Humerus",
        supported_modality="radiograph_image",
        checkpoint_location=_mura_checkpoint("humerus"),
        model_architecture="mura_densenet121",
        class_label_mapping=MURA_BINARY_CLASSES,
        expected_preprocessing=MURA_PREPROCESSING,
        model_version="MURA_XR_HUMERUS_DenseNet121_research_checkpoint",
        source_provenance="MURA region-specific research checkpoint; performance not independently validated by MediSense AI",
        supported_prediction_task="Humerus radiograph binary normal/abnormal research screening score; not a diagnosis",
        is_supported=True,
        validation_status="RESEARCH_MODEL_NOT_CLINICALLY_VALIDATED",
    ),
    "clavicle": XRayModelRegistryEntry(
        canonical_region="clavicle",
        display_name="Clavicle",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Musculoskeletal benchmark (unconfigured)",
        supported_prediction_task="Clavicular fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "knee": XRayModelRegistryEntry(
        canonical_region="knee",
        display_name="Knee",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="OAI Osteoarthritis Initiative benchmark (unconfigured)",
        supported_prediction_task="Knee joint osteoarthritis / joint space narrowing grading (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "lower leg": XRayModelRegistryEntry(
        canonical_region="lower leg",
        display_name="Lower Leg (Tibia/Fibula)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Musculoskeletal benchmark (unconfigured)",
        supported_prediction_task="Tibial/fibular fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "ankle": XRayModelRegistryEntry(
        canonical_region="ankle",
        display_name="Ankle",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Musculoskeletal benchmark (unconfigured)",
        supported_prediction_task="Ankle mortise fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "foot": XRayModelRegistryEntry(
        canonical_region="foot",
        display_name="Foot",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Musculoskeletal benchmark (unconfigured)",
        supported_prediction_task="Metatarsal / phalangeal fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "pelvis": XRayModelRegistryEntry(
        canonical_region="pelvis",
        display_name="Pelvis",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Pelvic benchmark (unconfigured)",
        supported_prediction_task="Pelvic ring disruption detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "hip": XRayModelRegistryEntry(
        canonical_region="hip",
        display_name="Hip",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Hip fracture benchmark (unconfigured)",
        supported_prediction_task="Femoral neck / intertrochanteric fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "femur": XRayModelRegistryEntry(
        canonical_region="femur",
        display_name="Femur",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Musculoskeletal benchmark (unconfigured)",
        supported_prediction_task="Femoral shaft fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "spine": XRayModelRegistryEntry(
        canonical_region="spine",
        display_name="Spine (Vertebral Column)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Spine radiograph benchmark (unconfigured)",
        supported_prediction_task="Vertebral compression fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "cervical spine": XRayModelRegistryEntry(
        canonical_region="cervical spine",
        display_name="Cervical Spine (Neck)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Cervical spine radiograph benchmark (unconfigured)",
        supported_prediction_task="Cervical fracture / subluxation detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "thoracic spine": XRayModelRegistryEntry(
        canonical_region="thoracic spine",
        display_name="Thoracic Spine (Mid-Back)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Thoracic spine benchmark (unconfigured)",
        supported_prediction_task="Thoracic vertebral compression detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "lumbar spine": XRayModelRegistryEntry(
        canonical_region="lumbar spine",
        display_name="Lumbar Spine (Lower Back)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Lumbar spine radiograph benchmark (unconfigured)",
        supported_prediction_task="Lumbar spondylolisthesis / compression detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "sacrum": XRayModelRegistryEntry(
        canonical_region="sacrum",
        display_name="Sacrum & Coccyx",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Sacrococcygeal benchmark (unconfigured)",
        supported_prediction_task="Sacral / coccygeal fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "skull": XRayModelRegistryEntry(
        canonical_region="skull",
        display_name="Skull (Cranium)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Cranial radiograph benchmark (unconfigured)",
        supported_prediction_task="Cranial vault fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "facial bones": XRayModelRegistryEntry(
        canonical_region="facial bones",
        display_name="Facial Bones",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Maxillofacial benchmark (unconfigured)",
        supported_prediction_task="Orbital / zygomatic fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "mandible": XRayModelRegistryEntry(
        canonical_region="mandible",
        display_name="Mandible (Jaw)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Dental / mandibular benchmark (unconfigured)",
        supported_prediction_task="Mandibular fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "sinuses": XRayModelRegistryEntry(
        canonical_region="sinuses",
        display_name="Paranasal Sinuses",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Sinus radiograph benchmark (unconfigured)",
        supported_prediction_task="Sinus opacification / fluid level detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "neck": XRayModelRegistryEntry(
        canonical_region="neck",
        display_name="Soft Tissue Neck",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Cervical airway benchmark (unconfigured)",
        supported_prediction_task="Upper airway narrowing detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "ribs": XRayModelRegistryEntry(
        canonical_region="ribs",
        display_name="Ribs (Bony Thorax)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Thoracic cage benchmark (unconfigured)",
        supported_prediction_task="Rib fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "sternum": XRayModelRegistryEntry(
        canonical_region="sternum",
        display_name="Sternum (Breastbone)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Thoracic skeleton benchmark (unconfigured)",
        supported_prediction_task="Sternal body fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "finger": XRayModelRegistryEntry(
        canonical_region="finger",
        display_name="Finger / Thumb",
        supported_modality="radiograph_image",
        checkpoint_location=_mura_checkpoint("finger"),
        model_architecture="mura_densenet121",
        class_label_mapping=MURA_BINARY_CLASSES,
        expected_preprocessing=MURA_PREPROCESSING,
        model_version="MURA_XR_FINGER_DenseNet121_research_checkpoint",
        source_provenance="MURA finger research checkpoint; performance not independently validated by MediSense AI",
        supported_prediction_task="Finger radiograph binary normal/abnormal research screening score; not a diagnosis",
        is_supported=True,
        validation_status="RESEARCH_MODEL_NOT_CLINICALLY_VALIDATED",
    ),
    "fingers": XRayModelRegistryEntry(
        canonical_region="fingers",
        display_name="Fingers & Thumb",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Digital musculoskeletal benchmark (unconfigured)",
        supported_prediction_task="Phalangeal fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "abdomen": XRayModelRegistryEntry(
        canonical_region="abdomen",
        display_name="Abdomen (KUB)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Abdominal radiograph benchmark (unconfigured)",
        supported_prediction_task="Pneumoperitoneum / bowel obstruction detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "calcaneus": XRayModelRegistryEntry(
        canonical_region="calcaneus",
        display_name="Calcaneus (Heel)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Hindfoot benchmark (unconfigured)",
        supported_prediction_task="Calcaneal fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "toes": XRayModelRegistryEntry(
        canonical_region="toes",
        display_name="Toes (Digits of Foot)",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Forefoot benchmark (unconfigured)",
        supported_prediction_task="Toe phalangeal fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
}

def _apply_environment_model_overrides() -> None:
    """Enable a region only when a real checkpoint, architecture and labels are supplied.

    For each canonical region, configure:
      XRAY_<REGION>_CHECKPOINT=path/to/weights.pt
      XRAY_<REGION>_MODEL_ARCHITECTURE=ResNet18
      XRAY_<REGION>_CLASSES=Class A,Class B
      XRAY_<REGION>_MODEL_VERSION=dataset-and-version

    The labels must be the exact ordered output labels used to train the checkpoint.
    This configuration does not train a model or validate clinical performance.
    """
    global SUPPORTED_REGIONS
    configured = {"chest"} | {
        region for region, entry in XRAY_MODEL_REGISTRY.items() if entry.is_supported
    }
    for region, entry in list(XRAY_MODEL_REGISTRY.items()):
        prefix = f"XRAY_{region.upper()}_"
        checkpoint = (os.getenv(prefix + "CHECKPOINT", "") or "").strip()
        architecture = (os.getenv(prefix + "MODEL_ARCHITECTURE", "") or "").strip()
        raw_classes = (os.getenv(prefix + "CLASSES", "") or "").strip()
        classes = [label.strip() for label in raw_classes.split(",") if label.strip()]
        version = (os.getenv(prefix + "MODEL_VERSION", "") or "").strip() or None

        # Never infer labels from a filename or auto-enable a region from its name.
        if not (checkpoint and architecture and classes):
            continue
        normalized_architecture = architecture.lower().replace("-", "_")
        if normalized_architecture in {"mura_densenet121", "densenet121", "mura-densenet121"} and (region == "wrist" or region in MURA_REGION_CHECKPOINTS):
            configured_architecture = "mura_densenet121"
            preprocessing = MURA_PREPROCESSING
        elif "resnet18" in normalized_architecture:
            configured_architecture = "ResNet18"
            preprocessing = STANDARD_PREPROCESSING
        else:
            logger.warning("Ignoring %s override: architecture %r is not implemented for this region.", region, architecture)
            continue
        XRAY_MODEL_REGISTRY[region] = replace(
            entry,
            checkpoint_location=checkpoint,
            model_architecture=configured_architecture,
            class_label_mapping=classes,
            expected_preprocessing=preprocessing,
            model_version=version or f"{region}_resnet18_env_configured",
            source_provenance=(os.getenv(prefix + "SOURCE", "") or "User-configured checkpoint; provenance not independently verified"),
            is_supported=True,
            validation_status="CONFIGURED_PENDING_VALIDATION",
        )
        configured.add(region)
    SUPPORTED_REGIONS = configured


_apply_environment_model_overrides()

# Alias resolution mapping
REGION_ALIASES: dict[str, str] = {
    # Chest aliases
    "chest": "chest",
    "cxr": "chest",
    "chest x-ray": "chest",
    "chest xray": "chest",
    "chest-xray": "chest",
    "chest radiograph": "chest",
    "thorax": "chest",
    "lungs": "chest",
    "pulmonary": "chest",
    # Broad anatomy labels are recognized but must not select a specific regional model.
    "extremity": "extremity",
    "upper extremity": "upper extremity",
    "lower extremity": "lower extremity",
    "head region": "head region",
    "leg": "lower extremity",
    # Exact classifier labels with no configured compatible regional model.
    "nasopharynx": "nasopharynx",
    "si joint": "sacroiliac joint",
    "sacroiliac joint": "sacroiliac joint",
    "temporal bone": "temporal bone",
    "pns": "sinuses",
    # Wrist aliases
    "wrist": "wrist",
    "wrist x-ray": "wrist",
    "wrist xray": "wrist",
    "carpus": "wrist",
    "carpal": "wrist",
    # Hand aliases
    "hand": "hand",
    "hand x-ray": "hand",
    "hand xray": "hand",
        "metacarpal": "hand",
    # Forearm
    "forearm": "forearm",
    "radius": "forearm",
    "ulna": "forearm",
    # Elbow
    "elbow": "elbow",
    # Shoulder
    "shoulder": "shoulder",
    "shoulder joint": "shoulder",
    # Humerus
    "humerus": "humerus",
    # The explicit phrase "upper arm" denotes the humerus region. Keep the
    # generic term "arm" broad so it cannot silently select a humerus model.
    "upper arm": "humerus",
    # "arm" is broad; do not silently treat the whole arm as a humerus X-ray.
    "arm": "upper extremity",
    # Clavicle
    "clavicle": "clavicle",
    "collarbone": "clavicle",
    # Knee
    "knee": "knee",
    "knee joint": "knee",
    "patella": "knee",
    # Lower leg
    "lower leg": "lower leg",
    "lower-leg": "lower leg",
    "tibia": "lower leg",
    "fibula": "lower leg",
    "shin": "lower leg",
    # Ankle
    "ankle": "ankle",
    "ankle joint": "ankle",
    # Foot
    "foot": "foot",
    "feet": "foot",
        "metatarsal": "foot",
    # Pelvis
    "pelvis": "pelvis",
    "pelvic": "pelvis",
    # Hip
    "hip": "hip",
    "hip joint": "hip",
    # Femur
    "femur": "femur",
    "thigh": "femur",
    # Spine
    "spine": "spine",
    "vertebra": "spine",
    "back": "spine",
    "cervical spine": "cervical spine",
    "c-spine": "cervical spine",
    "cervical": "cervical spine",
    "neck spine": "cervical spine",
    "thoracic spine": "thoracic spine",
    "t-spine": "thoracic spine",
    "dorsal spine": "thoracic spine",
    "lumbar spine": "lumbar spine",
    "l-spine": "lumbar spine",
    "lumbar": "lumbar spine",
    "lumbosacral": "spine",
    "lower back": "spine",
    "low back": "spine",
    "sacrum": "sacrum",
    "coccyx": "sacrum",
    "tailbone": "sacrum",
    "sacrococcygeal": "sacrum",
    # Head & Neck
    # The catalogue's generic Head search option resolves to the Skull study.
    # Classifier-predicted HEAD remains broad via _ANATOMY_CLASS_BROAD_LABELS.
    "head": "skull",
    "skull": "skull",
    "cranium": "skull",
    "calvarium": "skull",
    "head x-ray": "skull",
    "head xray": "skull",
    "skull x-ray": "skull",
    "skull xray": "skull",
    "facial bones": "facial bones",
    "facial bone": "facial bones",
    "face": "facial bones",
    "facial": "facial bones",
    "orbits": "facial bones",
    "orbit": "facial bones",
    "zygoma": "facial bones",
    "cheekbone": "facial bones",
    "mandible": "mandible",
    "jaw": "mandible",
    "chin": "mandible",
    "tmj": "mandible",
    "sinuses": "sinuses",
    "sinus": "sinuses",
    "paranasal sinuses": "sinuses",
    "neck": "neck",
    "throat": "neck",
    "adenoids": "neck",
    # Ribs & Sternum
    "ribs": "ribs",
    "rib": "ribs",
    "ribcage": "ribs",
    "bony thorax": "ribs",
    "sternum": "sternum",
    "breastbone": "sternum",
    # Digits
    "fingers": "finger",
    "finger": "finger",
    "thumb": "finger",
    "digits": "finger",
    "toes": "toes",
    "toe": "toes",
    "calcaneus": "calcaneus",
    "heel": "calcaneus",
    "heel bone": "calcaneus",
    # Abdomen
    "abdomen": "abdomen",
    "belly": "abdomen",
    "stomach": "abdomen",
    "tummy": "abdomen",
    "kub": "abdomen",
    "bowel": "abdomen",
}


def normalize_region_name(raw_region: Optional[str]) -> tuple[str, bool]:
    """Normalize a region without silently choosing between multiple body parts.

    Exact aliases are preferred. A descriptive phrase is accepted only when its
    recognized anatomical aliases resolve to one canonical region. If a phrase
    mentions more than one region (for example, "chest and knee X-ray"), it is
    rejected instead of routing it to whichever alias happens to be checked first.
    """
    if raw_region is None:
        return "unknown", False

    cleaned = str(raw_region).strip().lower().replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "unknown", False

    if cleaned in REGION_ALIASES:
        return REGION_ALIASES[cleaned], True
    if cleaned in XRAY_MODEL_REGISTRY:
        return cleaned, True

    matches: list[tuple[int, str]] = []
    for alias in sorted(REGION_ALIASES, key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        if re.search(pattern, cleaned):
            matches.append((len(alias), REGION_ALIASES[alias]))

    # Reject phrases that refer to multiple distinct anatomical regions.
    canonical_matches = {canonical for _, canonical in matches}
    if len(canonical_matches) == 1:
        return next(iter(canonical_matches)), True
    if len(canonical_matches) > 1:
        logger.warning(
            "Ambiguous X-ray region phrase rejected: '%s' -> %s",
            raw_region,
            sorted(canonical_matches),
        )
        return "unknown", False

    return cleaned, False



def get_model_registry_entry(region: str) -> Optional[XRayModelRegistryEntry]:
    """Retrieve the model registry entry for a canonical or aliased region."""
    canonical, _ = normalize_region_name(region)
    return XRAY_MODEL_REGISTRY.get(canonical)


def get_all_registry_entries() -> list[dict[str, Any]]:
    """Return serialized status dictionary for all registered anatomical regions."""
    return [entry.to_dict() for entry in XRAY_MODEL_REGISTRY.values()]


# Some labels in the 33-class anatomy classifier describe a broad area rather than
# a specific study compatible with one of the configured regional disease models.
# These must be displayed honestly without routing to a narrower anatomy model.
_ANATOMY_CLASS_BROAD_LABELS: dict[str, str] = {
    "extremity": "extremity",
    "upper extremity": "upper extremity",
    "lower extremity": "lower extremity",
    "arm": "upper extremity",
    "leg": "lower extremity",
    "head": "head region",
    # The classifier's generic SPINE class does not distinguish cervical,
    # thoracic, or lumbar studies and must not select a narrower spine model.
    "spine": "spine",
}


def _normalize_anatomy_classifier_label(label: Any) -> tuple[str, bool, bool]:
    """Normalize a 33-class model label without narrowing broad anatomy.

    Returns (canonical_region, recognized, is_broad). The classifier's ARM/LEG/HEAD
    labels are deliberately not mapped to humerus/femur/skull because that would
    claim a more specific region than the model actually predicted.
    """
    cleaned = re.sub(r"\s+", " ", str(label or "").strip().lower().replace("_", " "))
    if not cleaned:
        return "unknown", False, False
    broad_region = _ANATOMY_CLASS_BROAD_LABELS.get(cleaned)
    if broad_region:
        return broad_region, True, True
    canonical, recognized = normalize_region_name(cleaned)
    return canonical, recognized, False


def _broad_anatomy_route_result(
    region: str,
    anatomy_result: dict[str, Any],
    confidence: Any,
    *,
    classifier_gate_passed: bool = False,
    is_broad_label: bool = True,
) -> dict[str, Any]:
    """Represent broad-label acceptance without routing to a specific region.

    This helper is called after the confidence and top-two ambiguity gates have
    been evaluated. A broad label can therefore be accepted as a broad-region
    classifier result, but it cannot establish a specific anatomy such as Wrist,
    Hand, Elbow, Skull, or Lumbar Spine and cannot select a regional model.
    """
    display = region.replace("_", " ").title()
    broad_accepted = bool(classifier_gate_passed and is_broad_label)
    try:
        confidence_label = f"{float(confidence):.1%}" if confidence is not None else "not available"
    except (TypeError, ValueError, OverflowError):
        confidence_label = "not available"

    if broad_accepted:
        message = (
            f"The automatic classifier accepted the broad region {display} "
            f"(score: {confidence_label}). It did not identify a specific body region. "
            "Select the correct specific region before regional interpretation. No regional disease model was run. "
            "The score is not a calibrated clinical probability."
        )
        verification_method = "classifier_confidence_and_margin_broad_region"
    else:
        message = (
            f"The classifier candidate {display} does not establish a specific X-ray region. "
            "Choose the correct specific body region before regional interpretation; no regional model was run."
        )
        verification_method = None

    return {
        # Keep the broad candidate available for display/debugging, but never
        # describe it as a detected specific anatomical region.
        "region": region,
        "canonical_region": region,
        "candidate_region": region,
        "detected_region": None,
        "supported": False,
        "is_supported": False,
        "is_available": False,
        "is_recognized": True,
        "is_broad_region": bool(is_broad_label),
        "region_source": "image_classifier",
        "classification_status": "completed",
        # Specific-site confirmation remains required even if the broad-label
        # classifier gate passed. This status must not authorize regional inference.
        "anatomy_verification_status": "confirmation_required",
        "anatomy_independently_verified": False,
        "anatomy_classifier_accepted": broad_accepted,
        "broad_region_classifier_accepted": broad_accepted,
        "specific_region_confirmation_required": True,
        "anatomy_verification_method": verification_method,
        "body_part_confidence": confidence,
        "anatomy_detection": anatomy_result,
        "reason_code": "broad_anatomy_label" if is_broad_label else "anatomy_specificity_unverified",
        "registry_entry": None,
        "message": message,
    }


# Lazy singleton: load the anatomy classifier only when an image is routed without
# a user-supplied region. This keeps import/startup of unrelated API modules cheap.
_BODY_PART_CLASSIFIER = None
_BODY_PART_CLASSIFIER_LOCK = threading.Lock()
_UPPER_EXTREMITY_CLASSIFIER = None
_UPPER_EXTREMITY_CLASSIFIER_LOCK = threading.Lock()


def _get_upper_extremity_classifier():
    """Return the optional 7-class upper-extremity anatomy classifier."""
    global _UPPER_EXTREMITY_CLASSIFIER
    if _UPPER_EXTREMITY_CLASSIFIER is None:
        with _UPPER_EXTREMITY_CLASSIFIER_LOCK:
            if _UPPER_EXTREMITY_CLASSIFIER is None:
                from app.ai.upper_extremity_anatomy_classifier import UpperExtremityAnatomyClassifier
                _UPPER_EXTREMITY_CLASSIFIER = UpperExtremityAnatomyClassifier()
    return _UPPER_EXTREMITY_CLASSIFIER


def _refine_broad_upper_extremity_candidate(
    image_path: str,
    anatomy_result: dict[str, Any],
) -> dict[str, Any]:
    """Try a dedicated 7-class sub-classifier only for broad upper-limb outputs.

    The primary 33-class model is weak at separating a wrist from the broad
    UPPER_EXTREMITY category. A separately trained 7-class classifier can refine
    that result. If its checkpoint is absent, fails, or does not pass the same
    provisional score/margin gates, the original broad prediction is preserved.
    No label is inferred from a filename or forcibly promoted.
    """
    raw_candidate = anatomy_result.get("candidate_body_part") or anatomy_result.get("body_part_label")
    candidate = str(raw_candidate or "").strip().upper().replace(" ", "_")
    if candidate not in {"UPPER_EXTREMITY", "ARM"}:
        return anatomy_result

    primary_snapshot = {
        "candidate_body_part": anatomy_result.get("candidate_body_part") or anatomy_result.get("body_part_label"),
        "confidence": anatomy_result.get("body_part_confidence"),
        "second_candidate_body_part": anatomy_result.get("second_candidate_body_part"),
        "second_candidate_confidence": anatomy_result.get("second_candidate_confidence"),
        "top_two_margin": anatomy_result.get("top_two_margin"),
        "top_predictions": anatomy_result.get("top_predictions", []),
        "model_name": anatomy_result.get("model_name"),
    }
    try:
        classifier = _get_upper_extremity_classifier()
        # Use the untouched uploaded image for the specialist classifier.
        # Annotation cleanup is not validated against this model's training pipeline
        # and must not be applied here behind the primary classifier's opt-in setting.
        # The original file is never modified.
        refined = classifier.predict_file(image_path)
        refined["annotation_cleanup"] = {
            "applied": False,
            "reason": "disabled_unvalidated_preprocessing",
        }
    except FileNotFoundError as exc:
        anatomy_result["upper_extremity_refinement"] = {
            "status": "unavailable",
            "reason_code": "upper_extremity_specialist_checkpoint_missing",
            "message": str(exc),
            "primary_candidate": primary_snapshot["candidate_body_part"],
            "top_predictions": [],
        }
        logger.info("Upper-extremity sub-classifier not installed; retaining broad candidate")
        return anatomy_result
    except Exception as exc:
        anatomy_result["upper_extremity_refinement"] = {
            "status": "failed",
            "reason_code": "upper_extremity_specialist_inference_failed",
            "message": f"{type(exc).__name__}: {exc}",
            "primary_candidate": primary_snapshot["candidate_body_part"],
            "top_predictions": [],
        }
        logger.exception("Upper-extremity sub-classification failed; retaining broad candidate")
        return anatomy_result

    accepted = bool(refined.get("accepted"))
    specialist_result = {
        "status": "accepted" if accepted else "unverified",
        "reason_code": None if accepted else "upper_extremity_specialist_below_gate",
        "primary_candidate": primary_snapshot["candidate_body_part"],
        "primary_confidence": primary_snapshot["confidence"],
        "candidate_body_part": refined.get("candidate_body_part"),
        "confidence": refined.get("confidence"),
        "second_candidate_body_part": refined.get("second_candidate_body_part"),
        "second_confidence": refined.get("second_confidence"),
        "top_two_margin": refined.get("top_two_margin"),
        "top_predictions": refined.get("top_predictions", []),
        "confidence_threshold": refined.get("confidence_threshold"),
        "min_confidence_margin": refined.get("min_confidence_margin"),
        "confidence_threshold_validated": False,
        "model_name": refined.get("model_name"),
        "model_sha256": refined.get("model_sha256"),
        "preprocessing_id": refined.get("preprocessing_id"),
        "message": refined.get("message"),
    }
    if not accepted:
        anatomy_result["upper_extremity_refinement"] = specialist_result
        return anatomy_result

    # Preserve original full-body inference for debugging/traceability, then use
    # the specific-region classifier's own output contract for downstream routing.
    anatomy_result["primary_classifier_result"] = primary_snapshot
    anatomy_result["upper_extremity_refinement"] = specialist_result
    anatomy_result["body_part_label"] = refined["candidate_body_part"]
    anatomy_result["candidate_body_part"] = refined["candidate_body_part"]
    anatomy_result["body_part_confidence"] = float(refined["confidence"])
    anatomy_result["second_candidate_body_part"] = refined.get("second_candidate_body_part")
    anatomy_result["second_candidate_confidence"] = refined.get("second_confidence")
    anatomy_result["top_two_margin"] = refined.get("top_two_margin")
    anatomy_result["confidence_threshold"] = refined.get("confidence_threshold", 0.60)
    anatomy_result["min_confidence_margin"] = refined.get("min_confidence_margin", 0.10)
    anatomy_result["top_predictions"] = refined.get("top_predictions", [])
    anatomy_result["candidate_is_specific_anatomy"] = True
    anatomy_result["refined_by_model_name"] = refined.get("model_name")
    anatomy_result["refined_by_model_sha256"] = refined.get("model_sha256")
    anatomy_result["refined_preprocessing_id"] = refined.get("preprocessing_id")
    anatomy_result["status"] = "completed"
    anatomy_result["message"] = (
        f"The primary classifier returned {primary_snapshot['candidate_body_part']} and a dedicated "
        f"upper-extremity classifier refined the candidate to {refined['candidate_body_part']} "
        f"({float(refined['confidence']):.1%} specialist score). The specialist score is not a clinical probability."
    )
    return anatomy_result


def _get_body_part_classifier():
    """Load and cache the pretrained image-based anatomy classifier."""
    global _BODY_PART_CLASSIFIER
    if _BODY_PART_CLASSIFIER is not None:
        return _BODY_PART_CLASSIFIER

    with _BODY_PART_CLASSIFIER_LOCK:
        if _BODY_PART_CLASSIFIER is None:
            from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier

            classifier = XRayAnatomyClassifier()
            classifier.load()
            _BODY_PART_CLASSIFIER = classifier
    return _BODY_PART_CLASSIFIER


def route_xray(image_path: str, declared_region: Optional[str] = None) -> dict[str, Any]:
    """Route an X-ray by image-detected anatomy when no region is supplied.

    The automatic path uses the separate body-part classifier first. It only
    reaches the existing regional model registry when the classifier's configured
    confidence and ambiguity gates are passed. The ordinary upload endpoint calls
    this function without a declared region. `declared_region` is reserved for an
    explicit post-upload user confirmation/correction path; it is never inferred
    from the filename and does not itself verify clinical findings.
    """
    if not isinstance(declared_region, str) or not declared_region.strip():
        try:
            anatomy_classifier = _get_body_part_classifier()
        except Exception as exc:
            logger.exception("Unable to load the automatic X-ray anatomy classifier")
            return {
                "region": "unknown",
                "canonical_region": "unknown",
                "supported": False,
                "is_supported": False,
                "is_available": False,
                "is_recognized": False,
                "region_source": "image_classifier",
                "anatomy_verification_status": "unavailable",
                "reason_code": "anatomy_classifier_unavailable",
                "registry_entry": None,
                "diagnostic_message": str(exc),
                "message": (
                    "Automatic X-ray type recognition is currently unavailable. "
                    "No regional model was selected; please retry after the model is configured."
                ),
            }

        try:
            anatomy_result = anatomy_classifier.predict_file(image_path)
        except Exception as exc:
            logger.exception("Automatic X-ray anatomy classification failed")
            return {
                "region": "unknown",
                "canonical_region": "unknown",
                "supported": False,
                "is_supported": False,
                "is_available": False,
                "is_recognized": False,
                "region_source": "image_classifier",
                "anatomy_verification_status": "failed",
                "reason_code": "anatomy_classification_failed",
                "registry_entry": None,
                "diagnostic_message": str(exc),
                "message": (
                    "The X-ray image could not be classified automatically. "
                    "No regional model was selected. Please upload a readable X-ray image."
                ),
            }

        # If the 33-class model only identifies the broad upper limb, ask an
        # optional dedicated seven-class classifier to distinguish wrist/hand/
        # elbow/forearm/finger/humerus/shoulder. It never guesses when the specialist
        # checkpoint is absent or its candidate fails the provisional gates.
        anatomy_result = _refine_broad_upper_extremity_candidate(image_path, anatomy_result)

        # Some older classifier outputs mark an otherwise high-scoring candidate
        # as unverified solely because an independent-confirmation flag is false.
        # Do not use that flag as a second confidence gate. Re-evaluate the actual
        # score, margin, and label below. Failed/unavailable classifier executions
        # still fail closed.
        classifier_status = str(anatomy_result.get("status") or "").strip().lower()
        detected_label = anatomy_result.get("body_part_label")
        candidate_from_classifier = anatomy_result.get("candidate_body_part")
        if not isinstance(detected_label, str) or not detected_label.strip():
            detected_label = candidate_from_classifier
        if classifier_status in {"failed", "unavailable", "error"} or not isinstance(detected_label, str) or not detected_label.strip():
            candidate = candidate_from_classifier or detected_label
            confidence = anatomy_result.get("body_part_confidence")
            return {
                "region": "unknown",
                "canonical_region": "unknown",
                "candidate_region": str(candidate).lower().replace("_", " ") if candidate else None,
                "supported": False,
                "is_supported": False,
                "is_available": False,
                "is_recognized": False,
                "region_source": "image_classifier",
                "classification_status": classifier_status or "unknown",
                "anatomy_verification_status": "unverified",
                "anatomy_independently_verified": False,
                "anatomy_classifier_accepted": False,
                "anatomy_verification_method": None,
                "body_part_confidence": confidence,
                "anatomy_detection": anatomy_result,
                "reason_code": "anatomy_unverified",
                "registry_entry": None,
                "message": anatomy_result.get(
                    "message",
                    "The image body part could not be verified confidently. No regional model was selected.",
                ),
            }

        confidence = anatomy_result.get("body_part_confidence")

        # Defense in depth: never route to a regional interpretation model solely
        # because an upstream classifier marked its output completed. Enforce the
        # configured confidence threshold here too, and reject close top-two scores
        # when the classifier provides them. This does not change the model or lower
        # the threshold; it prevents ambiguous anatomy from reaching disease inference.
        try:
            confidence_value = float(confidence)
            if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
                confidence_value = None
        except (TypeError, ValueError, OverflowError):
            confidence_value = None

        threshold_value = anatomy_result.get(
            "confidence_threshold",
            anatomy_result.get("min_confidence", 0.60),
        )
        try:
            threshold_value = float(threshold_value)
        except (TypeError, ValueError, OverflowError):
            threshold_value = 0.60
        if not (0.0 <= threshold_value <= 1.0):
            threshold_value = 0.60

        candidate_list = (
            anatomy_result.get("top_predictions")
            or anatomy_result.get("top_candidates")
            or anatomy_result.get("candidates")
            or []
        )
        candidate_scores: list[float] = []
        if isinstance(candidate_list, list):
            for item in candidate_list[:5]:
                if not isinstance(item, dict):
                    continue
                raw_score = item.get("probability", item.get("score", item.get("confidence")))
                try:
                    score = float(raw_score)
                except (TypeError, ValueError, OverflowError):
                    continue
                if 0.0 <= score <= 1.0:
                    candidate_scores.append(score)

        # The classifier's configured margin is the source of truth. Support the
        # older key for backward compatibility, but prefer the new key introduced
        # by the updated anatomy classifier.
        min_margin_value = anatomy_result.get(
            "top_two_margin_threshold",
            anatomy_result.get("min_confidence_margin", 0.10),
        )
        try:
            min_margin_value = float(min_margin_value)
        except (TypeError, ValueError, OverflowError):
            min_margin_value = 0.10
        if not math.isfinite(min_margin_value) or not (0.0 <= min_margin_value <= 1.0):
            min_margin_value = 0.10

        candidate_scores.sort(reverse=True)
        reported_margin = anatomy_result.get("top_two_margin")
        try:
            reported_margin_value = float(reported_margin)
            if not math.isfinite(reported_margin_value) or not (0.0 <= reported_margin_value <= 1.0):
                reported_margin_value = None
        except (TypeError, ValueError, OverflowError):
            reported_margin_value = None

        # Prefer the explicit classifier-reported margin. If absent, calculate it
        # from candidate scores. Missing margin data must not be fabricated.
        if reported_margin_value is not None:
            effective_margin = reported_margin_value
        elif len(candidate_scores) >= 2:
            effective_margin = candidate_scores[0] - candidate_scores[1]
        else:
            effective_margin = None

        low_confidence = confidence_value is None or confidence_value < threshold_value
        ambiguous_top_two = (
            effective_margin is not None
            and effective_margin < min_margin_value
        )
        if low_confidence or ambiguous_top_two:
            candidate = anatomy_result.get("candidate_body_part") or detected_label
            if ambiguous_top_two and not low_confidence:
                reason_code = "ambiguous_anatomy_prediction"
            elif low_confidence and classifier_status == "completed":
                reason_code = "anatomy_confidence_below_threshold"
            else:
                reason_code = "anatomy_unverified"
            if ambiguous_top_two and not low_confidence:
                message = (
                    "The two leading body-region predictions are too close to select a region reliably. "
                    "No regional interpretation model was selected. Please confirm the region or upload a clearer X-ray."
                )
            else:
                message = (
                    f"The candidate body region did not meet the required confidence threshold ({threshold_value:.0%}). "
                    "No regional interpretation model was selected. Please confirm the region or upload a clearer X-ray."
                )
            return {
                "region": "unknown",
                "canonical_region": "unknown",
                "candidate_region": str(candidate).strip().lower().replace("_", " ") if candidate else None,
                "supported": False,
                "is_supported": False,
                "is_available": False,
                "is_recognized": False,
                "region_source": "image_classifier",
                "classification_status": "completed",
                "anatomy_verification_status": "unverified",
                "anatomy_independently_verified": False,
                "anatomy_classifier_accepted": False,
                "anatomy_verification_method": None,
                "body_part_confidence": confidence_value,
                "anatomy_detection": anatomy_result,
                "reason_code": reason_code,
                "registry_entry": None,
                "message": message,
            }

        detected_region, detected_region_recognized, is_broad_label = (
            _normalize_anatomy_classifier_label(detected_label)
        )
        if not detected_region_recognized:
            logger.error("Body-part classifier returned an unknown label: %r", detected_label)
            return {
                "region": "unknown",
                "canonical_region": "unknown",
                "candidate_region": str(detected_label).strip().lower().replace("_", " "),
                "supported": False,
                "is_supported": False,
                "is_available": False,
                "is_recognized": False,
                "region_source": "image_classifier",
                "classification_status": "completed",
                "anatomy_verification_status": "failed",
                "anatomy_independently_verified": False,
                "body_part_confidence": confidence,
                "anatomy_detection": anatomy_result,
                "reason_code": "unsupported_classifier_label",
                "registry_entry": None,
                "message": (
                    f"The anatomy classifier returned an unrecognized label ({detected_label!r}). "
                    "No regional interpretation model was selected."
                ),
            }

        # A broad label is a valid anatomy-classifier output, but it is not enough
        # to select a narrower region-specific model (e.g. ARM is not equivalent
        # to humerus, HEAD is not equivalent to skull, LEG is not equivalent to femur,
        # and generic SPINE does not distinguish cervical/thoracic/lumbar studies).
        if is_broad_label:
            return _broad_anatomy_route_result(
                detected_region, anatomy_result, confidence,
                classifier_gate_passed=True, is_broad_label=True,
            )

        # A specific class can pass the automatic classifier gate. This is not
        # independent clinical verification, and the score must not be described as
        # a clinical probability. Low-confidence and ambiguous labels returned above.
        candidate_is_specific = anatomy_result.get("candidate_is_specific_anatomy")
        if candidate_is_specific is False:
            return _broad_anatomy_route_result(
                detected_region, anatomy_result, confidence,
                classifier_gate_passed=False, is_broad_label=False,
            )

        # If the anatomy is recognized but no region-specific model is configured,
        # report that limitation directly. Do not mislabel a known region (e.g. knee)
        # as a bad classifier label or imply that user confirmation would make a
        # nonexistent model available.
        configured_entry = XRAY_MODEL_REGISTRY.get(detected_region)
        if configured_entry is None or not configured_entry.is_supported:
            # A registry entry may exist for a known anatomical label even when no
            # compatible regional model is supported. Report that limitation before
            # asking for confirmation: confirmation cannot make a missing model available.
            display = detected_region.replace("_", " ").title()
            return {
                "region": detected_region,
                "canonical_region": detected_region,
                "candidate_region": detected_region,
                "detected_region": detected_region,
                "supported": False,
                "is_supported": False,
                "is_available": False,
                "is_recognized": True,
                "region_source": "image_classifier",
                "classification_status": "completed",
                # A classifier score is not independent anatomy verification.
                # Confirmation is required even when this region is recognized.
                "anatomy_verification_status": "confirmation_required",
                "anatomy_independently_verified": False,
                "anatomy_classifier_accepted": False,
                "anatomy_verification_method": None,
                "interpretation_status": "unavailable",
                "body_part_confidence": confidence,
                "anatomy_detection": anatomy_result,
                "reason_code": "unsupported_region",
                "registry_entry": None,
                "message": (
                    f"Anatomy classification accepted: {display} ({float(confidence):.1%} classifier score). "
                    "Automated abnormality interpretation is unavailable because no compatible region-specific model is configured. "
                    "The classifier score is not clinical confirmation or a diagnosis."
                ),
            }

        # A model prediction is only a candidate. Require explicit user confirmation
        # of the specific anatomy before any regional abnormality model is run.
        # This prevents a confident but incorrect body-part prediction from routing
        # the image into an incompatible model.
        display = detected_region.replace("_", " ").title()
        return {
            "region": detected_region,
            "canonical_region": detected_region,
            "candidate_region": detected_region,
            "detected_region": detected_region,
            "supported": False,
            "is_supported": False,
            "is_available": False,
            "is_recognized": True,
            "region_source": "image_classifier",
            "classification_status": "completed",
            "anatomy_verification_status": "confirmation_required",
            "anatomy_independently_verified": False,
            "anatomy_classifier_accepted": False,
            "anatomy_verification_method": None,
            "specific_region_confirmation_required": True,
            "interpretation_status": "not_run_pending_anatomy_confirmation",
            "body_part_confidence": confidence,
            "anatomy_detection": anatomy_result,
            "reason_code": "anatomy_confirmation_required",
            "registry_entry": None,
            "message": (
                f"The image classifier suggests {display} "
                f"(score: {float(confidence):.1%}), but this is not independent verification. "
                "Confirm the specific X-ray region before regional interpretation. "
                "No regional abnormality model was run."
            ),
        }
        # Intentionally unreachable until an explicit confirmation flow is wired in.
        routed_result = route_xray(image_path, declared_region=detected_region)
        routed_result["region_source"] = "image_classifier"
        routed_result["detected_region"] = detected_region
        routed_result["classification_status"] = "completed"
        routed_result["anatomy_verification_status"] = "completed"
        routed_result["anatomy_independently_verified"] = False
        routed_result["anatomy_classifier_accepted"] = True
        routed_result["anatomy_verification_method"] = "classifier_threshold"
        routed_result["interpretation_status"] = str(routed_result.get("status") or "unavailable").lower()
        routed_result["body_part_confidence"] = confidence
        routed_result["anatomy_detection"] = anatomy_result
        try:
            confidence_label = f"{float(confidence):.1%}" if confidence is not None else "not available"
        except (TypeError, ValueError):
            confidence_label = "not available"
        routed_result["message"] = (
            f"Automatically classified this image as {detected_region.replace('_', ' ').title()} "
            f"(classifier score: {confidence_label}). "
            f"{routed_result.get('message', '')}"
        ).strip()
        return routed_result

    canonical_region, is_recognized = normalize_region_name(declared_region)
    entry = XRAY_MODEL_REGISTRY.get(canonical_region)

    if not is_recognized:
        logger.info("Unrecognized anatomical region declared: '%s'", declared_region)
        return {
            "region": canonical_region,
            "canonical_region": canonical_region,
            "supported": False,
            "is_supported": False,
            "is_available": False,
            "is_recognized": False,
            "region_source": "declared_region",
            "reason_code": "unrecognized_region",
            "registry_entry": None,
            "message": (
                f"The supplied region '{declared_region or 'unknown'}' could not be recognized. "
                "No regional interpretation model was selected."
            ),
        }

    if entry is None:
        # Known anatomy labels without a configured region-specific inference model
        # should be reported as recognized-but-unavailable, not as bad classifier labels.
        is_broad = canonical_region in {
            "extremity", "upper extremity", "lower extremity", "head region"
        }
        display = canonical_region.replace("_", " ").title()
        return {
            "region": canonical_region,
            "canonical_region": canonical_region,
            "supported": False,
            "is_supported": False,
            "is_available": False,
            "is_recognized": True,
            "region_source": "declared_region",
            "reason_code": "broad_anatomy_label" if is_broad else "unsupported_region",
            "registry_entry": None,
            "message": (
                f"The region {display} was recognized, but no compatible region-specific image interpretation model "
                "is configured. No disease finding was inferred. A radiology report can be added separately."
            ),
        }

    is_supported = entry.is_supported
    is_available, avail_reason = entry.check_availability()

    if is_supported and is_available:
        return {
            "region": canonical_region,
            "canonical_region": canonical_region,
            "supported": True,
            "is_supported": True,
            "is_available": True,
            "is_recognized": True,
            "registry_entry": entry.to_dict(),
            "message": (
                f"Routed to {entry.display_name} model adapter "
                f"({'TorchXRayVision DenseNet121' if _uses_torchxrayvision() else entry.model_architecture}). "
                + (
                    "Research-model scores are not calibrated probabilities or a diagnosis; "
                    "clinical interpretation by a qualified professional is required."
                    if _uses_torchxrayvision()
                    else ""
                )
            ),
        }

    if is_supported and not is_available:
        # Do not report every failure as a missing checkpoint: the file may exist but
        # be empty, unreadable, incompatible, or fail safe deserialization.
        lower_reason = str(avail_reason).lower()
        if "checkpoint_path_not_found" in lower_reason:
            reason_code = "missing_checkpoint"
            user_reason = "the trained model checkpoint was not found at the configured path"
        elif "checkpoint_path_invalid" in lower_reason:
            reason_code = "invalid_checkpoint_path"
            user_reason = "the configured model checkpoint path is invalid or the file is empty"
        elif "checkpoint_path_unreadable" in lower_reason:
            reason_code = "checkpoint_unreadable"
            user_reason = "the configured model checkpoint could not be read"
        elif "checkpoint_integrity_failed" in lower_reason:
            reason_code = "checkpoint_integrity_failed"
            user_reason = "the checkpoint failed safe integrity or class-dimension validation"
        elif "torchxrayvision_unavailable" in lower_reason:
            reason_code = "torchxrayvision_unavailable"
            user_reason = "the selected TorchXRayVision model dependency is unavailable"
        else:
            reason_code = "checkpoint_verification_failed"
            user_reason = "the checkpoint could not be verified for this model"

        logger.warning(
            "X-ray model unavailable for region=%s reason_code=%s detail=%s",
            canonical_region,
            reason_code,
            avail_reason,
        )
        return {
            "region": canonical_region,
            "canonical_region": canonical_region,
            "supported": True,
            "is_supported": True,
            "is_available": False,
            "is_recognized": True,
            "registry_entry": entry.to_dict(),
            "reason_code": reason_code,
            "diagnostic_message": str(avail_reason),
            "message": (
                f"X-ray received successfully. Automated interpretation for {entry.display_name} X-ray "
                f"is currently unavailable because {user_reason}. "
                "If you have the associated radiology report, upload it for supported text-based analysis."
            ),
        }

    # Region is recognized in human anatomy but unconfigured in CV models
    return {
        "region": canonical_region,
        "canonical_region": canonical_region,
        "supported": False,
        "is_supported": False,
        "is_available": False,
        "is_recognized": True,
        "registry_entry": entry.to_dict(),
        "reason_code": "unsupported_region",
        "message": (
            f"X-ray received. Automated image interpretation for {entry.display_name} is not configured. "
            "If you have the associated radiology report, upload it for supported text-based analysis."
        ),
    }

