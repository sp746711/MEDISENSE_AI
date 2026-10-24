"""X-ray body-region router and region-aware model registry for MediSense AI.

Defines canonical anatomical regions, alias normalizations, model architecture specifications,
expected preprocessing, class mappings, checkpoint paths, and availability disclosures.
Enforces strict medical-safety boundaries: only anatomical regions with verified, compatible
trained models are marked operational.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import re
from typing import Any, Optional

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Supported regions for computer-vision model evaluation
SUPPORTED_REGIONS: set[str] = {"chest"}  # Expand only when verified trained models exist

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

        if self.canonical_region != "chest" and "resnet18" not in architecture:
            return False, (
                "unsupported_model_architecture: This inference service supports configured ResNet18 "
                "state-dict models for non-chest regions; the requested architecture is not implemented."
            )

        configured_path = ""
        if self.canonical_region == "chest":
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
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Stanford MURA Musculoskeletal Radiographs (unconfigured)",
        supported_prediction_task="Musculoskeletal abnormality / fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "hand": XRayModelRegistryEntry(
        canonical_region="hand",
        display_name="Hand",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Stanford MURA (unconfigured)",
        supported_prediction_task="Musculoskeletal abnormality detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "forearm": XRayModelRegistryEntry(
        canonical_region="forearm",
        display_name="Forearm",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Stanford MURA (unconfigured)",
        supported_prediction_task="Forearm fracture / deformity detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "elbow": XRayModelRegistryEntry(
        canonical_region="elbow",
        display_name="Elbow",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Stanford MURA (unconfigured)",
        supported_prediction_task="Elbow joint effusion / fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "shoulder": XRayModelRegistryEntry(
        canonical_region="shoulder",
        display_name="Shoulder",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Stanford MURA (unconfigured)",
        supported_prediction_task="Shoulder dislocation / fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
    ),
    "humerus": XRayModelRegistryEntry(
        canonical_region="humerus",
        display_name="Humerus",
        supported_modality="radiograph_image",
        checkpoint_location=None,
        model_architecture=None,
        class_label_mapping=[],
        expected_preprocessing=None,
        model_version=None,
        source_provenance="Stanford MURA (unconfigured)",
        supported_prediction_task="Humerus shaft fracture detection (unconfigured)",
        is_supported=False,
        validation_status="UNCONFIGURED",
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
    "arm": "humerus",
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
    "fingers": "fingers",
    "finger": "fingers",
    "thumb": "fingers",
    "digits": "fingers",
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


def route_xray(image_path: str, declared_region: Optional[str] = None) -> dict[str, Any]:
    """Route an X-ray upload based on anatomical region and model registry availability."""
    canonical_region, is_recognized = normalize_region_name(declared_region)
    entry = XRAY_MODEL_REGISTRY.get(canonical_region)

    if not is_recognized or entry is None:
        logger.info("Unrecognized anatomical region declared: '%s'", declared_region)
        return {
            "region": canonical_region,
            "canonical_region": canonical_region,
            "supported": False,
            "is_recognized": False,
            "registry_entry": None,
            "message": (
                f"X-ray received. Automated interpretation for '{declared_region or 'unknown'}' is currently unavailable. "
                "If you have the associated radiology report, upload it for supported text-based analysis."
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

