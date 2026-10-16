"""X-ray body-region router and region-aware model registry for MediSense AI.

Defines canonical anatomical regions, alias normalizations, model architecture specifications,
expected preprocessing, class mappings, checkpoint paths, and availability disclosures.
Enforces strict medical-safety boundaries: only anatomical regions with verified, compatible
trained models are marked operational.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
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
        """Verify whether the configured checkpoint file physically exists on disk."""
        if not self.is_supported:
            return (
                False,
                f"Automated interpretation for {self.canonical_region} radiography is currently unsupported (no computer-vision model configured for this anatomical region).",
            )
        if not self.checkpoint_location:
            return False, "Checkpoint path not configured."

        settings = get_settings()
        configured_path = (settings.xray_chest_checkpoint or "").strip()
        if configured_path:
            p = Path(configured_path)
            if not p.is_absolute():
                p = settings.xray_model_path / p
        else:
            p = settings.xray_model_path / self.checkpoint_location

        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return True, f"Trained checkpoint verified at: {p}"
        return (
            False,
            f"Model checkpoint not configured on disk; place PyTorch weights in {p} or set XRAY_CHEST_CHECKPOINT in .env.",
        )

    def to_dict(self) -> dict[str, Any]:
        available, avail_msg = self.check_availability()
        d = asdict(self)
        d["is_available"] = available
        d["availability_message"] = avail_msg
        d["availability_status"] = (
            "AVAILABLE"
            if available
            else ("UNAVAILABLE_MISSING_CHECKPOINT" if self.is_supported else "UNSUPPORTED_REGION")
        )
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
        source_provenance="NIH ChestX-ray14 / CheXpert benchmark reference",
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
        display_name="Spine (Cervical / Lumbar)",
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
    "thoracic": "chest",
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
    "fingers": "hand",
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
    # Ankle
    "ankle": "ankle",
    "ankle joint": "ankle",
    # Foot
    "foot": "foot",
    "feet": "foot",
    "toes": "foot",
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
    "cervical spine": "spine",
    "lumbar spine": "spine",
    "thoracic spine": "spine",
    "back": "spine",
}


def normalize_region_name(raw_region: Optional[str]) -> tuple[str, bool]:
    """Normalize input region string to its canonical registry identifier.

    Returns:
        tuple[str, bool]: (canonical_region, is_recognized)
    """
    if not raw_region or not str(raw_region).strip():
        return "unknown", False

    cleaned = str(raw_region).strip().lower()
    cleaned = cleaned.replace("_", " ").strip()

    if cleaned in REGION_ALIASES:
        return REGION_ALIASES[cleaned], True

    if cleaned in XRAY_MODEL_REGISTRY:
        return cleaned, True

    # Substring heuristic for compound inputs like "x-ray of the right knee"
    for alias, canonical in REGION_ALIASES.items():
        if alias in cleaned:
            return canonical, True

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
            "message": f"Routed to {entry.display_name} model adapter ({entry.model_architecture}).",
        }

    if is_supported and not is_available:
        # Region is supported by pipeline architecture, but weights are absent on disk
        return {
            "region": canonical_region,
            "canonical_region": canonical_region,
            "supported": True,
            "is_supported": True,
            "is_available": False,
            "is_recognized": True,
            "registry_entry": entry.to_dict(),
            "reason_code": "missing_checkpoint",
            "message": (
                f"X-ray received successfully. Automated interpretation for {entry.display_name} X-ray is currently unavailable "
                "as model weights are not configured on disk. If you have the associated radiology report, upload it for supported text-based analysis."
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
            f"X-ray received. Automated interpretation for {entry.display_name} X-ray is currently unavailable "
            "as the diagnostic vision model is specialized for chest radiography. "
            "If you have the associated radiology report, upload it for supported text-based analysis."
        ),
    }

