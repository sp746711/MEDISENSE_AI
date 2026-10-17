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


def verify_checkpoint_integrity(
    checkpoint_path: Path, expected_classes: int = len(CHEST_CLASSES)
) -> tuple[bool, str]:
    """Verify that a checkpoint file is a valid, deserializable PyTorch model/state_dict with matching class architecture."""
    if not checkpoint_path.exists():
        return False, f"Checkpoint file missing: {checkpoint_path}"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        return False, f"Checkpoint file is empty: {checkpoint_path}"

    try:
        import torch
        import torch.nn as nn

        # 1. Attempt deserialization
        try:
            loaded = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        except Exception:
            try:
                loaded = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            except Exception as load_exc:
                return False, f"Checkpoint loading failed (corrupted or invalid PyTorch file): {load_exc}"

        # 2. Check architecture compatibility if nn.Module
        if isinstance(loaded, nn.Module):
            if hasattr(loaded, "fc") and hasattr(loaded.fc, "out_features"):
                if loaded.fc.out_features != expected_classes:
                    return (
                        False,
                        f"Checkpoint output dimension mismatch: model has {loaded.fc.out_features} classes, expected {expected_classes}.",
                    )
            return True, f"Trained checkpoint module verified at: {checkpoint_path}"

        # 3. Check architecture compatibility if state_dict
        if isinstance(loaded, dict):
            sd = loaded.get("state_dict", loaded)
            if not isinstance(sd, dict) or not sd:
                return False, f"Checkpoint dictionary does not contain valid state_dict tensor mapping."

            # Verify classification head weight shape
            if "fc.weight" in sd:
                out_classes = sd["fc.weight"].shape[0]
                if out_classes != expected_classes:
                    return (
                        False,
                        f"Checkpoint output dimension mismatch: fc.weight has {out_classes} classes, expected {expected_classes}.",
                    )
            elif "fc.bias" in sd:
                out_classes = sd["fc.bias"].shape[0]
                if out_classes != expected_classes:
                    return (
                        False,
                        f"Checkpoint output dimension mismatch: fc.bias has {out_classes} classes, expected {expected_classes}.",
                    )
            else:
                has_weights = any(k.endswith(".weight") for k in sd.keys())
                if not has_weights:
                    return False, "Checkpoint state_dict contains no tensor weights."

            return True, f"Trained checkpoint weights verified at: {checkpoint_path}"

        return False, f"Unsupported checkpoint object type: {type(loaded).__name__} (expected nn.Module or dict state_dict)."
    except Exception as exc:
        return False, f"Checkpoint integrity check failed: {exc}"


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
        """Verify whether the configured checkpoint exists on disk and contains verified compatible weights."""
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

        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            return (
                False,
                f"Model checkpoint not configured on disk; place PyTorch weights in {p} or set XRAY_CHEST_CHECKPOINT in .env.",
            )

        # Deep verification of deserialization, architecture, and class mapping
        return verify_checkpoint_integrity(p, expected_classes=len(self.class_label_mapping))

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
    "shin": "lower leg",
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
    "back": "spine",
    "cervical spine": "spine",
    "c-spine": "spine",
    "cervical": "spine",
    "neck spine": "spine",
    "thoracic spine": "spine",
    "t-spine": "spine",
    "thoracic": "spine",
    "dorsal spine": "spine",
    "lumbar spine": "spine",
    "l-spine": "spine",
    "lumbar": "spine",
    "lumbosacral": "spine",
    "lower back": "spine",
    "low back": "spine",
    "sacrum": "spine",
    "coccyx": "spine",
    "tailbone": "spine",
    "sacrococcygeal": "spine",
    # Head & Neck
    "head": "skull",
    "skull": "skull",
    "cranium": "skull",
    "calvarium": "skull",
    "brain": "skull",
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

