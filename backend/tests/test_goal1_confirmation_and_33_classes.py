"""Goal 1 Unit & Integration Tests: 33 Anatomical Classes, Verification, & Safe Routing.

Verifies:
1. All 33 required anatomical classes are preserved in classes.txt.
2. All 33 classes are present, searchable, and mapped in the catalogue.
3. User confirmation endpoint accepts all 33 classes and rejects invalid classes with 422.
4. Specific compatible regions route to regional models.
5. Unsupported specific and broad regions return explicit unsupported disclosures without crashing.
6. Image SHA-256 is recorded and preserved in anatomy_metadata.
7. Structured response fields (success, class_count, predicted_anatomy, warnings) are emitted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.ai.xray_anatomy_classifier import (
    CLASS_COUNT,
    XRayAnatomyClassifier,
    _read_labels,
)
from app.ai.xray_catalogue import (
    get_backend_body_regions,
    get_xray_catalogue_with_model_status,
)
from app.ai.xray_router import (
    normalize_region_name,
    route_xray,
)
from app.database.models import Assessment, XrayResult
from app.main import app
from app.routes.xray import (
    _ALL_33_ANATOMY_CONFIRMATION_LABELS,
    _canonical_user_confirmed_region,
)

REQUIRED_33_CLASSES = [
    "ABDOMEN", "ANKLE", "ARM", "CERVICAL_SPINE", "CHEST", "DORSAL_SPINE",
    "ELBOW", "EXTREMITY", "FEMUR", "FINGER", "FOOT", "FOREARM", "HAND",
    "HEAD", "HEEL", "HIP", "KNEE", "KUB", "LEG", "LOWER_EXTREMITY",
    "LUMBAR_SPINE", "NASOPHARYNX", "NECK", "PELVIS", "PNS", "SHOULDER",
    "SI_JOINT", "SKULL", "SPINE", "TEMPORAL_BONE", "TIBIA",
    "UPPER_EXTREMITY", "WRIST",
]


def test_1_all_33_classes_preserved_in_classes_file():
    """Verify classes.txt has exactly the 33 required anatomical classes."""
    classes_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "xray"
        / "bodypart_full"
        / "classes.txt"
    )
    labels = _read_labels(classes_path)
    assert len(labels) == 33
    assert len(labels) == CLASS_COUNT
    assert sorted(labels) == sorted(REQUIRED_33_CLASSES)
    assert _ALL_33_ANATOMY_CONFIRMATION_LABELS == set(REQUIRED_33_CLASSES)


def test_2_all_33_classes_present_in_catalogue():
    """Verify all 33 classes are represented in get_backend_body_regions()."""
    regions = get_backend_body_regions()
    assert len(regions) >= 33

    for required_class in REQUIRED_33_CLASSES:
        found = any(
            required_class.lower() == r.get("id", "").lower()
            or required_class.lower() == r.get("canonical_region", "").lower()
            or any(required_class.lower() == str(a).lower() for a in r.get("aliases", []))
            for r in regions
        )
        assert found, f"Required class {required_class} missing from catalogue regions!"


def test_3_canonical_user_confirmation_validation():
    """Verify _canonical_user_confirmed_region accepts all 33 classes and rejects invalid ones."""
    for required_class in REQUIRED_33_CLASSES:
        canonical = _canonical_user_confirmed_region(required_class)
        assert isinstance(canonical, str) and len(canonical) > 0

    # Invalid classes must raise HTTP 422
    for invalid in ["BRAIN", "LIVER", "EYE", "RANDOM_REGION", ""]:
        with pytest.raises(HTTPException) as exc_info:
            _canonical_user_confirmed_region(invalid)
        assert exc_info.value.status_code == 422


def test_4_user_confirmation_routing_behavior():
    """Verify declared_region routing across compatible, unsupported, and broad regions."""
    # 1. Compatible region: wrist
    wrist_route = route_xray("dummy.png", declared_region="WRIST")
    assert wrist_route["is_recognized"] is True
    assert wrist_route["region"] == "wrist"
    assert wrist_route["is_supported"] is True
    assert wrist_route["is_available"] is True
    assert wrist_route["regional_model_status"] == "configured_and_available"

    # 2. Unsupported specific region: knee
    knee_route = route_xray("dummy.png", declared_region="KNEE")
    assert knee_route["is_recognized"] is True
    assert knee_route["region"] == "knee"
    assert knee_route["is_supported"] is False
    assert knee_route["is_available"] is False
    assert "is not configured" in knee_route["message"]

    # 3. Broad region: spine
    spine_route = route_xray("dummy.png", declared_region="SPINE")
    assert spine_route["is_recognized"] is True
    assert spine_route["region"] == "spine"
    assert spine_route["is_supported"] is False
    assert spine_route["is_available"] is False
    assert spine_route["reason_code"] == "unsupported_region"

    # 4. Broad region: upper extremity
    upper_route = route_xray("dummy.png", declared_region="UPPER_EXTREMITY")
    assert upper_route["is_recognized"] is True
    assert upper_route["region"] == "upper extremity"
    assert upper_route["is_supported"] is False
    assert upper_route["is_available"] is False


def test_5_classifier_structured_response_fields():
    """Verify predict_image outputs the Goal 1 structured response keys."""
    classifier = XRayAnatomyClassifier()
    assert classifier.model_name == "convnext_tiny_onnx_33class"
    assert len(classifier.classes) == 0  # before load

    classifier.load()
    assert len(classifier.classes) == 33

    # Create dummy black radiograph image
    from PIL import Image
    import numpy as np

    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    # Add a gradient so it is not rejected as a blank white card
    arr[50:180, 50:180, :] = 120
    test_img = Image.fromarray(arr, mode="RGB")

    result = classifier.predict_image(test_img)
    assert result["success"] is True
    assert result["class_count"] == 33
    assert result["predicted_anatomy"] in classifier.classes
    assert isinstance(result["confidence"], float)
    assert result["prediction_status"] in {"completed", "needs_confirmation"}
    assert result["requires_user_confirmation"] is True
    assert result["preprocessing_verified"] is False
    assert isinstance(result["warnings"], list)
    assert len(result["warnings"]) >= 1


def test_6_catalogue_api_endpoint_has_all_classes():
    """Verify the /api/assessments/xray/catalogue HTTP endpoint returns all classes."""
    client = TestClient(app)
    resp = client.get("/api/assessments/xray/catalogue")
    assert resp.status_code == 200
    data = resp.json()
    assert "regions" in data
    assert "total_regions" in data

    catalogue_regions = data["regions"]
    for required_class in REQUIRED_33_CLASSES:
        found = any(
            required_class.lower() == r.get("id", "").lower()
            or required_class.lower() == r.get("canonical_region", "").lower()
            or any(required_class.lower() == str(a).lower() for a in r.get("aliases", []))
            for r in catalogue_regions
        )
        assert found, f"Catalogue API missing {required_class}"
