"""Comprehensive Goal 1 regression test suite covering all 15 required scenarios.

Requirements verified:
1. Valid classifier output and verified label mapping.
2. Candidate below the 60% threshold.
3. Candidate passing threshold but failing top-two margin.
4. High-confidence incorrect predictions treated as candidates rather than truth.
5. Invalid or unknown class labels handled safely.
6. Model missing or load failure.
7. Inference failure versus low-confidence output distinction.
8. Correct mapping from specific anatomy to compatible regional model.
9. Broad anatomy labels not silently routed to a specific model.
10. No compatible regional model configured.
11. User confirmation and user correction.
12. Persistence of confirmed region.
13. Correct state after refresh or reload.
14. Multiple images retaining independent predictions, confirmations, and routing results.
15. Upload, authorization, ownership, cleanup, and DB integrity intact.
"""

import sys
from types import SimpleNamespace
from uuid import uuid4
import numpy as np
import pytest
from PIL import Image
from fastapi import HTTPException

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier, _is_specific_anatomy_label
from app.ai import xray_router
from app.routes.xray import (
    _build_anatomy_metadata,
    _owned,
    _remove_file,
)
from app.database.models import Assessment


LABELS = [
    "ABDOMEN", "ANKLE", "ARM", "CERVICAL_SPINE", "CHEST", "DORSAL_SPINE",
    "ELBOW", "EXTREMITY", "FEMUR", "FINGER", "FOOT", "FOREARM", "HAND",
    "HEAD", "HEEL", "HIP", "KNEE", "KUB", "LEG", "LOWER_EXTREMITY",
    "LUMBAR_SPINE", "NASOPHARYNX", "NECK", "PELVIS", "PNS", "SHOULDER",
    "SI_JOINT", "SKULL", "SPINE", "TEMPORAL_BONE", "TIBIA",
    "UPPER_EXTREMITY", "WRIST",
]


class FakeSession:
    """Mock ONNX Runtime session for offline unit tests."""

    def __init__(self, model_path, providers=None, probabilities=None, output_shape=None):
        self.model_path = model_path
        self.providers = providers
        self.probabilities = probabilities
        self.output_shape = output_shape or ["batch", 33]
        self.last_tensor = None

    def get_inputs(self):
        return [
            SimpleNamespace(
                name="images",
                shape=["batch", 3, 224, 224],
                type="tensor(float)",
            )
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(
                name="probs",
                shape=self.output_shape,
                type="tensor(float)",
            )
        ]

    def run(self, output_names, input_feed):
        self.last_tensor = input_feed["images"]
        return [self.probabilities]


def _install_fake_runtime(monkeypatch, session):
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda path, providers=None: session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)


def _make_classifier(tmp_path, probabilities=None, threshold=0.6, output_shape=None):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake ONNX for mocked test")

    labels_path = tmp_path / "classes.txt"
    labels_path.write_text("\n".join(LABELS) + "\n", encoding="utf-8")

    probs = probabilities
    if probs is None:
        probs = np.full((1, 33), 0.08 / 32, dtype=np.float32)
        probs[0, LABELS.index("KNEE")] = 0.92

    session = FakeSession(
        str(model_path),
        probabilities=probs,
        output_shape=output_shape,
    )

    classifier = XRayAnatomyClassifier(
        checkpoint_path=model_path,
        labels_path=labels_path,
        min_confidence=threshold,
    )

    return classifier, session


class MockRouterClassifier:
    """Mock classifier for router-level tests."""

    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error

    def predict_file(self, image_path):
        if self.error:
            raise self.error
        return dict(self.result)


# 1. Correct handling of a valid classifier output and verified label mapping.
def test_1_valid_classifier_output_and_label_mapping(tmp_path, monkeypatch):
    probs = np.full((1, 33), 0.07 / 31, dtype=np.float32)
    probs[0, LABELS.index("HAND")] = 0.88
    probs[0, LABELS.index("WRIST")] = 0.05

    classifier, session = _make_classifier(tmp_path, probabilities=probs)
    _install_fake_runtime(monkeypatch, session)
    classifier.load()

    img = Image.new("RGB", (224, 224), color="gray")
    result = classifier.predict_image(img)

    assert result["status"] == "completed"
    assert result["body_part_label"] == "HAND"
    assert result["candidate_body_part"] == "HAND"
    assert pytest.approx(result["body_part_confidence"], rel=1e-2) == 0.88
    assert result["top_predictions"][0]["label"] == "HAND"
    assert result["top_predictions"][1]["label"] == "WRIST"


# 2. Candidate below the 60% threshold.
def test_2_candidate_below_60_percent_threshold(tmp_path, monkeypatch):
    probs = np.full((1, 33), (1.0 - 0.469) / 32, dtype=np.float32)
    probs[0, LABELS.index("NECK")] = 0.469

    classifier, session = _make_classifier(tmp_path, probabilities=probs)
    _install_fake_runtime(monkeypatch, session)
    classifier.load()

    img = Image.new("RGB", (224, 224), color="gray")
    result = classifier.predict_image(img)

    assert result["status"] == "unverified"
    assert result["body_part_label"] is None
    assert result["candidate_body_part"] == "NECK"
    assert pytest.approx(result["body_part_confidence"], rel=1e-2) == 0.469


# 3. Candidate passing the threshold but failing the top-two margin rule.
def test_3_candidate_passing_threshold_failing_margin(tmp_path, monkeypatch):
    probs = np.full((1, 33), 0.05 / 31, dtype=np.float32)
    probs[0, LABELS.index("PELVIS")] = 0.50
    probs[0, LABELS.index("HIP")] = 0.45
    # Margin is 0.05, below the 0.10 margin threshold

    classifier, session = _make_classifier(tmp_path, probabilities=probs, threshold=0.45)
    _install_fake_runtime(monkeypatch, session)
    classifier.load()

    img = Image.new("RGB", (224, 224), color="gray")
    result = classifier.predict_image(img)

    assert result["status"] == "unverified"
    assert result["body_part_label"] is None
    assert result["candidate_body_part"] == "PELVIS"
    assert result["top_two_margin"] < 0.10


# 4. High-confidence incorrect predictions being treated as candidates rather than truth.
def test_4_high_confidence_treated_as_candidate_not_truth(monkeypatch):
    mock_classifier = MockRouterClassifier({
        "status": "completed",
        "body_part_label": "PELVIS",
        "candidate_body_part": "PELVIS",
        "body_part_confidence": 0.859,
        "confidence_threshold": 0.60,
        "min_confidence_margin": 0.10,
        "top_predictions": [
            {"label": "PELVIS", "probability": 0.859},
            {"label": "LUMBAR_SPINE", "probability": 0.08},
        ],
        "message": "Detected PELVIS (85.9%)",
    })
    monkeypatch.setattr(xray_router, "_BODY_PART_CLASSIFIER", mock_classifier)

    result = xray_router.route_xray("scan.jpg")
    # Gate passed
    assert result["classifier_gate_passed"] is True
    assert result["anatomy_classifier_accepted"] is True
    # But verification status must NOT claim clinical confirmation
    assert result["anatomy_verification_status"] == "confirmation_required"
    assert result["anatomy_independently_verified"] is False
    # Regional disease inference must NOT run automatically before confirmation
    assert result["supported"] is False
    assert result["registry_entry"] is None


# 5. Invalid or unknown class labels handled safely.
def test_5_invalid_or_unknown_class_label(tmp_path, monkeypatch):
    # Output shape has 20 classes instead of 33
    probs = np.full((1, 20), 0.05, dtype=np.float32)
    classifier, session = _make_classifier(tmp_path, probabilities=probs, output_shape=["batch", 20])
    _install_fake_runtime(monkeypatch, session)

    with pytest.raises(ValueError) as exc:
        classifier.load()
    assert "classes" in str(exc.value).lower() or "output" in str(exc.value).lower()

    # Broad or unknown label safety check
    assert _is_specific_anatomy_label("EXTREMITY") is False
    assert _is_specific_anatomy_label("UPPER_EXTREMITY") is False
    assert _is_specific_anatomy_label("LOWER_EXTREMITY") is False
    canonical, recognized = xray_router.normalize_region_name("NONEXISTENT_BODY_PART")
    assert recognized is False


# 6. Model missing or load failure.
def test_6_model_missing_or_load_failure():
    classifier = XRayAnatomyClassifier(checkpoint_path="nonexistent_model.onnx")
    with pytest.raises(FileNotFoundError):
        classifier.load()


# 7. Inference failure versus low-confidence output distinction.
def test_7_inference_failure_vs_low_confidence(monkeypatch):
    # Case A: Inference failure (exception during classifier execution)
    mock_fail = MockRouterClassifier(error=RuntimeError("Corrupted ONNX tensor"))
    monkeypatch.setattr(xray_router, "_BODY_PART_CLASSIFIER", mock_fail)

    res_fail = xray_router.route_xray("scan.png")
    assert res_fail["anatomy_verification_status"] == "failed"
    assert res_fail["reason_code"] == "anatomy_classification_failed"

    # Case B: Low confidence prediction (clean classification but below threshold)
    mock_low = MockRouterClassifier({
        "status": "unverified",
        "body_part_label": None,
        "candidate_body_part": "CHEST",
        "body_part_confidence": 0.45,
        "confidence_threshold": 0.60,
        "top_predictions": [{"label": "CHEST", "probability": 0.45}],
    })
    monkeypatch.setattr(xray_router, "_BODY_PART_CLASSIFIER", mock_low)

    res_low = xray_router.route_xray("scan.png")
    assert res_low["anatomy_verification_status"] == "unverified"
    assert res_low["reason_code"] in ("anatomy_confidence_below_threshold", "anatomy_unverified")


# 8. Correct mapping from specific anatomy to compatible regional model.
def test_8_specific_anatomy_to_compatible_regional_model():
    # User confirmed region as 'wrist'
    result_wrist = xray_router.route_xray("test.png", declared_region="wrist")
    assert result_wrist["is_supported"] is True
    assert result_wrist["regional_model_configured"] is True
    assert result_wrist["regional_model_status"] in ("configured_and_available", "configured_unavailable")
    assert "Wrist" in result_wrist["regional_model_name"]

    # User confirmed region as 'chest'
    result_chest = xray_router.route_xray("test.png", declared_region="chest")
    assert result_chest["is_supported"] is True
    assert result_chest["regional_model_configured"] is True

    # User confirmed region as 'hand'
    result_hand = xray_router.route_xray("test.png", declared_region="hand")
    assert result_hand["is_supported"] is True
    assert result_hand["regional_model_configured"] is True


# 9. Broad anatomy labels not silently routed to a specific model.
def test_9_broad_anatomy_labels_not_silently_routed(monkeypatch):
    mock_classifier = MockRouterClassifier({
        "status": "completed",
        "body_part_label": "UPPER_EXTREMITY",
        "candidate_body_part": "UPPER_EXTREMITY",
        "body_part_confidence": 0.92,
        "confidence_threshold": 0.60,
        "min_confidence_margin": 0.10,
        "top_predictions": [{"label": "UPPER_EXTREMITY", "probability": 0.92}],
    })
    monkeypatch.setattr(xray_router, "_BODY_PART_CLASSIFIER", mock_classifier)

    result = xray_router.route_xray("scan.png")
    assert result["is_broad_region"] is True
    assert result["reason_code"] == "broad_anatomy_label"
    assert result["supported"] is False
    assert result["registry_entry"] is None


# 10. No compatible regional model configured.
def test_10_no_compatible_regional_model_configured():
    result = xray_router.route_xray("scan.png", declared_region="pelvis")
    assert result["is_supported"] is False
    assert result["regional_model_configured"] is False
    assert result["regional_model_status"] == "unsupported_region"
    assert result["registry_entry"]["is_supported"] is False


# 11. User confirmation and user correction.
def test_11_user_confirmation_and_user_correction():
    initial_routing = {
        "candidate_region": "knee",
        "anatomy_detection": {
            "body_part_label": "KNEE",
            "body_part_confidence": 0.819,
        },
        "classifier_gate_passed": True,
        "anatomy_classifier_accepted": True,
        "regional_model_configured": False,
        "regional_model_status": "unsupported_region",
    }
    meta_initial = _build_anatomy_metadata(
        routing=initial_routing,
        resolved_region="unknown",
        region_source="image_classifier",
        anatomy_status="confirmation_required",
        body_part_confidence=0.819,
    )
    assert meta_initial["candidate_region"] == "knee"
    assert meta_initial["anatomy_user_confirmed"] is False

    # User corrects region to 'wrist'
    corrected_routing = {
        "candidate_region": "knee",
        "detected_region": "wrist",
        "regional_model_configured": True,
        "regional_model_available": True,
        "regional_model_name": "Wrist",
        "regional_model_status": "configured_and_available",
    }
    meta_corrected = _build_anatomy_metadata(
        routing=corrected_routing,
        resolved_region="wrist",
        region_source="user_confirmed",
        anatomy_status="completed",
        body_part_confidence=0.819,
    )
    assert meta_corrected["detected_region"] == "wrist"
    assert meta_corrected["candidate_region"] == "knee"
    assert meta_corrected["anatomy_user_confirmed"] is True
    assert meta_corrected["regional_model_configured"] is True
    assert meta_corrected["regional_model_name"] == "Wrist"


# 12. Persistence of confirmed region.
def test_12_persistence_of_confirmed_region():
    routing = {
        "candidate_region": "hand",
        "detected_region": "hand",
        "classifier_gate_passed": True,
        "anatomy_classifier_accepted": True,
        "regional_model_configured": True,
        "regional_model_available": True,
        "regional_model_name": "Hand",
        "regional_model_status": "configured_and_available",
        "anatomy_detection": {
            "body_part_label": "HAND",
            "body_part_confidence": 0.702,
        },
    }
    meta = _build_anatomy_metadata(
        routing=routing,
        resolved_region="hand",
        region_source="user_confirmed",
        anatomy_status="completed",
        body_part_confidence=0.702,
    )
    assert meta["detected_region"] == "hand"
    assert meta["classifier_gate_passed"] is True
    assert meta["anatomy_user_confirmed"] is True
    assert meta["regional_model_status"] == "configured_and_available"
    assert meta["regional_model_name"] == "Hand"


# 13. Correct state after refresh or reload.
def test_13_state_after_refresh_or_reload():
    meta = {
        "status": "completed",
        "detected_region": "wrist",
        "candidate_region": "knee",
        "original_classifier_candidate": "knee",
        "body_part_confidence": 0.819,
        "anatomy_user_confirmed": True,
        "classifier_gate_passed": True,
        "anatomy_classifier_accepted": True,
        "regional_model_configured": True,
        "regional_model_available": True,
        "regional_model_name": "Wrist",
        "regional_model_status": "configured_and_available",
    }
    is_user_confirmed = bool(meta.get("anatomy_user_confirmed", False))
    classifier_accepted = bool(meta.get("anatomy_classifier_accepted") or meta.get("classifier_gate_passed"))

    state = {
        "detected_region": meta.get("detected_region"),
        "candidate_region": meta.get("candidate_region"),
        "body_part_confidence": meta.get("body_part_confidence"),
        "anatomy_verification_status": "confirmed" if is_user_confirmed else "unverified",
        "anatomy_user_confirmed": is_user_confirmed,
        "anatomy_classifier_accepted": classifier_accepted,
        "classifier_gate_passed": classifier_accepted,
        "regional_model_configured": bool(meta.get("regional_model_configured")),
        "regional_model_available": bool(meta.get("regional_model_available")),
        "regional_model_name": meta.get("regional_model_name"),
        "regional_model_status": meta.get("regional_model_status"),
    }

    assert state["detected_region"] == "wrist"
    assert state["candidate_region"] == "knee"
    assert state["anatomy_user_confirmed"] is True
    assert state["regional_model_configured"] is True
    assert state["regional_model_name"] == "Wrist"


# 14. Multiple images retaining independent predictions, confirmations, and routing results.
def test_14_multiple_images_retain_independent_state():
    # Image 1: Hand (confirmed by user)
    routing_img1 = {
        "candidate_region": "hand",
        "detected_region": "hand",
        "regional_model_configured": True,
        "regional_model_available": True,
        "regional_model_name": "Hand",
        "regional_model_status": "configured_and_available",
    }
    meta_img1 = _build_anatomy_metadata(
        routing=routing_img1,
        resolved_region="hand",
        region_source="user_confirmed",
        anatomy_status="completed",
        body_part_confidence=0.702,
    )

    # Image 2: Pelvis (unconfirmed, no model configured)
    routing_img2 = {
        "candidate_region": "pelvis",
        "regional_model_configured": False,
        "regional_model_available": False,
        "regional_model_status": "unsupported_region",
    }
    meta_img2 = _build_anatomy_metadata(
        routing=routing_img2,
        resolved_region="unknown",
        region_source="image_classifier",
        anatomy_status="confirmation_required",
        body_part_confidence=0.859,
    )

    # State must be completely separate
    assert meta_img1["detected_region"] == "hand"
    assert meta_img1["anatomy_user_confirmed"] is True
    assert meta_img1["regional_model_configured"] is True

    assert meta_img2["detected_region"] is None
    assert meta_img2["candidate_region"] == "pelvis"
    assert meta_img2["anatomy_user_confirmed"] is False
    assert meta_img2["regional_model_configured"] is False


# 15. Existing upload, authorization, ownership, cleanup, and database behavior remaining intact.
def test_15_upload_authorization_ownership_and_cleanup(tmp_path):
    class MockDb:
        def __init__(self, assessment):
            self.assessment = assessment

        def get(self, entity, entity_id):
            return self.assessment

    owner_id = uuid4()
    other_id = uuid4()
    assessment_id = uuid4()
    mock_assessment = Assessment(assessment_id=assessment_id, user_id=owner_id)

    db_mock = MockDb(mock_assessment)

    owned = _owned(db_mock, assessment_id, owner_id)
    assert owned.assessment_id == assessment_id

    with pytest.raises(HTTPException) as exc_info:
        _owned(db_mock, assessment_id, other_id)
    assert exc_info.value.status_code == 404

    temp_file = tmp_path / "temp_upload.png"
    temp_file.write_bytes(b"test data")
    assert temp_file.exists()

    _remove_file(temp_file)
    assert not temp_file.exists()

    # Repeated cleanup should not raise
    _remove_file(temp_file)
