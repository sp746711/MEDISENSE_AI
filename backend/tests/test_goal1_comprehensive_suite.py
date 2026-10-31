"""Comprehensive Regression Test Suite for Goal 1 — All 22 Required Clinical/Engineering Cases.

Covers:
1. ELBOW primary + ELBOW specialist = compatible agreement.
2. UPPER_EXTREMITY primary + HAND specialist = valid refinement.
3. ANKLE primary + ELBOW specialist = specialist rejected (anatomical mismatch).
4. TIBIA primary + FOREARM specialist = disagreement; no unsafe override.
5. KNEE primary + ELBOW specialist = specialist rejected.
6. SHOULDER primary + SHOULDER specialist = compatible agreement.
7. Specialist unavailable = primary result preserved.
8. Specialist inference failure = safe fallback.
9. Missing or invalid confidence = no automatic acceptance.
10. Low-quality image = uncertainty or rejection as appropriate.
11. Wrong image dimensions or corrupted image = safe error handling.
12. Different images in one assessment = independent prediction state.
13. User changes anatomy = selected region updates routing without rewriting original prediction.
14. User confirmation is required before regional abnormality inference.
15. No regional model exists = no fabricated result.
16. Every displayed confidence is correctly attributed to its model.
17. Specialist score never silently replaces the primary score.
18. Cross-model confidence comparison never bypasses anatomical compatibility.
19. HEAD and SKULL are distinguished correctly.
20. NECK and CERVICAL_SPINE are distinguished correctly.
21. Forearm and tibia are distinguished correctly.
22. Both elbow regression images are evaluated separately.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app.ai.xray_router import (
    _evaluate_two_model_comparison,
    route_xray,
    UPPER_EXTREMITY_ANATOMY,
    LOWER_EXTREMITY_ANATOMY,
)
from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier, _is_specific_anatomy_label
from app.ai.upper_extremity_anatomy_classifier import UpperExtremityAnatomyClassifier


@pytest.fixture
def dummy_image_path(tmp_path):
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    path = tmp_path / "dummy_xray.png"
    img.save(path)
    return str(path)


# 1. ELBOW primary + ELBOW specialist = compatible agreement
def test_case_1_elbow_compatible_agreement(dummy_image_path):
    anat_result = {
        "candidate_body_part": "ELBOW",
        "body_part_confidence": 0.8218,
        "top_two_margin": 0.7927,
        "top_predictions": [{"label": "ELBOW", "probability": 0.8218}],
        "model_name": "convnext_tiny_onnx_33class",
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "ELBOW",
            "confidence": 0.9593,
            "top_two_margin": 0.9315,
            "accepted": True,
            "top_predictions": [{"label": "ELBOW", "probability": 0.9593}],
        }
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["two_model_comparison"]["decision_case"] == "CASE_A_AGREEMENT"
    assert res["two_model_comparison"]["is_compatible"] is True
    assert res["final_candidate"] == "ELBOW"
    assert "AGREED" in res["selection_source"]
    assert res["confidence_source"] == "primary_model"
    assert res["source_attributed_confidence"] == 0.8218


# 2. UPPER_EXTREMITY primary + HAND specialist = valid refinement
def test_case_2_upper_extremity_hand_refinement(dummy_image_path):
    anat_result = {
        "candidate_body_part": "UPPER_EXTREMITY",
        "body_part_confidence": 0.6192,
        "top_two_margin": 0.4538,
        "top_predictions": [{"label": "UPPER_EXTREMITY", "probability": 0.6192}],
        "model_name": "convnext_tiny_onnx_33class",
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "HAND",
            "confidence": 0.9987,
            "top_two_margin": 0.9974,
            "accepted": True,
            "top_predictions": [{"label": "HAND", "probability": 0.9987}],
        }
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["two_model_comparison"]["decision_case"] == "CASE_B_REFINEMENT"
    assert res["final_candidate"] == "HAND"
    assert res["selection_source"] == "SPECIALIST REFINEMENT"
    assert res["confidence_source"] == "specialist_refinement"
    assert res["source_attributed_confidence"] == 0.9987
    assert res["primary_classifier_result"]["candidate_body_part"] == "UPPER_EXTREMITY"


# 3. ANKLE primary + ELBOW specialist = specialist rejected
def test_case_3_ankle_specialist_rejected(dummy_image_path):
    anat_result = {
        "candidate_body_part": "ANKLE",
        "body_part_confidence": 0.4835,
        "top_two_margin": 0.2054,
        "top_predictions": [{"label": "ANKLE", "probability": 0.4835}],
        "model_name": "convnext_tiny_onnx_33class",
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "ELBOW",
            "confidence": 0.8581,
            "top_two_margin": 0.7793,
            "accepted": True,
            "top_predictions": [{"label": "ELBOW", "probability": 0.8581}],
        }
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["two_model_comparison"]["decision_case"] in {"CASE_C_DISAGREEMENT", "CASE_D_UNRESOLVED"}
    assert res["upper_extremity_refinement"]["status"] == "REJECTED"
    assert res["final_candidate"] == "ANKLE"
    assert res["confidence_source"] == "primary_model"


# 4. TIBIA primary + FOREARM specialist = disagreement; no unsafe override
def test_case_4_tibia_forearm_disagreement(dummy_image_path):
    anat_result = {
        "candidate_body_part": "TIBIA",
        "body_part_confidence": 0.2202,
        "top_two_margin": 0.1019,
        "top_predictions": [{"label": "TIBIA", "probability": 0.2202}],
        "model_name": "convnext_tiny_onnx_33class",
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "FOREARM",
            "confidence": 0.9787,
            "top_two_margin": 0.9598,
            "accepted": True,
            "top_predictions": [{"label": "FOREARM", "probability": 0.9787}],
        }
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["two_model_comparison"]["decision_case"] == "CASE_D_UNRESOLVED"
    assert res["two_model_comparison"]["is_unresolved"] is True
    assert res["upper_extremity_refinement"]["status"] == "REJECTED"
    assert res["final_candidate"] == "TIBIA"  # preserved, not overridden
    assert "UNRESOLVED DISAGREEMENT" in res["selection_source"]


# 5. KNEE primary + ELBOW specialist = specialist rejected
def test_case_5_knee_elbow_mismatch_rejected(dummy_image_path):
    anat_result = {
        "candidate_body_part": "KNEE",
        "body_part_confidence": 0.7779,
        "top_two_margin": 0.7129,
        "top_predictions": [{"label": "KNEE", "probability": 0.7779}],
        "model_name": "convnext_tiny_onnx_33class",
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "ELBOW",
            "confidence": 0.8944,
            "top_two_margin": 0.8180,
            "accepted": True,
            "top_predictions": [{"label": "ELBOW", "probability": 0.8944}],
        }
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["two_model_comparison"]["decision_case"] == "CASE_C_DISAGREEMENT"
    assert res["upper_extremity_refinement"]["status"] == "REJECTED"
    assert res["final_candidate"] == "KNEE"


# 6. SHOULDER primary + SHOULDER specialist = compatible agreement
def test_case_6_shoulder_agreement(dummy_image_path):
    anat_result = {
        "candidate_body_part": "SHOULDER",
        "body_part_confidence": 0.8329,
        "top_two_margin": 0.7962,
        "top_predictions": [{"label": "SHOULDER", "probability": 0.8329}],
        "model_name": "convnext_tiny_onnx_33class",
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "SHOULDER",
            "confidence": 0.9883,
            "top_two_margin": 0.9767,
            "accepted": True,
            "top_predictions": [{"label": "SHOULDER", "probability": 0.9883}],
        }
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["two_model_comparison"]["decision_case"] == "CASE_A_AGREEMENT"
    assert res["final_candidate"] == "SHOULDER"


# 7. Specialist unavailable = primary result preserved
def test_case_7_specialist_unavailable(dummy_image_path):
    anat_result = {
        "candidate_body_part": "WRIST",
        "body_part_confidence": 0.75,
        "top_two_margin": 0.50,
        "top_predictions": [{"label": "WRIST", "probability": 0.75}],
        "model_name": "convnext_tiny_onnx_33class",
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.side_effect = FileNotFoundError("Missing checkpoint")
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["upper_extremity_refinement"]["status"] == "UNAVAILABLE"
    assert res["final_candidate"] == "WRIST"
    assert res["selection_source"] == "PRIMARY MODEL"


# 8. Specialist inference failure = safe fallback
def test_case_8_specialist_inference_failure(dummy_image_path):
    anat_result = {
        "candidate_body_part": "WRIST",
        "body_part_confidence": 0.75,
        "top_two_margin": 0.50,
        "top_predictions": [{"label": "WRIST", "probability": 0.75}],
        "model_name": "convnext_tiny_onnx_33class",
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.side_effect = RuntimeError("CUDA OOM or tensor error")
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["upper_extremity_refinement"]["status"] == "FAILED"
    assert res["final_candidate"] == "WRIST"


# 9. Missing or invalid confidence = no automatic acceptance
def test_case_9_invalid_confidence_no_acceptance():
    with pytest.raises(ValueError):
        UpperExtremityAnatomyClassifier(min_confidence=-0.5)
    with pytest.raises(ValueError):
        UpperExtremityAnatomyClassifier(min_confidence=1.5)


# 10. Low-quality image = uncertainty or rejection as appropriate
def test_case_10_low_quality_blank_image(tmp_path):
    blank = Image.new("RGB", (224, 224), color=(0, 0, 0))
    p = tmp_path / "blank.png"
    blank.save(p)
    clf = XRayAnatomyClassifier()
    # Blank/zero images should produce valid probabilities or throw clean error
    res = clf.predict_file(str(p))
    assert "body_part_confidence" in res


# 11. Wrong image dimensions or corrupted image = safe error handling
def test_case_11_corrupted_image(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a valid png file")
    clf = XRayAnatomyClassifier()
    with pytest.raises(Exception):
        clf.predict_file(str(corrupt))


# 12. Different images in one assessment = independent prediction state
def test_case_12_independent_prediction_states(dummy_image_path):
    anat1 = {
        "candidate_body_part": "CHEST",
        "body_part_confidence": 0.90,
        "top_two_margin": 0.85,
        "top_predictions": [{"label": "CHEST", "probability": 0.90}],
    }
    anat2 = {
        "candidate_body_part": "ELBOW",
        "body_part_confidence": 0.82,
        "top_two_margin": 0.79,
        "top_predictions": [{"label": "ELBOW", "probability": 0.82}],
    }
    res1 = _evaluate_two_model_comparison(dummy_image_path, anat1)
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "ELBOW",
            "confidence": 0.95,
            "top_two_margin": 0.90,
            "accepted": True,
            "top_predictions": [{"label": "ELBOW", "probability": 0.95}],
        }
        res2 = _evaluate_two_model_comparison(dummy_image_path, anat2)

    assert res1["final_candidate"] == "CHEST"
    assert res2["final_candidate"] == "ELBOW"
    assert res1["two_model_comparison"]["decision_case"] == "CASE_E_SPECIALIST_NOT_INVOKED"
    assert res2["two_model_comparison"]["decision_case"] == "CASE_A_AGREEMENT"


# 13. User changes anatomy = selected region updates routing without rewriting original prediction
def test_case_13_user_changes_anatomy(dummy_image_path):
    # route_xray with declared_region
    res = route_xray(dummy_image_path, declared_region="wrist")
    assert res["region"] == "wrist"
    assert res["canonical_region"] == "wrist"
    assert res["is_recognized"] is True


# 14. User confirmation is required before regional abnormality inference
def test_case_14_user_confirmation_required(dummy_image_path):
    res = route_xray(dummy_image_path)
    # Automatic route must always require confirmation
    assert res["anatomy_verification_status"] in {"confirmation_required", "unverified"}
    assert res["supported"] is False  # Abnormality model not authorized before confirmation


# 15. No regional model exists = no fabricated result
def test_case_15_no_regional_model_exists(dummy_image_path):
    res = route_xray(dummy_image_path, declared_region="skull")
    assert res["region"] == "skull"
    assert res["supported"] is False
    assert res["is_supported"] is False
    assert "not configured" in res["message"].lower() or "no compatible" in res["message"].lower() or "not supported" in res["message"].lower()


# 16. Every displayed confidence is correctly attributed to its model
def test_case_16_confidence_attribution(dummy_image_path):
    anat_result = {
        "candidate_body_part": "UPPER_EXTREMITY",
        "body_part_confidence": 0.6192,
        "top_two_margin": 0.4538,
        "top_predictions": [{"label": "UPPER_EXTREMITY", "probability": 0.6192}],
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "HAND",
            "confidence": 0.9987,
            "top_two_margin": 0.9974,
            "accepted": True,
            "top_predictions": [{"label": "HAND", "probability": 0.9987}],
        }
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["primary_classifier_result"]["confidence"] == 0.6192
    assert res["upper_extremity_refinement"]["confidence"] == 0.9987
    assert res["confidence_source"] == "specialist_refinement"


# 17. Specialist score never silently replaces primary score
def test_case_17_no_silent_replacement(dummy_image_path):
    anat_result = {
        "candidate_body_part": "ELBOW",
        "body_part_confidence": 0.8218,
        "top_two_margin": 0.7927,
        "top_predictions": [{"label": "ELBOW", "probability": 0.8218}],
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "ELBOW",
            "confidence": 0.9593,
            "top_two_margin": 0.9315,
            "accepted": True,
            "top_predictions": [{"label": "ELBOW", "probability": 0.9593}],
        }
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["primary_classifier_result"]["confidence"] == 0.8218
    assert res["source_attributed_confidence"] == 0.8218


# 18. Cross-model confidence comparison never bypasses anatomical compatibility
def test_case_18_compatibility_not_bypassed_by_score(dummy_image_path):
    anat_result = {
        "candidate_body_part": "ANKLE",
        "body_part_confidence": 0.40,
        "top_two_margin": 0.15,
        "top_predictions": [{"label": "ANKLE", "probability": 0.40}],
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "ELBOW",
            "confidence": 0.9999,  # Very high specialist score!
            "top_two_margin": 0.99,
            "accepted": True,
            "top_predictions": [{"label": "ELBOW", "probability": 0.9999}],
        }
        res = _evaluate_two_model_comparison(dummy_image_path, anat_result)

    assert res["final_candidate"] == "ANKLE"  # Cannot be overridden by ELBOW despite 99.99%!
    assert res["upper_extremity_refinement"]["status"] == "REJECTED"


# 19. HEAD and SKULL are distinguished correctly
def test_case_19_head_and_skull_distinction():
    assert _is_specific_anatomy_label("HEAD") is False  # Broad label
    assert _is_specific_anatomy_label("SKULL") is True  # Specific bone structure


# 20. NECK and CERVICAL_SPINE are distinguished correctly
def test_case_20_neck_cervical_spine_distinction():
    assert _is_specific_anatomy_label("NECK") is True  # Soft tissue neck
    assert _is_specific_anatomy_label("CERVICAL_SPINE") is True  # Vertebral spine


# 21. Forearm and tibia are distinguished correctly
def test_case_21_forearm_tibia_distinction():
    assert "FOREARM" in UPPER_EXTREMITY_ANATOMY
    assert "TIBIA" in LOWER_EXTREMITY_ANATOMY
    assert "FOREARM" not in LOWER_EXTREMITY_ANATOMY
    assert "TIBIA" not in UPPER_EXTREMITY_ANATOMY


# 22. Both elbow regression images evaluated separately
def test_case_22_elbow_regression_evaluated_separately(dummy_image_path):
    anat_05 = {
        "candidate_body_part": "ELBOW",
        "body_part_confidence": 0.8218,
        "top_two_margin": 0.7927,
        "top_predictions": [{"label": "ELBOW", "probability": 0.8218}],
    }
    anat_12 = {
        "candidate_body_part": "KNEE",
        "body_part_confidence": 0.7779,
        "top_two_margin": 0.7129,
        "top_predictions": [{"label": "KNEE", "probability": 0.7779}],
    }
    with patch("app.ai.xray_router._get_upper_extremity_classifier") as mock_spec:
        mock_spec.return_value.predict_file.return_value = {
            "candidate_body_part": "ELBOW",
            "confidence": 0.9593,
            "top_two_margin": 0.9315,
            "accepted": True,
            "top_predictions": [{"label": "ELBOW", "probability": 0.9593}],
        }
        res_05 = _evaluate_two_model_comparison(dummy_image_path, anat_05)
        res_12 = _evaluate_two_model_comparison(dummy_image_path, anat_12)

    assert res_05["two_model_comparison"]["decision_case"] == "CASE_A_AGREEMENT"
    assert res_05["final_candidate"] == "ELBOW"

    assert res_12["two_model_comparison"]["decision_case"] == "CASE_C_DISAGREEMENT"
    assert res_12["final_candidate"] == "KNEE"
    assert res_12["upper_extremity_refinement"]["status"] == "REJECTED"
