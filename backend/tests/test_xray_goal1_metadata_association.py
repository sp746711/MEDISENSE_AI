"""Goal 1 regression tests for anatomy persistence and explicit report pairing.

Run from backend root:
    .venv\\Scripts\\python.exe -m pytest tests\\test_xray_goal1_metadata_association.py -v
"""
from app.routes.xray import (
    _build_anatomy_metadata,
    _build_association_safety,
)


def test_anatomy_metadata_persists_confidence_candidate_and_classifier_output():
    routing = {
        "candidate_region": "knee",
        "anatomy_detection": {
            "body_part_label": "KNEE",
            "body_part_confidence": 0.91,
            "model_version": "convnext_tiny_onnx_33class",
            "top_candidates": [
                {"label": "KNEE", "score": 0.91},
                {"label": "LEG", "score": 0.05},
            ],
        },
    }
    metadata = _build_anatomy_metadata(
        routing=routing,
        resolved_region="knee",
        region_source="image_classifier",
        anatomy_status="completed",
        body_part_confidence=0.91,
    )
    assert metadata["detected_region"] == "knee"
    assert metadata["candidate_region"] == "knee"
    assert metadata["body_part_confidence"] == 0.91
    assert metadata["classifier_version"] == "convnext_tiny_onnx_33class"
    assert metadata["top_candidates"][0]["label"] == "KNEE"
    assert metadata["classifier_output"]["body_part_label"] == "KNEE"


def test_anatomy_metadata_does_not_mark_unknown_region_as_detected():
    metadata = _build_anatomy_metadata(
        routing={"anatomy_detection": {"candidate_body_part": "hand", "body_part_confidence": 0.44}},
        resolved_region="unknown",
        region_source="image_classifier",
        anatomy_status="unverified",
        body_part_confidence=0.44,
    )
    assert metadata["detected_region"] is None
    assert metadata["candidate_region"] == "hand"
    assert metadata["status"] == "UNVERIFIED"


def test_confirmed_pair_with_matching_recognized_region_can_combine():
    result = _build_association_safety(
        status_value="confirmed",
        image_region="chest",
        report_region="chest",
        region_recognized=True,
    )
    assert result["association_explicitly_confirmed"] is True
    assert result["region_check"] == "MATCH"
    assert result["can_combine"] is True


def test_matching_regions_without_explicit_confirmation_cannot_combine():
    result = _build_association_safety(
        status_value="pending",
        image_region="chest",
        report_region="chest",
        region_recognized=True,
    )
    assert result["region_compatible"] is True
    assert result["association_explicitly_confirmed"] is False
    assert result["can_combine"] is False


def test_confirmed_pair_with_region_mismatch_stays_separate():
    result = _build_association_safety(
        status_value="confirmed",
        image_region="chest",
        report_region="knee",
        region_recognized=True,
    )
    assert result["status"] == "CONFIRMED"
    assert result["region_check"] == "MISMATCH"
    assert result["can_combine"] is False


def test_unknown_or_unrecognized_regions_stay_unverified():
    result = _build_association_safety(
        status_value="confirmed",
        image_region="unknown",
        report_region="chest",
        region_recognized=True,
    )
    assert result["region_check"] == "UNVERIFIED"
    assert result["can_combine"] is False

    result2 = _build_association_safety(
        status_value="confirmed",
        image_region="chest",
        report_region="odd label",
        region_recognized=False,
    )
    assert result2["region_check"] == "UNVERIFIED"
    assert result2["can_combine"] is False


def test_rejected_pair_never_combines_even_if_regions_match():
    result = _build_association_safety(
        status_value="rejected",
        image_region="chest",
        report_region="chest",
        region_recognized=True,
    )
    assert result["region_check"] == "MATCH"
    assert result["can_combine"] is False

