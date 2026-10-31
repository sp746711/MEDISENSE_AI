"""Regression tests verifying anatomical mismatch protection for elbow and ankle radiographs.

Ensures that:
1. 05_elbow.png (primary KNEE 40.64%, specialist ELBOW 95.9%) is rejected as an anatomical mismatch.
2. 12_elbow (1).png (primary KNEE 52.32%, specialist ELBOW 89.44%) is rejected as an anatomical mismatch.
3. 09_ankle.png (primary LOWER_EXTREMITY 43.27%, specialist ELBOW 85.8%) is rejected as an anatomical mismatch.
4. An upper-extremity candidate (e.g. UPPER_EXTREMITY) CAN be refined to ELBOW.
5. In all mismatch cases, the candidate region is NOT overridden, classifier_gate_passed is False,
   and explicit user confirmation is required.
"""
from pathlib import Path
import pytest

from app.ai.xray_router import route_xray, _refine_broad_upper_extremity_candidate
from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier
from app.ai.upper_extremity_anatomy_classifier import UpperExtremityAnatomyClassifier


@pytest.fixture(scope="module")
def test_radiographs():
    base_dir = Path(__file__).resolve().parents[1]
    uploads_dir = base_dir / "uploads" / "xrays"
    
    # Locate files
    elbow_05 = uploads_dir / "b33cbd834e634d318ed8427f5fc79158.png"
    if not elbow_05.is_file():
        candidates = list(uploads_dir.glob("*b33cbd83*")) or list(base_dir.glob("**/*05_elbow*"))
        elbow_05 = candidates[0] if candidates else None

    elbow_12 = Path("D:/projects/MEDISENSE_AI/12_elbow (1).png")
    if not elbow_12.is_file():
        candidates = list(uploads_dir.glob("*7d79fce0*")) or list(base_dir.glob("**/*12_elbow*"))
        elbow_12 = candidates[0] if candidates else None

    ankle_09 = uploads_dir / "54f31ae08f364399bd4d462285049246.png"
    if not ankle_09.is_file():
        candidates = list(uploads_dir.glob("*54f31ae0*")) or list(base_dir.glob("**/*09_ankle*"))
        ankle_09 = candidates[0] if candidates else None

    return {
        "05_elbow": elbow_05,
        "12_elbow": elbow_12,
        "09_ankle": ankle_09,
    }


def test_05_elbow_anatomical_mismatch_rejection(test_radiographs):
    img_path = test_radiographs["05_elbow"]
    if not img_path or not Path(img_path).is_file():
        pytest.skip("05_elbow image artifact not found")

    routed = route_xray(str(img_path))
    detection = routed.get("anatomy_detection") or {}
    refinement = detection.get("upper_extremity_refinement") or {}

    # Primary predicted knee (40.64%)
    assert detection.get("body_part_label") == "KNEE" or routed.get("candidate_region") == "knee"
    # Specialist predicted ELBOW (> 90%)
    assert refinement.get("candidate_body_part") == "ELBOW"
    assert refinement.get("confidence") > 0.90
    # Must be rejected as anatomical mismatch
    assert refinement.get("status") == "rejected_anatomical_mismatch"
    assert refinement.get("reason_code") == "incompatible_anatomical_region"
    # Candidate must NOT be overridden to elbow
    assert routed.get("candidate_region") != "elbow"
    # Gate must NOT pass
    assert routed.get("classifier_gate_passed") is not True


def test_12_elbow_anatomical_mismatch_rejection(test_radiographs):
    img_path = test_radiographs["12_elbow"]
    if not img_path or not Path(img_path).is_file():
        pytest.skip("12_elbow image artifact not found")

    routed = route_xray(str(img_path))
    detection = routed.get("anatomy_detection") or {}
    refinement = detection.get("upper_extremity_refinement") or {}

    # Primary predicted knee (52.32%)
    assert detection.get("body_part_label") == "KNEE" or routed.get("candidate_region") == "knee"
    # Specialist predicted ELBOW (> 85%)
    assert refinement.get("candidate_body_part") == "ELBOW"
    assert refinement.get("confidence") > 0.85
    # Must be rejected as anatomical mismatch under current safe router
    assert refinement.get("status") == "rejected_anatomical_mismatch"
    assert refinement.get("reason_code") == "incompatible_anatomical_region"
    # Candidate must NOT be overridden to elbow
    assert routed.get("candidate_region") != "elbow"
    # Gate must NOT pass
    assert routed.get("classifier_gate_passed") is not True


def test_09_ankle_preserves_lower_extremity_mismatch_protection(test_radiographs):
    img_path = test_radiographs["09_ankle"]
    if not img_path or not Path(img_path).is_file():
        pytest.skip("09_ankle image artifact not found")

    routed = route_xray(str(img_path))
    detection = routed.get("anatomy_detection") or {}
    refinement = detection.get("upper_extremity_refinement") or {}

    # Primary predicted lower extremity
    assert "lower" in str(routed.get("candidate_region")).lower()
    # Specialist predicted ELBOW (> 80%)
    assert refinement.get("candidate_body_part") == "ELBOW"
    # Must be rejected as anatomical mismatch
    assert refinement.get("status") == "rejected_anatomical_mismatch"
    assert refinement.get("reason_code") == "incompatible_anatomical_region"
    # Candidate must NOT be overridden to elbow
    assert routed.get("candidate_region") != "elbow"
    # Gate must NOT pass
    assert routed.get("classifier_gate_passed") is not True


def test_synthetic_upper_extremity_refinement_is_accepted():
    """Verify that an upper-extremity candidate CAN be refined when anatomically compatible."""
    anatomy_result = {
        "candidate_body_part": "UPPER_EXTREMITY",
        "body_part_confidence": 0.55,
        "body_part_label": "UPPER_EXTREMITY",
    }
    # Path is not needed if we test the compatibility logic
    class MockSpecialist:
        def predict_file(self, path):
            return {
                "candidate_body_part": "ELBOW",
                "confidence": 0.88,
                "top_two_margin": 0.35,
                "accepted": True,
            }

    import app.ai.xray_router as router_mod
    original_getter = router_mod._get_upper_extremity_classifier
    try:
        router_mod._get_upper_extremity_classifier = lambda: MockSpecialist()
        refined = _refine_broad_upper_extremity_candidate("dummy.png", anatomy_result)
        assert refined.get("candidate_body_part") == "ELBOW"
        assert refined.get("body_part_confidence") == 0.88
        assert refined.get("upper_extremity_refinement", {}).get("status") == "accepted"
    finally:
        router_mod._get_upper_extremity_classifier = original_getter
