import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Set up paths
backend_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(backend_dir))

from app.ai.xray_router import (
    route_xray,
    _refine_broad_upper_extremity_candidate,
    UPPER_EXTREMITY_ANATOMY,
    LOWER_EXTREMITY_ANATOMY,
)

def run_tests():
    print("=" * 75)
    print("GOAL 1 FORENSIC VERIFICATION: STRICT ANATOMICAL BOUNDARY ENFORCEMENT")
    print("=" * 75)
    
    all_passed = True
    uploads_dir = backend_dir / "uploads" / "xrays"

    # -------------------------------------------------------------------------
    # PART 1: REAL RADIOGRAPH FILE ROUTING TESTS
    # -------------------------------------------------------------------------
    print("\n[PART 1: REAL RADIOGRAPH INFERENCE & ROUTING]")
    real_test_cases = [
        {
            "name": "09_ankle.png (LOWER_EXTREMITY 43.27% -> ELBOW 85.8% rejected)",
            "file": "54f31ae08f364399bd4d462285049246.png",
            "expected_primary": "LOWER_EXTREMITY",
            "expected_final_region": "lower extremity",
            "expected_specialist_status": "rejected_anatomical_mismatch",
            "expected_gate_passed": False,
            "expected_reg_model_status": None,
        },
        {
            "name": "12_elbow (1).png (KNEE 52.32% -> ELBOW 89.44% rejected, must NOT become ELBOW)",
            "file": "7d79fce06a214d0badc4f1c702473718.png",
            "expected_primary": "KNEE",
            "expected_final_region": "knee",
            "expected_specialist_status": "rejected_anatomical_mismatch",
            "expected_gate_passed": False,
            "expected_reg_model_status": None,
        },
        {
            "name": "08_knee.png (KNEE 60.13% confident -> specialist NOT invoked, remains knee)",
            "file": "10653bb005a8444a8c7f30b894af6bea.png",
            "expected_primary": "KNEE",
            "expected_final_region": "knee",
            "expected_specialist_status": None,
            "expected_gate_passed": True,
            "expected_reg_model_status": "unsupported_region",
        },
        {
            "name": "15_ankle.png (ANKLE 55.45% -> ELBOW rejected, remains ankle)",
            "file": "b0a43092ef5645b09bd64fbf754fe571.png",
            "expected_primary": "ANKLE",
            "expected_final_region": "ankle",
            "expected_specialist_status": "rejected_anatomical_mismatch",
            "expected_gate_passed": False,
            "expected_reg_model_status": None,
        },
    ]

    for tc in real_test_cases:
        path = uploads_dir / tc["file"]
        print(f"\n--- Testing: {tc['name']} ---")
        if not path.is_file():
            print(f"File not found: {path}")
            continue

        result = route_xray(str(path))
        cand_region = result.get("candidate_region")
        conf = result.get("body_part_confidence")
        gate_passed = bool(result.get("classifier_gate_passed") or result.get("anatomy_classifier_accepted"))
        conf_req = bool(result.get("specific_region_confirmation_required") or (not gate_passed))
        reg_model_status = result.get("regional_model_status")
        reg_model_name = result.get("regional_model_name")
        anatomy_output = result.get("anatomy_detection") or {}
        primary_res = anatomy_output.get("primary_classifier_result", {})
        spec_res = anatomy_output.get("upper_extremity_refinement") or {}

        print(f"Primary classifier: {primary_res.get('candidate_body_part')} ({primary_res.get('confidence')})")
        if spec_res:
            print(f"Specialist refinement: status={spec_res.get('status')}, specialist_candidate={spec_res.get('candidate_body_part')}, confidence={spec_res.get('confidence')}")
        else:
            print("Specialist refinement: None (bypassed for confident lower-extremity candidate)")
        print(f"Candidate region: {cand_region} (conf={conf})")
        print(f"Classifier gate passed: {gate_passed}")
        print(f"Confirmation required: {conf_req}")
        print(f"Regional model status: {reg_model_status} ({reg_model_name})")

        # Assertions
        if tc["expected_final_region"] != cand_region:
            print(f"FAIL: Expected candidate_region '{tc['expected_final_region']}', got '{cand_region}'")
            all_passed = False
        else:
            print(f"PASS: candidate_region matches '{tc['expected_final_region']}'")

        if tc["expected_specialist_status"] is None:
            if spec_res.get("status") is not None:
                print(f"FAIL: Expected no specialist invocation (None), got status '{spec_res.get('status')}'")
                all_passed = False
            else:
                print("PASS: Specialist was correctly NOT invoked on confident primary candidate")
        else:
            if spec_res.get("status") != tc["expected_specialist_status"]:
                print(f"FAIL: Expected specialist status '{tc['expected_specialist_status']}', got '{spec_res.get('status')}'")
                all_passed = False
            else:
                print(f"PASS: specialist status matches '{tc['expected_specialist_status']}'")

        if gate_passed != tc["expected_gate_passed"]:
            print(f"FAIL: Expected classifier_gate_passed={tc['expected_gate_passed']}, got {gate_passed}")
            all_passed = False
        else:
            print(f"PASS: classifier_gate_passed matches {tc['expected_gate_passed']}")

    # -------------------------------------------------------------------------
    # PART 2: DIRECT ANATOMICAL COMPATIBILITY UNIT REGRESSION TESTS
    # -------------------------------------------------------------------------
    print("\n\n[PART 2: DIRECT ANATOMICAL COMPATIBILITY UNIT TESTS]")

    mock_specialist = MagicMock()
    mock_specialist.predict_file.return_value = {
        "candidate_body_part": "ELBOW",
        "confidence": 0.895,
        "second_candidate_body_part": "WRIST",
        "second_confidence": 0.05,
        "top_two_margin": 0.845,
        "accepted": True,
        "model_name": "upper_extremity_specialist_convnext",
    }

    unit_test_cases = [
        {
            "desc": "LOWER_EXTREMITY -> ELBOW: rejected as anatomical mismatch",
            "primary": {"candidate_body_part": "LOWER_EXTREMITY", "body_part_confidence": 0.45, "status": "unverified"},
            "expect_specialist_status": "rejected_anatomical_mismatch",
            "expect_final_candidate": "LOWER_EXTREMITY",
        },
        {
            "desc": "ANKLE -> ELBOW: rejected as anatomical mismatch",
            "primary": {"candidate_body_part": "ANKLE", "body_part_confidence": 0.55, "status": "unverified"},
            "expect_specialist_status": "rejected_anatomical_mismatch",
            "expect_final_candidate": "ANKLE",
        },
        {
            "desc": "KNEE (below threshold 52%) -> ELBOW: rejected as anatomical mismatch, must not become ELBOW",
            "primary": {"candidate_body_part": "KNEE", "body_part_confidence": 0.52, "status": "unverified"},
            "expect_specialist_status": "rejected_anatomical_mismatch",
            "expect_final_candidate": "KNEE",
        },
        {
            "desc": "KNEE (above threshold 65%) -> ELBOW: specialist must not override KNEE",
            "primary": {"candidate_body_part": "KNEE", "body_part_confidence": 0.65, "status": "completed"},
            "expect_specialist_status": None, # Should be bypassed
            "expect_final_candidate": "KNEE",
        },
        {
            "desc": "Valid upper-extremity refinement (UPPER_EXTREMITY -> ELBOW): accepted",
            "primary": {"candidate_body_part": "UPPER_EXTREMITY", "body_part_confidence": 0.55, "status": "completed"},
            "expect_specialist_status": "accepted",
            "expect_final_candidate": "ELBOW",
        },
    ]

    with patch("app.ai.xray_router._get_upper_extremity_classifier", return_value=mock_specialist):
        for tc in unit_test_cases:
            print(f"\n--- Checking: {tc['desc']} ---")
            dummy_path = "dummy.png"
            primary_dict = dict(tc["primary"])
            out = _refine_broad_upper_extremity_candidate(dummy_path, primary_dict)

            spec_status = out.get("upper_extremity_refinement", {}).get("status") if out.get("upper_extremity_refinement") else None
            final_cand = out.get("candidate_body_part")

            print(f"Primary: {tc['primary']['candidate_body_part']} ({tc['primary']['body_part_confidence']})")
            print(f"Specialist status: {spec_status} (Expected: {tc['expect_specialist_status']})")
            print(f"Final candidate: {final_cand} (Expected: {tc['expect_final_candidate']})")

            if spec_status != tc["expect_specialist_status"]:
                print(f"FAIL: Expected specialist status '{tc['expect_specialist_status']}', got '{spec_status}'")
                all_passed = False
            else:
                print("PASS: specialist status matches expectation")

            if final_cand != tc["expect_final_candidate"]:
                print(f"FAIL: Expected final candidate '{tc['expect_final_candidate']}', got '{final_cand}'")
                all_passed = False
            else:
                print("PASS: final candidate matches expectation")

    print("\n" + "=" * 75)
    if all_passed:
        print("ALL VERIFICATION AND REGRESSION TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED.")
    print("=" * 75)
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
