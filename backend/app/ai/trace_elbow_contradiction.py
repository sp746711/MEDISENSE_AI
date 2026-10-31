import os
import sys
import json
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(backend_dir))

from app.ai.xray_router import route_xray, UPPER_EXTREMITY_ANATOMY, LOWER_EXTREMITY_ANATOMY
from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier
from app.ai.upper_extremity_anatomy_classifier import _default_model_path, UpperExtremityAnatomyClassifier

print("=" * 80)
print("RUNTIME TRACE: ELBOW CONTRADICTION INVESTIGATION")
print("=" * 80)

# Identify test images
images = {
    "05_elbow.png": backend_dir / "uploads" / "xrays" / "b33cbd834e634d318ed8427f5fc79158.png",
    "12_elbow (1).png": Path("D:/projects/MEDISENSE_AI/12_elbow (1).png"),
    "09_ankle.png": backend_dir / "uploads" / "xrays" / "54f31ae08f364399bd4d462285049246.png",
}

for name, path in images.items():
    if not path.is_file():
        # Check alternative locations
        found = list(backend_dir.glob(f"**/*{name.split('.')[0]}*"))
        print(f"File {name} not found directly at {path}, found: {found}")
        if found:
            path = found[0]
            images[name] = path

primary_clf = XRayAnatomyClassifier()
specialist_clf = UpperExtremityAnatomyClassifier()

for name, path in images.items():
    print(f"\n" + "-" * 70)
    print(f"IMAGE: {name} ({path})")
    print("-" * 70)

    # 1. Primary classifier directly
    raw_primary = primary_clf.predict_file(str(path))
    raw_class = raw_primary.get("body_part_label") or raw_primary.get("candidate_body_part")
    raw_conf = raw_primary.get("body_part_confidence")
    norm_candidate = str(raw_class or "").strip().upper().replace(" ", "_")

    # 2. Check is_ambiguous_hinge
    # Note: In earlier code: is_ambiguous_hinge = (norm_candidate == "KNEE" and raw_conf < 0.60)
    # In current code: variable has been completely removed from xray_router.py (evaluates to False)
    is_ambiguous_hinge = False

    # 3. Specialist directly
    spec_out = specialist_clf.predict_file(str(path))
    spec_class = spec_out.get("candidate_body_part")
    spec_conf = spec_out.get("confidence")
    spec_margin = spec_out.get("top_two_margin")

    # 4. Anatomical compatibility
    is_upper = norm_candidate in UPPER_EXTREMITY_ANATOMY or norm_candidate in {"UPPER_EXTREMITY", "ARM", "EXTREMITY"}
    is_lower = norm_candidate in LOWER_EXTREMITY_ANATOMY
    
    if is_lower:
        compat_result = "INCOMPATIBLE (Lower extremity candidate cannot be overridden by upper extremity specialist)"
        rejection_cond = "is_lower_extremity is True (candidate in LOWER_EXTREMITY_ANATOMY) -> status = 'rejected_anatomical_mismatch'"
    elif is_upper:
        compat_result = "COMPATIBLE"
        rejection_cond = "None (Specialist accepted if confidence >= 0.60 and margin >= 0.10)"
    else:
        compat_result = "INAPPLICABLE (Non-extremity body part)"
        rejection_cond = "Specialist not invoked"

    # 5. Full router execution
    routed = route_xray(str(path))
    refinement = (routed.get("anatomy_detection") or {}).get("upper_extremity_refinement") or {}
    
    final_cand = routed.get("canonical_region") or routed.get("region")
    final_conf = routed.get("body_part_confidence")
    gate_passed = routed.get("classifier_gate_passed")
    routing_status = routed.get("status")

    print(f"1. Raw Primary Class & Confidence:        {raw_class} ({float(raw_conf):.4f})")
    print(f"2. Normalized Primary Candidate:          {norm_candidate}")
    print(f"3. is_ambiguous_hinge:                    {is_ambiguous_hinge} (legacy exception removed)")
    print(f"4. Specialist Class & Confidence:         {spec_class} ({float(spec_conf):.4f})")
    print(f"5. Specialist Margin:                     {float(spec_margin):.4f}")
    print(f"6. Anatomical Compatibility Result:       {compat_result}")
    print(f"7. Exact Rejection / Acceptance Condition: {rejection_cond}")
    print(f"8. Refinement Status in Router:           {refinement.get('status')}")
    print(f"9. Final Candidate & Confidence:          {final_cand} ({float(final_conf):.4f})")
    print(f"10. Classifier Gate Passed:               {gate_passed}")
    print(f"11. Routing Status:                       {routing_status}")
