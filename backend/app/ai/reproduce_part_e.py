import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(backend_dir))

from app.ai.xray_router import route_xray
from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier
from app.ai.upper_extremity_anatomy_classifier import UpperExtremityAnatomyClassifier

print("=" * 80)
print("PART E: LIVE REPRODUCTION OF CRITICAL REGRESSION IMAGES")
print("=" * 80)

# Paths to the 5 critical images
target_images = {
    "05_elbow.png": backend_dir / "uploads/xrays/b33cbd834e634d318ed8427f5fc79158.png",
    "12_elbow (1).png": Path("D:/projects/MEDISENSE_AI/12_elbow (1).png"),
    "09_ankle.png": backend_dir / "uploads/xrays/54f31ae08f364399bd4d462285049246.png",
    "15_ankle.png": backend_dir / "uploads/xrays/b0a43092ef5645b09bd64fbf754fe571.png",
    "08_knee.png": backend_dir / "uploads/xrays/10653bb005a8444a8c7f30b894af6bea.png",
}

# Resolve alternative paths if needed
for name, p in target_images.items():
    if not p.is_file():
        matches = list(backend_dir.glob(f"**/*{name.split('.')[0]}*"))
        if matches:
            target_images[name] = matches[0]

primary_clf = XRayAnatomyClassifier(preprocessing_mode="center_crop")
spec_clf = UpperExtremityAnatomyClassifier()

for name, path in target_images.items():
    print(f"\n--- {name} ({path.name}) ---")
    
    # 1. Primary
    p_res = primary_clf.predict_file(str(path))
    p_class = p_res.get("candidate_body_part")
    p_conf = float(p_res.get("body_part_confidence") or 0.0)
    p_margin = float(p_res.get("top_two_margin") or 0.0)
    top3 = [(x['label'], round(float(x['probability']), 4)) for x in p_res.get('top_predictions', [])[:3]]
    
    # 2. Specialist
    s_res = spec_clf.predict_file(str(path))
    s_class = s_res.get("candidate_body_part")
    s_conf = float(s_res.get("confidence") or 0.0)
    s_margin = float(s_res.get("top_two_margin") or 0.0)

    # 3. Router
    routed = route_xray(str(path))
    detection = routed.get("anatomy_detection") or {}
    refinement = detection.get("upper_extremity_refinement") or {}
    cand_region = routed.get("candidate_region") or routed.get("region")
    gate_passed = routed.get("classifier_gate_passed")
    status = routed.get("status")
    ref_status = refinement.get("status")

    print(f"Primary Output:     {p_class} ({p_conf:.2%}), margin={p_margin:.4f}")
    print(f"Primary Top-3:      {top3}")
    print(f"Specialist Output:  {s_class} ({s_conf:.2%}), margin={s_margin:.4f}")
    print(f"Specialist Status:  {ref_status}")
    print(f"Final Candidate:    {cand_region}")
    print(f"Gate Passed:        {gate_passed}")
    print(f"Router Status:      {status}")
