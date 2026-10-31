import sys
from pathlib import Path
import hashlib
from PIL import Image

backend_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(backend_dir))

from app.ai.xray_router import route_xray
from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier

cases = [
    ("05_elbow.png", backend_dir / "uploads/xrays/b33cbd834e634d318ed8427f5fc79158.png", "ELBOW"),
    ("12_elbow (1).png", Path("D:/projects/MEDISENSE_AI/12_elbow (1).png"), "ELBOW"),
    ("09_ankle.png", backend_dir / "uploads/xrays/54f31ae08f364399bd4d462285049246.png", "ANKLE"),
    ("15_ankle.png", backend_dir / "uploads/xrays/b0a43092ef5645b09bd64fbf754fe571.png", "ANKLE"),
    ("03_cervical_spine.png", backend_dir / "uploads/xrays/03_cervical_spine.png", "CERVICAL_SPINE"),
    ("19_hand.png", backend_dir / "uploads/xrays/f5766c844fca47709d5450bce3f68cab.png", "HAND"),
    ("Normal Hand and Wrist X-ray.png", backend_dir / "uploads/xrays/1a651f4f4b1f473ab44ed7261f9334a4.png", "WRIST"),
]

print("=" * 90)
print("GOAL 1: 7 MANDATORY REGRESSION CASES FORENSIC EVALUATION")
print("=" * 90)

clf_letterbox = XRayAnatomyClassifier(preprocessing_mode="letterbox")
clf_letterbox.load()

clf_crop = XRayAnatomyClassifier(preprocessing_mode="center_crop")
clf_crop.load()

for name, p, expected in cases:
    if not p.is_file():
        # search
        candidates = list(backend_dir.glob(f"**/*{p.name}*"))
        if candidates:
            p = candidates[0]
        else:
            print(f"NOT FOUND: {name} at {p}")
            continue

    h = hashlib.sha256(p.read_bytes()).hexdigest()
    with Image.open(p) as img:
        dims = f"{img.width}x{img.height}"
        fmt = img.format

    res = route_xray(str(p))
    det = res.get("anatomy_detection") or {}
    spec = det.get("upper_extremity_refinement") or {}

    with Image.open(p) as im:
        raw_im = im.convert("RGB")
        probs_crop = clf_crop._run_inference_probabilities(raw_im)
        rank_crop = clf_crop._rank_probabilities(probs_crop)
        probs_lb = clf_letterbox._run_inference_probabilities(raw_im)
        rank_lb = clf_letterbox._rank_probabilities(probs_lb)

    print(f"\nImage: {name} | Verified Ground-Truth: {expected}")
    print(f"  Dimensions: {dims} | Format: {fmt} | SHA256: {h}")
    print(f"  [center_crop Primary] Top-1: {rank_crop['candidate']} ({rank_crop['confidence']*100:.2f}%) | Margin: {rank_crop['margin']*100:.2f}%")
    top3_crop = ", ".join([f"{x['label']} ({x['probability']*100:.1f}%)" for x in rank_crop['top_predictions'][:3]])
    print(f"    Top-3: {top3_crop}")
    print(f"  [letterbox Primary]   Top-1: {rank_lb['candidate']} ({rank_lb['confidence']*100:.2f}%) | Margin: {rank_lb['margin']*100:.2f}%")
    top3_lb = ", ".join([f"{x['label']} ({x['probability']*100:.1f}%)" for x in rank_lb['top_predictions'][:3]])
    print(f"    Top-3: {top3_lb}")
    print(f"  [Specialist Refinement]")
    print(f"    Invoked: {spec.get('invoked', False)} | Specialist Class: {spec.get('candidate_body_part', 'N/A')} ({spec.get('confidence', 0.0)*100:.2f}%)")
    print(f"    Compatibility Decision / Status: {spec.get('status', 'not_invoked')}")
    print(f"    Rejection Reason: {spec.get('rejection_reason', 'none')}")
    print(f"  [Router Decision & Gating]")
    print(f"    Final Candidate: {det.get('candidate_body_part') or res.get('candidate_region')} | Status: {res.get('status')}")
    print(f"    Confidence-Gate Result: {det.get('prediction_status')} | Verification Status: {res.get('anatomy_verification_status')}")
    print(f"    User Confirmation Required: {res.get('requires_user_confirmation', True)}")
    print(f"    Regional Model Routing: {res.get('model_type_selected') or 'None (Unavailable / Unconfirmed)'}")
    print("-" * 90)
