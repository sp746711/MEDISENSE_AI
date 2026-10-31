import sys
from pathlib import Path
sys.path.insert(0, str(Path("backend").resolve()))
from PIL import Image

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier
from app.ai.upper_extremity_anatomy_classifier import UpperExtremityAnatomyClassifier

image_map = {
    "12_elbow (1).png": ("81a009576f264985a15b63329ac51275.png", "ELBOW"),
    "13_forearm.png": ("dffc9cc5831245838edaea9b59811340.png", "FOREARM"),
    "09_ankle.png": ("e860ad152bfb4531ba03c606444a709e.png", "ANKLE"),
    "05_elbow.png": ("492e1df41ca74eb3a9f5373ce2c1128e.png", "ELBOW"),
    "foot-without-labels-copy-2.jpg": ("ee3a87cb306e4c7fbb9e2e27a905a73f.jpg", "FOOT"),
    "02_skull.png": ("d470ed0ad5544affad1eafa5d0f97064.png", "SKULL"),
    "03_cervical_spine.png": ("bbbdb4622162493c8c5542c6b6d9ed7e.png", "CERVICAL_SPINE"),
    "11_cervical_spine.png": ("3680dfccb1464960a3dcf62424bd442f.png", "CERVICAL_SPINE"),
    "male_foot-(1).jpg": ("777a21f3759849f880f6baf5fc7e18ce.jpg", "FOOT"),
}

upload_dir = Path("backend/uploads/xrays")
modes = ["center_crop", "direct_resize", "letterbox", "dual_ensemble"]

print("=" * 100)
print(f"{'IMAGE':32} | {'MODE':14} | {'PRED':15} | {'CONF':6} | {'MARGIN':6} | {'TOP-3'}")
print("=" * 100)

spec = UpperExtremityAnatomyClassifier()

for friendly, (stored, gt) in image_map.items():
    p = upload_dir / stored
    if not p.exists():
        print(f"MISSING: {friendly} ({p})")
        continue
    
    with Image.open(p) as im:
        w, h = im.size
    print(f"\n--- {friendly} (GT: {gt}, {w}x{h}) ---")
    
    for mode in modes:
        clf = XRayAnatomyClassifier(preprocessing_mode=mode)
        res = clf.predict_file(str(p))
        cand = res["candidate_body_part"]
        conf = res["body_part_confidence"] * 100
        margin = res["top_two_margin"] * 100
        top3 = ", ".join([f"{tp['label']}: {tp['probability']*100:.1f}%" for tp in res["top_predictions"][:3]])
        verdict = "MATCH" if cand == gt else "MISMATCH"
        print(f"{friendly:32} | {mode:14} | {cand:15} | {conf:5.1f}% | {margin:5.1f}% | [{verdict:8}] Top3: {top3}")
        
    try:
        s_res = spec.predict_file(str(p))
        s_top3 = ", ".join([f"{tp['label']}: {tp['probability']*100:.1f}%" for tp in s_res["top_predictions"][:3]])
        print(f"{friendly:32} | {'specialist_7cl':14} | {s_res['candidate_body_part']:15} | {s_res['confidence']*100:5.1f}% | {s_res['top_two_margin']*100:5.1f}% | [{s_res['status']}] Top3: {s_top3}")
    except Exception as e:
        print(f"Specialist error: {e}")
