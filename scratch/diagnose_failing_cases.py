import sys
from pathlib import Path
sys.path.insert(0, str(Path("backend").resolve()))
import os
import hashlib
from PIL import Image
import numpy as np

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier
from app.ai.upper_extremity_anatomy_classifier import UpperExtremityAnatomyClassifier
from app.ai.evaluate_fullbody_dataset import unique_entries

def find_image_by_filename(target_name):
    # Check current directory, uploads directory, or unique_entries
    p = Path(target_name)
    if p.exists():
        return p
    p = Path("backend") / target_name
    if p.exists():
        return p
    for entry in unique_entries:
        if target_name in entry.get("filename", ""):
            ep = Path(entry["path"])
            if ep.exists():
                return ep
    # Search recursively in uploads
    for p in Path("backend/uploads").rglob("*.*"):
        if target_name in p.name:
            return p
    # Search in root
    for p in Path(".").glob(f"*{target_name}*"):
        return p
    return None

targets = [
    ("12_elbow (1).png", "ELBOW"),
    ("13_forearm.png", "FOREARM"),
    ("09_ankle.png", "ANKLE"),
    ("foot-without-labels-copy-2.jpg", "FOOT"),
    ("03_cervical_spine.png", "CERVICAL_SPINE"),
    ("11_cervical_spine.png", "CERVICAL_SPINE"),
    ("02_skull.png", "SKULL"),
    ("05_elbow.png", "ELBOW"),
]

modes = ["center_crop", "direct_resize", "letterbox", "dual_ensemble"]

print("=" * 80)
print("FORENSIC ANALYSIS OF FAILING IMAGES ACROSS PREPROCESSING MODES")
print("=" * 80)

specialist = UpperExtremityAnatomyClassifier()

for target_name, gt in targets:
    img_path = find_image_by_filename(target_name)
    print(f"\n--- Target: {target_name} (GT: {gt}) ---")
    if not img_path:
        print("  FILE NOT FOUND ON DISK!")
        continue
    print(f"  Path: {img_path} (Size: {img_path.stat().st_size} bytes)")
    
    with Image.open(img_path) as im:
        w, h = im.size
        print(f"  Dimensions: {w}x{h}, Mode: {im.mode}")
        
    for mode in modes:
        clf = XRayAnatomyClassifier(preprocessing_mode=mode)
        res = clf.predict_file(str(img_path))
        cand = res["candidate_body_part"]
        conf = res["body_part_confidence"]
        margin = res["top_two_margin"]
        top3 = ", ".join([f"{tp['label']}: {tp['probability']*100:.1f}%" for tp in res["top_predictions"][:3]])
        correct = "CORRECT" if cand == gt else "WRONG"
        print(f"  [{mode:14}] -> {cand:15} ({conf*100:5.1f}%, margin {margin*100:4.1f}%) [{correct:7}] | Top3: {top3}")

    # Also test specialist
    try:
        s_res = specialist.predict_file(str(img_path))
        s_top3 = ", ".join([f"{tp['label']}: {tp['probability']*100:.1f}%" for tp in s_res["top_predictions"][:3]])
        print(f"  [Specialist 7-cl] -> {s_res['candidate_body_part']:15} ({s_res['confidence']*100:5.1f}%, status={s_res['status']}) | Top3: {s_top3}")
    except Exception as e:
        print(f"  [Specialist 7-cl] -> Error: {e}")
