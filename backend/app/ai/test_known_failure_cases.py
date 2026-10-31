import sys
from pathlib import Path
from PIL import Image

backend_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(backend_dir))

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier

test_cases = [
    ("Case A: 05_elbow.png", backend_dir / "uploads" / "xrays" / "b33cbd834e634d318ed8427f5fc79158.png", "ELBOW"),
    ("Case B: 12_elbow (1).png", Path("D:/projects/MEDISENSE_AI/12_elbow (1).png"), "ELBOW"),
    ("Case C: 09_ankle.png", backend_dir / "uploads" / "xrays" / "54f31ae08f364399bd4d462285049246.png", "ANKLE"),
    ("Case D: 15_ankle.png", backend_dir / "uploads" / "xrays" / "b0a43092ef5645b09bd64fbf754fe571.png", "ANKLE"),
    ("Case E: 03_cervical_spine.png", None, "CERVICAL_SPINE"),
]

# Find path for Case E
cerv_paths = list(backend_dir.glob("**/03_cervical_spine*")) or list(Path("D:/projects/MEDISENSE_AI").glob("**/03_cervical_spine*"))
if cerv_paths:
    test_cases[4] = ("Case E: 03_cervical_spine.png", cerv_paths[0], "CERVICAL_SPINE")
else:
    # check in uploads
    cerv_stored = list((backend_dir / "uploads" / "xrays").glob("*ae945ad25b354e1c9b492368c45e3558*"))
    if cerv_stored:
        test_cases[4] = ("Case E: 03_cervical_spine.png (stored)", cerv_stored[0], "CERVICAL_SPINE")

clf_crop = XRayAnatomyClassifier(preprocessing_mode="center_crop")
clf_resize = XRayAnatomyClassifier(preprocessing_mode="direct_resize")

print("=" * 90)
print(f"{'IMAGE CASE':<30} | {'MODE':<12} | {'PREDICTED':<16} | {'CONF':<7} | {'GT':<15} | {'MATCH?'}")
print("=" * 90)

for label, p, gt in test_cases:
    if not p or not Path(p).is_file():
        # search
        found = list(backend_dir.glob(f"**/*{label.split(':')[1].strip().split('.')[0]}*"))
        if found:
            p = found[0]
        else:
            print(f"Skipping {label}: file not found ({p})")
            continue

    im = Image.open(p)
    w, h = im.size
    aspect = round(w / h, 2)
    
    # 1. Center crop
    res_crop = clf_crop.predict_file(str(p))
    top1_c = res_crop.get("candidate_body_part")
    conf_c = res_crop.get("body_part_confidence")
    match_c = "CORRECT" if top1_c == gt else "WRONG"

    # 2. Direct resize
    res_resize = clf_resize.predict_file(str(p))
    top1_r = res_resize.get("candidate_body_part")
    conf_r = res_resize.get("body_part_confidence")
    match_r = "CORRECT" if top1_r == gt else "WRONG"

    print(f"{label[:29]:<30} | {'center_crop':<12} | {top1_c:<16} | {conf_c:<7.1%} | {gt:<15} | {match_c} (Dims: {w}x{h}, AR: {aspect})")
    print(f"{'':<30} | {'direct_resize':<12} | {top1_r:<16} | {conf_r:<7.1%} | {gt:<15} | {match_r}")
    
    # Print top-3
    top3_c = [(x['label'], f"{x['probability']:.1%}") for x in res_crop.get('top_predictions', [])[:3]]
    top3_r = [(x['label'], f"{x['probability']:.1%}") for x in res_resize.get('top_predictions', [])[:3]]
    print(f"   Crop Top-3:   {top3_c}")
    print(f"   Resize Top-3: {top3_r}")
    print("-" * 90)
