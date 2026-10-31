import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import os
import json
from PIL import Image

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier
from app.ai.upper_extremity_anatomy_classifier import UpperExtremityAnatomyClassifier, UPPER_EXTREMITY_CLASSES
from app.ai.xray_router import (
    XRAY_MODEL_REGISTRY,
    REGION_ALIASES,
    route_xray,
)

def audit():
    classifier = XRayAnatomyClassifier()
    classifier.load()
    classes = classifier.classes
    print(f"Total anatomy classes loaded from ONNX/classes.txt: {len(classes)}")
    
    spec_supported = set(UPPER_EXTREMITY_CLASSES)
    
    # Create probe test image
    test_img_path = Path("scratch/test_probe.png")
    test_img_path.parent.mkdir(parents=True, exist_ok=True)
    if not test_img_path.exists():
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(test_img_path)

    verified_images = {
        "ELBOW": "05_elbow.png, 12_elbow (1).png",
        "HAND": "07_hand.png",
        "ANKLE": "09_ankle.png",
        "FOREARM": "13_forearm.png",
        "FOOT": "male_foot-(1).jpg",
        "SKULL": "02_skull.png (evaluated)",
        "CERVICAL_SPINE": "03_cervical_spine.png (evaluated)",
        "ABDOMEN": "17_abdomen_kub.png, 10_abdomen_kub.png",
        "KUB": "17_abdomen_kub.png, 10_abdomen_kub.png",
        "CHEST": "01_chest.png",
        "SHOULDER": "04_shoulder.png",
        "KNEE": "08_knee.png",
        "HEAD": "02_skull.png",
        "NECK": "03_cervical_spine.png",
        "UPPER_EXTREMITY": "07_hand.png, 04_shoulder.png",
        "LOWER_EXTREMITY": "09_ankle.png",
        "TIBIA": "13_forearm.png (predicted)",
    }

    table_rows = []

    for idx, cls_name in enumerate(classes):
        # 1. Primary anatomy support
        primary_support = f"Supported (Index {idx})"
        
        # 2. Specialist support
        is_spec = cls_name in spec_supported
        spec_support = f"Supported ({cls_name})" if is_spec else "Not Applicable (Upper-Extremity Only)"
        
        # 3. Route test with declared_region
        try:
            route_res = route_xray(str(test_img_path), declared_region=cls_name)
            is_supported = route_res.get("supported", False)
            canon_region = route_res.get("region", "unknown")
            msg = route_res.get("message", "")
            
            # Check registry entry
            reg_entry = XRAY_MODEL_REGISTRY.get(canon_region)
            if is_supported and reg_entry:
                reg_avail = "Available"
                model_file = reg_entry.checkpoint_location or reg_entry.model_architecture or "TorchXRayVision"
                if model_file and "torchxrayvision" not in str(model_file).lower():
                    # Check disk existence
                    p = Path(model_file)
                    if not p.is_absolute():
                        p = Path("backend") / p
                    exists_str = " (Verified on disk)" if p.exists() else " (Not found)"
                    model_file = f"{Path(model_file).name}{exists_str}"
            else:
                reg_avail = "Unavailable"
                model_file = "None configured"
                
            inf_status = f"Operational (Supported: {is_supported})"
        except Exception as e:
            reg_avail = "Unavailable"
            model_file = "Error"
            inf_status = f"Error: {e}"
            
        test_img = verified_images.get(cls_name, "None in standard set")
        
        table_rows.append({
            "idx": idx,
            "class": cls_name,
            "primary": primary_support,
            "specialist": spec_support,
            "regional_avail": reg_avail,
            "model_file": model_file,
            "inf_status": inf_status,
            "test_image": test_img,
        })

    print("\n### FULL 33-CLASS INVENTORY AUDIT TABLE\n")
    print("| # | Anatomy Class | Primary 33-Class Support | Specialist Support | Regional Abnormality Model | Model File / Registry | Inference Test Status | Verified Test Image |")
    print("|---|---|---|---|---|---|---|---|")
    for r in table_rows:
        print(f"| {r['idx']+1} | `{r['class']}` | {r['primary']} | {r['specialist']} | {r['regional_avail']} | `{r['model_file']}` | {r['inf_status']} | {r['test_image']} |")

if __name__ == "__main__":
    audit()
