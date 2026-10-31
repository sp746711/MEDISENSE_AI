import os
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

backend_dir = Path(__file__).resolve().parents[2]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

model_path = backend_dir / "models" / "xray" / "bodypart_full" / "model.onnx"
labels_path = backend_dir / "models" / "xray" / "bodypart_full" / "classes.txt"
uploads_dir = backend_dir / "uploads" / "xrays"

labels = [line.strip().upper() for line in labels_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

from app.ai.goal1_evaluation import VERIFIED_TEST_MANIFEST

test_10_images = [
    {"name": "01_chest.png", "stored": "65de04ed2358482392b4f93d2a1ac6e6.png", "truth": "CHEST"},
    {"name": "02_skull.png", "stored": "7ec9ae95e9f34c27919ae9764c6e3ac9.png", "truth": "SKULL"},
    {"name": "03_cervical_spine.png", "stored": "b420b368e93a422e893adfea5d1fbb30.png", "truth": "CERVICAL_SPINE"},
    {"name": "04_shoulder.png", "stored": "931f6bda95934d6480f6995a0e08b18b.png", "truth": "SHOULDER"},
    {"name": "05_elbow.png", "stored": "c31df909386b4956a1a0c80f7ffb825f.png", "truth": "ELBOW"},
    {"name": "06_wrist.png", "stored": "9fa2c9d58fbf4a0f8ce91c34878df301.png", "truth": "WRIST"},
    {"name": "07_hand.png", "stored": "acf006c78480402a9e1f7295995b45dc.png", "truth": "HAND"},
    {"name": "08_knee.png", "stored": "10653bb005a8444a8c7f30b894af6bea.png", "truth": "KNEE"},
    {"name": "09_ankle.png", "stored": "54f31ae08f364399bd4d462285049246.png", "truth": "ANKLE"},
    {"name": "10_abdomen_kub.png", "stored": "867c1e5a0fc54221887a2906e48b9a66.png", "truth": "ABDOMEN"},
]

def preprocess(img, mode="center_crop"):
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if mode == "center_crop":
        scale = 224 / min(w, h)
        new_w = max(224, int(round(w * scale)))
        new_h = max(224, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        left = (new_w - 224) // 2
        top = (new_h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
    elif mode == "direct_resize":
        img = img.resize((224, 224), Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)

def evaluate_dataset(dataset, title):
    print("\n" + "=" * 80)
    print(f"EVALUATION: {title}")
    print("=" * 80)
    print(f"{'Image':<30} | {'Truth':<15} | {'Center Crop Top-1':<22} | {'Direct Resize Top-1':<22}")
    print("-" * 100)

    cc_correct = 0
    dr_correct = 0
    total = 0

    for item in dataset:
        name = item.get("name") or item.get("original_name")
        stored = item.get("stored") or item.get("stored_name")
        truth = item.get("truth") or item.get("verified_ground_truth")
        path = uploads_dir / stored
        if not path.is_file():
            print(f"Missing: {name}")
            continue

        with Image.open(path) as img:
            t_cc = preprocess(img, "center_crop")
            out_cc = session.run([output_name], {input_name: t_cc})[0][0]
            top_cc = labels[int(np.argmax(out_cc))]
            score_cc = float(np.max(out_cc))

            t_dr = preprocess(img, "direct_resize")
            out_dr = session.run([output_name], {input_name: t_dr})[0][0]
            top_dr = labels[int(np.argmax(out_dr))]
            score_dr = float(np.max(out_dr))

        total += 1
        # Check anatomical equivalence: HEAD == SKULL, NECK == CERVICAL_SPINE, etc.
        # But for strict check:
        cc_match = (top_cc == truth)
        dr_match = (top_dr == truth)
        if cc_match: cc_correct += 1
        if dr_match: dr_correct += 1

        print(f"{name:<30} | {truth:<15} | {top_cc:<14} ({score_cc*100:4.1f}%) | {top_dr:<14} ({score_dr*100:4.1f}%)")

    print("-" * 100)
    print(f"Center Crop Exact Accuracy:   {cc_correct}/{total} ({cc_correct/total*100:.1f}%)")
    print(f"Direct Resize Exact Accuracy: {dr_correct}/{total} ({dr_correct/total*100:.1f}%)")

if __name__ == "__main__":
    evaluate_dataset(test_10_images, "10 USER TEST RADIOGRAPHS")
    evaluate_dataset(VERIFIED_TEST_MANIFEST, "20 VERIFIED REAL RADIOGRAPHS MANIFEST")
