import os
import sys
from pathlib import Path
from collections import defaultdict
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

# Canonical 10 test images uploaded by user in browser testing
TEN_TEST_IMAGES = [
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

def evaluate_set(dataset, mode_name):
    y_true = []
    y_pred = []
    scores = []
    margins = []
    records = []

    for item in dataset:
        name = item.get("name") or item.get("original_name")
        stored = item.get("stored") or item.get("stored_name")
        truth = item.get("truth") or item.get("verified_ground_truth")
        path = uploads_dir / stored
        if not path.is_file():
            continue

        with Image.open(path) as img:
            t = preprocess(img, mode=mode_name)
            out = session.run([output_name], {input_name: t})[0][0]
            top_order = np.argsort(out)[::-1]
            top1_lbl = labels[top_order[0]]
            top1_val = float(out[top_order[0]])
            top2_val = float(out[top_order[1]])
            margin = top1_val - top2_val

        y_true.append(truth)
        y_pred.append(top1_lbl)
        scores.append(top1_val)
        margins.append(margin)
        records.append({
            "name": name,
            "truth": truth,
            "pred": top1_lbl,
            "score": top1_val,
            "margin": margin,
            "correct": (top1_lbl == truth),
            "gate_passed": (top1_val >= 0.60 and margin >= 0.10),
        })

    # Metrics
    classes_in_set = sorted(list(set(y_true) | set(y_pred)))
    matrix = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    per_class = {}
    for c in sorted(list(set(y_true))):
        tp = matrix[c][c]
        fn = sum(matrix[c][p] for p in matrix[c] if p != c)
        fp = sum(matrix[t][c] for t in matrix if t != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = tp + fn
        per_class[c] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "support": support}

    total_correct = sum(1 for r in records if r["correct"])
    acc = total_correct / len(records) if records else 0.0

    # High confidence errors (score >= 0.60 and margin >= 0.10, but wrong label)
    hc_errors = [r for r in records if r["gate_passed"] and not r["correct"]]

    return {
        "accuracy": acc,
        "total": len(records),
        "correct": total_correct,
        "per_class": per_class,
        "matrix": matrix,
        "records": records,
        "hc_errors": hc_errors,
        "classes_in_set": classes_in_set,
    }

def print_evaluation(res, title):
    print("\n" + "=" * 90)
    print(f"EVALUATION REPORT: {title}")
    print("=" * 90)
    print(f"Overall Accuracy: {res['correct']}/{res['total']} ({res['accuracy']*100:.2f}%)")
    print(f"High-Confidence Errors (gate passed >= 60% but incorrect): {len(res['hc_errors'])}")
    if res['hc_errors']:
        for h in res['hc_errors']:
            print(f"  CRITICAL ERROR: {h['name']}: Truth={h['truth']}, Predicted={h['pred']} ({h['score']*100:.1f}%, margin={h['margin']*100:.1f}%)")
    else:
        print("  None. All errors were appropriately held for user confirmation.")

    print("\nPER-CLASS PERFORMANCE:")
    print(f"{'Class':<18} | {'Support':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for c, m in res['per_class'].items():
        print(f"{c:<18} | {m['support']:<8} | {m['precision']*100:8.1f}% | {m['recall']*100:8.1f}% | {m['f1']*100:8.1f}%")

    print("\nSAMPLE BREAKDOWN:")
    for r in res['records']:
        status = "CORRECT" if r["correct"] else ("HIGH_CONF_ERR" if r["gate_passed"] else "HELD_FOR_CONFIRM")
        print(f"  {r['name']:<32} | Truth: {r['truth']:<15} | Pred: {r['pred']:<15} ({r['score']*100:5.1f}%) | [{status}]")

def main():
    print("RUNNING REPRODUCIBLE EVALUATION ACROSS VERIFIED RADIOGRAPHS")
    
    # 1. 10 User Test Radiographs under center_crop (current default)
    res_10_cc = evaluate_set(TEN_TEST_IMAGES, "center_crop")
    print_evaluation(res_10_cc, "10 USER TEST RADIOGRAPHS (CENTER CROP - PRODUCTION DEFAULT)")

    # 2. 10 User Test Radiographs under direct_resize
    res_10_dr = evaluate_set(TEN_TEST_IMAGES, "direct_resize")
    print_evaluation(res_10_dr, "10 USER TEST RADIOGRAPHS (DIRECT RESIZE)")

    # 3. 20 Verified Manifest Images under center_crop
    res_20_cc = evaluate_set(VERIFIED_TEST_MANIFEST, "center_crop")
    print_evaluation(res_20_cc, "20 VERIFIED MANIFEST RADIOGRAPHS (CENTER CROP)")

    # 4. 20 Verified Manifest Images under direct_resize
    res_20_dr = evaluate_set(VERIFIED_TEST_MANIFEST, "direct_resize")
    print_evaluation(res_20_dr, "20 VERIFIED MANIFEST RADIOGRAPHS (DIRECT RESIZE)")

if __name__ == "__main__":
    main()
