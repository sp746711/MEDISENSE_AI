import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

backend_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(backend_dir))

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier, _read_labels, _default_labels_path, _default_model_path

classes_33 = _read_labels(_default_labels_path(_default_model_path()))

# Complete verified ground-truth dataset from repository artifacts
DATASET_ENTRIES = [
    # 20 Manifest Radiographs
    {"filename": "Artboard-1.png", "path": backend_dir / "uploads/xrays/d2b19d448e5d482db01f8b387eb2919a.png", "gt": "CHEST"},
    {"filename": "18_chest.png", "path": backend_dir / "uploads/xrays/ac61b6482f12421cbc5de22be8ac589f.png", "gt": "CHEST"},
    {"filename": "chest_xray.png", "path": backend_dir / "uploads/xrays/157ecfe677a94d6e9833c159f6ed2a8e.png", "gt": "CHEST"},
    {"filename": "19_hand.png", "path": backend_dir / "uploads/xrays/f5766c844fca47709d5450bce3f68cab.png", "gt": "HAND"},
    {"filename": "male_hand.jpeg", "path": backend_dir / "uploads/xrays/c22bb83329fb480bb9b7a1f02e01477a.jpg", "gt": "HAND"},
    {"filename": "Normal Hand and Wrist X-ray.png", "path": backend_dir / "uploads/xrays/1a651f4f4b1f473ab44ed7261f9334a4.png", "gt": "WRIST"},
    {"filename": "20_wrist.png", "path": backend_dir / "uploads/xrays/1403b6783a3947e5b7002c9c174887c3.png", "gt": "WRIST"},
    {"filename": "Left Wrist X-Ray_ PA and Lateral Views.png", "path": backend_dir / "uploads/xrays/17e8bde459e7417f8bdf0a79be22de45.png", "gt": "WRIST"},
    {"filename": "13_forearm.png", "path": backend_dir / "uploads/xrays/049742074cde4db3bc6066bc12f80438.png", "gt": "FOREARM"},
    {"filename": "12_elbow (1).png", "path": Path("D:/projects/MEDISENSE_AI/12_elbow (1).png"), "gt": "ELBOW"},
    {"filename": "elbow-without-labels-copy-1.jpg", "path": backend_dir / "uploads/xrays/f4605f42c33b4c48965a68fdc8969638.jpg", "gt": "ELBOW"},
    {"filename": "15_ankle.png", "path": backend_dir / "uploads/xrays/b0a43092ef5645b09bd64fbf754fe571.png", "gt": "ANKLE"},
    {"filename": "Right Ankle X-Ray Views.png", "path": backend_dir / "uploads/xrays/728673426fcf422c82bf5a593ba086e6.png", "gt": "ANKLE"},
    {"filename": "top-foot-without-labels-copy.jpg", "path": backend_dir / "uploads/xrays/b9857a5d7d0e47d489c8b7327afbabe2.jpg", "gt": "FOOT"},
    {"filename": "male_foot-(1).jpg", "path": backend_dir / "uploads/xrays/fd15ea3de8b54835bc732a1e18348665.jpg", "gt": "FOOT"},
    {"filename": "11_cervical_spine.png", "path": backend_dir / "uploads/xrays/ae945ad25b354e1c9b492368c45e3558.png", "gt": "CERVICAL_SPINE"},
    {"filename": "pelvis-without-labels-copy.jpg", "path": backend_dir / "uploads/xrays/222b2bca333647ed9d7ed2edaf445cea.jpg", "gt": "PELVIS"},
    {"filename": "hip-joint-with-no-labels.jpg", "path": backend_dir / "uploads/xrays/90143c1ccf4243cc84c463f50b019cad.jpg", "gt": "HIP"},
    {"filename": "17_abdomen_kub.png", "path": backend_dir / "uploads/xrays/e768a04631844eacbaca2eac8261649f.png", "gt": "ABDOMEN"},
    {"filename": "lumbar-spine-without-labels.jpg", "path": backend_dir / "uploads/xrays/0e00db9e44a643588de3838f9f620414.jpg", "gt": "LUMBAR_SPINE"},
    
    # 10 User Test Suite Radiographs
    {"filename": "01_chest.png", "path": backend_dir / "uploads/xrays/01_chest.png", "gt": "CHEST"},
    {"filename": "02_skull.png", "path": backend_dir / "uploads/xrays/02_skull.png", "gt": "SKULL"},
    {"filename": "03_cervical_spine.png", "path": backend_dir / "uploads/xrays/03_cervical_spine.png", "gt": "CERVICAL_SPINE"},
    {"filename": "04_shoulder.png", "path": backend_dir / "uploads/xrays/04_shoulder.png", "gt": "SHOULDER"},
    {"filename": "05_elbow.png", "path": backend_dir / "uploads/xrays/b33cbd834e634d318ed8427f5fc79158.png", "gt": "ELBOW"},
    {"filename": "06_wrist.png", "path": backend_dir / "uploads/xrays/06_wrist.png", "gt": "WRIST"},
    {"filename": "07_hand.png", "path": backend_dir / "uploads/xrays/07_hand.png", "gt": "HAND"},
    {"filename": "08_knee.png", "path": backend_dir / "uploads/xrays/08_knee.png", "gt": "KNEE"},
    {"filename": "09_ankle.png", "path": backend_dir / "uploads/xrays/54f31ae08f364399bd4d462285049246.png", "gt": "ANKLE"},
    {"filename": "10_abdomen_kub.png", "path": backend_dir / "uploads/xrays/10_abdomen_kub.png", "gt": "ABDOMEN"},

    # Additional Distinct Ground Truth Radiographs from Unique Audit
    {"filename": "0020e11e84c54adbabf3ba38174fb73a.png", "path": backend_dir / "uploads/xrays/0020e11e84c54adbabf3ba38174fb73a.png", "gt": "SKULL"},
    {"filename": "0bb2278bff65442182788cf476c97cd8.png", "path": backend_dir / "uploads/xrays/0bb2278bff65442182788cf476c97cd8.png", "gt": "KNEE"},
    {"filename": "0d280557f1c9460f8cfe2ae7a5abaf68.png", "path": backend_dir / "uploads/xrays/0d280557f1c9460f8cfe2ae7a5abaf68.png", "gt": "KNEE"},
    {"filename": "274fcfd8d96442c2a9356f9c2874ddb0.png", "path": backend_dir / "uploads/xrays/274fcfd8d96442c2a9356f9c2874ddb0.png", "gt": "CHEST"},
    {"filename": "4eb207e90e8842a39876eb8456805469.png", "path": backend_dir / "uploads/xrays/4eb207e90e8842a39876eb8456805469.png", "gt": "CHEST"},
]

# Deduplicate by SHA256
unique_entries = []
seen_hashes = set()

for entry in DATASET_ENTRIES:
    p = entry["path"]
    if not p or not Path(p).is_file():
        # Search by filename
        matches = list(backend_dir.glob(f"**/{entry['filename']}"))
        if matches:
            p = matches[0]
            entry["path"] = p
        else:
            continue
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if h not in seen_hashes:
        seen_hashes.add(h)
        entry["sha256"] = h
        unique_entries.append(entry)

print(f"Total Unique Verified Radiographs: {len(unique_entries)}")

def evaluate_pipeline(mode: str):
    clf = XRayAnatomyClassifier(preprocessing_mode=mode)
    
    y_true = []
    y_pred = []
    top3_correct = 0
    accepted_correct = 0
    accepted_total = 0
    rejected_total = 0
    high_conf_wrong = 0
    results = []

    for item in unique_entries:
        try:
            res = clf.predict_file(str(item["path"]))
        except ValueError as exc:
            # Card/placeholder rejection or invalid image
            results.append({
                "filename": item["filename"],
                "gt": item["gt"],
                "pred": "REJECTED_PLACEHOLDER",
                "conf": 0.0,
                "margin": 0.0,
                "is_top1": False,
                "is_top3": False,
                "gate_passed": False,
                "rejection_reason": str(exc)
            })
            rejected_total += 1
            y_true.append(item["gt"])
            y_pred.append("REJECTED_PLACEHOLDER")
            continue

        pred = res.get("candidate_body_part")
        conf = float(res.get("body_part_confidence") or 0.0)
        margin = float(res.get("top_two_margin") or 0.0)
        gt = item["gt"]

        top_preds = [x["label"] for x in res.get("top_predictions", [])[:3]]
        
        is_top1 = (pred == gt)
        is_top3 = (gt in top_preds)
        gate_passed = (conf >= 0.60 and margin >= 0.10)

        y_true.append(gt)
        y_pred.append(pred)

        if is_top3:
            top3_correct += 1
        if gate_passed:
            accepted_total += 1
            if is_top1:
                accepted_correct += 1
            else:
                high_conf_wrong += 1
        else:
            rejected_total += 1

        results.append({
            "filename": item["filename"],
            "gt": gt,
            "pred": pred,
            "conf": conf,
            "margin": margin,
            "is_top1": is_top1,
            "is_top3": is_top3,
            "gate_passed": gate_passed
        })

    total = len(y_true)
    top1_acc = np.mean([1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)])
    top3_acc = top3_correct / total
    abstention_rate = rejected_total / total
    accepted_acc = (accepted_correct / accepted_total) if accepted_total > 0 else 0.0

    # Per-class counts, precision, recall, F1
    eval_classes = sorted(list(set(y_true)))
    per_class = {}
    for c in eval_classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(1 for yt in y_true if yt == c)
        per_class[c] = {
            "support": support,
            "correct": tp,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    macro_prec = np.mean([v["precision"] for v in per_class.values()])
    macro_rec = np.mean([v["recall"] for v in per_class.values()])
    weighted_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / total

    return {
        "mode": mode,
        "total": total,
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "abstention_rate": abstention_rate,
        "accepted_total": accepted_total,
        "accepted_acc": accepted_acc,
        "high_conf_wrong": high_conf_wrong,
        "macro_f1": macro_f1,
        "macro_prec": macro_prec,
        "macro_rec": macro_rec,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "eval_classes": eval_classes,
        "unsupported_classes": sorted(list(set(classes_33) - set(eval_classes)))
    }

# Run both
crop_metrics = evaluate_pipeline("center_crop")
resize_metrics = evaluate_pipeline("direct_resize")

print("\n" + "=" * 80)
print("EVALUATION RESULTS COMPARISON ACROSS VERIFIED RADIOGRAPHS")
print("=" * 80)
print(f"{'Metric':<35} | {'center_crop':<18} | {'direct_resize':<18}")
print("-" * 80)
print(f"{'Total Verified Samples':<35} | {crop_metrics['total']:<18} | {resize_metrics['total']:<18}")
print(f"{'Top-1 Accuracy':<35} | {crop_metrics['top1_acc']:<18.2%} | {resize_metrics['top1_acc']:<18.2%}")
print(f"{'Top-3 Accuracy':<35} | {crop_metrics['top3_acc']:<18.2%} | {resize_metrics['top3_acc']:<18.2%}")
print(f"{'Macro Precision':<35} | {crop_metrics['macro_prec']:<18.2%} | {resize_metrics['macro_prec']:<18.2%}")
print(f"{'Macro Recall':<35} | {crop_metrics['macro_rec']:<18.2%} | {resize_metrics['macro_rec']:<18.2%}")
print(f"{'Macro F1-Score':<35} | {crop_metrics['macro_f1']:<18.2%} | {resize_metrics['macro_f1']:<18.2%}")
print(f"{'Weighted F1-Score':<35} | {crop_metrics['weighted_f1']:<18.2%} | {resize_metrics['weighted_f1']:<18.2%}")
print(f"{'Abstention Rate (Held for Conf)':<35} | {crop_metrics['abstention_rate']:<18.2%} | {resize_metrics['abstention_rate']:<18.2%}")
print(f"{'Accepted Count':<35} | {crop_metrics['accepted_total']:<18} | {resize_metrics['accepted_total']:<18}")
print(f"{'Accuracy Among Accepted':<35} | {crop_metrics['accepted_acc']:<18.2%} | {resize_metrics['accepted_acc']:<18.2%}")
print(f"{'High-Confidence Wrong (>60%)':<35} | {crop_metrics['high_conf_wrong']:<18} | {resize_metrics['high_conf_wrong']:<18}")

print("\n" + "=" * 80)
print("PER-CLASS METRICS (center_crop):")
print("=" * 80)
print(f"{'Class':<20} | {'Support':<8} | {'Correct':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 80)
for c, m in sorted(crop_metrics["per_class"].items()):
    print(f"{c:<20} | {m['support']:<8} | {m['correct']:<8} | {m['precision']:<10.1%} | {m['recall']:<10.1%} | {m['f1']:<10.1%}")

print("\n" + "=" * 80)
print(f"UNSUPPORTED / NOT EVALUATED CLASSES ({len(crop_metrics['unsupported_classes'])} classes):")
print("=" * 80)
for c in crop_metrics['unsupported_classes']:
    print(f"  - {c}: 0 verified samples (Status: NOT EVALUATED)")

# Save evaluation output
with open(backend_dir / "goal1_full_evaluation_report.json", "w") as f:
    json.dump({
        "crop_metrics": crop_metrics,
        "resize_metrics": resize_metrics
    }, f, indent=2, default=str)
print("\nSaved full evaluation report to backend/goal1_full_evaluation_report.json")
