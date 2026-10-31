"""Forensic investigation of verified real radiographs and model architectures for Goal 1."""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure backend root is in sys.path
backend_root = Path(__file__).resolve().parents[2]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier
from app.ai.goal1_evaluation import VERIFIED_TEST_MANIFEST

def main():
    print("=" * 85)
    print("GOAL 1 FORENSIC EVALUATION — 20 VERIFIED REAL RADIOGRAPHS ON 33-CLASS ONNX MODEL")
    print("=" * 85)

    clf = XRayAnatomyClassifier(preprocessing_mode="center_crop")
    clf.load()

    uploads_dir = backend_root / "uploads" / "xrays"

    correct_top1 = 0
    correct_top3 = 0
    gate_accepted_correct = 0
    gate_accepted_incorrect = 0
    held_for_confirmation = 0

    results = []

    for item in VERIFIED_TEST_MANIFEST:
        orig = item["original_name"]
        stored = item["stored_name"]
        true_anat = item["verified_ground_truth"]

        fp = uploads_dir / stored
        if not fp.exists():
            print(f"MISSING FILE: {orig} ({stored})")
            continue

        try:
            res = clf.predict_file(fp)
            with Image.open(fp) as im:
                raw_im = im.convert("RGB")
                probs = clf._run_inference_probabilities(raw_im)
            is_rejected = False
        except ValueError as err:
            res = {
                "predicted_anatomy": "REJECTED_NON_XRAY",
                "confidence": 0.0,
                "prediction_status": "rejected",
                "second_candidate_body_part": "NONE",
                "second_candidate_confidence": 0.0,
                "top_two_margin": 0.0,
            }
            probs = np.zeros(len(clf.classes))
            is_rejected = True

        top_indices = np.argsort(probs)[::-1]
        top1_label = clf.classes[top_indices[0]]
        top1_prob = float(probs[top_indices[0]])
        top3_labels = [clf.classes[top_indices[i]] for i in range(3)]

        is_top1 = (top1_label == true_anat)
        is_top3 = (true_anat in top3_labels)

        if is_top1:
            correct_top1 += 1
        if is_top3:
            correct_top3 += 1

        gate_status = res.get("prediction_status")
        if gate_status == "accepted":
            if is_top1:
                gate_accepted_correct += 1
            else:
                gate_accepted_incorrect += 1
        else:
            held_for_confirmation += 1

        true_idx = clf.classes.index(true_anat) if true_anat in clf.classes else -1
        true_rank = int(np.where(top_indices == true_idx)[0][0]) + 1 if true_idx != -1 else -1
        true_prob = float(probs[true_idx]) if true_idx != -1 else 0.0

        results.append({
            "original_name": orig,
            "true_anatomy": true_anat,
            "pred_top1": top1_label,
            "conf": top1_prob,
            "top3": [(clf.classes[top_indices[i]], float(probs[top_indices[i]])) for i in range(3)],
            "true_rank": true_rank,
            "true_prob": true_prob,
            "gate_status": gate_status,
            "is_top1": is_top1,
        })

        status_flag = "OK" if is_top1 else "MISMATCH"
        print(f"\n[{status_flag}] Image: {orig:<36} True: {true_anat}")
        print(f"   Top-1: {top1_label:<18} ({top1_prob*100:5.1f}%) | Gate: {gate_status}")
        print(f"   True Anatomy Rank: #{true_rank} with {true_prob*100:5.1f}%")
        top3_str = ", ".join([f"{lbl} ({p*100:.1f}%)" for lbl, p in [(clf.classes[top_indices[i]], float(probs[top_indices[i]])) for i in range(3)]])
        print(f"   Top-3: {top3_str}")

    total = len(results)
    print("\n" + "=" * 85)
    print("OVERALL SUMMARY METRICS")
    print("=" * 85)
    print(f"Total Evaluated Real Images: {total}")
    print(f"Top-1 Accuracy:              {correct_top1}/{total} ({correct_top1/total*100:.1f}%)")
    print(f"Top-3 Accuracy:              {correct_top3}/{total} ({correct_top3/total*100:.1f}%)")
    print(f"Gate Accepted & Correct:     {gate_accepted_correct}/{total} ({gate_accepted_correct/total*100:.1f}%)")
    print(f"Gate Accepted but Incorrect: {gate_accepted_incorrect}/{total} ({gate_accepted_incorrect/total*100:.1f}%)")
    print(f"Held for User Confirmation:  {held_for_confirmation}/{total} ({held_for_confirmation/total*100:.1f}%)")

if __name__ == "__main__":
    main()
