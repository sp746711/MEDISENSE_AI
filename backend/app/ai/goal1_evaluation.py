"""Goal 1 Evaluation Module — Verified Real-Data Radiograph Assessment.

This module provides a reproducible evaluation pipeline for the MediSense AI Goal 1
33-class ONNX anatomy classifier, assessing top-1 accuracy, top-2 accuracy,
confidence threshold compliance, margin gating, and regional routing availability.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier
from app.ai.xray_router import XRAY_MODEL_REGISTRY, normalize_region_name

logger = logging.getLogger(__name__)

# Ground-truth test manifest of verified real radiographs present in the repository
VERIFIED_TEST_MANIFEST: list[dict[str, str]] = [
    {
        "original_name": "Artboard-1.png",
        "stored_name": "d2b19d448e5d482db01f8b387eb2919a.png",
        "verified_ground_truth": "CHEST",
        "expected_regional_model": "torchxrayvision_densenet121",
    },
    {
        "original_name": "18_chest.png",
        "stored_name": "ac61b6482f12421cbc5de22be8ac589f.png",
        "verified_ground_truth": "CHEST",
        "expected_regional_model": "torchxrayvision_densenet121",
    },
    {
        "original_name": "chest_xray.png",
        "stored_name": "157ecfe677a94d6e9833c159f6ed2a8e.png",
        "verified_ground_truth": "CHEST",
        "expected_regional_model": "torchxrayvision_densenet121",
    },
    {
        "original_name": "19_hand.png",
        "stored_name": "f5766c844fca47709d5450bce3f68cab.png",
        "verified_ground_truth": "HAND",
        "expected_regional_model": "mura_densenet169",
    },
    {
        "original_name": "male_hand.jpeg",
        "stored_name": "c22bb83329fb480bb9b7a1f02e01477a.jpg",
        "verified_ground_truth": "HAND",
        "expected_regional_model": "mura_densenet169",
    },
    {
        "original_name": "Normal Hand and Wrist X-ray.png",
        "stored_name": "1a651f4f4b1f473ab44ed7261f9334a4.png",
        "verified_ground_truth": "WRIST",
        "expected_regional_model": "wrist_deep_ensemble",
    },
    {
        "original_name": "20_wrist.png",
        "stored_name": "1403b6783a3947e5b7002c9c174887c3.png",
        "verified_ground_truth": "WRIST",
        "expected_regional_model": "wrist_deep_ensemble",
    },
    {
        "original_name": "Left Wrist X-Ray_ PA and Lateral Views.png",
        "stored_name": "17e8bde459e7417f8bdf0a79be22de45.png",
        "verified_ground_truth": "WRIST",
        "expected_regional_model": "wrist_deep_ensemble",
    },
    {
        "original_name": "13_forearm.png",
        "stored_name": "049742074cde4db3bc6066bc12f80438.png",
        "verified_ground_truth": "FOREARM",
        "expected_regional_model": "mura_densenet169",
    },
    {
        "original_name": "12_elbow (1).png",
        "stored_name": "7d79fce06a214d0badc4f1c702473718.png",
        "verified_ground_truth": "ELBOW",
        "expected_regional_model": "mura_densenet169",
    },
    {
        "original_name": "elbow-without-labels-copy-1.jpg",
        "stored_name": "f4605f42c33b4c48965a68fdc8969638.jpg",
        "verified_ground_truth": "ELBOW",
        "expected_regional_model": "mura_densenet169",
    },
    {
        "original_name": "15_ankle.png",
        "stored_name": "b0a43092ef5645b09bd64fbf754fe571.png",
        "verified_ground_truth": "ANKLE",
        "expected_regional_model": None,
    },
    {
        "original_name": "Right Ankle X-Ray Views.png",
        "stored_name": "728673426fcf422c82bf5a593ba086e6.png",
        "verified_ground_truth": "ANKLE",
        "expected_regional_model": None,
    },
    {
        "original_name": "top-foot-without-labels-copy.jpg",
        "stored_name": "b9857a5d7d0e47d489c8b7327afbabe2.jpg",
        "verified_ground_truth": "FOOT",
        "expected_regional_model": None,
    },
    {
        "original_name": "male_foot-(1).jpg",
        "stored_name": "fd15ea3de8b54835bc732a1e18348665.jpg",
        "verified_ground_truth": "FOOT",
        "expected_regional_model": None,
    },
    {
        "original_name": "11_cervical_spine.png",
        "stored_name": "ae945ad25b354e1c9b492368c45e3558.png",
        "verified_ground_truth": "CERVICAL_SPINE",
        "expected_regional_model": None,
    },
    {
        "original_name": "pelvis-without-labels-copy.jpg",
        "stored_name": "222b2bca333647ed9d7ed2edaf445cea.jpg",
        "verified_ground_truth": "PELVIS",
        "expected_regional_model": None,
    },
    {
        "original_name": "hip-joint-with-no-labels.jpg",
        "stored_name": "90143c1ccf4243cc84c463f50b019cad.jpg",
        "verified_ground_truth": "HIP",
        "expected_regional_model": None,
    },
    {
        "original_name": "17_abdomen_kub.png",
        "stored_name": "e768a04631844eacbaca2eac8261649f.png",
        "verified_ground_truth": "ABDOMEN",
        "expected_regional_model": None,
    },
    {
        "original_name": "lumbar-spine-without-labels.jpg",
        "stored_name": "0e00db9e44a643588de3838f9f620414.jpg",
        "verified_ground_truth": "LUMBAR_SPINE",
        "expected_regional_model": None,
    },
]


@dataclass
class EvaluationSampleResult:
    original_name: str
    stored_name: str
    sha256: str
    verified_ground_truth: str
    predicted_top1: str
    top1_score: float
    predicted_top2: str
    top2_score: float
    margin: float
    is_correct_top1: bool
    is_correct_top2: bool
    gate_passed: bool
    regional_model_available: bool
    safe_outcome: str


def run_goal1_evaluation(
    uploads_dir: str | Path | None = None,
    preprocessing_mode: str = "center_crop",
    min_confidence: float = 0.60,
    min_top_two_margin: float = 0.10,
) -> dict[str, Any]:
    """Execute evaluation on the verified real-data radiograph manifest."""
    base_dir = Path(__file__).resolve().parents[2]
    uploads_path = Path(uploads_dir) if uploads_dir else base_dir / "uploads" / "xrays"

    classifier = XRayAnatomyClassifier(
        preprocessing_mode=preprocessing_mode,
        min_confidence=min_confidence,
    )
    classifier.load()

    results: list[EvaluationSampleResult] = []
    missing_count = 0

    for item in VERIFIED_TEST_MANIFEST:
        file_path = uploads_path / item["stored_name"]
        if not file_path.is_file():
            missing_count += 1
            logger.warning("Evaluation image missing: %s", item["original_name"])
            continue

        with file_path.open("rb") as f:
            file_sha = hashlib.sha256(f.read()).hexdigest()

        try:
            pred = classifier.predict_file(file_path)
            top1 = pred["candidate_body_part"]
            conf1 = pred["body_part_confidence"]
            top2_entry = pred["top_predictions"][1] if len(pred.get("top_predictions", [])) > 1 else {"label": "NONE", "probability": 0.0}
            top2 = top2_entry["label"]
            conf2 = top2_entry["probability"]
            margin = max(0.0, conf1 - conf2)
            is_rejected = False
        except ValueError as val_err:
            top1 = "REJECTED_BLANK_OR_NON_XRAY"
            conf1 = 0.0
            top2 = "NONE"
            conf2 = 0.0
            margin = 0.0
            is_rejected = True

        ground_truth = item["verified_ground_truth"].upper()
        # Normalization check for aliases: CERVICAL_SPINE and NECK are distinct
        is_correct_top1 = (top1.upper() == ground_truth)
        is_correct_top2 = is_correct_top1 or (top2.upper() == ground_truth)

        # Gate logic: threshold >= min_confidence, margin >= min_top_two_margin
        gate_passed = bool(not is_rejected and conf1 >= min_confidence and margin >= min_top_two_margin)

        # Regional model check
        norm_gt = normalize_region_name(ground_truth)
        reg_entry = XRAY_MODEL_REGISTRY.get(norm_gt) if norm_gt else None
        reg_avail = bool(reg_entry and reg_entry.check_availability()[0])

        if is_rejected:
            outcome = "REJECTED_NON_RADIOGRAPH"
        elif is_correct_top1 and gate_passed:
            outcome = "CORRECT_ACCEPTED"
        elif is_correct_top1 and not gate_passed:
            outcome = "CORRECT_HELD_FOR_CONFIRMATION"
        elif not is_correct_top1 and gate_passed:
            outcome = "INCORRECT_ACCEPTED_RISK"
        else:
            outcome = "INCORRECT_HELD_SAFELY"

        results.append(
            EvaluationSampleResult(
                original_name=item["original_name"],
                stored_name=item["stored_name"],
                sha256=file_sha,
                verified_ground_truth=ground_truth,
                predicted_top1=top1,
                top1_score=conf1,
                predicted_top2=top2,
                top2_score=conf2,
                margin=round(margin, 4),
                is_correct_top1=is_correct_top1,
                is_correct_top2=is_correct_top2,
                gate_passed=gate_passed,
                regional_model_available=reg_avail,
                safe_outcome=outcome,
            )
        )

    total_evaluated = len(results)
    correct_top1 = sum(1 for r in results if r.is_correct_top1)
    correct_top2 = sum(1 for r in results if r.is_correct_top2)
    accepted_by_gate = sum(1 for r in results if r.gate_passed)
    correct_accepted = sum(1 for r in results if r.safe_outcome == "CORRECT_ACCEPTED")
    incorrect_accepted = sum(1 for r in results if r.safe_outcome == "INCORRECT_ACCEPTED_RISK")
    held_for_confirmation = sum(1 for r in results if not r.gate_passed)

    summary = {
        "total_manifest_images": len(VERIFIED_TEST_MANIFEST),
        "total_evaluated": total_evaluated,
        "missing_files": missing_count,
        "preprocessing_mode": preprocessing_mode,
        "min_confidence": min_confidence,
        "min_top_two_margin": min_top_two_margin,
        "top1_accuracy": round(correct_top1 / total_evaluated, 4) if total_evaluated else 0.0,
        "top2_accuracy": round(correct_top2 / total_evaluated, 4) if total_evaluated else 0.0,
        "gate_acceptance_rate": round(accepted_by_gate / total_evaluated, 4) if total_evaluated else 0.0,
        "correct_accepted_count": correct_accepted,
        "incorrect_accepted_count": incorrect_accepted,
        "held_for_confirmation_count": held_for_confirmation,
        "samples": [asdict(r) for r in results],
    }
    return summary


if __name__ == "__main__":
    import pprint
    rep = run_goal1_evaluation()
    pprint.pprint({k: v for k, v in rep.items() if k != "samples"})
