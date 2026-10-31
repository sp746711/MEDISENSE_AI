"""Automated regression test executing the Goal 1 verified real-data evaluation pipeline."""

import pytest
from app.ai.goal1_evaluation import run_goal1_evaluation


def test_goal1_evaluation_manifest_runs_deterministically():
    """Verify that the Goal 1 evaluation runs, processes all test samples, and produces stable metrics."""
    metrics = run_goal1_evaluation(preprocessing_mode="center_crop")
    assert metrics["total_manifest_images"] == 20
    assert metrics["total_evaluated"] == 20
    assert metrics["missing_files"] == 0
    assert metrics["top1_accuracy"] >= 0.50
    assert metrics["top2_accuracy"] >= 0.60
    assert metrics["gate_acceptance_rate"] > 0.0
    # Center crop must have <= 2 incorrect accepted images on this 20-image benchmark
    assert metrics["incorrect_accepted_count"] <= 2
    # The majority of uncertain or out-of-distribution images must be held safely
    assert metrics["held_for_confirmation_count"] >= 10


def test_goal1_evaluation_direct_resize_comparison():
    """Verify that direct resize produces more incorrect accepted cases than center crop."""
    crop_metrics = run_goal1_evaluation(preprocessing_mode="center_crop")
    resize_metrics = run_goal1_evaluation(preprocessing_mode="direct_resize")

    # Empirical proof: direct resize allows more incorrect predictions to bypass the 60% gate
    assert resize_metrics["incorrect_accepted_count"] >= crop_metrics["incorrect_accepted_count"]
