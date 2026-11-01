"""Unit Tests for Multimodal Condition Prediction Architecture and Data Integrity.

Verifies:
1. Multimodal forward pass across full modalities.
2. Missing-modality forward pass with explicit masks (e.g. symptoms-only).
3. Non-leakage: perturbations in missing modalities (mask=0) do not affect outputs.
4. Authentic data availability check correctly guards against unvalidated training.
5. Baseline unimodal symptom model remains intact and separate.
"""

from pathlib import Path
import joblib
import pytest
import torch

from app.ai.multimodal_condition_model import (
    CANONICAL_RAD_FINDINGS,
    CONDITION_CLASSES,
    MultimodalConditionPredictor,
)
from scripts.train_multimodal_model import check_authentic_data_availability


def test_multimodal_model_full_modalities_forward():
    """Verify forward pass when all 4 modalities are provided."""
    model = MultimodalConditionPredictor(num_classes=24, hidden_dim=64)
    model.eval()

    batch_size = 2
    max_symptoms = 8

    symptom_tokens = torch.randint(1, 100, (batch_size, max_symptoms))
    symptom_features = torch.randn(batch_size, max_symptoms, 3)
    lab_features = torch.randn(batch_size, 21 * 5)
    rad_features = torch.randn(batch_size, len(CANONICAL_RAD_FINDINGS) * 3)
    xray_features = torch.randn(batch_size, 11 * 3)
    xray_eligibility = torch.ones(batch_size, 1)
    modality_mask = torch.ones(batch_size, 4)  # All 4 active

    with torch.no_grad():
        logits, is_ev = model(
            symptom_tokens,
            symptom_features,
            lab_features,
            rad_features,
            xray_features,
            xray_eligibility,
            modality_mask,
        )

    assert logits.shape == (batch_size, 24)
    assert not torch.isnan(logits).any()
    assert is_ev.shape == (batch_size, 1)
    assert (is_ev == 1.0).all()


def test_multimodal_model_symptoms_only_forward():
    """Verify forward pass when patient only provides symptoms (labs/rad/xray omitted)."""
    model = MultimodalConditionPredictor(num_classes=24, hidden_dim=64)
    model.eval()

    batch_size = 2
    max_symptoms = 6

    symptom_tokens = torch.randint(1, 100, (batch_size, max_symptoms))
    symptom_features = torch.randn(batch_size, max_symptoms, 3)
    # Empty/zeroed placeholders for missing modalities
    lab_features = torch.zeros(batch_size, 21 * 5)
    rad_features = torch.zeros(batch_size, len(CANONICAL_RAD_FINDINGS) * 3)
    xray_features = torch.zeros(batch_size, 11 * 3)
    xray_eligibility = torch.zeros(batch_size, 1)
    # Mask indicates only symptom modality is present: [1, 0, 0, 0]
    modality_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    with torch.no_grad():
        logits, is_ev = model(
            symptom_tokens,
            symptom_features,
            lab_features,
            rad_features,
            xray_features,
            xray_eligibility,
            modality_mask,
        )

    assert logits.shape == (batch_size, 24)
    assert not torch.isnan(logits).any()
    assert (is_ev == 1.0).all()


def test_missing_modality_non_leakage():
    """Verify that changes in missing modalities (mask=0) do NOT alter the model prediction."""
    model = MultimodalConditionPredictor(num_classes=24, hidden_dim=64)
    model.eval()

    batch_size = 1
    max_symptoms = 5

    symptom_tokens = torch.randint(1, 100, (batch_size, max_symptoms))
    symptom_features = torch.randn(batch_size, max_symptoms, 3)

    # Labs are missing (mask = [1, 0, 0, 0])
    modality_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    lab_1 = torch.zeros(batch_size, 21 * 5)
    rad_1 = torch.zeros(batch_size, len(CANONICAL_RAD_FINDINGS) * 3)
    xray_1 = torch.zeros(batch_size, 11 * 3)
    elig_1 = torch.zeros(batch_size, 1)

    # Alter missing lab features drastically
    lab_2 = torch.randn(batch_size, 21 * 5) * 10.0

    with torch.no_grad():
        out1, _ = model(symptom_tokens, symptom_features, lab_1, rad_1, xray_1, elig_1, modality_mask)
        out2, _ = model(symptom_tokens, symptom_features, lab_2, rad_1, xray_1, elig_1, modality_mask)

    # Because lab modality was masked out (mask=0), gate zeroed it out. Predictions must match exactly!
    assert torch.allclose(out1, out2, atol=1e-5)


def test_authentic_data_guard_blocks_unvalidated_training():
    """Verify that training on unlinked/nonexistent datasets is safely guarded and blocked."""
    data_dir = Path("backend/data/multimodal_linked")
    audit = check_authentic_data_availability(data_dir)

    assert audit["is_available"] is False
    assert len(audit["missing_prerequisites"]) > 0
    assert "MIMIC" in audit["roadmap_action"]


def test_baseline_unimodal_symptom_model_intact():
    """Verify that existing symptom-only baseline model is preserved and loads cleanly."""
    baseline_path = Path("models/conditions/condition_pipeline.joblib")
    assert baseline_path.exists()

    pipeline = joblib.load(baseline_path)
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")
    # Verify baseline classes match authentic 24 conditions
    assert len(pipeline.classes_) == 24
