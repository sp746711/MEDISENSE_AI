"""Unit Tests for Multimodal Condition Prediction Architecture and Data Integrity.

Verifies:
1. 5-modality forward pass across symptoms, labs, radiology, X-ray, and clinical examination.
2. Missing-modality forward pass with explicit masks.
3. Non-leakage: perturbations in missing modalities (mask=0) do not alter predictions.
4. Clinical examination encoder: changes in physical exam change model inputs.
5. Authentic data availability check correctly guards against unvalidated training.
6. Baseline unimodal symptom model remains intact and separate.
"""

from pathlib import Path
import joblib
import pytest
import torch

from app.ai.multimodal_condition_model import (
    CANONICAL_CLINICAL_EXAM_FINDINGS,
    CANONICAL_LAB_TESTS,
    CANONICAL_RAD_FINDINGS,
    CANONICAL_XRAY_PATHOLOGIES,
    CONDITION_CLASSES,
    MultimodalConditionPredictor,
)
from scripts.train_multimodal_model import check_authentic_data_availability


def test_multimodal_model_full_5_modalities_forward():
    """Verify forward pass when all 5 modalities (symptoms, labs, rad, xray, exam) are active."""
    model = MultimodalConditionPredictor(num_classes=24, hidden_dim=64)
    model.eval()

    batch_size = 2
    max_symptoms = 8

    symptom_tokens = torch.randint(1, 100, (batch_size, max_symptoms))
    symptom_features = torch.randn(batch_size, max_symptoms, 6)
    lab_features = torch.randn(batch_size, len(CANONICAL_LAB_TESTS) * 5)
    rad_features = torch.randn(batch_size, len(CANONICAL_RAD_FINDINGS) * 4)
    exam_features = torch.randn(batch_size, len(CANONICAL_CLINICAL_EXAM_FINDINGS) * 3)
    xray_features = torch.randn(batch_size, len(CANONICAL_XRAY_PATHOLOGIES) * 3)
    xray_anatomy = torch.randn(batch_size, 33)
    xray_eligibility = torch.ones(batch_size, 1)
    modality_mask = torch.ones(batch_size, 5)  # All 5 active

    with torch.no_grad():
        logits, is_ev = model(
            symptom_tokens,
            symptom_features,
            lab_features,
            rad_features,
            xray_features,
            exam_features,
            xray_anatomy,
            xray_eligibility,
            modality_mask,
        )

    assert logits.shape == (batch_size, 24)
    assert not torch.isnan(logits).any()
    assert is_ev.shape == (batch_size, 1)
    assert (is_ev == 1.0).all()


def test_multimodal_model_symptoms_only_forward():
    """Verify forward pass when patient only provides symptoms (labs/rad/xray/exam omitted)."""
    model = MultimodalConditionPredictor(num_classes=24, hidden_dim=64)
    model.eval()

    batch_size = 2
    max_symptoms = 6

    symptom_tokens = torch.randint(1, 100, (batch_size, max_symptoms))
    symptom_features = torch.randn(batch_size, max_symptoms, 6)
    # Empty/zeroed placeholders for missing modalities
    lab_features = torch.zeros(batch_size, len(CANONICAL_LAB_TESTS) * 5)
    rad_features = torch.zeros(batch_size, len(CANONICAL_RAD_FINDINGS) * 4)
    exam_features = torch.zeros(batch_size, len(CANONICAL_CLINICAL_EXAM_FINDINGS) * 3)
    xray_features = torch.zeros(batch_size, len(CANONICAL_XRAY_PATHOLOGIES) * 3)
    xray_anatomy = torch.zeros(batch_size, 33)
    xray_eligibility = torch.zeros(batch_size, 1)
    # Mask indicates only symptom modality is present: [1, 0, 0, 0, 0]
    modality_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])

    with torch.no_grad():
        logits, is_ev = model(
            symptom_tokens,
            symptom_features,
            lab_features,
            rad_features,
            xray_features,
            exam_features,
            xray_anatomy,
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
    symptom_features = torch.randn(batch_size, max_symptoms, 6)

    # Labs and Clinical Exam are missing (mask = [1, 0, 0, 0, 0])
    modality_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])

    lab_1 = torch.zeros(batch_size, len(CANONICAL_LAB_TESTS) * 5)
    rad_1 = torch.zeros(batch_size, len(CANONICAL_RAD_FINDINGS) * 4)
    exam_1 = torch.zeros(batch_size, len(CANONICAL_CLINICAL_EXAM_FINDINGS) * 3)
    xray_1 = torch.zeros(batch_size, len(CANONICAL_XRAY_PATHOLOGIES) * 3)
    xray_anat = torch.zeros(batch_size, 33)
    elig_1 = torch.zeros(batch_size, 1)

    # Alter missing lab features and exam features drastically
    lab_2 = torch.randn(batch_size, len(CANONICAL_LAB_TESTS) * 5) * 10.0
    exam_2 = torch.randn(batch_size, len(CANONICAL_CLINICAL_EXAM_FINDINGS) * 3) * 5.0

    with torch.no_grad():
        out1, _ = model(symptom_tokens, symptom_features, lab_1, rad_1, xray_1, exam_1, xray_anat, elig_1, modality_mask)
        out2, _ = model(symptom_tokens, symptom_features, lab_2, rad_1, xray_1, exam_2, xray_anat, elig_1, modality_mask)

    # Because lab and exam modalities were masked out (mask=0), gates zeroed them out. Predictions match exactly!
    assert torch.allclose(out1, out2, atol=1e-5)


def test_clinical_examination_influences_model_when_active():
    """Verify that when clinical examination mask is active, changing exam findings alters model outputs (Problem A)."""
    model = MultimodalConditionPredictor(num_classes=24, hidden_dim=64)
    model.eval()

    batch_size = 1
    max_symptoms = 4

    symptom_tokens = torch.randint(1, 100, (batch_size, max_symptoms))
    symptom_features = torch.randn(batch_size, max_symptoms, 6)
    lab_features = torch.randn(batch_size, len(CANONICAL_LAB_TESTS) * 5)
    rad_features = torch.randn(batch_size, len(CANONICAL_RAD_FINDINGS) * 4)
    xray_features = torch.randn(batch_size, len(CANONICAL_XRAY_PATHOLOGIES) * 3)
    xray_anat = torch.zeros(batch_size, 33)
    xray_elig = torch.ones(batch_size, 1)

    # Mask with active clinical exam: [1, 1, 1, 1, 1]
    modality_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

    exam_tender = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])  # Tenderness present
    exam_rom = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])     # Reduced ROM present

    with torch.no_grad():
        out_tender, _ = model(symptom_tokens, symptom_features, lab_features, rad_features, xray_features, exam_tender, xray_anat, xray_elig, modality_mask)
        out_rom, _ = model(symptom_tokens, symptom_features, lab_features, rad_features, xray_features, exam_rom, xray_anat, xray_elig, modality_mask)

    # Active clinical examination must alter the fused representation and logits
    assert not torch.allclose(out_tender, out_rom, atol=1e-4)


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
    assert len(pipeline.classes_) == 24
