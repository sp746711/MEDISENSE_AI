"""Unit Tests for MediSense AI Goal 2 Step 6.12 Corrections.

Verifies:
1. Fix 1: Acute heart failure ICD label rules (428.1, 428.0, I50.1, I50.9 require additional evidence; BNP alone insufficient; acute accepted; chronic rejected).
2. Fix 2: COPD exacerbation label eligibility (J44.1 accepted; J44.0 requires exacerbation doc; J44.0 does not infer pneumonia; 491.22 requires evidence; chronic rejected; secondary accepted).
3. Fix 3: Condition-specific fusion isolation (changing X-ray cannot change COPD logit, but changes Pneumonia logit).
4. Fix 4: Separate evidence roles (ModalityEvidenceRole).
5. Fix 5: Missing / failed / unavailable modalities fail-closed without negative bias.
6. Fix 6: Clinical examination timing and provenance controls (admission window [-24h, +24h], note types, vital signs separation, four-state assertions).
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
import torch

from app.ai.multimodal_condition_model import (
    CANONICAL_CLINICAL_EXAM_FINDINGS,
    CANONICAL_LAB_TESTS,
    CANONICAL_RAD_FINDINGS,
    CANONICAL_XRAY_PATHOLOGIES,
    CHEST_MULTIMODAL_CLASSES,
    CHEST_CONDITION_ELIGIBILITY_MATRIX,
    MultimodalConditionPredictor,
)
from app.ai.multimodal_dataset_ingestion import (
    AcuteHeartFailureLabelEvaluator,
    COPDExacerbationLabelEvaluator,
    ClinicalExamTimingProvenanceValidator,
    LabelDecision,
    ModalityEvidenceRole,
)


# ==============================================================================
# FIX 1 TESTS: ACUTE HEART FAILURE ICD MAPPINGS
# ==============================================================================

def test_heart_failure_428_1_requires_additional_evidence():
    """Rule 1: ICD-9 428.1 (Left heart failure) does NOT independently establish acute decompensated HF."""
    res = AcuteHeartFailureLabelEvaluator.evaluate("428.1", icd_version=9)
    assert res["decision"] == LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE
    assert res["target"] is None


def test_heart_failure_unspecified_codes_require_additional_evidence():
    """Rule 2: 428.0, I50.1, and I50.9 do NOT automatically establish acute decompensated HF."""
    for code, ver in [("428.0", 9), ("I50.1", 10), ("I50.9", 10)]:
        res = AcuteHeartFailureLabelEvaluator.evaluate(code, icd_version=ver)
        assert res["decision"] == LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE
        assert res["target"] is None


def test_heart_failure_bnp_alone_insufficient_for_unspecified_codes():
    """Rule 3: Elevated BNP/NT-proBNP alone cannot convert an unspecified billing code into confirmed acute HF."""
    res = AcuteHeartFailureLabelEvaluator.evaluate(
        "I50.9",
        icd_version=10,
        clinical_context={"bnp_elevated": True, "nt_pro_bnp_elevated": True},
    )
    assert res["decision"] == LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE
    assert "bnp_alone_insufficient" in res["reason"]


def test_heart_failure_unspecified_code_confirmed_with_provider_documentation():
    """Rule 4: Unspecified codes accepted only when corroborated by provider documented acute decompensation."""
    res = AcuteHeartFailureLabelEvaluator.evaluate(
        "I50.9",
        icd_version=10,
        clinical_context={"provider_documented_acute_decompensation": True},
    )
    assert res["decision"] == LabelDecision.ACCEPT
    assert res["target"] == "Acute Decompensated Heart Failure"


def test_heart_failure_definitive_acute_codes_accepted():
    """Definitive acute and acute-on-chronic codes are accepted unconditionally."""
    acute_codes = [
        ("428.21", 9), ("428.23", 9), ("428.31", 9), ("428.33", 9), ("428.41", 9), ("428.43", 9),
        ("I50.21", 10), ("I50.23", 10), ("I50.31", 10), ("I50.33", 10), ("I50.41", 10), ("I50.43", 10),
    ]
    for code, ver in acute_codes:
        res = AcuteHeartFailureLabelEvaluator.evaluate(code, icd_version=ver)
        assert res["decision"] == LabelDecision.ACCEPT
        assert res["target"] == "Acute Decompensated Heart Failure"
        assert res["is_acute"] is True


def test_heart_failure_chronic_and_noncardiac_codes_rejected():
    """Chronic compensated codes and non-cardiogenic pulmonary edema are strictly rejected."""
    rejected_codes = [
        ("428.22", 9), ("428.32", 9), ("428.42", 9), ("518.4", 9),
        ("I50.22", 10), ("I50.32", 10), ("I50.42", 10), ("J81.0", 10),
    ]
    for code, ver in rejected_codes:
        res = AcuteHeartFailureLabelEvaluator.evaluate(code, icd_version=ver)
        assert res["decision"] == LabelDecision.REJECT
        assert res["target"] is None


# ==============================================================================
# FIX 2 TESTS: COPD EXACERBATION LABEL ELIGIBILITY
# ==============================================================================

def test_copd_j44_1_qualifies_as_candidate():
    """J44.1 qualifies as acute COPD exacerbation candidate."""
    res = COPDExacerbationLabelEvaluator.evaluate("J44.1", icd_version=10)
    assert res["decision"] == LabelDecision.ACCEPT
    assert res["target"] == "Acute COPD Exacerbation"
    assert res["copd_exacerbation"] is True


def test_copd_j44_0_without_exacerbation_documentation():
    """J44.0 means COPD with acute lower resp infection; does NOT automatically become acute COPD exacerbation."""
    res = COPDExacerbationLabelEvaluator.evaluate("J44.0", icd_version=10)
    assert res["decision"] == LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE
    assert res["copd_exacerbation"] is False
    assert res["pneumonia_inferred"] is False


def test_copd_j44_0_with_independently_documented_exacerbation():
    """J44.0 with independently documented exacerbation establishes COPD exacerbation."""
    res = COPDExacerbationLabelEvaluator.evaluate(
        "J44.0",
        icd_version=10,
        clinical_context={"documented_exacerbation": True},
    )
    assert res["decision"] == LabelDecision.ACCEPT
    assert res["target"] == "Acute COPD Exacerbation"
    assert res["copd_exacerbation"] is True
    assert res["pneumonia_inferred"] is False


def test_copd_j44_0_with_independent_pneumonia():
    """J44.0 does not infer pneumonia; pneumonia must be independently established."""
    res = COPDExacerbationLabelEvaluator.evaluate(
        "J44.0",
        icd_version=10,
        clinical_context={"independent_pneumonia": True},
    )
    assert res["independent_pneumonia_present"] is True
    assert res["pneumonia_inferred"] is False


def test_copd_491_22_requires_clinical_documentation():
    """ICD-9 491.22 (acute bronchitis) must not automatically establish COPD exacerbation without clinical doc."""
    res_no_doc = COPDExacerbationLabelEvaluator.evaluate("491.22", icd_version=9)
    assert res_no_doc["decision"] == LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE
    assert res_no_doc["copd_exacerbation"] is False

    res_doc = COPDExacerbationLabelEvaluator.evaluate(
        "491.22",
        icd_version=9,
        clinical_context={"documented_exacerbation": True},
    )
    assert res_doc["decision"] == LabelDecision.ACCEPT
    assert res_doc["copd_exacerbation"] is True


def test_copd_unspecified_codes_rejected():
    """Unspecified and chronic baseline COPD codes (496, J44.9) are rejected for acute exacerbation."""
    for code, ver in [("496", 9), ("J44.9", 10), ("492.8", 9), ("J43.9", 10)]:
        res = COPDExacerbationLabelEvaluator.evaluate(code, icd_version=ver)
        assert res["decision"] == LabelDecision.REJECT
        assert res["copd_exacerbation"] is False


def test_copd_secondary_diagnoses_preserved():
    """Secondary COPD diagnoses (seq_num > 1) with valid acute codes are preserved as secondary targets."""
    res = COPDExacerbationLabelEvaluator.evaluate("J44.1", icd_version=10, seq_num=2)
    assert res["decision"] == LabelDecision.ACCEPT
    assert res["is_secondary"] is True
    assert res["copd_exacerbation"] is True


# ==============================================================================
# FIX 3 TESTS: CONDITION-SPECIFIC FUSION ISOLATION
# ==============================================================================

def test_copd_xray_isolation_cannot_alter_prediction():
    """
    Mandatory Test: Changing only the X-ray embedding CANNOT change the COPD positive prediction
    when all other inputs and conditions are held constant.
    Also proves that Pneumonia prediction DOES change under X-ray perturbations.
    """
    model = MultimodalConditionPredictor(
        num_classes=5,
        hidden_dim=64,
        condition_eligibility_mask=CHEST_CONDITION_ELIGIBILITY_MATRIX,
    )
    model.eval()

    batch_size = 2
    max_symptoms = 6

    # Fixed common inputs across all modalities
    symptom_tokens = torch.randint(1, 100, (batch_size, max_symptoms))
    symptom_features = torch.randn(batch_size, max_symptoms, 6)
    lab_features = torch.randn(batch_size, len(CANONICAL_LAB_TESTS) * 5)
    rad_features = torch.randn(batch_size, len(CANONICAL_RAD_FINDINGS) * 4)
    exam_features = torch.randn(batch_size, len(CANONICAL_CLINICAL_EXAM_FINDINGS) * 3)
    xray_anatomy = torch.randn(batch_size, 33)
    xray_eligibility = torch.ones(batch_size, 1)
    modality_mask = torch.ones(batch_size, 5)  # All 5 modalities active

    # 1. Baseline forward pass with initial X-ray features
    xray_features_1 = torch.randn(batch_size, len(CANONICAL_XRAY_PATHOLOGIES) * 3)
    with torch.no_grad():
        logits_1, _ = model(
            symptom_tokens,
            symptom_features,
            lab_features,
            rad_features,
            xray_features_1,
            exam_features,
            xray_anatomy,
            xray_eligibility,
            modality_mask,
            condition_eligibility_mask=CHEST_CONDITION_ELIGIBILITY_MATRIX,
        )

    # 2. Second forward pass with perturbed X-ray features (extreme values + random noise)
    xray_features_2 = xray_features_1 + 100.0 * torch.randn_like(xray_features_1)
    with torch.no_grad():
        logits_2, _ = model(
            symptom_tokens,
            symptom_features,
            lab_features,
            rad_features,
            xray_features_2,
            exam_features,
            xray_anatomy,
            xray_eligibility,
            modality_mask,
            condition_eligibility_mask=CHEST_CONDITION_ELIGIBILITY_MATRIX,
        )

    # Class 4 is Acute COPD Exacerbation (where X-ray is RULE_OUT_ONLY -> Diagnostic eligibility = 0.0)
    copd_logits_1 = logits_1[:, 4]
    copd_logits_2 = logits_2[:, 4]

    # Verify MATHEMATICAL ZERO IMPACT on COPD prediction
    assert torch.allclose(copd_logits_1, copd_logits_2, atol=1e-7), (
        f"COPD logit changed under X-ray perturbation! diff={torch.abs(copd_logits_1 - copd_logits_2).max()}"
    )

    # Class 0 is Pneumonia (where X-ray is DIAGNOSTICALLY_RELEVANT -> Diagnostic eligibility = 1.0)
    pneumonia_logits_1 = logits_1[:, 0]
    pneumonia_logits_2 = logits_2[:, 0]

    # Verify that Pneumonia DOES change under X-ray perturbation
    assert not torch.allclose(pneumonia_logits_1, pneumonia_logits_2, atol=1e-4), (
        "Pneumonia logit should change under X-ray perturbation, but remained unchanged!"
    )


# ==============================================================================
# FIX 4 & 5 TESTS: EVIDENCE ROLES AND MISSING/FAILED MODALITY HANDLING
# ==============================================================================

def test_evidence_roles_constants_exist():
    """Verify explicit evidence roles exist and are distinct."""
    assert ModalityEvidenceRole.POSITIVE_DIAGNOSTIC != ModalityEvidenceRole.NEGATIVE_OR_RULE_OUT
    assert ModalityEvidenceRole.CONTEXTUAL != ModalityEvidenceRole.UNAVAILABLE
    assert ModalityEvidenceRole.UNVERIFIED != ModalityEvidenceRole.FAILED


def test_missing_all_modalities_fails_closed():
    """Verify that when all modalities are missing (modality_mask all 0), model returns zero evidence."""
    model = MultimodalConditionPredictor(num_classes=5, hidden_dim=64)
    model.eval()

    batch_size = 2
    symptom_tokens = torch.randint(1, 100, (batch_size, 4))
    symptom_features = torch.randn(batch_size, 4, 6)
    lab_features = torch.zeros(batch_size, len(CANONICAL_LAB_TESTS) * 5)
    rad_features = torch.zeros(batch_size, len(CANONICAL_RAD_FINDINGS) * 4)
    exam_features = torch.zeros(batch_size, len(CANONICAL_CLINICAL_EXAM_FINDINGS) * 3)
    xray_features = torch.zeros(batch_size, len(CANONICAL_XRAY_PATHOLOGIES) * 3)
    xray_anatomy = torch.zeros(batch_size, 33)
    xray_eligibility = torch.zeros(batch_size, 1)
    modality_mask = torch.zeros(batch_size, 5)  # All missing

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

    assert (is_ev == 0.0).all()
    assert (logits == 0.0).all()


# ==============================================================================
# FIX 6 TESTS: CLINICAL EXAMINATION TIMING & PROVENANCE CONTROLS
# ==============================================================================

def test_exam_timing_within_admission_window():
    """Physical exam charted within [-24h, +24h] of admission is accepted."""
    admit_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    valid_note_time = (admit_time + timedelta(hours=2)).isoformat()

    exam_record = {
        "note_type": "ED_INITIAL_ASSESSMENT",
        "charttime": valid_note_time,
        "findings": [{"finding": "crackles", "state": "PRESENT"}],
        "extraction_pipeline_verified": True,
    }
    res = ClinicalExamTimingProvenanceValidator.validate_exam_record(exam_record, admit_time=admit_time)
    assert res["eligible"] is True
    assert res["exam_eligible"] == 1.0


def test_exam_timing_outside_admission_window_rejected():
    """Physical exam charted >24h after admission is rejected (post-treatment status)."""
    admit_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    late_note_time = (admit_time + timedelta(hours=48)).isoformat()

    exam_record = {
        "note_type": "ADMISSION_HP",
        "charttime": late_note_time,
        "findings": [{"finding": "crackles", "state": "PRESENT"}],
        "extraction_pipeline_verified": True,
    }
    res = ClinicalExamTimingProvenanceValidator.validate_exam_record(exam_record, admit_time=admit_time)
    assert res["eligible"] is False
    assert res["reason"] == "note_time_outside_admission_window"


def test_discharge_summary_rejected_unless_initial_hp():
    """Discharge summary note type is rejected unless verified as initial H&P section."""
    admit_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    exam_record = {
        "note_type": "DISCHARGE_SUMMARY",
        "charttime": admit_time.isoformat(),
        "findings": [{"finding": "wheezing", "state": "PRESENT"}],
    }
    res = ClinicalExamTimingProvenanceValidator.validate_exam_record(exam_record, admit_time=admit_time)
    assert res["eligible"] is False
    assert "discharge_summary" in res["reason"]


def test_vital_signs_separation_enforced():
    """Vital signs passed as physical exam findings trigger an immediate violation."""
    admit_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    exam_record = {
        "note_type": "ED_INITIAL_ASSESSMENT",
        "charttime": admit_time.isoformat(),
        "findings": [{"finding": "heartrate", "state": "PRESENT"}],  # Vital sign, not physical exam!
    }
    res = ClinicalExamTimingProvenanceValidator.validate_exam_record(exam_record, admit_time=admit_time)
    assert res["eligible"] is False
    assert "vital_signs_separation_violation" in res["reason"]


def test_unverified_nlp_pipeline_fails_closed():
    """Unvalidated NLP extraction pipeline defaults to exam_eligible = 0.0 fail-closed."""
    admit_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    exam_record = {
        "note_type": "ED_INITIAL_ASSESSMENT",
        "charttime": admit_time.isoformat(),
        "findings": [{"finding": "crackles", "state": "PRESENT"}],
        "extraction_pipeline_verified": False,  # Pipeline not verified
    }
    res = ClinicalExamTimingProvenanceValidator.validate_exam_record(exam_record, admit_time=admit_time)
    assert res["eligible"] is False
    assert res["exam_eligible"] == 0.0
    assert "unvalidated_nlp_pipeline" in res["reason"]
