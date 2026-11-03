"""
Test suite for Goal 2 Step 6.13:
Residual ICD, Fusion Safety & Clinical Examination Corrections

Mandatory Tests Covered:
1. 428.1 consistent classification (description, version, consistent REQUIRE_ADDITIONAL_EVIDENCE rule).
2. 491.21 versus 491.22 separation (no broad prefix matching, 491.22 requires clinical doc).
3. J44.0 versus J44.1 separation (J44.0 requires doc, does not infer pneumonia).
4. All-masked fusion without NaN (zero NaNs, safe softmax, safe GMU gating).
5. All modalities unavailable (DATA_UNAVAILABLE state, abstention, no fabricated probabilities).
6. No eligible positive evidence (condition-specific isolation returns DATA_UNAVAILABLE for ineligible conditions).
7. Rule-out evidence isolation (NEGATIVE_OR_RULE_OUT cannot contribute to positive diagnostic head).
8. Contextual evidence isolation (CONTEXTUAL cannot activate positive score).
9. Unverified evidence rejection (UNVERIFIED cannot become verified without approved provenance).
10. Failed evidence handling (FAILED cannot become negative finding).
11. Unsupported modality handling (UNSUPPORTED cannot be treated as normal).
12. Initial examination versus progress note (routine progress notes rejected without initial provenance).
13. Examination before and after treatment (tracks precedes_treatment).
14. Discharge-summary rejection (rejected unless independently verified admission H&P section).
15. Unknown versus undocumented examination findings (preserves NOT_DOCUMENTED without converting to ABSENT).
16. Symptom-only fallback eligibility (fallback rejected without usable, verified symptoms).
"""

import pytest
import torch
from datetime import datetime, timedelta, timezone

from app.ai.multimodal_dataset_ingestion import (
    LabelDecision,
    ModalityEvidenceRole,
    ICD_CODE_METADATA,
    AcuteHeartFailureLabelEvaluator,
    COPDExacerbationLabelEvaluator,
    EvidenceRolePolicy,
    ClinicalExamTimingProvenanceValidator,
)
from app.ai.multimodal_condition_model import (
    CHEST_MULTIMODAL_CLASSES,
    CHEST_CONDITION_ELIGIBILITY_MATRIX,
    CANONICAL_LAB_TESTS,
    CANONICAL_RAD_FINDINGS,
    CANONICAL_CLINICAL_EXAM_FINDINGS,
    CANONICAL_XRAY_PATHOLOGIES,
    GatedMultimodalFusion,
    MultimodalConditionPredictor,
    MultimodalInferenceService,
    MultimodalPatientRecord,
    ClinicalSymptomObservation,
    ModalityState,
    safe_masked_softmax,
)


# ==============================================================================
# TEST 1: 428.1 CONSISTENT CLASSIFICATION
# ==============================================================================

def test_01_heart_failure_428_1_consistent_classification():
    """
    Verify ICD-9 428.1:
    - Description is 'Left heart failure'
    - Coding version is 'ICD-9-CM v32'
    - Default decision is consistently REQUIRE_ADDITIONAL_EVIDENCE (no conflicting REJECT)
    - BNP alone cannot confirm it
    - Provider documented acute decompensation confirms it
    """
    # 1. Base call without context -> REQUIRE_ADDITIONAL_EVIDENCE
    res_base = AcuteHeartFailureLabelEvaluator.evaluate("428.1", icd_version=9)
    assert res_base["decision"] == LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE
    assert res_base["description"] == "Left heart failure"
    assert res_base["coding_version"] == "ICD-9-CM v32"
    assert res_base["is_acute"] is False
    assert res_base["target"] is None

    # 2. Call with BNP alone -> still REQUIRE_ADDITIONAL_EVIDENCE
    res_bnp = AcuteHeartFailureLabelEvaluator.evaluate(
        "428.1",
        icd_version=9,
        clinical_context={"bnp_elevated": True},
    )
    assert res_bnp["decision"] == LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE
    assert res_bnp["is_acute"] is False

    # 3. Call with provider documentation of acute decompensation -> ACCEPT
    res_doc = AcuteHeartFailureLabelEvaluator.evaluate(
        "428.1",
        icd_version=9,
        clinical_context={"provider_documented_acute_decompensation": True},
    )
    assert res_doc["decision"] == LabelDecision.ACCEPT
    assert res_doc["target"] == "Acute Decompensated Heart Failure"
    assert res_doc["is_acute"] is True

    # Check official metadata entry
    meta = ICD_CODE_METADATA.get((9, "428.1"))
    assert meta is not None
    assert meta["description"] == "Left heart failure"
    assert meta["default_decision"] == LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE


# ==============================================================================
# TEST 2: 491.21 VERSUS 491.22 SEPARATION
# ==============================================================================

def test_02_copd_491_21_vs_491_22_separation():
    """
    Verify exact descriptions and strict separation:
    - 491.21: Obstructive chronic bronchitis with (acute) exacerbation -> candidate ACCEPT
    - 491.22: Obstructive chronic bronchitis with acute bronchitis -> REQUIRE_ADDITIONAL_EVIDENCE
    - Broad prefix matching (e.g. '491.2') is strictly forbidden
    """
    # 491.21: Definite exacerbation
    res_21 = COPDExacerbationLabelEvaluator.evaluate("491.21", icd_version=9)
    assert res_21["decision"] == LabelDecision.ACCEPT
    assert res_21["target"] == "Acute COPD Exacerbation"
    assert res_21["description"] == "Obstructive chronic bronchitis with (acute) exacerbation"
    assert res_21["coding_version"] == "ICD-9-CM v32"
    assert res_21["copd_exacerbation"] is True

    # 491.22: Acute bronchitis -> requires documentation
    res_22 = COPDExacerbationLabelEvaluator.evaluate("491.22", icd_version=9)
    assert res_22["decision"] == LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE
    assert res_22["target"] is None
    assert res_22["description"] == "Obstructive chronic bronchitis with acute bronchitis"
    assert res_22["copd_exacerbation"] is False

    # 491.22 with documentation -> ACCEPT
    res_22_doc = COPDExacerbationLabelEvaluator.evaluate(
        "491.22",
        icd_version=9,
        clinical_context={"documented_exacerbation": True},
    )
    assert res_22_doc["decision"] == LabelDecision.ACCEPT
    assert res_22_doc["target"] == "Acute COPD Exacerbation"

    # Truncated or broad prefix '491.2' is rejected
    res_broad = COPDExacerbationLabelEvaluator.evaluate("491.2", icd_version=9)
    assert res_broad["decision"] == LabelDecision.UNMAPPED


# ==============================================================================
# TEST 3: J44.0 VERSUS J44.1 SEPARATION
# ==============================================================================

def test_03_copd_j44_0_vs_j44_1_separation():
    """
    Verify:
    - J44.1: COPD with (acute) exacerbation -> candidate ACCEPT
    - J44.0: COPD with (acute) lower respiratory infection -> REQUIRE_ADDITIONAL_EVIDENCE
    - J44.0 does NOT infer pneumonia
    """
    res_1 = COPDExacerbationLabelEvaluator.evaluate("J44.1", icd_version=10)
    assert res_1["decision"] == LabelDecision.ACCEPT
    assert res_1["description"] == "Chronic obstructive pulmonary disease with (acute) exacerbation"
    assert res_1["coding_version"] == "ICD-10-CM FY2027"

    res_0 = COPDExacerbationLabelEvaluator.evaluate("J44.0", icd_version=10)
    assert res_0["decision"] == LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE
    assert res_0["description"] == "Chronic obstructive pulmonary disease with (acute) lower respiratory infection"
    assert res_0["coding_version"] == "ICD-10-CM FY2027"
    assert res_0["copd_exacerbation"] is False
    assert res_0["pneumonia_inferred"] is False


# ==============================================================================
# TEST 4: ALL-MASKED FUSION WITHOUT NAN
# ==============================================================================

def test_04_all_masked_fusion_without_nan():
    """Verify that when all modalities are masked, fusion produces no NaNs or Infs."""
    hidden_dim = 32
    fusion = GatedMultimodalFusion(hidden_dim=hidden_dim, num_modalities=5, num_conditions=5)
    fusion.eval()

    batch_size = 3
    mod_embeds = [torch.randn(batch_size, hidden_dim) for _ in range(5)]
    all_zero_mask = torch.zeros(batch_size, 5)

    with torch.no_grad():
        # Shared fusion
        fused_shared, is_ev = fusion(mod_embeds, all_zero_mask)
        assert not torch.isnan(fused_shared).any()
        assert not torch.isinf(fused_shared).any()
        assert (fused_shared == 0.0).all()
        assert (is_ev == 0.0).all()

        # Condition-specific fusion
        fused_cond, is_cond_ev = fusion(
            mod_embeds,
            all_zero_mask,
            condition_eligibility_mask=CHEST_CONDITION_ELIGIBILITY_MATRIX,
        )
        assert not torch.isnan(fused_cond).any()
        assert not torch.isinf(fused_cond).any()
        assert (fused_cond == 0.0).all()
        assert (is_cond_ev == 0.0).all()

    # Verify safe_masked_softmax with all-negative infinity
    logits = torch.tensor([[-float("inf"), -float("inf"), -float("inf")]])
    mask = torch.tensor([[0.0, 0.0, 0.0]])
    weights = safe_masked_softmax(logits, mask)
    assert not torch.isnan(weights).any()
    assert (weights == 0.0).all()


# ==============================================================================
# TEST 5: ALL MODALITIES UNAVAILABLE
# ==============================================================================

def test_05_all_modalities_unavailable():
    """Verify model returns explicit DATA_UNAVAILABLE state when all modalities are missing."""
    model = MultimodalConditionPredictor(num_classes=5, hidden_dim=32)
    model.eval()

    batch_size = 1
    s_tokens = torch.randint(1, 100, (batch_size, 4))
    s_feat = torch.zeros(batch_size, 4, 6)
    l_feat = torch.zeros(batch_size, len(CANONICAL_LAB_TESTS) * 5)
    r_feat = torch.zeros(batch_size, len(CANONICAL_RAD_FINDINGS) * 4)
    x_feat = torch.zeros(batch_size, len(CANONICAL_XRAY_PATHOLOGIES) * 3)
    mod_mask = torch.zeros(batch_size, 5)

    with torch.no_grad():
        result = model.evaluate_condition_differential(
            s_tokens,
            s_feat,
            l_feat,
            r_feat,
            x_feat,
            modality_mask=mod_mask,
            condition_eligibility_mask=CHEST_CONDITION_ELIGIBILITY_MATRIX,
        )

    assert result["status"] == "DATA_UNAVAILABLE"
    assert result["abstain"] is True
    assert len(result["evaluated_conditions"]) == 0
    # Every condition must be explicitly DATA_UNAVAILABLE, not 0.0 or uniform 0.2
    for cond_name, c_state in result["condition_states"].items():
        assert c_state["state"] == "DATA_UNAVAILABLE"
        assert c_state["probability"] is None
        assert c_state["has_positive_evidence"] is False


# ==============================================================================
# TEST 6: NO ELIGIBLE POSITIVE EVIDENCE
# ==============================================================================

def test_06_no_eligible_positive_evidence():
    """
    Encounter has ONLY X-ray available (modality_mask = [0, 0, 0, 1, 0]).
    For COPD, X-ray is RULE_OUT_ONLY -> eligibility is 0.0.
    Verify COPD returns DATA_UNAVAILABLE while Pneumonia returns EVALUATED.
    """
    model = MultimodalConditionPredictor(num_classes=5, hidden_dim=32)
    model.eval()

    batch_size = 1
    s_tokens = torch.randint(1, 100, (batch_size, 4))
    s_feat = torch.zeros(batch_size, 4, 6)
    l_feat = torch.zeros(batch_size, len(CANONICAL_LAB_TESTS) * 5)
    r_feat = torch.zeros(batch_size, len(CANONICAL_RAD_FINDINGS) * 4)
    x_feat = torch.randn(batch_size, len(CANONICAL_XRAY_PATHOLOGIES) * 3)
    x_anat = torch.randn(batch_size, 33)
    x_elig = torch.ones(batch_size, 1)

    # Encounter has ONLY X-ray
    xray_only_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])

    with torch.no_grad():
        result = model.evaluate_condition_differential(
            s_tokens,
            s_feat,
            l_feat,
            r_feat,
            x_feat,
            xray_anatomy_probs=x_anat,
            xray_pathology_eligible=x_elig,
            modality_mask=xray_only_mask,
            condition_eligibility_mask=CHEST_CONDITION_ELIGIBILITY_MATRIX,
        )

    assert result["status"] == "PARTIALLY_EVALUATED"
    # COPD must be DATA_UNAVAILABLE because its only active modality (X-ray) is ineligible for positive diagnosis
    copd_state = result["condition_states"]["Acute COPD Exacerbation"]
    assert copd_state["state"] == "DATA_UNAVAILABLE"
    assert copd_state["probability"] is None
    assert copd_state["has_positive_evidence"] is False

    # Pneumonia has positive X-ray evidence -> must be EVALUATED
    pneumonia_state = result["condition_states"]["Pneumonia"]
    assert pneumonia_state["state"] == "EVALUATED"
    assert pneumonia_state["probability"] is not None
    assert pneumonia_state["has_positive_evidence"] is True


# ==============================================================================
# TEST 7: RULE-OUT EVIDENCE ISOLATION
# ==============================================================================

def test_07_rule_out_evidence_isolation():
    """Verify that rule-out evidence cannot contribute to positive diagnostic scoring."""
    role = EvidenceRolePolicy.determine_modality_role(
        modality="xray",
        condition="Acute COPD Exacerbation",
        modality_state=ModalityState.VERIFIED,
        is_verified=True,
    )
    assert role == ModalityEvidenceRole.NEGATIVE_OR_RULE_OUT
    assert EvidenceRolePolicy.can_contribute_to_positive_diagnosis(role, is_condition_eligible=False) is False


# ==============================================================================
# TEST 8: CONTEXTUAL EVIDENCE ISOLATION
# ==============================================================================

def test_08_contextual_evidence_isolation():
    """Verify contextual evidence cannot silently activate positive disease score."""
    role = EvidenceRolePolicy.determine_modality_role(
        modality="radiology",
        condition="Pneumonia",
        modality_state=ModalityState.VERIFIED,
        finding_type="contextual",
        is_verified=True,
    )
    assert role == ModalityEvidenceRole.CONTEXTUAL
    assert EvidenceRolePolicy.can_contribute_to_positive_diagnosis(role) is False


# ==============================================================================
# TEST 9: UNVERIFIED EVIDENCE REJECTION
# ==============================================================================

def test_09_unverified_evidence_rejection():
    """Verify that unverified evidence cannot become verified without approved provenance."""
    role = EvidenceRolePolicy.determine_modality_role(
        modality="xray",
        condition="Pneumonia",
        modality_state=ModalityState.UNVERIFIED,
        is_verified=False,
    )
    assert role == ModalityEvidenceRole.UNVERIFIED
    assert EvidenceRolePolicy.can_contribute_to_positive_diagnosis(role) is False


# ==============================================================================
# TEST 10: FAILED EVIDENCE HANDLING
# ==============================================================================

def test_10_failed_evidence_handling():
    """Verify that failed modality (e.g. failed lab assay) cannot be treated as negative finding."""
    role = EvidenceRolePolicy.determine_modality_role(
        modality="laboratory",
        condition="Pneumonia",
        modality_state=ModalityState.FAILED,
        is_verified=True,
    )
    assert role == ModalityEvidenceRole.FAILED
    assert EvidenceRolePolicy.can_contribute_to_positive_diagnosis(role) is False


# ==============================================================================
# TEST 11: UNSUPPORTED MODALITY HANDLING
# ==============================================================================

def test_11_unsupported_modality_handling():
    """Verify unsupported modality is not treated as normal result."""
    role = EvidenceRolePolicy.determine_modality_role(
        modality="genomics",
        condition="Pneumonia",
        modality_state=ModalityState.UNSUPPORTED,
    )
    assert role == ModalityEvidenceRole.UNSUPPORTED
    assert EvidenceRolePolicy.can_contribute_to_positive_diagnosis(role) is False


# ==============================================================================
# TEST 12: INITIAL EXAMINATION VERSUS PROGRESS NOTE
# ==============================================================================

def test_12_initial_examination_versus_progress_note():
    """
    Verify routine progress notes are rejected as initial examinations
    unless explicit initial examination section provenance is verified.
    """
    admit_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    chart_time = (admit_time + timedelta(hours=2)).isoformat()

    # 1. Routine progress note without initial section -> REJECTED
    rec_routine = {
        "note_id": "PN101",
        "note_type": "PROGRESS_NOTE",
        "note_section": "DAILY PROGRESS UPDATE",
        "charttime": chart_time,
        "extraction_pipeline_verified": True,
    }
    res_routine = ClinicalExamTimingProvenanceValidator.validate_exam_record(rec_routine, admit_time=admit_time)
    assert res_routine["eligible"] is False
    assert res_routine["reason"] == "routine_progress_note_rejected_without_initial_provenance"

    # 2. Progress note with explicitly verified initial exam section -> ELIGIBLE
    rec_initial_sec = {
        "note_id": "PN102",
        "note_type": "PROGRESS_NOTE",
        "note_section": "INITIAL ADMISSION PHYSICAL EXAMINATION",
        "is_initial_exam_section": True,
        "charttime": chart_time,
        "extraction_pipeline_verified": True,
    }
    res_initial = ClinicalExamTimingProvenanceValidator.validate_exam_record(rec_initial_sec, admit_time=admit_time)
    assert res_initial["eligible"] is True
    assert res_initial["exam_eligible"] == 1.0


# ==============================================================================
# TEST 13: EXAMINATION BEFORE AND AFTER TREATMENT
# ==============================================================================

def test_13_examination_before_and_after_treatment():
    """Verify validator tracks whether note precedes or follows relevant treatment."""
    admit_time = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)
    treatment_time = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Exam charted BEFORE treatment (11:00 <= 12:00)
    rec_pre = {
        "note_type": "ADMISSION_HP",
        "charttime": (admit_time + timedelta(hours=1)).isoformat(),
        "extraction_pipeline_verified": True,
    }
    res_pre = ClinicalExamTimingProvenanceValidator.validate_exam_record(
        rec_pre,
        admit_time=admit_time,
        treatment_time=treatment_time,
    )
    assert res_pre["precedes_treatment"] is True

    # Exam charted AFTER treatment (14:00 > 12:00)
    rec_post = {
        "note_type": "ADMISSION_HP",
        "charttime": (admit_time + timedelta(hours=4)).isoformat(),
        "extraction_pipeline_verified": True,
    }
    res_post = ClinicalExamTimingProvenanceValidator.validate_exam_record(
        rec_post,
        admit_time=admit_time,
        treatment_time=treatment_time,
    )
    assert res_post["precedes_treatment"] is False


# ==============================================================================
# TEST 14: DISCHARGE SUMMARY REJECTION
# ==============================================================================

def test_14_discharge_summary_rejection():
    """Discharge summary rejected unless explicitly proven to be admission H&P section."""
    admit_time = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)
    chart_time = (admit_time + timedelta(hours=2)).isoformat()

    rec_dc = {
        "note_type": "DISCHARGE_SUMMARY",
        "note_section": "HOSPITAL COURSE SUMMARY",
        "charttime": chart_time,
        "extraction_pipeline_verified": True,
    }
    res_dc = ClinicalExamTimingProvenanceValidator.validate_exam_record(rec_dc, admit_time=admit_time)
    assert res_dc["eligible"] is False
    assert res_dc["reason"] == "discharge_summary_not_verified_as_initial_hp_section"


# ==============================================================================
# TEST 15: UNKNOWN VERSUS UNDOCUMENTED EXAMINATION FINDINGS
# ==============================================================================

def test_15_unknown_versus_undocumented_examination_findings():
    """Verify that NOT_DOCUMENTED findings are preserved and NEVER converted into ABSENT."""
    admit_time = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)
    chart_time = (admit_time + timedelta(hours=1)).isoformat()

    rec = {
        "note_type": "ADMISSION_HP",
        "charttime": chart_time,
        "findings": [
            {"finding": "wheezing", "state": "PRESENT"},
            {"finding": "crackles", "state": "ABSENT"},
            {"finding": "jvd", "state": "UNKNOWN"},
            {"finding": "pedal_edema", "state": "NOT_DOCUMENTED"},
        ],
        "extraction_pipeline_verified": True,
    }
    res = ClinicalExamTimingProvenanceValidator.validate_exam_record(rec, admit_time=admit_time)
    assert res["eligible"] is True
    sanitized = {f["finding"]: f["state"] for f in res["sanitized_findings"]}
    assert sanitized["wheezing"] == "PRESENT"
    assert sanitized["crackles"] == "ABSENT"
    assert sanitized["jvd"] == "UNKNOWN"
    assert sanitized["pedal_edema"] == "NOT_DOCUMENTED"
    assert sanitized["pedal_edema"] != "ABSENT"


# ==============================================================================
# TEST 16: SYMPTOM-ONLY FALLBACK ELIGIBILITY
# ==============================================================================

def test_16_symptom_only_fallback_eligibility():
    """Verify that symptom-only fallback is rejected without usable, verified symptom input."""
    service = MultimodalInferenceService()

    # 1. Record with empty symptoms -> insufficient_evidence
    rec_empty = MultimodalPatientRecord(subject_id="SUBJ_01")
    res_empty = service.predict(rec_empty)
    assert res_empty["status"] == "insufficient_evidence"

    # 2. Record with unverified symptoms -> fallback rejected / DATA_UNAVAILABLE
    rec_unverified = MultimodalPatientRecord(
        subject_id="SUBJ_02",
        symptoms=[ClinicalSymptomObservation(symptom="cough", canonical="cough", severity="MODERATE", duration_days=3)],
        provenance={"symptoms_verified": False},
    )
    res_unv = service.predict(rec_unverified)
    assert res_unv["status"] == "DATA_UNAVAILABLE"
    assert res_unv["model_scope"] == "fallback_rejected"
    assert res_unv["abstain"] is True

    # 3. Record with verified symptoms -> allowed to fallback to unimodal symptom baseline
    rec_verified = MultimodalPatientRecord(
        subject_id="SUBJ_03",
        symptoms=[ClinicalSymptomObservation(symptom="cough", canonical="cough", severity="MODERATE", duration_days=3)],
        provenance={"symptoms_verified": True},
    )
    res_ver = service.predict(rec_verified)
    assert res_ver["status"] == "blocked_untrained"
    assert res_ver["model_scope"] == "unimodal_symptom_baseline_active"
