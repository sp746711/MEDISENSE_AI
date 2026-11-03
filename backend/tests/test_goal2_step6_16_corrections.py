"""Unit and regression tests for Goal 2 Step 6.16: Independent Correction and Verification of Step 6.15.

Tests:
1. Laboratory encounter linkage rules:
   - subject_id + charttime alone NEVER qualifies as VERIFIED_ENCOUNTER_LINK.
   - Timestamp proximity without encounter key remains TEMPORAL_CANDIDATE_LINK (lab_eligible = 0.0).
   - Conflicting encounter identifiers (hadm_id mismatch) are strictly REJECTED.
   - Direct matching hadm_id or verified crosswalk achieves VERIFIED_ENCOUNTER_LINK.
   - LaboratoryLinkageValidator fails closed against automatic promotion.
2. Clinical evidence roles:
   - Six distinct clinical roles: DIAGNOSTIC, SUPPORTING, CONTEXTUAL, SEVERITY, RULE_OUT, NONSPECIFIC.
   - Nonspecific inflammatory markers (WBC, neutrophils, CRP, procalcitonin) cannot independently establish pneumonia.
   - BNP / NT-proBNP cannot independently establish acute heart failure (SUPPORTING role).
   - Renal markers (creatinine, BUN) cannot independently establish disease (CONTEXTUAL/SEVERITY role).
   - Chest X-ray in COPD is strictly RULE_OUT with zero positive diagnostic contribution.
3. Modality-specific temporal policies:
   - ClinicalTemporalPolicy enforces distinct presentation windows for symptoms, labs, imaging, and exam.
   - Pre-admission outpatient labs (>24h prior) and post-discharge labs are rejected.
   - Clinical exam must precede acute treatment.
"""

from datetime import datetime, timezone
import pytest

from app.ai.multimodal_dataset_ingestion import (
    AuthenticMultimodalDatasetIngester,
    ClinicalTemporalPolicy,
    CrosswalkSigningAuthority,
    EvidenceRolePolicy,
    LaboratoryLinkageValidator,
    ModalityEvidenceRole,
    create_authenticated_imaging_crosswalk,
    validate_lab_provenance,
)


# ==============================================================================
# 1. LABORATORY ENCOUNTER LINKAGE TESTS (Correction 1)
# ==============================================================================

def test_01_lab_subject_and_charttime_alone_is_temporal_candidate():
    """
    Step 6.16 Rule: subject_id + charttime ALONE must never qualify as VERIFIED_ENCOUNTER_LINK.
    Unverified temporal matches must remain TEMPORAL_CANDIDATE_LINK with lab_eligible = 0.0.
    """
    admit_dt = datetime(2026, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
    disch_dt = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    # Detached lab record with subject_id and charttime, but NO hadm_id
    raw_lab = {
        "subject_id": "10002430",
        "charttime": "2026-01-10T14:00:00Z",
        "wbc": 14.5,
    }

    status, source, elig, reason = validate_lab_provenance(
        subject_id="10002430",
        hadm_id="21400215",
        lab_data=raw_lab,
        admit_time=admit_dt,
        disch_time=disch_dt,
        is_temporal_match=True,
    )

    assert status == "TEMPORAL_CANDIDATE_LINK"
    assert elig == 0.0
    assert "subject_and_charttime_proximity_only" in reason
    assert LaboratoryLinkageValidator.can_promote_temporal_candidate() is False


def test_02_lab_conflicting_encounter_id_rejected():
    """
    Step 6.16 Rule: Conflicting encounter identifiers (hadm_id mismatch) must be strictly REJECTED.
    """
    admit_dt = datetime(2026, 1, 10, 10, 0, 0, tzinfo=timezone.utc)

    # Lab belongs to a different encounter hadm_id
    conflicting_lab = {
        "subject_id": "10002430",
        "hadm_id": "99999999",  # Conflicting hadm_id!
        "charttime": "2026-01-10T12:00:00Z",
        "wbc": 12.0,
    }

    status, source, elig, reason = validate_lab_provenance(
        subject_id="10002430",
        hadm_id="21400215",
        lab_data=conflicting_lab,
        admit_time=admit_dt,
    )

    assert status == "REJECTED"
    assert elig == 0.0
    assert "lab_encounter_mismatch" in reason


def test_03_lab_conflicting_subject_id_rejected():
    """
    Step 6.16 Rule: Conflicting patient identifiers (subject_id mismatch) must be strictly REJECTED.
    """
    conflicting_lab = {
        "subject_id": "WRONG_PATIENT",
        "hadm_id": "21400215",
        "wbc": 11.0,
    }

    status, source, elig, reason = validate_lab_provenance(
        subject_id="10002430",
        hadm_id="21400215",
        lab_data=conflicting_lab,
    )

    assert status == "REJECTED"
    assert elig == 0.0
    assert "lab_subject_mismatch" in reason


def test_04_lab_verified_encounter_link_via_matching_hadm():
    """
    Step 6.16 Rule: Direct matching hadm_id within encounter window achieves VERIFIED_ENCOUNTER_LINK.
    """
    admit_dt = datetime(2026, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
    disch_dt = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    verified_lab = {
        "subject_id": "10002430",
        "hadm_id": "21400215",  # Explicit matching hadm_id
        "charttime": "2026-01-10T12:00:00Z",
        "wbc": 14.5,
    }

    res = LaboratoryLinkageValidator.validate_lab_record(
        subject_id="10002430",
        hadm_id="21400215",
        lab_data=verified_lab,
        admit_time=admit_dt,
        disch_time=disch_dt,
    )

    assert res["linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert res["lab_eligible"] == 1.0
    assert res["is_verified"] is True
    assert res["is_temporal_candidate"] is False


def test_05_lab_verified_encounter_link_via_authenticated_crosswalk():
    """
    Step 6.16 Rule: Verified encounter crosswalk achieves VERIFIED_ENCOUNTER_LINK.
    """
    crosswalk = {
        "subject_id": "10002430",
        "hadm_id": "21400215",
        "source": "MIMIC_HOSP_LAB_SPECIMEN_CROSSWALK",
        "source_records_verified": True,
        "verified": True,
    }

    lab_with_crosswalk = {
        "subject_id": "10002430",
        "charttime": "2026-01-10T12:00:00Z",
        "wbc": 14.5,
        "lab_hadm_crosswalk": crosswalk,
    }

    status, source, elig, reason = validate_lab_provenance(
        subject_id="10002430",
        hadm_id="21400215",
        lab_data=lab_with_crosswalk,
    )

    assert status == "VERIFIED_ENCOUNTER_LINK"
    assert elig == 1.0
    assert source == "MIMIC_HOSP_LAB_SPECIMEN_CROSSWALK"


def test_06_ingester_masks_temporal_candidate_lab_records():
    """
    Verify AuthenticMultimodalDatasetIngester masks lab modality when lab is only a temporal candidate.
    """
    ingester = AuthenticMultimodalDatasetIngester()

    # Record has symptoms and a detached lab without hadm_id (subject_id + charttime only)
    rec = {
        "subject_id": "10002010",
        "hadm_id": "20003010",
        "target_condition": "Pneumonia",
        "admittime": "2026-03-01T12:00:00Z",
        "symptoms": ["cough", "fever"],
        # Detached lab with only subject_id and charttime
        "laboratory": {
            "subject_id": "10002010",
            "charttime": "2026-03-01T14:00:00Z",
            "wbc": 13.0,
        },
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]

    # Symptoms are verified, but lab must be masked out because it lacks an encounter key!
    assert r["modality_mask"][0] == 1.0  # Symptoms active
    assert r["modality_mask"][1] == 0.0  # Lab masked out!
    assert r["lab_eligible"] == 0.0
    assert r["lab_linkage_status"] == "TEMPORAL_CANDIDATE_LINK"
    assert r["temporal_linkage_evidence"]["lab_encounter_linked"] is False


# ==============================================================================
# 2. CLINICAL EVIDENCE ROLES TESTS (Correction 2)
# ==============================================================================

def test_07_evidence_role_classification_pneumonia():
    """
    Step 6.16: Verify evidence role classification for Pneumonia.
    Nonspecific markers cannot independently establish disease.
    """
    # 1. Nonspecific inflammatory markers
    wbc_role = EvidenceRolePolicy.classify_evidence_role("Pneumonia", "laboratory", "White Blood Cell Count (WBC)")
    crp_role = EvidenceRolePolicy.classify_evidence_role("Pneumonia", "laboratory", "C-Reactive Protein (CRP)")
    procal_role = EvidenceRolePolicy.classify_evidence_role("Pneumonia", "laboratory", "Procalcitonin")

    assert wbc_role == ModalityEvidenceRole.NONSPECIFIC
    assert crp_role == ModalityEvidenceRole.NONSPECIFIC
    assert procal_role == ModalityEvidenceRole.NONSPECIFIC

    # Must NOT independently establish disease!
    assert EvidenceRolePolicy.can_independently_establish_disease(wbc_role) is False
    assert EvidenceRolePolicy.can_independently_establish_disease(crp_role) is False
    assert EvidenceRolePolicy.can_independently_establish_disease(procal_role) is False

    # 2. Severity marker
    pao2_role = EvidenceRolePolicy.classify_evidence_role("Pneumonia", "laboratory", "PaO2 blood gas")
    assert pao2_role == ModalityEvidenceRole.SEVERITY
    assert EvidenceRolePolicy.can_contribute_to_severity(pao2_role) is True
    assert EvidenceRolePolicy.can_independently_establish_disease(pao2_role) is False

    # 3. Diagnostic imaging finding
    cxr_role = EvidenceRolePolicy.classify_evidence_role("Pneumonia", "xray", "consolidation")
    assert cxr_role == ModalityEvidenceRole.DIAGNOSTIC
    assert EvidenceRolePolicy.can_independently_establish_disease(cxr_role) is True

    # 4. Supporting symptoms
    sym_role = EvidenceRolePolicy.classify_evidence_role("Pneumonia", "symptoms", "cough with fever")
    assert sym_role == ModalityEvidenceRole.SUPPORTING
    assert EvidenceRolePolicy.can_support_candidate(sym_role) is True
    assert EvidenceRolePolicy.can_independently_establish_disease(sym_role) is False


def test_08_evidence_role_classification_heart_failure():
    """
    Step 6.16: Verify evidence role classification for Acute Decompensated Heart Failure.
    BNP alone is supporting, cannot independently establish acute HF.
    """
    bnp_role = EvidenceRolePolicy.classify_evidence_role("Acute Decompensated Heart Failure", "laboratory", "BNP")
    assert bnp_role == ModalityEvidenceRole.SUPPORTING
    assert EvidenceRolePolicy.can_independently_establish_disease(bnp_role) is False
    assert EvidenceRolePolicy.can_support_candidate(bnp_role) is True

    creat_role = EvidenceRolePolicy.classify_evidence_role("Acute Decompensated Heart Failure", "laboratory", "Serum Creatinine")
    assert creat_role == ModalityEvidenceRole.CONTEXTUAL
    assert EvidenceRolePolicy.can_independently_establish_disease(creat_role) is False

    edema_role = EvidenceRolePolicy.classify_evidence_role("Acute Decompensated Heart Failure", "xray", "pulmonary edema")
    assert edema_role == ModalityEvidenceRole.DIAGNOSTIC
    assert EvidenceRolePolicy.can_independently_establish_disease(edema_role) is True


def test_09_evidence_role_classification_copd_xray_rule_out():
    """
    Step 6.16: For Acute COPD Exacerbation, chest X-ray is strictly RULE_OUT.
    Cannot enter positive diagnostic prediction pathway.
    """
    xray_role = EvidenceRolePolicy.classify_evidence_role("Acute COPD Exacerbation", "xray", "chest_xray")
    assert xray_role == ModalityEvidenceRole.RULE_OUT
    assert EvidenceRolePolicy.can_independently_establish_disease(xray_role) is False
    assert EvidenceRolePolicy.can_contribute_to_positive_diagnosis(xray_role, is_condition_eligible=False) is False


def test_10_evidence_role_classification_pneumothorax():
    """
    Step 6.16: For Spontaneous Pneumothorax, X-ray is DIAGNOSTIC, symptoms are SUPPORTING,
    and serum labs are CONTEXTUAL/NONSPECIFIC (labs do not diagnose pneumothorax).
    """
    pleural_line_role = EvidenceRolePolicy.classify_evidence_role("Spontaneous Pneumothorax", "xray", "visceral pleural line")
    assert pleural_line_role == ModalityEvidenceRole.DIAGNOSTIC
    assert EvidenceRolePolicy.can_independently_establish_disease(pleural_line_role) is True

    lab_role = EvidenceRolePolicy.classify_evidence_role("Spontaneous Pneumothorax", "laboratory", "Serum Creatinine")
    assert lab_role == ModalityEvidenceRole.CONTEXTUAL
    assert EvidenceRolePolicy.can_independently_establish_disease(lab_role) is False


# ==============================================================================
# 3. MODALITY-SPECIFIC TEMPORAL POLICIES TESTS (Correction 3)
# ==============================================================================

def test_11_temporal_policy_laboratory_presentation_vs_monitoring():
    """
    Step 6.16: Distinguish acute presentation lab panel from inpatient monitoring.
    """
    admit_dt = datetime(2026, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
    disch_dt = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    # 1. Presentation lab (2 hours after admission)
    pres_state = ClinicalTemporalPolicy.evaluate_modality_temporal_state(
        modality="laboratory",
        timestamp=datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        admit_time=admit_dt,
        disch_time=disch_dt,
    )
    assert pres_state["is_within_window"] is True
    assert pres_state["is_presentation_phase"] is True
    assert pres_state["temporal_category"] == "PRESENTATION_PANEL"

    # 2. Longitudinal inpatient monitoring lab (72 hours after admission)
    mon_state = ClinicalTemporalPolicy.evaluate_modality_temporal_state(
        modality="laboratory",
        timestamp=datetime(2026, 1, 13, 10, 0, 0, tzinfo=timezone.utc),
        admit_time=admit_dt,
        disch_time=disch_dt,
    )
    assert mon_state["is_within_window"] is True
    assert mon_state["is_presentation_phase"] is False
    assert mon_state["temporal_category"] == "INPATIENT_MONITORING"

    # 3. Pre-admission outpatient lab (48 hours prior) -> Rejected for acute encounter panel
    out_state = ClinicalTemporalPolicy.evaluate_modality_temporal_state(
        modality="laboratory",
        timestamp=datetime(2026, 1, 8, 10, 0, 0, tzinfo=timezone.utc),
        admit_time=admit_dt,
        disch_time=disch_dt,
    )
    assert out_state["is_within_window"] is False
    assert out_state["temporal_category"] == "PRE_ENCOUNTER_OUTPATIENT"

    # 4. Post-discharge lab -> Rejected
    post_state = ClinicalTemporalPolicy.evaluate_modality_temporal_state(
        modality="laboratory",
        timestamp=datetime(2026, 1, 16, 10, 0, 0, tzinfo=timezone.utc),
        admit_time=admit_dt,
        disch_time=disch_dt,
    )
    assert post_state["is_within_window"] is False
    assert post_state["temporal_category"] == "POST_DISCHARGE"


def test_12_temporal_policy_clinical_exam_precedes_treatment():
    """
    Step 6.16: Physical examination must precede therapeutic intervention.
    """
    admit_dt = datetime(2026, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
    treatment_dt = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Exam precedes treatment (1 hour after admission, 1 hour before treatment) -> PASS
    valid_exam = ClinicalTemporalPolicy.evaluate_modality_temporal_state(
        modality="clinical_exam",
        timestamp=datetime(2026, 1, 10, 11, 0, 0, tzinfo=timezone.utc),
        admit_time=admit_dt,
        treatment_time=treatment_dt,
    )
    assert valid_exam["is_within_window"] is True
    assert valid_exam["temporal_category"] == "INITIAL_EXAM"

    # 2. Exam follows treatment (3 hours after treatment) -> REJECTED as initial exam
    post_tx_exam = ClinicalTemporalPolicy.evaluate_modality_temporal_state(
        modality="clinical_exam",
        timestamp=datetime(2026, 1, 10, 15, 0, 0, tzinfo=timezone.utc),
        admit_time=admit_dt,
        treatment_time=treatment_dt,
    )
    assert post_tx_exam["is_within_window"] is False
    assert post_tx_exam["temporal_category"] == "POST_TREATMENT_EXAM"
    assert post_tx_exam["exclusion_reason"] == "exam_follows_treatment"


def test_13_patient_level_split_isolation_and_zero_leakage():
    """
    Step 6.16 Addendum: Verify patient-level partitioning guarantees zero patient leakage.
    Ensures all encounters/studies for a subject_id remain in the same partition.
    """
    from app.ai.multimodal_dataset_ingestion import (
        partition_patient_level_splits,
        validate_patient_split_isolation,
    )

    # 10 mock patients, each with 2 encounters and 2 imaging studies
    mock_records = []
    for p in range(1, 11):
        sub_id = f"sub_{p:03d}"
        for e in range(1, 3):
            mock_records.append({
                "subject_id": sub_id,
                "hadm_id": f"hadm_{p}_{e}",
                "study_id": f"study_{p}_{e}",
                "target_condition": "Pneumonia" if p % 2 == 0 else "Acute Decompensated Heart Failure",
                "anchor_year_group": "2014 - 2016",
            })

    splits = partition_patient_level_splits(
        records=mock_records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # Verify patient counts
    assert splits["train_patients_count"] == 7
    assert splits["val_patients_count"] == 1
    assert splits["test_patients_count"] == 2
    assert splits["isolation_verified"] is True

    # Validate strict zero-leakage using dedicated validator
    isolation_manifest = validate_patient_split_isolation(
        splits["train"], splits["val"], splits["test"]
    )
    assert isolation_manifest["isolated"] is True
    assert isolation_manifest["leakage_subjects"] == []

    # Verify encounter and study integrity: all records of a patient belong to same split
    train_subs = {r["subject_id"] for r in splits["train"]}
    val_subs = {r["subject_id"] for r in splits["val"]}
    test_subs = {r["subject_id"] for r in splits["test"]}

    assert train_subs.isdisjoint(val_subs)
    assert train_subs.isdisjoint(test_subs)
    assert val_subs.isdisjoint(test_subs)


def test_14_temporal_evaluation_with_anchor_year_group():
    """
    Step 6.16 Addendum: Verify temporal out-of-time evaluation via anchor_year_group.
    Distinguishes patient-held-out (in-distribution) from temporal holdout (OOD).
    """
    from app.ai.multimodal_dataset_ingestion import partition_patient_level_splits

    mock_records = [
        {"subject_id": "sub_01", "target_condition": "Pneumonia", "anchor_year_group": "2008 - 2010"},
        {"subject_id": "sub_02", "target_condition": "Pneumonia", "anchor_year_group": "2008 - 2010"},
        {"subject_id": "sub_03", "target_condition": "Pneumonia", "anchor_year_group": "2011 - 2013"},
        {"subject_id": "sub_04", "target_condition": "Pneumonia", "anchor_year_group": "2014 - 2016"},
        {"subject_id": "sub_05", "target_condition": "Spontaneous Pneumothorax", "anchor_year_group": "2017 - 2019"}, # Latest decile
        {"subject_id": "sub_06", "target_condition": "Spontaneous Pneumothorax", "anchor_year_group": "2017 - 2019"}, # Latest decile
    ]

    splits = partition_patient_level_splits(
        records=mock_records,
        train_ratio=0.75,
        val_ratio=0.25,
        test_ratio=0.0,
        temporal_holdout_anchor_year_group="2017 - 2019",
    )

    assert splits["evaluation_type"] == "temporal_out_of_time"
    # Patients from 2017 - 2019 must be in test set
    test_subs = {r["subject_id"] for r in splits["test"]}
    assert test_subs == {"sub_05", "sub_06"}

    # Rare class warning must note Spontaneous Pneumothorax has < 3 cases
    warnings = splits["rare_class_warnings"]
    assert any("Spontaneous Pneumothorax" in w for w in warnings)

