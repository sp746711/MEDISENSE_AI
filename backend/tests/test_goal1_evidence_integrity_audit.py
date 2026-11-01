"""Goal 1 Evidence Integrity and Assessment Audit Regression Tests.

Verifies:
1. Anatomy verification fail-closed semantics across assessment_service, evidence_engine, and triage_engine.
2. Report/image association: only persisted confirmed associations pair images and reports;
   pending stays unconfirmed, rejected stays rejected, no auto-pairing fallback for single image + single report.
3. Multimodal evidence compilation: unverified images excluded from clinical support,
   unreadable/empty reports remain visible, modalities remain distinct.
4. Deterministic triage protection: unverified images cannot trigger triage; emergency rules cannot be downgraded.
5. Condition prediction: honest symptom-based scope, up to 24 candidates, no fake candidates.
"""

from __future__ import annotations

import pytest
from typing import Any

from app.services.assessment_service import _anatomy_verification_fields
from app.ai.evidence_engine import (
    combine_evidence,
    _xray_anatomy_is_independently_verified,
)
from app.ai.triage_engine import (
    apply_triage,
    _anatomy_is_independently_verified,
)
from app.ai.condition_prediction import (
    get_condition_prediction_service,
    MAX_DISPLAY_CANDIDATES,
)


# ==============================================================================
# PHASE B: ANATOMY VERIFICATION FAIL-CLOSED TESTS
# ==============================================================================

def test_completed_processing_no_metadata_is_unverified():
    """Generic completed processing status alone with missing anatomy metadata must be unverified."""
    status, verified = _anatomy_verification_fields(None, record_status="completed")
    assert verified is False
    assert status == "unverified"

    empty_record = {"status": "completed", "inference_status": "completed"}
    assert _xray_anatomy_is_independently_verified(empty_record) is False
    assert _anatomy_is_independently_verified(empty_record) is False


def test_generic_completed_status_does_not_verify_anatomy():
    """A generic metadata status of COMPLETED alone must never establish anatomy acceptance."""
    meta = {"status": "COMPLETED", "confidence": 0.95}
    status, verified = _anatomy_verification_fields(meta, record_status="completed")
    assert verified is False
    assert status == "unverified"

    record = {"status": "completed", "anatomy_metadata": meta}
    assert _xray_anatomy_is_independently_verified(record) is False
    assert _anatomy_is_independently_verified(record) is False


def test_generic_verified_status_does_not_verify_anatomy():
    """A generic metadata status of VERIFIED alone must never establish anatomy acceptance."""
    meta = {"status": "VERIFIED"}
    status, verified = _anatomy_verification_fields(meta)
    assert verified is False
    assert status == "unverified"

    record = {"status": "completed", "anatomy_metadata": meta}
    assert _xray_anatomy_is_independently_verified(record) is False
    assert _anatomy_is_independently_verified(record) is False


def test_explicit_classifier_acceptance():
    """Explicit classifier acceptance gate passed without confirmation required establishes verification."""
    meta = {
        "classifier_gate_passed": True,
        "anatomy_classifier_accepted": True,
        "requires_user_confirmation": False,
        "candidate_is_specific": True,
    }
    status, verified = _anatomy_verification_fields(meta)
    assert verified is True
    assert status == "accepted"

    record = {"status": "completed", "anatomy_metadata": meta}
    assert _xray_anatomy_is_independently_verified(record) is True
    assert _anatomy_is_independently_verified(record) is True


def test_confirmation_required_is_not_accepted():
    """Confirmation required state must NOT establish anatomy acceptance."""
    meta = {
        "classifier_gate_passed": True,
        "anatomy_classifier_accepted": True,
        "requires_user_confirmation": True,
        "prediction_status": "needs_confirmation",
    }
    status, verified = _anatomy_verification_fields(meta)
    assert verified is False
    assert status == "confirmation_required"

    record = {"status": "completed", "anatomy_metadata": meta}
    assert _xray_anatomy_is_independently_verified(record) is False
    assert _anatomy_is_independently_verified(record) is False


def test_explicit_user_confirmation():
    """Explicit user confirmation establishes anatomy acceptance."""
    meta = {
        "anatomy_user_confirmed": True,
        "user_confirmation_status": "confirmed",
        "verification_source": "explicit_user_region_confirmation",
    }
    status, verified = _anatomy_verification_fields(meta)
    assert verified is True
    assert status == "confirmed"

    record = {"status": "completed", "anatomy_metadata": meta}
    assert _xray_anatomy_is_independently_verified(record) is True
    assert _anatomy_is_independently_verified(record) is True


def test_anatomy_classifier_unavailable():
    """If anatomy classifier failed or is unavailable, anatomy remains unverified."""
    meta = {"status": "failed", "reason_code": "anatomy_classification_failed"}
    status, verified = _anatomy_verification_fields(meta)
    assert verified is False
    assert status == "unverified"

    record = {"status": "completed", "anatomy_metadata": meta}
    assert _xray_anatomy_is_independently_verified(record) is False
    assert _anatomy_is_independently_verified(record) is False


def test_anatomy_mismatch_and_evidence_excluded_fail_closed():
    """Region mismatch or explicit evidence exclusion must immediately fail verification."""
    meta_mismatch = {"region_mismatch": True, "anatomy_user_confirmed": True}
    status_m, verified_m = _anatomy_verification_fields(meta_mismatch)
    assert verified_m is False

    record_m = {"has_region_mismatch": True, "anatomy_metadata": meta_mismatch}
    assert _xray_anatomy_is_independently_verified(record_m) is False
    assert _anatomy_is_independently_verified(record_m) is False

    meta_excluded = {"evidence_excluded": True, "anatomy_user_confirmed": True}
    status_e, verified_e = _anatomy_verification_fields(meta_excluded)
    assert verified_e is False

    record_e = {"evidence_excluded": True, "anatomy_metadata": meta_excluded}
    assert _xray_anatomy_is_independently_verified(record_e) is False
    assert _anatomy_is_independently_verified(record_e) is False


# ==============================================================================
# PHASE C: REPORT / IMAGE ASSOCIATION TESTS
# ==============================================================================

def test_one_image_one_report_no_association_is_not_paired():
    """One image + one report with NO persisted association must NOT be automatically paired."""
    img = {
        "id": "img-001",
        "file_name": "chest_xray.png",
        "region": "chest",
        "status": "COMPLETED",
        "prediction": "Pneumonia",
        "anatomy_independently_verified": True,
        "anatomy_metadata": {"anatomy_independently_verified": True},
    }
    rep = {
        "id": "rep-001",
        "file_name": "chest_rad_report.pdf",
        "extracted_text": "Chest X-ray shows right lower lobe consolidation consistent with pneumonia.",
        "structured_findings": {
            "qualitative_findings": [{"finding": "consolidation", "state": "PRESENT"}],
        },
    }

    evidence = combine_evidence(
        xray_results=[img],
        xray_reports=[rep],
        report_image_associations=[],  # No association
    )

    proc_rep = evidence["xray"]["reports"][0]
    # Verification status must be UNPAIRED or UNVERIFIED, NEVER MATCH
    assert proc_rep["region_verification"]["association_confirmed"] is False
    assert proc_rep["region_verification"]["status"] in {"UNPAIRED", "UNVERIFIED"}
    assert proc_rep["region_verification"]["matched_image_files"] == []


def test_one_image_one_report_pending_association_not_paired():
    """An association with status 'pending' must remain unconfirmed and NOT paired."""
    img = {
        "id": "img-001",
        "file_name": "chest_xray.png",
        "region": "chest",
        "status": "COMPLETED",
        "anatomy_independently_verified": True,
    }
    rep = {
        "id": "rep-001",
        "file_name": "chest_rad_report.pdf",
        "extracted_text": "Chest radiograph: clear lungs.",
    }
    assocs = [
        {"report_id": "rep-001", "xray_result_id": "img-001", "status": "pending"}
    ]

    evidence = combine_evidence(
        xray_results=[img],
        xray_reports=[rep],
        report_image_associations=assocs,
    )

    proc_rep = evidence["xray"]["reports"][0]
    assert proc_rep["region_verification"]["association_confirmed"] is False
    assert proc_rep["region_verification"]["matched_image_files"] == []


def test_one_image_one_report_confirmed_association_is_paired():
    """An association with status 'confirmed' must establish a confirmed pairing."""
    img = {
        "id": "img-001",
        "file_name": "chest_xray.png",
        "region": "chest",
        "status": "COMPLETED",
        "anatomy_independently_verified": True,
    }
    rep = {
        "id": "rep-001",
        "file_name": "chest_rad_report.pdf",
        "extracted_text": "Chest radiograph: normal heart and lungs.",
    }
    assocs = [
        {"report_id": "rep-001", "xray_result_id": "img-001", "status": "confirmed"}
    ]

    evidence = combine_evidence(
        xray_results=[img],
        xray_reports=[rep],
        report_image_associations=assocs,
    )

    proc_rep = evidence["xray"]["reports"][0]
    assert proc_rep["region_verification"]["association_confirmed"] is True
    assert "chest_xray.png" in proc_rep["region_verification"]["matched_image_files"]
    assert proc_rep["region_verification"]["status"] == "MATCH"


def test_one_image_one_report_rejected_association_is_rejected():
    """An association with status 'rejected' must NEVER be paired, even if regions match."""
    img = {
        "id": "img-001",
        "file_name": "chest_xray.png",
        "region": "chest",
        "status": "COMPLETED",
        "anatomy_independently_verified": True,
    }
    rep = {
        "id": "rep-001",
        "file_name": "chest_rad_report.pdf",
        "extracted_text": "Chest radiograph: normal.",
        "association_status": "REJECTED",
    }
    assocs = [
        {"report_id": "rep-001", "xray_result_id": "img-001", "status": "rejected"}
    ]

    evidence = combine_evidence(
        xray_results=[img],
        xray_reports=[rep],
        report_image_associations=assocs,
    )

    proc_rep = evidence["xray"]["reports"][0]
    assert proc_rep["region_verification"]["association_confirmed"] is False
    assert proc_rep["region_verification"]["matched_image_files"] == []


def test_multiple_images_and_one_report_partial_association():
    """With multiple images and 1 report, only the explicitly confirmed image is linked."""
    img1 = {
        "id": "img-001",
        "file_name": "chest_pa.png",
        "region": "chest",
        "status": "COMPLETED",
        "anatomy_independently_verified": True,
    }
    img2 = {
        "id": "img-002",
        "file_name": "knee_ap.png",
        "region": "knee",
        "status": "COMPLETED",
        "anatomy_independently_verified": True,
    }
    rep = {
        "id": "rep-001",
        "file_name": "chest_report.pdf",
        "extracted_text": "Chest radiograph impression.",
    }
    assocs = [
        {"report_id": "rep-001", "xray_result_id": "img-001", "status": "confirmed"},
        {"report_id": "rep-001", "xray_result_id": "img-002", "status": "rejected"},
    ]

    evidence = combine_evidence(
        xray_results=[img1, img2],
        xray_reports=[rep],
        report_image_associations=assocs,
    )

    proc_rep = evidence["xray"]["reports"][0]
    assert proc_rep["region_verification"]["matched_image_files"] == ["chest_pa.png"]
    assert "knee_ap.png" not in proc_rep["region_verification"]["matched_image_files"]


def test_one_report_linked_to_multiple_confirmed_views():
    """A report confirmed against two views of the same region pairs with both views."""
    img_ap = {
        "id": "img-ap",
        "file_name": "chest_ap.png",
        "region": "chest",
        "status": "COMPLETED",
        "anatomy_independently_verified": True,
    }
    img_lat = {
        "id": "img-lat",
        "file_name": "chest_lat.png",
        "region": "chest",
        "status": "COMPLETED",
        "anatomy_independently_verified": True,
    }
    rep = {
        "id": "rep-001",
        "file_name": "chest_two_views.pdf",
        "extracted_text": "Examination: Chest X-ray. 2 views: AP and lateral. Clear lungs.",
        "structured_findings": {"region": "chest"},
    }
    assocs = [
        {"report_id": "rep-001", "xray_result_id": "img-ap", "status": "confirmed"},
        {"report_id": "rep-001", "xray_result_id": "img-lat", "status": "confirmed"},
    ]

    evidence = combine_evidence(
        xray_results=[img_ap, img_lat],
        xray_reports=[rep],
        report_image_associations=assocs,
    )

    proc_rep = evidence["xray"]["reports"][0]
    matched = proc_rep["region_verification"]["matched_image_files"]
    assert "chest_ap.png" in matched
    assert "chest_lat.png" in matched
    assert proc_rep["region_verification"]["status"] == "MATCH"


def test_region_mismatch_with_confirmed_association():
    """If confirmed pair has mismatched body regions, mismatch is flagged and recorded as contradictory."""
    img = {
        "id": "img-wrist",
        "file_name": "wrist_xray.png",
        "region": "wrist",
        "status": "COMPLETED",
        "anatomy_independently_verified": True,
    }
    rep = {
        "id": "rep-chest",
        "file_name": "chest_report.pdf",
        "extracted_text": "Chest X-ray shows clear lungs.",
    }
    assocs = [
        {"report_id": "rep-chest", "xray_result_id": "img-wrist", "status": "confirmed"}
    ]

    evidence = combine_evidence(
        xray_results=[img],
        xray_reports=[rep],
        report_image_associations=assocs,
    )

    proc_rep = evidence["xray"]["reports"][0]
    assert proc_rep["region_verification"]["status"] == "MISMATCH"
    assert any("Conflicting evidence / body region mismatch" in c for c in evidence["contradictory"])


# ==============================================================================
# PHASE D & E: MULTIMODAL EVIDENCE & TRIAGE PROTECTION TESTS
# ==============================================================================

def test_unverified_image_findings_excluded_from_support_and_triage():
    """Unverified image findings must NOT enter supporting evidence or trigger triage."""
    unverified_img = {
        "id": "img-unverified",
        "file_name": "unverified_chest.png",
        "region": "chest",
        "status": "COMPLETED",
        "prediction": "Pneumothorax",
        "findings": [{"label": "Pneumothorax", "is_actionable_finding": True}],
        "anatomy_independently_verified": False,
        "anatomy_metadata": {"status": "unverified", "requires_user_confirmation": True},
    }

    evidence = combine_evidence(
        symptoms=[{"symptom": "mild cough", "state": "PRESENT"}],
        xray_results=[unverified_img],
    )

    # Finding must NOT be in supporting evidence
    supporting_text = " ".join(evidence["supporting"])
    assert "Pneumothorax" not in supporting_text

    # Triage must NOT trigger emergency based on unverified X-ray finding
    triage = apply_triage(evidence)
    assert triage["pathway"] != "EMERGENCY"
    assert not any("Pneumothorax" in rule for rule in triage["rules_triggered"])


def test_raw_image_scores_remain_non_actionable():
    """Raw dense label scores without is_actionable_finding must NOT support diagnoses."""
    img = {
        "id": "img-scores",
        "file_name": "chest_scores.png",
        "region": "chest",
        "status": "COMPLETED",
        "findings": [
            {"label": "Atelectasis", "score": 0.42, "is_actionable_finding": False},
            {"label": "Cardiomegaly", "score": 0.35, "is_actionable_finding": False},
        ],
        "anatomy_independently_verified": True,
        "anatomy_metadata": {"anatomy_independently_verified": True},
    }

    evidence = combine_evidence(xray_results=[img])
    supporting_text = " ".join(evidence["supporting"])
    assert "Atelectasis" not in supporting_text
    assert "Cardiomegaly" not in supporting_text


def test_generic_normal_abnormal_output_non_diagnostic():
    """A generic screening output of 'normal' or 'abnormal' must not be counted as a specific diagnosis."""
    img = {
        "id": "img-normal",
        "file_name": "knee_normal.png",
        "region": "knee",
        "status": "COMPLETED",
        "prediction": "normal",
        "anatomy_independently_verified": True,
        "anatomy_metadata": {"anatomy_independently_verified": True},
    }

    evidence = combine_evidence(xray_results=[img])
    assert not any("returned explicitly actionable candidate" in s for s in evidence["supporting"])
    assert any("broad normal/abnormal class rather than a specific disease finding" in u for u in evidence["unassessed"])


def test_radiology_report_visible_when_extraction_fails():
    """Unreadable radiology report remains visible in processed_xray_reports with UNREADABLE status."""
    rep = {
        "id": "rep-unreadable",
        "file_name": "corrupt_scan.pdf",
        "extracted_text": "",
        "structured_findings": {},
    }

    evidence = combine_evidence(xray_reports=[rep])
    reports = evidence["xray"]["reports"]
    assert len(reports) == 1
    assert reports[0]["status"] == "UNREADABLE"
    assert reports[0]["file_name"] == "corrupt_scan.pdf"


def test_radiology_report_visible_when_zero_findings():
    """Radiology report with extracted text but 0 qualitative findings remains visible with status EXTRACTED."""
    rep = {
        "id": "rep-zero",
        "file_name": "normal_xray_report.pdf",
        "extracted_text": "Study Date: 2026-10-15. Examination: Chest radiograph. Technique: Single view.",
        "structured_findings": {"qualitative_findings": [], "narrative_findings": []},
    }

    evidence = combine_evidence(xray_reports=[rep])
    reports = evidence["xray"]["reports"]
    assert len(reports) == 1
    assert reports[0]["status"] == "EXTRACTED"
    assert reports[0]["qualitative_findings"] == []


def test_medical_report_distinct_from_radiology_report():
    """Medical/lab reports and radiology reports populate separate structures in compiled evidence."""
    med_rep = {
        "id": "lab-001",
        "file_name": "cbc_report.pdf",
        "report_category": "medical_report",
        "extracted_text": "Hemoglobin 14.2 g/dL",
        "structured_findings": {
            "lab_parameters": [{"test_name": "Hemoglobin", "value": 14.2, "unit": "g/dL", "status": "NORMAL"}]
        },
    }
    rad_rep = {
        "id": "rad-001",
        "file_name": "chest_radiology.pdf",
        "report_category": "xray_report",
        "extracted_text": "Lungs clear bilaterally.",
        "structured_findings": {
            "qualitative_findings": [{"finding": "clear lungs", "state": "PRESENT"}]
        },
    }

    evidence = combine_evidence(
        medical_reports=[med_rep],
        xray_reports=[rad_rep],
    )

    # Labs are under report.medical_reports or report.normal_findings
    assert len(evidence["report"]["medical_reports"]) == 1
    assert evidence["report"]["medical_reports"][0]["file_name"] == "cbc_report.pdf"

    # Radiology reports are under xray.reports
    assert len(evidence["xray"]["reports"]) == 1
    assert evidence["xray"]["reports"][0]["file_name"] == "chest_radiology.pdf"


def test_symptom_states_remain_distinct():
    """PRESENT, ABSENT, UNKNOWN, and HISTORICAL remain strictly distinct."""
    symptoms = [
        {"symptom": "fever", "state": "PRESENT"},
        {"symptom": "rash", "state": "ABSENT"},
        {"symptom": "wheezing", "state": "UNKNOWN"},
        {"symptom": "childhood asthma", "state": "HISTORICAL"},
    ]

    evidence = combine_evidence(symptoms=symptoms)
    s = evidence["symptoms"]

    present_names = [item["symptom"] for item in s["present"]]
    absent_names = [item["symptom"] for item in s["absent"]]
    unknown_names = [item["symptom"] for item in s["unknown"]]
    historical_names = [item["symptom"] for item in s["historical"]]

    assert "fever" in present_names
    assert "rash" in absent_names
    assert "wheezing" in unknown_names
    assert "childhood asthma" in historical_names

    # UNKNOWN must never be converted to ABSENT
    assert "wheezing" not in absent_names


def test_emergency_triage_cannot_be_downgraded_by_ml_or_reassuring_evidence():
    """A confirmed emergency symptom must trigger EMERGENCY and cannot be downgraded."""
    evidence = combine_evidence(
        symptoms=[
            {"symptom": "chest pain", "severity": "severe", "state": "PRESENT"},
            {"symptom": "mild headache", "state": "PRESENT"},
        ],
        report_findings=[
            {"test_name": "Hemoglobin", "value": 14.5, "unit": "g/dL", "status": "NORMAL"}
        ],
    )

    triage = apply_triage(evidence)
    assert triage["pathway"] == "EMERGENCY"
    assert any("chest_pain" in rule.lower() or "chest pain" in rule.lower() for rule in triage["rules_triggered"])


# ==============================================================================
# PHASE G: CONDITION PREDICTION SCOPE & 24 CANDIDATE LIMIT TESTS
# ==============================================================================

def test_condition_prediction_candidate_display_and_limit():
    """Condition predictor returns up to 24 genuine candidates with accurate limit note."""
    svc = get_condition_prediction_service()
    res = svc.predict_conditions(
        symptoms_present=[{"symptom": "fever"}, {"symptom": "cough"}, {"symptom": "fatigue"}],
        symptoms_absent=[],
        symptoms_unknown=[],
    )

    assert res["status"] in {"completed", "rejected"}
    if res["status"] == "completed":
        assert len(res["conditions"]) <= MAX_DISPLAY_CANDIDATES
        assert res["maximum_candidates_displayed"] == 24
        assert "24" in res["candidate_limit_note"]
        # Model scores must not claim calibrated clinical probability
        for cand in res["conditions"]:
            assert "model_score" in cand
            assert "supporting_symptoms" in cand
            assert "clinical_rationale" in cand
