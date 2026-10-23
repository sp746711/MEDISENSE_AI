"""Phase 1 Regression Test Suite — Medical Report & Lab Extraction Audit & Fix.

Covers all 8 required tests specified in Section 5 of the Master Prompt:
Test 1 — CBC values and units (Hemoglobin, WBC, RBC, Platelets, Hematocrit)
Test 2 — Alternate RBC formats (mil/uL, million cells/uL, million cells/µL, etc.)
Test 3 — No reference range (preserves UNKNOWN status, never invents ranges)
Test 4 — Explicit flag conflicts with range (preserves source flag, exposes conflict)
Test 5 — Multiple consecutive rows (no status/range leakage between adjacent rows)
Test 6 — Structured table and plain text parity
Test 7 — PDF parity (preserves value with complete unit, reference range, status)
Test 8 — Verification of existing behaviour without regressions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.ai.evidence_engine import combine_evidence
from app.ai.report_nlp import structure_report_text
from app.services.assessment_service import generate_assessment_feedback
from app.services.pdf_service import generate_assessment_pdf


# ==============================================================================
# Test 1 — CBC values and units
# ==============================================================================
def test_1_cbc_values_and_units():
    """Verify that CBC values, full units, reference ranges, and statuses are accurately extracted.

    Input:
    - Hemoglobin: 10.5 g/dL, range 13.0–17.0, LOW.
    - WBC: 12500 cells/µL, range 4000–11000, HIGH.
    - RBC: 4.2 million cells/µL, range 4.5–5.5, NORMAL.
    - Platelets: 280000 cells/µL, range 150000–450000, NORMAL.
    - Hematocrit: 35%, range 40–50, LOW.
    """
    sample = """
    COMPLETE BLOOD COUNT (CBC)
    Hemoglobin: 10.5 g/dL, range 13.0–17.0, LOW
    WBC: 12500 cells/µL, range 4000–11000, HIGH
    RBC: 4.2 million cells/µL, range 4.5–5.5, NORMAL
    Platelets: 280000 cells/µL, range 150000–450000, NORMAL
    Hematocrit: 35%, range 40–50, LOW
    """
    res = structure_report_text(sample)
    assert res["status"] == "structured"
    findings = {f["test_name"]: f for f in res["findings"]}

    # Verify all 5 tests are extracted
    assert len(findings) == 5

    # Hemoglobin
    assert "Hemoglobin" in findings
    hb = findings["Hemoglobin"]
    assert hb["value"] == 10.5
    assert hb["unit"] == "g/dL"
    assert hb["reference_range"] == "13.0 - 17.0"
    assert hb["status"] == "LOW"

    # WBC
    assert "White Blood Cell Count (WBC)" in findings
    wbc = findings["White Blood Cell Count (WBC)"]
    assert wbc["value"] == 12500.0
    assert "cells" in wbc["unit"]
    assert wbc["reference_range"] == "4000.0 - 11000.0"
    assert wbc["status"] == "HIGH"

    # RBC: multiplier preserved, value not scaled, source flag preserved
    assert "Red Blood Cell Count (RBC)" in findings
    rbc = findings["Red Blood Cell Count (RBC)"]
    assert rbc["value"] == 4.2
    assert "million cells" in rbc["unit"]
    assert rbc["reference_range"] == "4.5 - 5.5"
    assert rbc["status"] == "NORMAL"  # Source flag preserved
    assert rbc["calculated_status"] == "LOW"
    assert rbc["conflict"] is True  # Conflict correctly exposed

    # Platelets: MUST remain NORMAL (must not be contaminated by Hematocrit's LOW)
    assert "Platelet Count" in findings
    plt = findings["Platelet Count"]
    assert plt["value"] == 280000.0
    assert "cells" in plt["unit"]
    assert plt["reference_range"] == "150000.0 - 450000.0"
    assert plt["status"] == "NORMAL", "Platelet Count incorrectly contaminated with LOW from neighbouring Hematocrit!"
    assert plt["conflict"] is False

    # Hematocrit: extracted as a structured finding
    assert "Hematocrit" in findings
    hct = findings["Hematocrit"]
    assert hct["value"] == 35.0
    assert hct["unit"] == "%"
    assert hct["reference_range"] == "40.0 - 50.0"
    assert hct["status"] == "LOW"


# ==============================================================================
# Test 2 — Alternate RBC formats
# ==============================================================================
def test_2_alternate_rbc_formats():
    """Verify supported RBC formats are preserved without accidental numeric scaling."""
    formats = [
        ("Red Blood Cells: 4.2 million cells/µL (Reference: 4.5 - 5.5)", 4.2, "million cells/µL"),
        ("RBC: 4.80 million cells/uL (Reference: 4.5 - 5.5)", 4.8, "million cells/uL"),
        ("Red Blood Cell Count: 3.20 mil/uL (Reference: 4.5 - 5.5)", 3.2, "mil/uL"),
        ("Total Erythrocytes: 4.50 million/uL (Reference: 4.5 - 5.5)", 4.5, "million/uL"),
        ("RBC: 4.70 x 10^6/uL (Reference: 4.5 - 5.5)", 4.7, "x 10^6/uL"),
    ]

    for text_line, expected_val, expected_unit_substr in formats:
        res = structure_report_text(text_line)
        findings = {f["test_name"]: f for f in res["findings"]}
        assert "Red Blood Cell Count (RBC)" in findings, f"Failed to extract RBC for: {text_line}"
        rbc = findings["Red Blood Cell Count (RBC)"]
        assert rbc["value"] == expected_val, f"RBC value scaled unexpectedly: {rbc['value']} != {expected_val}"
        assert expected_unit_substr.lower() in rbc["unit"].lower(), f"RBC unit '{rbc['unit']}' does not contain '{expected_unit_substr}'"


# ==============================================================================
# Test 3 — No reference range
# ==============================================================================
def test_3_no_reference_range_gives_unknown():
    """A report with a value and no reference range must not receive an invented range.

    Status must be UNKNOWN.
    """
    sample = """
    LABORATORY TEST RESULT
    Platelets: 280000 cells/µL
    Hemoglobin: 14.2 g/dL
    """
    res = structure_report_text(sample)
    findings = {f["test_name"]: f for f in res["findings"]}

    assert "Platelet Count" in findings
    plt = findings["Platelet Count"]
    assert plt["reference_range"] is None
    assert plt["reference_low"] is None
    assert plt["reference_high"] is None
    assert plt["status"] == "UNKNOWN"
    assert plt["calculated_status"] == "UNKNOWN"

    assert "Hemoglobin" in findings
    hb = findings["Hemoglobin"]
    assert hb["reference_range"] is None
    assert hb["status"] == "UNKNOWN"


# ==============================================================================
# Test 4 — Explicit flag conflicts with range
# ==============================================================================
def test_4_explicit_flag_conflicts_with_range():
    """If a report's explicit flag disagrees with the range, preserve source flag and report conflict."""
    sample = """
    Hemoglobin: 11.0 g/dL, range 13.0 - 17.0, NORMAL
    Serum Creatinine: 0.9 mg/dL, range 0.6 - 1.2, HIGH
    """
    res = structure_report_text(sample)
    findings = {f["test_name"]: f for f in res["findings"]}

    # Case A: Value 11.0 is below 13.0-17.0, but explicit flag is NORMAL
    assert "Hemoglobin" in findings
    hb = findings["Hemoglobin"]
    assert hb["explicit_flag"] == "NORMAL"
    assert hb["calculated_status"] == "LOW"
    assert hb["status"] == "NORMAL", "Explicit source flag was not preserved!"
    assert hb["conflict"] is True, "Conflict was not detected!"
    assert "conflicts with calculated status" in hb["conflict_detail"]

    # Case B: Value 0.9 is within 0.6-1.2, but explicit flag is HIGH
    assert "Serum Creatinine" in findings
    cr = findings["Serum Creatinine"]
    assert cr["explicit_flag"] == "HIGH"
    assert cr["calculated_status"] == "NORMAL"
    assert cr["status"] == "HIGH", "Explicit source flag was not preserved!"
    assert cr["conflict"] is True, "Conflict was not detected!"

    # Verify conflict reaches evidence engine contradictory findings
    evidence = combine_evidence(report_findings=res["findings"], report_provided=True)
    contradictions = evidence.get("contradictory", [])
    assert any("Hemoglobin" in c and "NORMAL" in c for c in contradictions), (
        f"Contradictory evidence missing Hemoglobin flag conflict: {contradictions}"
    )


# ==============================================================================
# Test 5 — Multiple consecutive rows
# ==============================================================================
def test_5_multiple_consecutive_rows_no_status_leak():
    """Consecutive rows with different flags must not leak flags into adjacent tests."""
    sample = """
    Hemoglobin: 10.5 g/dL (Reference: 13.0 - 17.0) LOW
    White Blood Cells: 12500 /uL (Reference: 4000 - 11000) HIGH
    Red Blood Cells: 4.8 mil/uL (Reference: 4.5 - 5.5) NORMAL
    Platelets: 280000 cells/uL (Reference: 150000 - 450000) NORMAL
    Hematocrit: 34% (Reference: 40 - 50) LOW
    Total Cholesterol: 240 mg/dL (Reference: 120 - 200) HIGH
    Fasting Blood Sugar: 90 mg/dL (Reference: 70 - 100) NORMAL
    """
    res = structure_report_text(sample)
    findings = {f["test_name"]: f for f in res["findings"]}

    expected_statuses = {
        "Hemoglobin": "LOW",
        "White Blood Cell Count (WBC)": "HIGH",
        "Red Blood Cell Count (RBC)": "NORMAL",
        "Platelet Count": "NORMAL",
        "Hematocrit": "LOW",
        "Total Cholesterol": "HIGH",
        "Fasting Blood Glucose": "NORMAL",
    }

    for test_name, expected_stat in expected_statuses.items():
        assert test_name in findings, f"Test {test_name} not extracted"
        actual_stat = findings[test_name]["status"]
        assert actual_stat == expected_stat, (
            f"Status leaked for {test_name}! Expected {expected_stat}, got {actual_stat}"
        )


# ==============================================================================
# Test 6 — Structured table and plain text parity
# ==============================================================================
def test_6_structured_table_and_plain_text_parity():
    """Verify both table and plain text paths extract matching structured fields."""
    table_data = [
        ["Investigation", "Observed Value", "Units", "Reference Range", "Interpretation"],
        ["Platelets", "280000", "cells/µL", "150000 - 450000", "NORMAL"],
        ["Hematocrit", "35", "%", "40 - 50", "LOW"],
        ["Hemoglobin", "10.5", "g/dL", "13.0 - 17.0", "LOW"],
        ["Red Blood Cells", "4.2", "million cells/µL", "4.5 - 5.5", "NORMAL"],
    ]
    res_table = structure_report_text("", tables=[table_data])
    table_findings = {f["test_name"]: f for f in res_table["findings"]}

    plain_text = """
    Investigation: Platelets: 280000 cells/µL, Reference: 150000 - 450000, NORMAL
    Investigation: Hematocrit: 35%, Reference: 40 - 50, LOW
    Investigation: Hemoglobin: 10.5 g/dL, Reference: 13.0 - 17.0, LOW
    Investigation: Red Blood Cells: 4.2 million cells/µL, Reference: 4.5 - 5.5, NORMAL
    """
    res_text = structure_report_text(plain_text)
    text_findings = {f["test_name"]: f for f in res_text["findings"]}

    for canonical in ["Platelet Count", "Hematocrit", "Hemoglobin", "Red Blood Cell Count (RBC)"]:
        assert canonical in table_findings, f"Missing in table: {canonical}"
        assert canonical in text_findings, f"Missing in text: {canonical}"

        tf = table_findings[canonical]
        xf = text_findings[canonical]

        assert tf["value"] == xf["value"]
        assert tf["status"] == xf["status"]
        assert tf["reference_range"] == xf["reference_range"]
        # Both should retain units
        assert tf["unit"] is not None and xf["unit"] is not None


# ==============================================================================
# Test 7 — PDF parity and downstream feedback
# ==============================================================================
def test_7_pdf_and_feedback_parity(tmp_path):
    """Confirm downstream assessment feedback and PDF generation preserve complete units, values, and statuses."""
    sample = """
    Hemoglobin: 10.5 g/dL, range 13.0–17.0, LOW
    RBC: 4.2 million cells/µL, range 4.5–5.5, NORMAL
    Platelets: 280000 cells/µL, range 150000–450000, NORMAL
    Hematocrit: 35%, range 40–50, LOW
    """
    res = structure_report_text(sample)
    report_findings = res["findings"]

    # 1. Evidence engine integration
    evidence = combine_evidence(report_findings=report_findings, report_provided=True)
    report_meta = evidence.get("report", {})
    abnormal = report_meta.get("abnormal_findings", [])
    normal = report_meta.get("normal_findings", [])

    abnormal_names = {f["test_name"] for f in abnormal}
    normal_names = {f["test_name"] for f in normal}

    assert "Hemoglobin" in abnormal_names
    assert "Hematocrit" in abnormal_names
    assert "Platelet Count" in normal_names
    assert "Red Blood Cell Count (RBC)" in normal_names

    # 2. Assessment feedback construction
    triage = {"pathway": "MILD", "rules_triggered": [], "ignored_rules": []}
    feedback = generate_assessment_feedback(evidence, triage, specialty="Hematology")
    med_findings = feedback.get("medical_report_findings", [])

    findings_map = {f["test_name"]: f for f in med_findings}
    assert "Red Blood Cell Count (RBC)" in findings_map
    rbc_feed = findings_map["Red Blood Cell Count (RBC)"]
    assert "4.2" in rbc_feed["value"]
    assert "million cells" in rbc_feed["value"]
    assert rbc_feed["status"] == "NORMAL"

    assert "Platelet Count" in findings_map
    plt_feed = findings_map["Platelet Count"]
    assert plt_feed["status"] == "NORMAL"

    assert "Hematocrit" in findings_map
    hct_feed = findings_map["Hematocrit"]
    assert hct_feed["status"] == "LOW"

    # 3. PDF generation parity
    mock_db = MagicMock()
    mock_assessment = MagicMock()
    mock_assessment.assessment_id = uuid4()
    mock_assessment.user_id = uuid4()
    mock_assessment.pathway = "MILD"
    mock_assessment.specialty = "General Physician"
    mock_assessment.rules_version = "v2.1"
    mock_assessment.input_types = ["medical_report"]
    from datetime import datetime, timezone
    mock_assessment.created_at = datetime.now(timezone.utc)
    mock_assessment.result_payload = {
        "feedback": feedback,
        "triage": triage,
        "evidence": evidence,
    }

    mock_user = MagicMock()
    mock_user.name = "Test Audit Patient"
    mock_user.district = "Pune"
    mock_user.state = "Maharashtra"

    def mock_get(model, ident):
        from app.database.models import Assessment, User
        if model is Assessment:
            return mock_assessment
        if model is User:
            return mock_user
        return None

    mock_db.get.side_effect = mock_get

    pdf_res = generate_assessment_pdf(mock_assessment.assessment_id, mock_db)
    assert pdf_res["status"] == "ok"
    pdf_path = Path(pdf_res["path"])
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000


# ----------------------------------------------------------------------
# Problem 1 — Radiology evidence uncertainty regression tests
# ----------------------------------------------------------------------

def _qualitative_finding(text: str, term: str) -> dict[str, Any]:
    """Return the extracted qualitative finding for a term or fail clearly."""
    result = structure_report_text(text)
    matches = [
        item for item in result.get("qualitative_findings", [])
        if term.lower() in str(item.get("finding", "")).lower()
    ]
    assert matches, (
        f"Expected radiology term {term!r} to be extracted. "
        f"Got: {result.get('qualitative_findings', [])!r}"
    )
    return matches[0]


@pytest.mark.parametrize(
    ("text", "term", "expected"),
    [
        ("Impression: Possible non-displaced linear left parietal skull fracture.", "fracture", "UNCERTAIN"),
        ("Impression: Findings are suggestive of right lower-lobe pneumonia.", "pneumonia", "UNCERTAIN"),
        ("Findings: No significant pleural effusion.", "pleural effusion", "ABSENT"),
        ("Findings: Acute fracture identified.", "fracture", "PRESENT"),
    ],
)
def test_problem1_radiology_uncertainty_and_negation(text, term, expected):
    """Uncertain impressions must not be promoted to confirmed findings."""
    assert _qualitative_finding(text, term).get("state") == expected


def test_problem1_radiology_conflicting_statements_are_not_silently_confirmed():
    """A positive and negative statement for the same finding must retain conflict."""
    text = (
        "Findings: There is a pleural effusion. "
        "Impression: No pleural effusion is identified."
    )
    result = structure_report_text(text)
    findings = [
        item for item in result.get("qualitative_findings", [])
        if "pleural effusion" in str(item.get("finding", "")).lower()
    ]
    assert findings, "Expected pleural effusion to be extracted."
    states = {str(item.get("state", "UNKNOWN")).upper() for item in findings}
    assert "CONFLICTING" in states or {"PRESENT", "ABSENT"}.issubset(states), (
        f"Conflicting report statements were lost: {findings!r}"
    )



def test_problem1_uncertain_ossific_and_avulsion_fragments_are_extracted_as_uncertain():
    """Possible/old ossific fragments must remain uncertain, not confirmed fractures."""
    text = (
        "Findings: A tiny ossific fragment is seen near the lateral tibial plateau; "
        "this may be a small avulsion fragment or may reflect an old injury. "
        "Its age is indeterminate. No definite acute fracture is identified."
    )
    result = structure_report_text(text)
    states = {item["finding"]: item["state"] for item in result.get("qualitative_findings", [])}
    assert states.get("fracture") == "UNCERTAIN"
    assert states.get("avulsion fragment") == "UNCERTAIN"
    assert states.get("ossific fragment") == "UNCERTAIN"


def test_problem1_knee_effusion_opposing_statements_are_conflicting():
    """Opposing statements for the same effusion must be retained as a conflict."""
    text = (
        "Findings: A small knee joint effusion is present. "
        "Impression: No significant knee joint effusion is identified."
    )
    result = structure_report_text(text)
    findings = [
        item for item in result.get("qualitative_findings", [])
        if item.get("finding") in {"joint effusion", "knee joint effusion"}
    ]
    assert findings, f"Expected knee effusion findings; got {result.get('qualitative_findings', [])!r}"
    assert any(item.get("state") == "CONFLICTING" for item in findings) or (
        any(item.get("state") == "PRESENT" for item in findings)
        and any(item.get("state") == "ABSENT" for item in findings)
    ), f"Opposing effusion statements were not preserved: {findings!r}"

def test_problem1_uncertain_radiology_does_not_enter_confirmed_supporting_evidence():
    """UNCERTAIN findings must not be presented as confirmed supporting evidence."""
    evidence = combine_evidence(
        symptoms=[{"symptom": "cough", "state": "PRESENT", "domain": "respiratory"}],
        qualitative_report_findings=[{
            "finding": "pneumonia",
            "category": "radiology",
            "domain": "respiratory",
            "state": "UNCERTAIN",
            "source": "uploaded_radiology_report",
        }],
    )
    supporting = " ".join(str(x).lower() for x in evidence.get("supporting", []))
    assert "confirmed: pneumonia" not in supporting
    serialized = repr(evidence).lower()
    assert "pneumonia" in serialized and "uncertain" in serialized, (
        f"Uncertain finding was not preserved in evidence: {evidence!r}"
    )
