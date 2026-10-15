"""Unit tests for Medical Report OCR and NLP structuring."""

from app.ai.report_nlp import structure_report_text
from app.ai.report_ocr import extract_text_from_file


def test_structure_report_text_lab_parameters():
    sample = """
    LABORATORY TEST REPORT
    Hemoglobin: 11.2 g/dL (Reference: 13.0 - 17.0)
    Fasting Blood Sugar: 145 mg/dL (Reference: 70 - 100)
    Serum Creatinine: 1.1 mg/dL (Reference: 0.6 - 1.2)
    Total Bilirubin: 0.8 mg/dL (Reference: 0.2 - 1.2)
    """
    res = structure_report_text(sample)
    assert res["status"] == "structured"
    findings_by_name = {f["test_name"]: f for f in res["findings"]}

    assert "Hemoglobin" in findings_by_name
    assert findings_by_name["Hemoglobin"]["value"] == 11.2
    assert findings_by_name["Hemoglobin"]["interpretation"] == "LOW"

    assert "Fasting Blood Glucose" in findings_by_name
    assert findings_by_name["Fasting Blood Glucose"]["value"] == 145.0
    assert findings_by_name["Fasting Blood Glucose"]["interpretation"] == "HIGH"

    assert "Serum Creatinine" in findings_by_name
    assert findings_by_name["Serum Creatinine"]["interpretation"] == "NORMAL"


def test_structure_report_text_empty():
    res = structure_report_text("")
    assert res["status"] == "unavailable"
    assert res["findings"] == []


def test_extract_text_nonexistent_file():
    res = extract_text_from_file("nonexistent_report_file.pdf")
    assert res["status"] == "unavailable"
    assert res["extracted_text"] is None


def test_structure_report_text_wbc_rbc_singular_and_plural():
    # Test plural forms
    plural_sample = """
    COMPLETE BLOOD COUNT
    White Blood Cells: 14,500 /uL (Reference: 4000 - 11000)
    Red Blood Cells: 3.20 mil/uL (Reference: 4.5 - 5.5)
    Platelets: 85,000 /uL (Reference: 150000 - 450000)
    """
    res_plural = structure_report_text(plural_sample)
    assert res_plural["status"] == "structured"
    plural_findings = {f["test_name"]: f for f in res_plural["findings"]}
    assert "White Blood Cell Count (WBC)" in plural_findings
    assert plural_findings["White Blood Cell Count (WBC)"]["value"] == 14500.0
    assert plural_findings["White Blood Cell Count (WBC)"]["interpretation"] == "HIGH"
    assert "Red Blood Cell Count (RBC)" in plural_findings
    assert plural_findings["Red Blood Cell Count (RBC)"]["value"] == 3.2
    assert plural_findings["Red Blood Cell Count (RBC)"]["interpretation"] == "LOW"

    # Test singular forms
    singular_sample = """
    COMPLETE BLOOD COUNT
    White Blood Cell Count: 6,800 /uL (Reference: 4000 - 11000)
    Red Blood Cell Count: 4.90 mil/uL (Reference: 4.5 - 5.5)
    """
    res_singular = structure_report_text(singular_sample)
    assert res_singular["status"] == "structured"
    singular_findings = {f["test_name"]: f for f in res_singular["findings"]}
    assert "White Blood Cell Count (WBC)" in singular_findings
    assert singular_findings["White Blood Cell Count (WBC)"]["value"] == 6800.0
    assert singular_findings["White Blood Cell Count (WBC)"]["interpretation"] == "NORMAL"
    assert "Red Blood Cell Count (RBC)" in singular_findings
    assert singular_findings["Red Blood Cell Count (RBC)"]["value"] == 4.9
    assert singular_findings["Red Blood Cell Count (RBC)"]["interpretation"] == "NORMAL"
