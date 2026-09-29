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
