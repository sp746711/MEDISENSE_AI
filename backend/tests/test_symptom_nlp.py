"""Unit tests for Symptom NLP engine."""

from app.ai.symptom_nlp import extract_symptoms


def test_extract_symptoms_present_and_absent():
    text = "I have cough and fever for 3 days. I don't have chest pain."
    result = extract_symptoms(text)
    assert result["status"] == "extracted"
    symptoms = {s["symptom"]: s for s in result["symptoms"]}

    assert "cough" in symptoms
    assert symptoms["cough"]["state"] == "PRESENT"
    assert "3 days" in (symptoms["cough"]["duration"] or "")

    assert "fever" in symptoms
    assert symptoms["fever"]["state"] == "PRESENT"

    assert "chest pain" in symptoms
    assert symptoms["chest pain"]["state"] == "ABSENT"


def test_extract_symptoms_severity():
    text = "I have severe headache since yesterday without vomiting."
    result = extract_symptoms(text)
    symptoms = {s["symptom"]: s for s in result["symptoms"]}

    assert "headache" in symptoms
    assert symptoms["headache"]["state"] == "PRESENT"
    assert symptoms["headache"]["severity"] == "severe"

    assert "vomiting" in symptoms
    assert symptoms["vomiting"]["state"] == "ABSENT"


def test_extract_symptoms_empty():
    result = extract_symptoms("")
    assert result["status"] == "empty"
    assert result["symptoms"] == []


def test_extract_cough_character_and_mucus():
    text = "The symptoms gradually started about 5 days ago. The cough is mostly dry but sometimes produces a small amount of mucus."
    result = extract_symptoms(text)
    symptoms = {s["symptom"]: s for s in result["symptoms"]}

    assert "cough" in symptoms
    cough = symptoms["cough"]
    assert cough["state"] == "PRESENT"
    assert "5 days" in (cough.get("duration") or "")
    assert cough.get("onset") == "GRADUAL"
    assert cough.get("character") == "mostly dry" or cough.get("type") == "mostly dry"
    assert cough.get("type") != "productive"
    assert "mucus" in (cough.get("mucus") or cough.get("qualifier") or cough.get("context") or "")
    # Check that unrelated symptoms are not created or contaminated
    assert "mucus" not in symptoms.get("fever", {})

