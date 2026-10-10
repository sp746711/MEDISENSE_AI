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


def test_regression_a_mild_fever_and_sore_throat():
    """Requirement A: 'I have a mild fever and sore throat.'
    Expected: fever = PRESENT / MILD, sore throat = PRESENT / UNKNOWN (no severity leakage).
    """
    res = extract_symptoms("I have a mild fever and sore throat.")
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    assert "fever" in s_map
    assert s_map["fever"]["state"] == "PRESENT"
    assert s_map["fever"]["severity"] == "MILD"

    assert "sore throat" in s_map
    assert s_map["sore throat"]["state"] == "PRESENT"
    assert s_map["sore throat"]["severity"] == "UNKNOWN"


def test_regression_b_tired_and_mild_chest_discomfort_when_coughing():
    """Requirement B: 'I feel tired and have mild chest discomfort when coughing.'
    Expected: fatigue = PRESENT / UNKNOWN, chest discomfort = PRESENT / MILD / trigger coughing.
    """
    res = extract_symptoms("I feel tired and have mild chest discomfort when coughing.")
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    # Fatigue / tiredness
    fatigue = s_map.get("tiredness") or s_map.get("fatigue")
    assert fatigue is not None
    assert fatigue["state"] == "PRESENT"
    assert fatigue["severity"] == "UNKNOWN"

    # Chest discomfort
    assert "chest discomfort" in s_map
    cd = s_map["chest discomfort"]
    assert cd["state"] == "PRESENT"
    assert cd["severity"] == "MILD"
    assert cd.get("trigger") == "coughing"


def test_regression_c_do_not_feel_confused():
    """Requirement C: 'I do not feel confused.'
    Expected: confusion = ABSENT (canonical concept).
    """
    res = extract_symptoms("I do not feel confused.")
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    assert "confusion" in s_map
    assert s_map["confusion"]["state"] == "ABSENT"


def test_regression_d_no_severe_breathing_difficulty():
    """Requirement D: 'I do not have severe breathing difficulty.'
    Expected: severe breathing difficulty = ABSENT (exact negation scope preserved).
    """
    res = extract_symptoms("I do not have severe breathing difficulty.")
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    assert "severe breathing difficulty" in s_map
    assert s_map["severe breathing difficulty"]["state"] == "ABSENT"
    # Must NOT create difficulty breathing PRESENT
    assert s_map.get("difficulty breathing", {}).get("state") != "PRESENT"


def test_regression_e_headache_duration_scoping():
    """Requirement E: 'I have had a headache for 2 days. I feel nauseous.'
    Expected: headache duration = 2 days, nausea duration = UNKNOWN (None).
    """
    res = extract_symptoms("I have had a headache for 2 days. I feel nauseous.")
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    assert "headache" in s_map
    assert s_map["headache"]["state"] == "PRESENT"
    assert "2 days" in (s_map["headache"]["duration"] or "")

    assert "nausea" in s_map
    assert s_map["nausea"]["state"] == "PRESENT"
    assert s_map["nausea"]["duration"] is None


def test_regression_f_chest_discomfort_not_erased_by_negative_chest_pain_at_rest():
    """Requirement F: 'I have mild chest discomfort when coughing. I do not have chest pain when resting.'
    Expected BOTH: chest discomfort = PRESENT / MILD / trigger coughing, chest pain at rest = ABSENT.
    The negative finding must not erase the positive finding.
    """
    res = extract_symptoms("I have mild chest discomfort when coughing. I do not have chest pain when resting.")
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    assert "chest discomfort" in s_map
    assert s_map["chest discomfort"]["state"] == "PRESENT"
    assert s_map["chest discomfort"]["severity"] == "MILD"
    assert s_map["chest discomfort"].get("trigger") == "coughing"

    assert "chest pain at rest" in s_map
    assert s_map["chest pain at rest"]["state"] == "ABSENT"


