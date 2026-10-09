"""Regression tests for Headache and Respiratory demonstration cases."""

from app.ai.symptom_nlp import extract_symptoms

HEADACHE_TEXT = """I have had a severe headache for 2 days, mainly on the right side of my head.
The pain feels throbbing and becomes worse when I am exposed to bright light or loud sounds.
I feel nauseous but I have not vomited.
I feel tired and have difficulty concentrating.
The headache started gradually yesterday morning.
I have had similar headaches once or twice before.
I do not have a fever.
I do not have weakness or numbness in my arms or legs.
I do not have difficulty speaking.
I have not fainted and I do not have a seizure."""

RESPIRATORY_TEXT = """I have had fever for 3 days. My temperature is around 101°F.
I have a dry cough, sore throat, headache and tiredness.
I also feel mild chest discomfort when coughing.
The symptoms started gradually 3 days ago.
I have mild difficulty sleeping because of the cough.
I do not have severe breathing difficulty.
I do not have chest pain when resting.
No vomiting or fainting."""


def test_headache_regression_suite():
    res = extract_symptoms(HEADACHE_TEXT)
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    # Headache qualifiers
    assert s_map["headache"]["status"] == "PRESENT"
    assert s_map["headache"]["severity"].lower() == "severe"
    assert "2 days" in s_map["headache"]["duration"]
    assert s_map["headache"]["laterality"] == "RIGHT"
    assert s_map["headache"]["quality"] == "THROBBING"

    # Associated findings
    assert s_map["photophobia"]["status"] == "PRESENT"
    assert s_map["phonophobia"]["status"] == "PRESENT"
    assert s_map["nausea"]["status"] == "PRESENT"
    assert s_map["vomiting"]["status"] == "ABSENT"
    assert s_map["fatigue"]["status"] == "PRESENT"
    assert s_map["concentration difficulty"]["status"] == "PRESENT"
    assert s_map["gradual onset"]["status"] == "PRESENT"
    assert s_map["previous similar headache"]["status"] == "PRESENT"

    # Explicit negatives
    assert s_map["fever"]["status"] == "ABSENT"
    assert s_map["weakness"]["status"] == "ABSENT"
    assert s_map["numbness"]["status"] == "ABSENT"
    assert s_map["speech difficulty"]["status"] == "ABSENT"
    assert s_map["fainting"]["status"] == "ABSENT"
    assert s_map["seizure"]["status"] == "ABSENT"

    # Zero hallucination check
    assert "shortness of breath" not in s_map or s_map["shortness of breath"]["status"] != "PRESENT"


def test_respiratory_regression_suite():
    res_resp = extract_symptoms(RESPIRATORY_TEXT)
    r_map = {s["symptom"]: s for s in res_resp["symptoms"]}

    assert r_map["fever"]["status"] == "PRESENT"
    assert "3 days" in r_map["fever"]["duration"]
    assert r_map["temperature"]["status"] == "PRESENT"
    assert "101" in r_map["temperature"]["context"]
    assert r_map["cough"]["status"] == "PRESENT"
    assert r_map["cough"]["type"] == "dry"
    assert r_map["sore throat"]["status"] == "PRESENT"
    assert r_map["headache"]["status"] == "PRESENT"
    assert r_map["headache"]["severity"] == "UNKNOWN"
    assert r_map["tiredness"]["status"] == "PRESENT"
    assert r_map["tiredness"]["severity"] == "UNKNOWN"
    assert r_map["chest discomfort"]["status"] == "PRESENT"
    assert r_map["chest discomfort"]["severity"].lower() == "mild"
    assert r_map["chest discomfort"]["trigger"] == "coughing"
    assert r_map["sleep difficulty"]["status"] == "PRESENT"
    assert r_map["sleep difficulty"]["severity"].lower() == "mild"

    # Explicit negatives
    assert r_map["severe breathing difficulty"]["status"] == "ABSENT"
    assert r_map["chest pain at rest"]["status"] == "ABSENT"
    assert r_map["vomiting"]["status"] == "ABSENT"
    assert r_map["fainting"]["status"] == "ABSENT"
