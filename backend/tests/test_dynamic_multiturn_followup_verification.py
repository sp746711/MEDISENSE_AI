"""Comprehensive multi-turn dynamic follow-up verification across 5 clinical presentations.

Mandatory requirements tested:
1. Knee injury: dynamic questions, limb movement, mechanism, deformity, no headache leakage.
2. Headache: sudden onset (thunderclap), weakness/numbness, speech difficulty, duration.
3. Chest pain: breathing difficulty, duration, fever, no limb deformity questions.
4. Abdominal pain: location, onset, severity, vomiting, no knee/headache questions.
5. Ankle injury: weight bearing/movement, twisting mechanism, deformity/swelling, duration.
6. Changing previous answers dynamically shifts the next question.
7. Answered questions are never repeated.
8. NOT_ASKED is never converted to ABSENT.
"""

import pytest
from app.ai.symptom_nlp import extract_symptoms
from app.ai.followup_engine import suggest_followups, apply_followup_answer, is_target_addressed


def test_knee_injury_dynamic_multiturn_flow():
    """Knee injury presentation: questions adapt dynamically across multiple turns."""
    # Turn 1: Initial presentation
    sym_res = extract_symptoms("I twisted my right knee while playing soccer yesterday and it hurts.")
    symptoms = sym_res["symptoms"]

    q1_res = suggest_followups(symptoms=symptoms, raw_text="I twisted my right knee while playing soccer yesterday and it hurts.")
    assert q1_res["needed"] is True
    assert len(q1_res["questions"]) > 0
    q1 = q1_res["questions"][0]
    
    # Must be MSK/knee relevant, NEVER headache or abdominal
    assert "knee" in q1["question"].lower() or "movement" in q1["question"].lower() or "bear weight" in q1["question"].lower()
    assert "headache" not in q1["question"].lower()
    assert "abdomen" not in q1["question"].lower()
    assert q1["id"] in {"q_limb_movement", "q_injury_mechanism", "q_severe_deformity", "q_duration"}

    # Turn 2: User answers movement question
    q1_id = q1["id"]
    updated_symptoms = apply_followup_answer(
        symptoms,
        question_id=q1_id,
        answer="I have reduced movement, cannot bend fully",
        target_symptom=q1.get("target_symptom") or q1.get("target"),
    )
    
    q2_res = suggest_followups(symptoms=updated_symptoms)
    if q2_res["needed"]:
        q2 = q2_res["questions"][0]
        # Never repeat q1!
        assert q2["id"] != q1_id
        assert "headache" not in q2["question"].lower()


def test_headache_dynamic_multiturn_flow():
    """Headache presentation: red-flag neurological questions prioritized."""
    sym_res = extract_symptoms("Severe throbbing headache on the right side.")
    symptoms = sym_res["symptoms"]

    q1_res = suggest_followups(symptoms=symptoms)
    assert q1_res["needed"] is True
    q1 = q1_res["questions"][0]

    # Sudden onset / thunderclap headache red flag is top priority
    assert q1["id"] == "q_sudden_onset"
    assert "sudden" in q1["question"].lower() or "peak" in q1["question"].lower()

    # Turn 2: Patient denies sudden onset
    updated_symptoms = apply_followup_answer(
        symptoms,
        question_id="q_sudden_onset",
        answer="No, it developed gradually over several hours",
        target_symptom="headache",
    )
    
    q2_res = suggest_followups(symptoms=updated_symptoms)
    assert q2_res["needed"] is True
    q2 = q2_res["questions"][0]
    assert q2["id"] != "q_sudden_onset"
    # Next priority: weakness/numbness or speech difficulty
    assert q2["id"] in {"q_neuro_weakness", "q_speech_difficulty", "q_duration"}


def test_chest_pain_dynamic_multiturn_flow():
    """Chest pain presentation: cardiopulmonary safety checks prioritized."""
    sym_res = extract_symptoms("I have chest pain and pressure.")
    symptoms = sym_res["symptoms"]

    q1_res = suggest_followups(symptoms=symptoms)
    assert q1_res["needed"] is True
    q1 = q1_res["questions"][0]

    # Breathing difficulty is top cardiopulmonary safety check
    assert q1["id"] == "q_breathing_difficulty"
    assert "breath" in q1["question"].lower()

    # Turn 2: User answers "No" to breathing difficulty
    updated_symptoms = apply_followup_answer(
        symptoms,
        question_id="q_breathing_difficulty",
        answer="No, breathing is fine",
        target_symptom="difficulty breathing",
    )

    q2_res = suggest_followups(symptoms=updated_symptoms)
    assert q2_res["needed"] is True
    q2 = q2_res["questions"][0]
    assert q2["id"] != "q_breathing_difficulty"
    # Never ask knee deformity for chest pain!
    assert "knee" not in q2["question"].lower()
    assert "limb" not in q2["question"].lower()


def test_abdominal_pain_dynamic_multiturn_flow():
    """Abdominal pain presentation: localization, onset, severity, GI red flags."""
    sym_res = extract_symptoms("I have abdominal cramps.")
    symptoms = sym_res["symptoms"]

    q1_res = suggest_followups(symptoms=symptoms)
    assert q1_res["needed"] is True
    q1 = q1_res["questions"][0]

    # Location question first if location is unknown
    assert q1["id"] == "q_abdominal_location"

    # Turn 2: User answers "Lower right side"
    updated_symptoms = apply_followup_answer(
        symptoms,
        question_id="q_abdominal_location",
        answer="Lower right side",
        target_symptom="abdominal location",
    )

    q2_res = suggest_followups(symptoms=updated_symptoms)
    assert q2_res["needed"] is True
    q2 = q2_res["questions"][0]
    assert q2["id"] != "q_abdominal_location"
    # Next: onset, severity, or vomiting
    assert q2["id"] in {"q_abdominal_onset", "q_severity", "q_vomiting", "q_duration"}


def test_ankle_injury_dynamic_multiturn_flow():
    """Ankle injury presentation: weight bearing, injury mechanism, deformity."""
    sym_res = extract_symptoms("I twisted my left ankle on the stairs and it is swollen.")
    symptoms = sym_res["symptoms"]

    q1_res = suggest_followups(symptoms=symptoms)
    assert q1_res["needed"] is True
    q1 = q1_res["questions"][0]

    # Must be ankle/MSK specific, NOT headache or chest pain
    assert q1["id"] in {"q_limb_movement", "q_injury_mechanism", "q_severe_deformity", "q_duration"}
    assert "chest" not in q1["question"].lower()
    assert "headache" not in q1["question"].lower()

    # Turn 2: User answers movement question
    updated_symptoms = apply_followup_answer(
        symptoms,
        question_id=q1["id"],
        answer="I cannot bear weight on my ankle",
        target_symptom="ankle pain",
    )

    q2_res = suggest_followups(symptoms=updated_symptoms)
    if q2_res["needed"]:
        q2 = q2_res["questions"][0]
        assert q2["id"] != q1["id"]


def test_changing_previous_answer_dynamically_shifts_next_question():
    """Changing an answer updates evidence and re-ranks the next question appropriately."""
    base_symptoms = extract_symptoms("I have abdominal pain.")["symptoms"]

    # Scenario A: Patient answers location is "Lower right side" -> next is onset or severity
    symptoms_a = apply_followup_answer(
        base_symptoms,
        question_id="q_abdominal_location",
        answer="Lower right side",
        target_symptom="abdominal location",
    )
    res_a = suggest_followups(symptoms=symptoms_a)
    q_id_a = res_a["questions"][0]["id"] if res_a["needed"] else None

    # Scenario B: Patient already knew onset ("sudden onset") in text -> location is asked first
    symptoms_b = extract_symptoms("I have sudden abdominal pain.")["symptoms"]
    res_b = suggest_followups(symptoms=symptoms_b)
    q_id_b = res_b["questions"][0]["id"] if res_b["needed"] else None

    assert q_id_b == "q_abdominal_location"

    # Now answer location for B -> since onset was ALREADY known, it does NOT ask onset, moves to severity/vomiting
    symptoms_b2 = apply_followup_answer(
        symptoms_b,
        question_id="q_abdominal_location",
        answer="Upper abdomen",
        target_symptom="abdominal location",
    )
    res_b2 = suggest_followups(symptoms=symptoms_b2)
    if res_b2["needed"]:
        assert res_b2["questions"][0]["id"] != "q_abdominal_onset"
        assert res_b2["questions"][0]["id"] in {"q_severity", "q_vomiting", "q_duration"}


def test_not_asked_never_interpreted_as_absent():
    """Unasked questions remain UNKNOWN / NOT_REPORTED and are never treated as ABSENT."""
    sym_res = extract_symptoms("I have right knee pain.")
    symptoms = sym_res["symptoms"]

    by_name = {s["symptom"]: s for s in symptoms}
    # Fever, shortness of breath, headache were NOT asked and NOT mentioned
    assert "fever" not in by_name or by_name["fever"]["state"] != "ABSENT"
    assert "chest pain" not in by_name or by_name["chest pain"]["state"] != "ABSENT"
    assert "difficulty breathing" not in by_name or by_name["difficulty breathing"]["state"] != "ABSENT"
