"""Controlled clinical follow-up question engine for MediSense AI.

Identifies missing critical information and generates targeted questions.
Never implements a simplistic 'one symptom = one disease' mapping.
Preserves user responses as PRESENT, ABSENT, or UNKNOWN evidence states.
"""

from __future__ import annotations

from typing import Any

CRITICAL_FOLLOWUP_DEFINITIONS = [
    {
        "id": "q_chest_pain",
        "target": "chest pain",
        "question": "Do you currently experience chest pain, tightness, or pressure?",
        "options": ["No", "Yes", "Not sure"],
        "category": "red_flag",
    },
    {
        "id": "q_shortness_of_breath",
        "target": "shortness of breath",
        "question": "Do you have difficulty breathing or shortness of breath at rest or with mild activity?",
        "options": ["No", "Yes", "Not sure"],
        "category": "red_flag",
    },
    {
        "id": "q_fever",
        "target": "fever",
        "question": "Do you currently have a fever, chills, or high body temperature?",
        "options": ["No", "Yes", "Not sure"],
        "category": "systemic",
    },
    {
        "id": "q_duration",
        "target": "duration",
        "question": "How long have you had these symptoms (e.g. 2 days, 1 week, over a month)?",
        "options": ["Under 3 days", "3 to 7 days", "1 to 2 weeks", "More than 2 weeks"],
        "category": "timeframe",
    },
]


def suggest_followups(
    symptoms: list[dict[str, Any]] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate controlled follow-up questions for unconfirmed critical factors."""
    symptoms = symptoms or []
    if evidence and "symptoms" in evidence:
        symptoms_present = evidence["symptoms"].get("present", [])
        symptoms_absent = evidence["symptoms"].get("absent", [])
    else:
        symptoms_present = [s for s in symptoms if s.get("state") == "PRESENT"]
        symptoms_absent = [s for s in symptoms if s.get("state") == "ABSENT"]

    known_names = {
        s.get("symptom", "").lower() for s in (symptoms_present + symptoms_absent)
    }

    questions: list[dict[str, Any]] = []

    # Check for critical red-flag symptoms not yet recorded as either PRESENT or ABSENT
    for item in CRITICAL_FOLLOWUP_DEFINITIONS:
        target = item["target"]
        if target == "duration":
            # Only ask duration if symptoms are present but duration is missing
            has_duration = any(s.get("duration") for s in symptoms_present)
            if symptoms_present and not has_duration:
                questions.append(item)
        else:
            if target not in known_names:
                questions.append(item)

    # Limit to maximum 3 controlled questions per step to avoid overwhelming user
    selected_questions = questions[:3]

    return {
        "needed": len(selected_questions) > 0,
        "questions": selected_questions,
        "message": (
            "Answering these quick safety questions will help clarify important missing information."
            if selected_questions
            else "No additional clarifying questions required."
        ),
    }


def apply_followup_answer(
    symptoms: list[dict[str, Any]],
    question_id: str,
    answer: str,
) -> list[dict[str, Any]]:
    """Update symptoms evidence list based on user's follow-up answer."""
    updated = list(symptoms)
    q_map = {q["id"]: q for q in CRITICAL_FOLLOWUP_DEFINITIONS}
    target_def = q_map.get(question_id)

    if not target_def:
        return updated

    target_name = target_def["target"]

    if target_name == "duration":
        for s in updated:
            if s.get("state") == "PRESENT" and not s.get("duration"):
                s["duration"] = answer
        return updated

    # Map Yes/No/Not sure to state
    ans_norm = answer.strip().lower()
    if ans_norm in {"yes", "true", "present"}:
        state = "PRESENT"
    elif ans_norm in {"no", "false", "absent"}:
        state = "ABSENT"
    else:
        state = "UNKNOWN"

    # Check if target symptom already in list
    for s in updated:
        if s.get("symptom", "").lower() == target_name:
            s["state"] = state
            return updated

    # Otherwise append new structured finding
    updated.append(
        {
            "symptom": target_name,
            "state": state,
            "duration": None,
            "severity": None,
            "body_area": "chest" if "chest" in target_name or "breath" in target_name else "whole body",
            "source": "followup_question",
        }
    )
    return updated
