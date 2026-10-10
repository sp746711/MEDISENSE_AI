"""Controlled clinical follow-up question engine for MediSense AI.

Identifies missing critical information and generates targeted questions.
Never implements a simplistic 'one symptom = one disease' mapping.
Preserves user responses as PRESENT, ABSENT, or UNKNOWN evidence states.
"""

from __future__ import annotations

import re
from typing import Any

CRITICAL_FOLLOWUP_DEFINITIONS = [
    # Respiratory / Cardiovascular domain
    {
        "id": "q_breathing_difficulty",
        "target": "difficulty breathing",
        "domain": "respiratory",
        "question": "Do you have shortness of breath or difficulty breathing at rest or with mild activity?",
        "options": ["No", "Yes", "Not sure"],
        "category": "red_flag",
    },
    {
        "id": "q_chest_pain",
        "target": "chest pain",
        "domain": "respiratory/cardiovascular",
        "question": "Do you currently experience chest pain, tightness, or pressure?",
        "options": ["No", "Yes", "Not sure"],
        "category": "red_flag",
    },
    {
        "id": "q_fever",
        "target": "fever",
        "domain": "respiratory/systemic",
        "question": "Do you currently have a fever, chills, or high body temperature?",
        "options": ["No", "Yes", "Not sure"],
        "category": "systemic",
    },

    # Neurological / Headache domain
    {
        "id": "q_sudden_onset",
        "target": "sudden onset",
        "domain": "neurological",
        "question": "Did your headache or symptoms start abruptly and reach peak severity within seconds to minutes?",
        "options": ["No, gradual onset", "Yes, sudden peak", "Not sure"],
        "category": "red_flag",
    },
    {
        "id": "q_neuro_weakness",
        "target": "numbness",
        "domain": "neurological",
        "question": "Do you have any weakness, numbness, or loss of sensation in your face, arms, or legs?",
        "options": ["No", "Yes", "Not sure"],
        "category": "red_flag",
    },
    {
        "id": "q_speech_difficulty",
        "target": "speech difficulty",
        "domain": "neurological",
        "question": "Are you experiencing difficulty speaking, slurred speech, or confusion?",
        "options": ["No", "Yes", "Not sure"],
        "category": "red_flag",
    },
    {
        "id": "q_fainting_seizure",
        "target": "fainting",
        "domain": "neurological",
        "question": "Have you experienced fainting, loss of consciousness, or seizures?",
        "options": ["No", "Yes", "Not sure"],
        "category": "red_flag",
    },
    {
        "id": "q_vomiting",
        "target": "vomiting",
        "domain": "neurological/gi",
        "question": "Have you experienced nausea or vomiting?",
        "options": ["No", "Yes", "Not sure"],
        "category": "symptom",
    },

    # Musculoskeletal / Injury domain
    {
        "id": "q_numbness_tingling",
        "target": "numbness",
        "domain": "musculoskeletal",
        "question": "Do you experience numbness, tingling, or 'pins and needles' in your fingers or limb?",
        "options": ["No", "Yes", "Not sure"],
        "category": "neurovascular",
    },
    {
        "id": "q_limb_movement",
        "target": "reduced movement",
        "domain": "musculoskeletal",
        "question": "Are you able to move your fingers, hand, or limb normally, or is movement reduced?",
        "options": ["Normal movement", "Reduced movement", "Cannot move limb"],
        "category": "functional",
    },
    {
        "id": "q_injury_mechanism",
        "target": "fall/trauma",
        "domain": "musculoskeletal",
        "question": "Did this injury occur after a fall, direct trauma, or sudden impact?",
        "options": ["Yes, fall or trauma", "No fall or trauma", "Not sure"],
        "category": "mechanism",
    },
    {
        "id": "q_severe_deformity",
        "target": "severe swelling/deformity",
        "domain": "musculoskeletal",
        "question": "Is there severe visible deformity or severe swelling over the affected joint?",
        "options": ["No", "Yes", "Not sure"],
        "category": "red_flag",
    },

    # General timeframe
    {
        "id": "q_duration",
        "target": "duration",
        "domain": "general",
        "question": "Approximately how long have you had these symptoms?",
        "options": ["Under 3 days", "3 to 7 days", "1 to 2 weeks", "More than 2 weeks"],
        "category": "timeframe",
    },
]


def is_target_addressed(target: str, symptoms: list[dict[str, Any]] | None = None) -> bool:
    """Check if a clinical target is already addressed by known symptoms."""
    symptoms = symptoms or []
    known_names = {s.get("symptom", "").lower() for s in symptoms}
    if target in {"vomiting", "vomiting (red flag)"}:
        # Nausea does NOT answer vomiting; explicit vomiting information required
        return any("vomit" in n for n in known_names)
    if target == "duration":
        return any(s.get("duration") for s in symptoms)
    if target in known_names:
        return True
    if target == "severe swelling/deformity" and any("swelling" in n or "deform" in n for n in known_names):
        return True
    if target == "chest pain" and any("chest" in n for n in known_names):
        return True
    if target == "difficulty breathing" and any("breath" in n or "dyspnea" in n for n in known_names):
        return True
    if target == "fever" and any("fever" in n or "temperature" in n for n in known_names):
        return True
    if target == "numbness" and any("numb" in n or "tingling" in n for n in known_names):
        return True
    if target == "reduced movement" and any("movement" in n or "motion" in n for n in known_names):
        return True
    if target == "fall/trauma" and any("fall" in n or "trauma" in n or "injury" in n for n in known_names):
        return True
    if target == "sudden onset":
        return any(s.get("onset") in {"GRADUAL", "SUDDEN"} for s in symptoms) or any("onset" in n for n in known_names)
    if target == "fainting" and any("faint" in n or "seizure" in n for n in known_names):
        return True
    if target == "speech difficulty" and any("speech" in n for n in known_names):
        return True
    return False


def suggest_followups(
    symptoms: list[dict[str, Any]] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate controlled domain-aware follow-up questions for unconfirmed critical factors."""
    symptoms = symptoms or []
    if evidence and "symptoms" in evidence:
        symptoms_present = evidence["symptoms"].get("present", [])
        symptoms_absent = evidence["symptoms"].get("absent", [])
    else:
        symptoms_present = [s for s in symptoms if s.get("state") == "PRESENT"]
        symptoms_absent = [s for s in symptoms if s.get("state") == "ABSENT"]

    all_symptoms = symptoms_present + symptoms_absent

    # Detect active clinical domains from present symptoms
    present_domains = {s.get("domain", "").lower() for s in symptoms_present}
    present_areas = {s.get("body_area", "").lower() for s in symptoms_present}
    present_names = {s.get("symptom", "").lower() for s in symptoms_present}

    is_msk = (
        "musculoskeletal" in present_domains
        or "injury/trauma" in present_domains
        or any(area in {"wrist", "hand", "joints", "muscles", "back/spine", "knee", "ankle", "shoulder", "elbow"} for area in present_areas)
        or any(k in present_names for k in {"wrist pain", "wrist injury", "fall/trauma", "swelling", "tenderness", "back pain", "arthralgia"})
    )

    is_neuro = (
        "neurological" in present_domains
        or any(area in {"head", "brain"} for area in present_areas)
        or any(k in present_names for k in {"headache", "migraine", "dizziness", "photophobia", "phonophobia"})
    )

    is_respiratory = (
        "respiratory" in present_domains
        or any(area in {"chest", "lungs", "throat"} for area in present_areas)
        or any(k in present_names for k in {"cough", "sore throat", "chest discomfort", "shortness of breath", "wheezing", "rhinorrhea"})
    )

    # Has duration already been established?
    has_duration = any(s.get("duration") for s in symptoms_present)

    questions: list[dict[str, Any]] = []

    # Filter candidate definitions by relevant domains with clinical priority
    candidate_defs: list[dict[str, Any]] = []

    # Prioritize neurological questions when headache/neurological presentation is active
    if is_neuro and any(s.get("symptom") in {"headache", "migraine"} for s in symptoms_present):
        candidate_defs.extend([
            d for d in CRITICAL_FOLLOWUP_DEFINITIONS if d["domain"] in {"neurological", "neurological/gi"}
        ])
    if is_msk:
        candidate_defs.extend([
            d for d in CRITICAL_FOLLOWUP_DEFINITIONS if d["domain"] == "musculoskeletal"
        ])
    if is_neuro and not any(s.get("symptom") in {"headache", "migraine"} for s in symptoms_present):
        candidate_defs.extend([
            d for d in CRITICAL_FOLLOWUP_DEFINITIONS if d["domain"] in {"neurological", "neurological/gi"}
        ])
    if is_respiratory:
        candidate_defs.extend([
            d for d in CRITICAL_FOLLOWUP_DEFINITIONS if d["domain"] in {"respiratory", "respiratory/cardiovascular", "respiratory/systemic"}
        ])

    # If no specific domain detected, use systemic safety checks
    if not candidate_defs:
        candidate_defs = [
            d for d in CRITICAL_FOLLOWUP_DEFINITIONS
            if d["id"] in {"q_chest_pain", "q_breathing_difficulty", "q_fever"}
        ]

    # Add duration check if duration is missing
    if symptoms_present and not has_duration:
        duration_def = next((d for d in CRITICAL_FOLLOWUP_DEFINITIONS if d["id"] == "q_duration"), None)
        if duration_def and duration_def not in candidate_defs:
            candidate_defs.append(duration_def)

    # Select unaddressed questions
    seen_ids = set()
    for item in candidate_defs:
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])
        if not is_target_addressed(item["target"], all_symptoms):
            questions.append(item)

    # Limit to maximum 3 questions
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
    updated = [dict(s) for s in symptoms]
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

    # q_sudden_onset: Map to onset attribute, NOT a fake symptom
    if question_id == "q_sudden_onset" or target_name == "sudden onset":
        ans_norm = answer.strip().lower()
        if "not sure" in ans_norm or "unsure" in ans_norm:
            onset_val = "UNKNOWN"
        elif "gradual" in ans_norm or re.search(r"\bno\b", ans_norm):
            onset_val = "GRADUAL"
        elif "sudden" in ans_norm or "peak" in ans_norm or re.search(r"\byes\b", ans_norm):
            onset_val = "SUDDEN"
        else:
            onset_val = "UNKNOWN"

        matched = False
        for s in updated:
            if s.get("state") == "PRESENT":
                s_name = s.get("symptom", "").lower()
                if s_name in {"headache", "migraine"} or s.get("domain") == "neurological":
                    s["onset"] = onset_val
                    matched = True
        if not matched:
            for s in updated:
                if s.get("state") == "PRESENT" and s.get("type") != "HISTORICAL" and s.get("symptom") not in {"gradual onset", "sudden onset"}:
                    s["onset"] = onset_val
                    break
        return updated

    # Map answers to state
    ans_norm = answer.strip().lower()
    if any(pos in ans_norm for pos in ["yes", "true", "present", "reduced", "cannot", "sudden"]):
        state = "PRESENT"
    elif any(neg in ans_norm for neg in ["no", "false", "absent", "normal"]):
        state = "ABSENT"
    else:
        state = "UNKNOWN"

    # Check if target symptom already in list
    for s in updated:
        if s.get("symptom", "").lower() == target_name:
            s["state"] = state
            return updated

    # Otherwise append new structured finding
    body_area = "whole body"
    if "chest" in target_name or "breath" in target_name:
        body_area = "chest"
    elif target_def.get("domain") == "musculoskeletal":
        body_area = "extremity/joints"
    elif target_def.get("domain") == "neurological":
        body_area = "head"

    updated.append(
        {
            "symptom": target_name,
            "state": state,
            "duration": None,
            "severity": None,
            "body_area": body_area,
            "source": "followup_question",
        }
    )
    return updated
