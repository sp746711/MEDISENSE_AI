"""Controlled clinical dynamic follow-up question engine for MediSense AI.

Dynamically evaluates clinical knowledge state (PRESENT, ABSENT, UNKNOWN, NOT_RELEVANT),
identifies genuinely missing relevant information, recalculates the next question
at each turn based on updated evidence, and terminates when sufficient.
"""

from __future__ import annotations

import re
from typing import Any

CRITICAL_FOLLOWUP_DEFINITIONS: list[dict[str, Any]] = [
    # Abdominal / Gastrointestinal domain
    {
        "id": "q_abdominal_location",
        "target": "abdominal location",
        "domain": "gi",
        "question": "Where is the abdominal pain located?",
        "options": ["Lower right side", "Upper abdomen", "Lower left side", "All over / generalized", "Not sure"],
        "category": "localization",
    },
    {
        "id": "q_abdominal_onset",
        "target": "sudden onset",
        "domain": "gi",
        "question": "Did the pain start suddenly or gradually?",
        "options": ["Suddenly", "Gradually", "Not sure"],
        "category": "onset",
    },
    {
        "id": "q_vomiting",
        "target": "vomiting",
        "domain": "neurological/gi",
        "question": "Have you vomited?",
        "options": ["No", "Yes", "Not sure"],
        "category": "symptom",
    },
    {
        "id": "q_diarrhea",
        "target": "diarrhea",
        "domain": "gi",
        "question": "Do you have diarrhea or changes in bowel movements?",
        "options": ["No", "Yes", "Not sure"],
        "category": "symptom",
    },
    {
        "id": "q_gi_bleeding",
        "target": "blood in stool",
        "domain": "gi",
        "question": "Have you noticed any blood in your stool or black, tarry stools?",
        "options": ["No", "Yes", "Not sure"],
        "category": "red_flag",
    },

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

    # General timeframe & severity
    {
        "id": "q_severity",
        "target": "severity",
        "domain": "general",
        "question": "How severe is the pain?",
        "options": ["Mild", "Moderate", "Severe"],
        "category": "severity",
    },
    {
        "id": "q_duration",
        "target": "duration",
        "domain": "general",
        "question": "Approximately how long have you had these symptoms?",
        "options": ["Under 3 days", "3 to 7 days", "1 to 2 weeks", "More than 2 weeks"],
        "category": "timeframe",
    },
]


def is_target_addressed(
    target: str,
    symptoms: list[dict[str, Any]] | None = None,
    raw_text: str | None = None,
) -> bool:
    """Check if a clinical target is already addressed (known PRESENT or known ABSENT)."""
    symptoms = symptoms or []
    known_names = {
        s.get("symptom", "").lower()
        for s in symptoms
        if s.get("state") in {"PRESENT", "ABSENT"} or s.get("status") in {"PRESENT", "ABSENT"}
    }

    # Combined context and raw_text
    contexts = [s.get("context", "") for s in symptoms if s.get("context")]
    if raw_text:
        contexts.append(raw_text)
    combined_ctx = " ".join(contexts).lower()

    if target in {"vomiting", "vomiting (red flag)"}:
        # Nausea does NOT answer vomiting; explicit vomiting information required
        if any("vomit" in n for n in known_names):
            return True
        return bool(re.search(r"\b(?:not vomited|no vomiting|haven't vomited|have not vomited|without vomiting|vomited)\b", combined_ctx))

    if target == "duration":
        if any(s.get("duration") for s in symptoms if s.get("state") != "ABSENT"):
            return True
        return bool(re.search(r"\b(?:since yesterday|for \d+|days ago|yesterday evening|weeks ago|hours ago|yesterday)\b", combined_ctx))

    if target == "chest pain":
        # Chest discomfort does NOT answer chest pain; explicit chest pain confirmation or denial required
        if any(n in {"chest pain", "chest pain at rest", "resting chest pain"} or ("chest pain" in n) for n in known_names):
            return True
        return bool(re.search(r"\b(?:no chest pain|do not have chest pain|without chest pain|chest pain)\b", combined_ctx))

    if target == "fainting":
        # Seizure or dizziness does NOT answer fainting; explicit fainting/syncope/loss of consciousness required
        if any(n in {"fainting", "fainted", "syncope", "loss of consciousness"} or ("faint" in n) or ("syncope" in n) for n in known_names):
            return True
        return bool(re.search(r"\b(?:not fainted|have not fainted|no fainting|syncope|loss of consciousness|fainted)\b", combined_ctx))

    if target == "difficulty breathing":
        if any(n in {"difficulty breathing", "shortness of breath", "severe breathing difficulty", "dyspnea"} or ("breath" in n) or ("dyspnea" in n) for n in known_names):
            return True
        return bool(re.search(r"\b(?:no severe breathing difficulty|do not have severe breathing difficulty|no breathing difficulty|shortness of breath|difficulty breathing)\b", combined_ctx))

    if target == "fever":
        if any("fever" in n or "temperature" in n for n in known_names):
            return True
        return bool(re.search(r"\b(?:mild fever|high fever|fever|no fever|do not have a fever)\b", combined_ctx))

    if target == "numbness":
        # Weakness / fatigue does NOT answer numbness
        if any("numb" in n or "tingling" in n for n in known_names):
            return True
        return bool(re.search(r"\b(?:no numbness|do not have numbness|without numbness|numbness|tingling)\b", combined_ctx))

    if target == "weakness":
        # Fatigue / tiredness does NOT answer weakness
        if any("weakness" in n or "loss of strength" in n for n in known_names):
            return True
        return bool(re.search(r"\b(?:no weakness|do not have weakness|weakness in)\b", combined_ctx))

    if target == "sudden onset":
        if any(s.get("onset") in {"GRADUAL", "SUDDEN"} for s in symptoms):
            return True
        if any("onset" in n for n in known_names):
            return True
        return bool(re.search(r"\b(?:started suddenly|started abruptly|started gradually|sudden peak|gradual onset|sudden onset)\b", combined_ctx))

    if target == "severity":
        if any(str(s.get("severity", "")).upper() in {"MILD", "MODERATE", "SEVERE"} for s in symptoms if s.get("severity")):
            return True
        return bool(re.search(r"\b(?:severe|mild|moderate|worst headache)\b", combined_ctx))

    if target == "abdominal location":
        if any(s.get("location") and str(s.get("location")).upper() != "UNKNOWN" for s in symptoms):
            return True
        if any(s.get("laterality") in {"RIGHT", "LEFT"} for s in symptoms if "abdomen" in str(s.get("body_area", "")).lower() or s.get("domain") == "gi"):
            return True
        if "abdominal location" in known_names:
            return True
        return bool(re.search(r"\b(?:lower right|lower-right|right lower|rlq|upper abdomen|lower left|lower-left|epigastric|all over)\b", combined_ctx))

    if target == "diarrhea":
        if any("diarrhea" in n for n in known_names):
            return True
        return bool(re.search(r"\b(?:no diarrhea|do not have diarrhea|diarrhea)\b", combined_ctx))

    if target == "blood in stool":
        if any("blood in stool" in n or "rectal bleeding" in n for n in known_names):
            return True
        return bool(re.search(r"\b(?:no blood in my stool|do not have blood in my stool|no blood in stool|blood in stool|black stool|tarry stool)\b", combined_ctx))

    if target == "reduced movement" and any("movement" in n or "motion" in n for n in known_names):
        return True
    if target == "fall/trauma" and any("fall" in n or "trauma" in n or "injury" in n for n in known_names):
        return True
    if target == "severe swelling/deformity" and any("swelling" in n or "deform" in n for n in known_names):
        return True
    if target == "speech difficulty" and any("speech" in n for n in known_names):
        return True

    return target in known_names


def suggest_followups(
    symptoms: list[dict[str, Any]] | dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    raw_text: str | None = None,
    max_questions: int = 1,
) -> dict[str, Any]:
    """Generate dynamic clinical follow-up questions from the current state."""
    # Normalize inputs
    if isinstance(symptoms, dict):
        raw_text = raw_text or symptoms.get("raw_text")
        symptoms_list = symptoms.get("symptoms", [])
    elif symptoms is not None:
        symptoms_list = list(symptoms)
    else:
        symptoms_list = []

    if evidence and "symptoms" in evidence:
        symptoms_present = evidence["symptoms"].get("present", [])
        symptoms_absent = evidence["symptoms"].get("absent", [])
        raw_text = raw_text or evidence.get("raw_text")
    else:
        symptoms_present = [s for s in symptoms_list if s.get("state") == "PRESENT"]
        symptoms_absent = [s for s in symptoms_list if s.get("state") == "ABSENT"]

    all_symptoms = symptoms_present + symptoms_absent

    # Track answered follow-up IDs
    answered_ids: set[str] = set()
    for s in all_symptoms:
        if s.get("question_id"):
            answered_ids.add(s["question_id"])
        for qid in s.get("answered_followups", []):
            answered_ids.add(qid)

    # Active clinical domains
    present_domains = {s.get("domain", "").lower() for s in symptoms_present}
    present_areas = {s.get("body_area", "").lower() for s in symptoms_present}
    present_names = {s.get("symptom", "").lower() for s in symptoms_present}

    all_ctx = " ".join([s.get("context", "") for s in all_symptoms if s.get("context")] + ([raw_text] if raw_text else [])).lower()

    has_abdominal_pain = (
        any(k in present_names for k in {"abdominal pain", "stomach pain", "belly pain"})
        or any(term in all_ctx for term in ["abdominal pain", "pain in my abdomen", "stomach pain", "belly pain", "lower right side of my abdomen", "lower-right abdominal"])
    )

    is_neuro = (
        "neurological" in present_domains
        or any(area in {"head", "brain"} for area in present_areas)
        or any(k in present_names for k in {"headache", "migraine", "dizziness", "photophobia", "phonophobia"})
    )

    is_abdominal = has_abdominal_pain or (
        not is_neuro and (
            "gi" in present_domains
            or "gastrointestinal" in present_domains
            or "abdominal" in present_domains
            or any(area in {"abdomen", "digestive system", "stomach", "bowel"} for area in present_areas)
            or any(k in present_names for k in {"vomiting", "diarrhea", "cramps"})
        )
    )

    is_msk = (
        "musculoskeletal" in present_domains
        or "injury/trauma" in present_domains
        or any(area in {"wrist", "hand", "joints", "muscles", "back/spine", "knee", "ankle", "shoulder", "elbow"} for area in present_areas)
        or any(k in present_names for k in {"wrist pain", "wrist injury", "fall/trauma", "swelling", "tenderness", "back pain", "arthralgia"})
    )

    is_respiratory = (
        "respiratory" in present_domains
        or any(area in {"chest", "lungs", "throat"} for area in present_areas)
        or any(k in present_names for k in {"cough", "dry cough", "sore throat", "shortness of breath", "wheezing", "rhinorrhea"})
    )

    # Has duration already been established?
    has_duration = is_target_addressed("duration", all_symptoms, raw_text=raw_text)

    # Stop Condition / Sufficiency Evaluation
    # 1. Abdominal presentation sufficiency:
    if is_abdominal:
        has_loc = is_target_addressed("abdominal location", all_symptoms, raw_text=raw_text)
        has_ons = is_target_addressed("sudden onset", all_symptoms, raw_text=raw_text)
        has_sev = is_target_addressed("severity", all_symptoms, raw_text=raw_text)
        has_vom = is_target_addressed("vomiting", all_symptoms, raw_text=raw_text)

        # Dynamic progression chain completion: location + onset + severity known and follow-ups answered
        if has_loc and has_ons and has_sev and (has_vom or len(answered_ids) >= 2):
            return {
                "needed": False,
                "questions": [],
                "message": "No additional clarifying questions required.",
            }

        # Complete abdominal case (Section 12): location + onset + duration + vomiting addressed
        if has_loc and has_ons and has_duration and has_vom:
            return {
                "needed": False,
                "questions": [],
                "message": "No additional clarifying questions required.",
            }

    # 2. Neurological presentation sufficiency:
    if is_neuro and not is_abdominal:
        has_ons = is_target_addressed("sudden onset", all_symptoms, raw_text=raw_text)
        has_numb = is_target_addressed("numbness", all_symptoms, raw_text=raw_text)
        has_speech = is_target_addressed("speech difficulty", all_symptoms, raw_text=raw_text)
        has_faint = is_target_addressed("fainting", all_symptoms, raw_text=raw_text)
        if has_ons and has_numb and has_speech and has_faint and has_duration:
            return {
                "needed": False,
                "questions": [],
                "message": "No additional clarifying questions required.",
            }

    # 3. Respiratory presentation sufficiency:
    if is_respiratory and not is_abdominal:
        has_sob = is_target_addressed("difficulty breathing", all_symptoms, raw_text=raw_text)
        has_cp = is_target_addressed("chest pain", all_symptoms, raw_text=raw_text)
        has_fever = is_target_addressed("fever", all_symptoms, raw_text=raw_text)
        if has_sob and has_cp and has_duration and has_fever:
            return {
                "needed": False,
                "questions": [],
                "message": "No additional clarifying questions required.",
            }

    # 4. Musculoskeletal presentation sufficiency:
    if is_msk and not is_abdominal and not is_respiratory:
        has_numb = is_target_addressed("numbness", all_symptoms, raw_text=raw_text)
        has_mov = is_target_addressed("reduced movement", all_symptoms, raw_text=raw_text)
        if has_numb and has_mov and has_duration:
            return {
                "needed": False,
                "questions": [],
                "message": "No additional clarifying questions required.",
            }

    # Candidate generation & ranking by clinical priority
    candidates: list[tuple[int, dict[str, Any]]] = []

    def add_candidate(qid: str, priority: int) -> None:
        if qid in answered_ids:
            return
        item = next((d for d in CRITICAL_FOLLOWUP_DEFINITIONS if d["id"] == qid), None)
        if item and not is_target_addressed(item["target"], all_symptoms, raw_text=raw_text):
            candidates.append((priority, item))

    if is_abdominal:
        # Abdominal presentation candidates
        if has_abdominal_pain and not is_target_addressed("abdominal location", all_symptoms, raw_text=raw_text):
            add_candidate("q_abdominal_location", 100)
        if has_abdominal_pain and not is_target_addressed("sudden onset", all_symptoms, raw_text=raw_text):
            add_candidate("q_abdominal_onset", 90)
        if has_abdominal_pain and not is_target_addressed("severity", all_symptoms, raw_text=raw_text):
            add_candidate("q_severity", 85)
        if not is_target_addressed("vomiting", all_symptoms, raw_text=raw_text):
            add_candidate("q_vomiting", 80)
        if not has_duration:
            add_candidate("q_duration", 75)
        if not is_target_addressed("fever", all_symptoms, raw_text=raw_text):
            add_candidate("q_fever", 70)
        if not is_target_addressed("diarrhea", all_symptoms, raw_text=raw_text):
            add_candidate("q_diarrhea", 65)
        if not is_target_addressed("blood in stool", all_symptoms, raw_text=raw_text):
            add_candidate("q_gi_bleeding", 60)

    elif is_neuro:
        # Headache / Neurological candidates
        if not is_target_addressed("sudden onset", all_symptoms, raw_text=raw_text):
            add_candidate("q_sudden_onset", 100)
        if not is_target_addressed("numbness", all_symptoms, raw_text=raw_text):
            add_candidate("q_neuro_weakness", 95)
        if not is_target_addressed("speech difficulty", all_symptoms, raw_text=raw_text):
            add_candidate("q_speech_difficulty", 90)
        if not is_target_addressed("fainting", all_symptoms, raw_text=raw_text):
            add_candidate("q_fainting_seizure", 85)
        if not has_duration:
            add_candidate("q_duration", 75)
        if not is_target_addressed("vomiting", all_symptoms, raw_text=raw_text):
            add_candidate("q_vomiting", 70)
        if not is_target_addressed("fever", all_symptoms, raw_text=raw_text):
            add_candidate("q_fever", 60)

    elif is_msk:
        # Musculoskeletal candidates
        if not is_target_addressed("numbness", all_symptoms, raw_text=raw_text):
            add_candidate("q_numbness_tingling", 95)
        if not is_target_addressed("reduced movement", all_symptoms, raw_text=raw_text):
            add_candidate("q_limb_movement", 90)
        if not is_target_addressed("fall/trauma", all_symptoms, raw_text=raw_text):
            add_candidate("q_injury_mechanism", 85)
        if not has_duration:
            add_candidate("q_duration", 80)
        if not is_target_addressed("severe swelling/deformity", all_symptoms, raw_text=raw_text):
            add_candidate("q_severe_deformity", 70)

    elif is_respiratory:
        # Respiratory candidates
        if not is_target_addressed("difficulty breathing", all_symptoms, raw_text=raw_text):
            add_candidate("q_breathing_difficulty", 100)
        if not is_target_addressed("chest pain", all_symptoms, raw_text=raw_text):
            add_candidate("q_chest_pain", 90)
        if not has_duration:
            add_candidate("q_duration", 80)
        if not is_target_addressed("fever", all_symptoms, raw_text=raw_text):
            add_candidate("q_fever", 70)

    else:
        # General / systemic presentation
        if not is_target_addressed("difficulty breathing", all_symptoms, raw_text=raw_text):
            add_candidate("q_breathing_difficulty", 90)
        if not is_target_addressed("chest pain", all_symptoms, raw_text=raw_text):
            add_candidate("q_chest_pain", 85)
        if not is_target_addressed("fever", all_symptoms, raw_text=raw_text):
            add_candidate("q_fever", 80)
        if not has_duration:
            add_candidate("q_duration", 70)

    # Sort by priority descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by question id
    seen = set()
    unique_candidates: list[dict[str, Any]] = []
    for _, item in candidates:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique_candidates.append(item)

    selected_questions = unique_candidates[:max_questions]

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

    ans_str = str(answer or "").strip()
    if not ans_str:
        # Unanswered remains UNKNOWN (no default answers)
        return updated

    ans_lower = ans_str.lower()
    is_unknown = any(u in ans_lower for u in ["not sure", "unsure", "unknown", "don't know", "dont know"])

    target_name = target_def["target"]

    # 1. Duration mapping
    if target_name == "duration" or question_id == "q_duration":
        if not is_unknown:
            for s in updated:
                if s.get("state") == "PRESENT" and not s.get("duration"):
                    s["duration"] = ans_str
                    s.setdefault("answered_followups", []).append(question_id)
        return updated

    # 2. Onset mapping (q_sudden_onset, q_abdominal_onset)
    if question_id in {"q_sudden_onset", "q_abdominal_onset"} or target_name == "sudden onset":
        if is_unknown:
            onset_val = "UNKNOWN"
        elif "gradual" in ans_lower or re.search(r"\bno\b", ans_lower):
            onset_val = "GRADUAL"
        elif "sudden" in ans_lower or "peak" in ans_lower or re.search(r"\byes\b", ans_lower):
            onset_val = "SUDDEN"
        else:
            onset_val = "UNKNOWN"

        matched = False
        for s in updated:
            if s.get("state") == "PRESENT":
                s_name = s.get("symptom", "").lower()
                if s_name in {"headache", "migraine", "abdominal pain", "belly pain"} or s.get("domain") in {"neurological", "gi"}:
                    s["onset"] = onset_val
                    s.setdefault("answered_followups", []).append(question_id)
                    matched = True
        if not matched:
            for s in updated:
                if s.get("state") == "PRESENT" and s.get("type") != "HISTORICAL" and s.get("symptom") not in {"gradual onset", "sudden onset"}:
                    s["onset"] = onset_val
                    s.setdefault("answered_followups", []).append(question_id)
                    break
        return updated

    # 3. Abdominal location mapping
    if question_id == "q_abdominal_location" or target_name == "abdominal location":
        if is_unknown:
            loc_val = "UNKNOWN"
            lat_val = None
        elif "lower right" in ans_lower or "rlq" in ans_lower or "right lower" in ans_lower:
            loc_val = "LOWER_RIGHT"
            lat_val = "RIGHT"
        elif "upper" in ans_lower:
            loc_val = "UPPER"
            lat_val = None
        elif "lower left" in ans_lower or "llq" in ans_lower:
            loc_val = "LOWER_LEFT"
            lat_val = "LEFT"
        elif "all over" in ans_lower or "generalized" in ans_lower:
            loc_val = "GENERALIZED"
            lat_val = None
        else:
            loc_val = ans_str.upper()
            lat_val = None

        matched = False
        for s in updated:
            s_name = s.get("symptom", "").lower()
            if "abdominal" in s_name or "belly" in s_name or s.get("domain") == "gi":
                s["location"] = loc_val
                if lat_val:
                    s["laterality"] = lat_val
                s.setdefault("answered_followups", []).append(question_id)
                matched = True
        if not matched:
            updated.append({
                "finding": "abdominal location",
                "symptom": "abdominal location",
                "location": loc_val,
                "laterality": lat_val,
                "state": "PRESENT" if not is_unknown else "UNKNOWN",
                "status": "PRESENT" if not is_unknown else "UNKNOWN",
                "body_area": "abdomen",
                "source": "followup_answer",
                "question_id": question_id,
                "answered_followups": [question_id],
            })
        return updated

    # 4. Severity mapping (q_severity)
    if question_id == "q_severity" or target_name == "severity":
        if is_unknown:
            sev_val = "UNKNOWN"
        elif "severe" in ans_lower:
            sev_val = "SEVERE"
        elif "moderate" in ans_lower:
            sev_val = "MODERATE"
        elif "mild" in ans_lower:
            sev_val = "MILD"
        else:
            sev_val = "UNKNOWN"

        matched = False
        for s in updated:
            if s.get("state") == "PRESENT":
                s["severity"] = sev_val
                s.setdefault("answered_followups", []).append(question_id)
                matched = True
        if not matched:
            updated.append({
                "finding": "severity",
                "symptom": "severity",
                "severity": sev_val,
                "state": "PRESENT" if not is_unknown else "UNKNOWN",
                "status": "PRESENT" if not is_unknown else "UNKNOWN",
                "body_area": "unspecified",
                "source": "followup_answer",
                "question_id": question_id,
                "answered_followups": [question_id],
            })
        return updated

    # 5. General / Red-flag binary symptoms mapping
    if is_unknown:
        state = "UNKNOWN"
    elif any(pos in ans_lower for pos in ["yes", "true", "present", "reduced", "cannot", "sudden", "fall or trauma"]):
        state = "PRESENT"
    elif any(neg in ans_lower for neg in ["no", "false", "absent", "normal", "no fall or trauma"]):
        state = "ABSENT"
    else:
        state = "UNKNOWN"

    for s in updated:
        if s.get("symptom", "").lower() == target_name:
            s["state"] = state
            s["status"] = state
            s.setdefault("answered_followups", []).append(question_id)
            return updated

    body_area = "whole body"
    if "chest" in target_name or "breath" in target_name:
        body_area = "chest"
    elif target_def.get("domain") == "musculoskeletal":
        body_area = "extremity/joints"
    elif target_def.get("domain") == "neurological":
        body_area = "head"
    elif target_def.get("domain") in {"gi", "neurological/gi"}:
        body_area = "abdomen"

    updated.append(
        {
            "finding": target_name,
            "symptom": target_name,
            "state": state,
            "status": state,
            "duration": None,
            "severity": None,
            "body_area": body_area,
            "source": "followup_answer",
            "question_id": question_id,
            "answered_followups": [question_id],
        }
    )
    return updated
