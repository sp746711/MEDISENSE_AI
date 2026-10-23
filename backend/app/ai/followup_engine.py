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

    # If any symptom explicitly has state UNKNOWN for this target, it is NOT addressed
    for s in symptoms:
        s_name = (s.get("symptom") or s.get("finding") or "").lower()
        if (target in s_name or s_name in target) and (s.get("state") == "UNKNOWN" or s.get("status") == "UNKNOWN"):
            return False

    # Check for explicit uncertainty expressions around this target in combined text
    if re.search(
        r"\b(?:not\s+sure|unsure|not\s+certain|uncertain|don't\s+know|dont\s+know|might\s+have|may\s+have|could\s+have)\s*(?:whether|if|about)?\s*(?:i\s+have\s+)?(?:a\s+)?"
        + re.escape(target)
        + r"\b",
        combined_ctx,
    ):
        return False

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
        return bool(re.search(r"\b(?:mild fever|high fever|have a fever|running a fever|no fever|do not have a fever)\b", combined_ctx))

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
        symptoms_unknown = evidence["symptoms"].get("unknown", [])
        raw_text = raw_text or evidence.get("raw_text")
    else:
        symptoms_present = [s for s in symptoms_list if s.get("state") == "PRESENT"]
        symptoms_absent = [s for s in symptoms_list if s.get("state") == "ABSENT"]
        symptoms_unknown = [s for s in symptoms_list if s.get("state") == "UNKNOWN"]

    all_symptoms = symptoms_present + symptoms_absent + symptoms_unknown

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

    def add_candidate(
        qid: str,
        priority: int,
        target_symptom: str | None = None,
        custom_question: str | None = None,
    ) -> None:
        if qid in answered_ids:
            return
        item = next((d for d in CRITICAL_FOLLOWUP_DEFINITIONS if d["id"] == qid), None)
        if item and not is_target_addressed(item["target"], all_symptoms, raw_text=raw_text):
            item_copy = dict(item)
            if target_symptom:
                item_copy["target_symptom"] = target_symptom
            if custom_question:
                item_copy["question"] = custom_question
            candidates.append((priority, item_copy))

    if is_abdominal:
        # Abdominal presentation candidates
        if has_abdominal_pain and not is_target_addressed("abdominal location", all_symptoms, raw_text=raw_text):
            add_candidate("q_abdominal_location", 100, target_symptom="abdominal pain")
        if has_abdominal_pain and not is_target_addressed("sudden onset", all_symptoms, raw_text=raw_text):
            add_candidate("q_abdominal_onset", 90, target_symptom="abdominal pain")
        if has_abdominal_pain and not is_target_addressed("severity", all_symptoms, raw_text=raw_text):
            add_candidate("q_severity", 85, target_symptom="abdominal pain", custom_question="How severe is the abdominal pain?")
        if not is_target_addressed("vomiting", all_symptoms, raw_text=raw_text):
            add_candidate("q_vomiting", 80)
        if not has_duration:
            add_candidate("q_duration", 75, target_symptom="abdominal pain", custom_question="Approximately how long have you had this abdominal pain?")
        if not is_target_addressed("fever", all_symptoms, raw_text=raw_text):
            add_candidate("q_fever", 70)
        if not is_target_addressed("diarrhea", all_symptoms, raw_text=raw_text):
            add_candidate("q_diarrhea", 65)
        if not is_target_addressed("blood in stool", all_symptoms, raw_text=raw_text):
            add_candidate("q_gi_bleeding", 60)

    elif is_neuro:
        # Headache / Neurological candidates
        neuro_primary = "headache" if "headache" in present_names or "migraine" in present_names else "headache"
        if not is_target_addressed("sudden onset", all_symptoms, raw_text=raw_text):
            add_candidate("q_sudden_onset", 100, target_symptom=neuro_primary)
        if not is_target_addressed("numbness", all_symptoms, raw_text=raw_text):
            add_candidate("q_neuro_weakness", 95)
        if not is_target_addressed("speech difficulty", all_symptoms, raw_text=raw_text):
            add_candidate("q_speech_difficulty", 90)
        if not is_target_addressed("fainting", all_symptoms, raw_text=raw_text):
            add_candidate("q_fainting_seizure", 85)
        if not has_duration:
            add_candidate("q_duration", 75, target_symptom=neuro_primary, custom_question=f"Approximately how long have you had this {neuro_primary}?")
        if not is_target_addressed("vomiting", all_symptoms, raw_text=raw_text):
            add_candidate("q_vomiting", 70)
        if not is_target_addressed("fever", all_symptoms, raw_text=raw_text):
            add_candidate("q_fever", 60)

    elif is_msk:
        # Musculoskeletal candidates
        msk_primary = "wrist pain" if "wrist pain" in present_names else "joint pain"
        if not is_target_addressed("numbness", all_symptoms, raw_text=raw_text):
            add_candidate("q_numbness_tingling", 95)
        if not is_target_addressed("reduced movement", all_symptoms, raw_text=raw_text):
            add_candidate("q_limb_movement", 90)
        if not is_target_addressed("fall/trauma", all_symptoms, raw_text=raw_text):
            add_candidate("q_injury_mechanism", 85)
        if not has_duration:
            add_candidate("q_duration", 80, target_symptom=msk_primary)
        if not is_target_addressed("severe swelling/deformity", all_symptoms, raw_text=raw_text):
            add_candidate("q_severe_deformity", 70)

    elif is_respiratory:
        # Respiratory candidates
        resp_primary = "cough" if "cough" in present_names or "dry cough" in present_names else "shortness of breath"
        if not is_target_addressed("difficulty breathing", all_symptoms, raw_text=raw_text):
            add_candidate("q_breathing_difficulty", 100)
        if not is_target_addressed("chest pain", all_symptoms, raw_text=raw_text):
            add_candidate("q_chest_pain", 90)
        if not has_duration:
            add_candidate("q_duration", 80, target_symptom=resp_primary)
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


def parse_binary_answer(
    ans_str: str,
    question_id: str,
    target_name: str,
    options: list[str] | None = None,
) -> str:
    """Parse follow-up binary/red-flag answer with strict negation precedence."""
    ans_clean = str(ans_str or "").strip()
    if not ans_clean:
        return "UNKNOWN"
    ans_lower = ans_clean.lower()

    # 1. Uncertainty check (must not be treated as YES or NO)
    if re.search(
        r"\b(?:not\s+sure|unsure|don't\s+know|dont\s+know|cannot\s+tell|can't\s+tell|not\s+certain|uncertain|unknown|hard\s+to\s+say|unclear|maybe|possibly)\b",
        ans_lower,
    ):
        return "UNKNOWN"

    # 2. Specific question patterns
    if question_id == "q_limb_movement" or target_name == "reduced movement":
        if re.search(r"\b(?:normal\s+movement|normal|can\s+move\s+(?:normally|fine|well)|no\s+problem\s+moving|no\s+reduction|without\s+restriction)\b", ans_lower):
            return "ABSENT"
        if re.search(r"\b(?:reduced\s+movement|cannot\s+move|can't\s+move|unable\s+to\s+move|difficulty\s+moving|limited\s+movement|can't\s+move\s+limb|cannot\s+move\s+limb)\b", ans_lower):
            return "PRESENT"

    if question_id == "q_injury_mechanism" or target_name == "fall/trauma":
        # Strict negation check first!
        if re.search(
            r"\b(?:no\s+fall\s+or\s+trauma|no\s+fall|no\s+trauma|no\s+injury|did\s+not\s+fall|didn't\s+fall|not\s+fallen|without\s+(?:any\s+)?(?:fall|trauma|injury)|have\s+not\s+fallen|haven't\s+fallen|no)\b",
            ans_lower,
        ):
            return "ABSENT"
        if re.search(
            r"\b(?:yes|fell|fallen|did\s+fall|i\s+fell|direct\s+trauma|after\s+a\s+fall|fall\s+or\s+trauma|impact)\b",
            ans_lower,
        ):
            return "PRESENT"

    # 3. Whole-answer negation precedence (handles "No", "No fall or trauma", "I did not fall", "I have not experienced this", etc.)
    if ans_lower in {"no", "false", "absent", "none", "denied", "denies", "nope", "negative", "not really", "not at all"}:
        return "ABSENT"
    if re.match(r"^(?:no\b|never\b|not\s+at\s+all\b|none\b|nope\b)", ans_lower):
        return "ABSENT"
    if re.search(
        r"\b(?:i\s+did\s+not|i\s+didn't|did\s+not|didn't|have\s+not|haven't|had\s+not|hadn't|do\s+not|don't|no\s+history\s+of|without\s+(?:any\s+)?|never\s+had|never\s+experienced|not\s+experienced|i\s+have\s+not\s+experienced|denies|denied)\b",
        ans_lower,
    ):
        return "ABSENT"

    # 4. Whole-answer affirmation (handles "Yes", "Yes, fall or trauma", "Yes, I fell", "I experienced the symptom", etc.)
    if ans_lower in {"yes", "true", "present", "yep", "yeah", "positive"}:
        return "PRESENT"
    if re.match(r"^(?:yes\b|yep\b|yeah\b|sure\b|definitely\b)", ans_lower):
        return "PRESENT"
    if re.search(
        r"\b(?:i\s+(?:did\s+)?experience(?:d)?|experienced\s+(?:the\s+symptom|this)|i\s+have\s+(?:this|it)|suffering\s+from|present)\b",
        ans_lower,
    ):
        return "PRESENT"

    # 5. Fallback for unhandled / ambiguous natural language: UNKNOWN
    return "UNKNOWN"


def apply_followup_answer(
    symptoms: list[dict[str, Any]],
    question_id: str,
    answer: str | dict[str, Any],
    target_symptom: str | None = None,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    """Update symptoms evidence list based on user's follow-up answer."""
    updated = [dict(s) for s in symptoms]

    # Handle rich answer dictionary
    if isinstance(answer, dict):
        target_symptom = target_symptom or answer.get("target_symptom")
        scope = scope or answer.get("scope")
        ans_str = str(answer.get("answer") or answer.get("value") or "").strip()
    else:
        ans_str = str(answer or "").strip()

    # Parse encoded target or base question ID (e.g. q_duration:headache, q_duration_headache)
    base_qid = question_id
    if ":" in question_id:
        parts = question_id.split(":", 1)
        base_qid = parts[0]
        if not target_symptom:
            target_symptom = parts[1]
    elif question_id.startswith("q_duration_") and question_id != "q_duration":
        base_qid = "q_duration"
        if not target_symptom:
            target_symptom = question_id[len("q_duration_"):]
    elif question_id.startswith("q_severity_") and question_id != "q_severity":
        base_qid = "q_severity"
        if not target_symptom:
            target_symptom = question_id[len("q_severity_"):]

    q_map = {q["id"]: q for q in CRITICAL_FOLLOWUP_DEFINITIONS}
    target_def = q_map.get(base_qid)

    if not target_def:
        return updated

    if not ans_str:
        # Unanswered remains UNKNOWN (no default answers)
        return updated

    ans_lower = ans_str.lower()
    is_unknown = bool(
        re.search(
            r"\b(?:not\s+sure|unsure|unknown|don't\s+know|dont\s+know|cannot\s+tell|can't\s+tell|hard\s+to\s+say)\b",
            ans_lower,
        )
    )

    target_name = target_def["target"]

    # 1. Duration mapping
    if target_name == "duration" or base_qid == "q_duration":
        if is_unknown:
            return updated

        # Case A: Explicit group scope ("all")
        if scope == "all":
            for s in updated:
                if s.get("state") == "PRESENT" and not s.get("duration"):
                    s["duration"] = ans_str
                    s["duration_scope"] = "all"
                    s.setdefault("answered_followups", []).append(question_id)
            return updated

        # Case B: Explicit target_symptom specified
        if target_symptom:
            t_clean = target_symptom.strip().lower()
            for s in updated:
                s_name = (s.get("symptom") or s.get("finding") or "").lower()
                if (s_name == t_clean or t_clean in s_name or s_name in t_clean) and s.get("state") == "PRESENT":
                    s["duration"] = ans_str
                    s.setdefault("answered_followups", []).append(question_id)
                    return updated

        # Case C: Unspecified target -> safely target primary chief/pain complaint without contaminating associated symptoms
        primary_candidates = []
        for s in updated:
            if s.get("state") == "PRESENT" and not s.get("duration"):
                s_name = (s.get("symptom") or s.get("finding") or "").lower()
                if any(p in s_name for p in ["pain", "ache", "migraine", "cough", "breath", "injury"]):
                    primary_candidates.append(s)

        if primary_candidates:
            primary_candidates[0]["duration"] = ans_str
            primary_candidates[0].setdefault("answered_followups", []).append(question_id)
            return updated

        present_s = [s for s in updated if s.get("state") == "PRESENT"]
        if len(present_s) == 1:
            present_s[0]["duration"] = ans_str
            present_s[0].setdefault("answered_followups", []).append(question_id)
            return updated

        return updated

    # 2. Onset mapping (q_sudden_onset, q_abdominal_onset)
    if base_qid in {"q_sudden_onset", "q_abdominal_onset"} or target_name == "sudden onset":
        if is_unknown:
            onset_val = "UNKNOWN"
        elif "gradual" in ans_lower or re.search(r"\bno\b", ans_lower):
            onset_val = "GRADUAL"
        elif "sudden" in ans_lower or "peak" in ans_lower or re.search(r"\byes\b", ans_lower):
            onset_val = "SUDDEN"
        else:
            onset_val = "UNKNOWN"

        if target_symptom:
            t_clean = target_symptom.strip().lower()
            for s in updated:
                s_name = (s.get("symptom") or s.get("finding") or "").lower()
                if (s_name == t_clean or t_clean in s_name or s_name in t_clean) and s.get("state") == "PRESENT":
                    s["onset"] = onset_val
                    s.setdefault("answered_followups", []).append(question_id)
                    return updated

        # A follow-up answer must update one finding, not every finding in the
        # same clinical domain. Prefer a present, non-historical primary finding.
        eligible = [
            s for s in updated
            if s.get("state") == "PRESENT"
            and str(s.get("type", "")).upper() != "HISTORICAL"
            and (s.get("symptom") or s.get("finding")) not in {"gradual onset", "sudden onset"}
        ]
        preferred_names = {"headache", "migraine", "abdominal pain", "belly pain", "stomach pain"}
        selected = next(
            (s for s in eligible
             if str(s.get("symptom") or s.get("finding") or "").strip().lower() in preferred_names),
            None,
        )
        if selected is None and eligible:
            selected = eligible[0]
        if selected is not None:
            selected["onset"] = onset_val
            selected.setdefault("answered_followups", [])
            if question_id not in selected["answered_followups"]:
                selected["answered_followups"].append(question_id)
        return updated

    # 3. Abdominal location mapping
    if base_qid == "q_abdominal_location" or target_name == "abdominal location":
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

        # Apply the location to one relevant abdominal complaint only. A
        # single answer must not be copied onto unrelated GI findings (e.g.
        # vomiting and diarrhea) or repeated across multiple symptom records.
        abdominal_candidates = [
            s for s in updated
            if s.get("state") == "PRESENT"
            and (
                "abdominal" in str(s.get("symptom") or s.get("finding") or "").lower()
                or "belly" in str(s.get("symptom") or s.get("finding") or "").lower()
                or "stomach pain" in str(s.get("symptom") or s.get("finding") or "").lower()
            )
            and str(s.get("type", "")).upper() != "HISTORICAL"
        ]
        selected = None
        if target_symptom:
            t_clean = target_symptom.strip().lower()
            selected = next(
                (s for s in abdominal_candidates
                 if str(s.get("symptom") or s.get("finding") or "").strip().lower() == t_clean),
                None,
            )
        if selected is None and abdominal_candidates:
            selected = abdominal_candidates[0]

        if selected is not None:
            selected["location"] = loc_val
            if lat_val:
                selected["laterality"] = lat_val
            selected.setdefault("answered_followups", [])
            if question_id not in selected["answered_followups"]:
                selected["answered_followups"].append(question_id)
        else:
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
    if base_qid == "q_severity" or target_name == "severity":
        if is_unknown:
            sev_val = "UNKNOWN"
        elif re.search(r"\bsevere\b", ans_lower):
            sev_val = "SEVERE"
        elif re.search(r"\bmoderate\b", ans_lower):
            sev_val = "MODERATE"
        elif re.search(r"\bmild\b", ans_lower):
            sev_val = "MILD"
        else:
            sev_val = "UNKNOWN"

        # Case A: Explicit target_symptom specified
        if target_symptom:
            t_clean = target_symptom.strip().lower()
            for s in updated:
                s_name = (s.get("symptom") or s.get("finding") or "").lower()
                if (s_name == t_clean or t_clean in s_name or s_name in t_clean) and s.get("state") == "PRESENT":
                    s["severity"] = sev_val
                    s.setdefault("answered_followups", []).append(question_id)
                    return updated

        # Case B: Target relevant pain complaint (never assign pain severity to nausea, vomiting, dizziness, etc.)
        pain_candidates = []
        for s in updated:
            if s.get("state") == "PRESENT":
                s_name = (s.get("symptom") or s.get("finding") or "").lower()
                if any(p in s_name for p in ["pain", "ache", "migraine", "cramp", "soreness"]):
                    pain_candidates.append(s)

        if pain_candidates:
            pain_candidates[0]["severity"] = sev_val
            pain_candidates[0].setdefault("answered_followups", []).append(question_id)
            return updated

        # No applicable symptom exists: do NOT create a misleading generic severity finding!
        return updated

    # 5. General / Red-flag binary symptoms mapping
    state = parse_binary_answer(ans_str, base_qid, target_name, target_def.get("options"))

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
