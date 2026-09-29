"""Symptom NLP engine for MediSense AI.

Extracts structured clinical symptoms from natural language text:
- State: PRESENT, ABSENT (via negation detection), UNKNOWN
- Duration (e.g. "3 days", "2 weeks")
- Severity (e.g. "mild", "moderate", "severe")
- Body area (e.g. "chest", "head", "abdomen", "throat")
- Context (e.g. "at night", "after eating")

Never guesses or invents symptoms not indicated in the text.
"""

from __future__ import annotations

import re
from typing import Any, Optional

NLP_VERSION = "symptom-nlp-v1.0"

# Common symptom lexicon mapping to canonical name and anatomical/system domain
SYMPTOM_LEXICON: dict[str, dict[str, str]] = {
    "cough": {"canonical": "cough", "domain": "respiratory", "body_area": "chest/throat"},
    "dry cough": {"canonical": "dry cough", "domain": "respiratory", "body_area": "chest/throat"},
    "wet cough": {"canonical": "productive cough", "domain": "respiratory", "body_area": "chest/throat"},
    "fever": {"canonical": "fever", "domain": "systemic", "body_area": "whole body"},
    "high fever": {"canonical": "high fever", "domain": "systemic", "body_area": "whole body"},
    "chills": {"canonical": "chills", "domain": "systemic", "body_area": "whole body"},
    "chest pain": {"canonical": "chest pain", "domain": "cardiovascular", "body_area": "chest"},
    "shortness of breath": {"canonical": "shortness of breath", "domain": "respiratory", "body_area": "chest/lungs"},
    "breathlessness": {"canonical": "shortness of breath", "domain": "respiratory", "body_area": "chest/lungs"},
    "difficulty breathing": {"canonical": "dyspnea", "domain": "respiratory", "body_area": "chest/lungs"},
    "wheezing": {"canonical": "wheezing", "domain": "respiratory", "body_area": "chest/lungs"},
    "headache": {"canonical": "headache", "domain": "neurological", "body_area": "head"},
    "migraine": {"canonical": "migraine", "domain": "neurological", "body_area": "head"},
    "dizziness": {"canonical": "dizziness", "domain": "neurological", "body_area": "head"},
    "fatigue": {"canonical": "fatigue", "domain": "systemic", "body_area": "whole body"},
    "tiredness": {"canonical": "fatigue", "domain": "systemic", "body_area": "whole body"},
    "sore throat": {"canonical": "sore throat", "domain": "ent", "body_area": "throat"},
    "throat pain": {"canonical": "sore throat", "domain": "ent", "body_area": "throat"},
    "runny nose": {"canonical": "rhinorrhea", "domain": "ent", "body_area": "nose"},
    "nasal congestion": {"canonical": "nasal congestion", "domain": "ent", "body_area": "nose"},
    "stomach pain": {"canonical": "abdominal pain", "domain": "gi", "body_area": "abdomen"},
    "abdominal pain": {"canonical": "abdominal pain", "domain": "gi", "body_area": "abdomen"},
    "nausea": {"canonical": "nausea", "domain": "gi", "body_area": "abdomen"},
    "vomiting": {"canonical": "vomiting", "domain": "gi", "body_area": "abdomen"},
    "diarrhea": {"canonical": "diarrhea", "domain": "gi", "body_area": "digestive system"},
    "constipation": {"canonical": "constipation", "domain": "gi", "body_area": "digestive system"},
    "back pain": {"canonical": "back pain", "domain": "bones", "body_area": "back/spine"},
    "joint pain": {"canonical": "arthralgia", "domain": "joints", "body_area": "joints"},
    "muscle ache": {"canonical": "myalgia", "domain": "musculoskeletal", "body_area": "muscles"},
    "body ache": {"canonical": "body ache", "domain": "systemic", "body_area": "whole body"},
    "rash": {"canonical": "skin rash", "domain": "skin", "body_area": "skin"},
    "itching": {"canonical": "pruritus", "domain": "skin", "body_area": "skin"},
    "loss of appetite": {"canonical": "anorexia", "domain": "gi", "body_area": "systemic"},
    "loss of taste": {"canonical": "ageusia", "domain": "ent", "body_area": "mouth/tongue"},
    "loss of smell": {"canonical": "anosmia", "domain": "ent", "body_area": "nose"},
    "palpitations": {"canonical": "palpitations", "domain": "cardiovascular", "body_area": "heart/chest"},
    "ear pain": {"canonical": "otalgia", "domain": "ent", "body_area": "ear"},
    "eye redness": {"canonical": "conjunctival injection", "domain": "eye", "body_area": "eyes"},
    "toothache": {"canonical": "dental pain", "domain": "dental", "body_area": "teeth/mouth"},
}

NEGATION_PATTERNS = [
    r"\b(?:no|not|don't|dont|doesn't|doesnt|didn't|didnt|never|without|denies|denied|free of|negative for)\b",
]

SEVERITY_PATTERNS = {
    "severe": r"\b(severe|intense|acute|excruciating|unbearable|high|worst)\b",
    "moderate": r"\b(moderate|considerable|medium)\b",
    "mild": r"\b(mild|slight|minimal|low|minor)\b",
}

DURATION_REGEX = re.compile(
    r"\b(?:for|since|last|past)?\s*(\d+|a|an|few|several|couple of)\s*(days?|weeks?|months?|hours?|years?)\b",
    re.IGNORECASE,
)


def extract_symptoms(raw_text: str) -> dict[str, Any]:
    """Extract structured symptoms from raw user text using rule-based NLP."""
    if not raw_text or not raw_text.strip():
        return {
            "status": "empty",
            "symptoms": [],
            "message": "No symptom text provided.",
            "source": "user_input",
            "nlp_version": NLP_VERSION,
        }

    cleaned = raw_text.strip()
    sentences = re.split(r"[.;\n]+", cleaned)
    extracted: list[dict[str, Any]] = []
    seen_canonicals: set[str] = set()

    # Global text duration detection if applicable
    global_duration_match = DURATION_REGEX.search(cleaned)
    global_duration = global_duration_match.group(0).strip() if global_duration_match else None

    # Global severity detection
    global_severity = None
    for sev_name, pattern in SEVERITY_PATTERNS.items():
        if re.search(pattern, cleaned, re.IGNORECASE):
            global_severity = sev_name
            break

    # Sort lexicon keys by length descending to match multi-word symptoms first
    sorted_keywords = sorted(SYMPTOM_LEXICON.keys(), key=len, reverse=True)

    for sentence in sentences:
        clause = sentence.strip()
        if not clause:
            continue

        # Check clause duration
        clause_duration_match = DURATION_REGEX.search(clause)
        clause_duration = clause_duration_match.group(0).strip() if clause_duration_match else global_duration

        # Check clause severity
        clause_severity = None
        for sev_name, pattern in SEVERITY_PATTERNS.items():
            if re.search(pattern, clause, re.IGNORECASE):
                clause_severity = sev_name
                break
        if not clause_severity:
            clause_severity = global_severity

        # Check for symptoms in this clause
        for kw in sorted_keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            match = re.search(pattern, clause, re.IGNORECASE)
            if not match:
                continue

            meta = SYMPTOM_LEXICON[kw]
            canonical = meta["canonical"]
            if canonical in seen_canonicals:
                continue

            # Check negation in the window before the symptom within the clause
            preceding_text = clause[: match.start()]
            is_negated = False
            for neg_pat in NEGATION_PATTERNS:
                if re.search(neg_pat, preceding_text, re.IGNORECASE):
                    is_negated = True
                    break

            state = "ABSENT" if is_negated else "PRESENT"
            seen_canonicals.add(canonical)

            extracted.append(
                {
                    "symptom": canonical,
                    "state": state,
                    "duration": clause_duration if state == "PRESENT" else None,
                    "severity": clause_severity if state == "PRESENT" else None,
                    "body_area": meta["body_area"],
                    "domain": meta["domain"],
                    "context": clause if len(clause) < 120 else clause[:117] + "...",
                    "source": "user_input",
                }
            )

    # If no symptoms from lexicon were detected, capture user input as general finding
    if not extracted and cleaned:
        extracted.append(
            {
                "symptom": cleaned[:80],
                "state": "UNKNOWN",
                "duration": global_duration,
                "severity": global_severity,
                "body_area": "unspecified",
                "domain": "general",
                "context": cleaned[:120],
                "source": "user_input",
            }
        )

    return {
        "status": "extracted",
        "symptoms": extracted,
        "raw_text": cleaned,
        "message": f"Successfully structured {len(extracted)} symptom finding(s).",
        "source": "user_input",
        "nlp_version": NLP_VERSION,
    }
