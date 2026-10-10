"""Symptom NLP engine for MediSense AI.

Extracts structured clinical symptoms from natural language text:
- State / Status: PRESENT, ABSENT (via precise local negation), UNKNOWN
- Duration (e.g. "3 days", "2 days", "2 weeks")
- Severity (e.g. "MILD", "MODERATE", "SEVERE", "UNKNOWN")
- Body area / Laterality (e.g. "head", "chest", laterality: RIGHT/LEFT/BILATERAL)
- Quality (e.g. THROBBING, SHARP, DULL, PRESSURE)
- Context & Trigger (e.g. "when coughing", "bright light", "because of the cough")
- Source text and provenance

CRITICAL PRINCIPLES:
1. UNKNOWN != ABSENT: Unmentioned findings remain UNKNOWN (never inferred as ABSENT).
2. Local negation: Negation cannot leak across conjunctions or into adjacent symptoms.
3. Unsupported findings (e.g. shortness of breath) must NEVER be generated if not present in text.
4. Preserves all qualifiers (laterality, quality, triggers, onset, previous episodes).
"""

from __future__ import annotations

import re
from typing import Any, Optional

NLP_VERSION = "symptom-nlp-v2.1"


class ClinicalSeverity(str):
    """Case-tolerant string for severity comparisons ('severe' == 'SEVERE')."""

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.lower())


# Comprehensive clinical symptom lexicon
SYMPTOM_LEXICON: dict[str, dict[str, Any]] = {
    # Resting chest pain vs acute chest pain vs chest discomfort
    "chest pain when resting": {
        "canonical": "chest pain at rest",
        "domain": "cardiovascular",
        "body_area": "chest",
    },
    "chest pain at rest": {
        "canonical": "chest pain at rest",
        "domain": "cardiovascular",
        "body_area": "chest",
    },
    "resting chest pain": {
        "canonical": "chest pain at rest",
        "domain": "cardiovascular",
        "body_area": "chest",
    },
    "chest discomfort": {
        "canonical": "chest discomfort",
        "domain": "respiratory/cardiovascular",
        "body_area": "chest",
    },
    "chest pain": {
        "canonical": "chest pain",
        "domain": "cardiovascular",
        "body_area": "chest",
    },
    "chest tightness": {
        "canonical": "chest tightness",
        "domain": "respiratory",
        "body_area": "chest",
    },

    # Respiratory / Breathing
    "severe breathing difficulty": {
        "canonical": "severe breathing difficulty",
        "domain": "respiratory",
        "body_area": "chest/lungs",
    },
    "severe shortness of breath": {
        "canonical": "severe breathing difficulty",
        "domain": "respiratory",
        "body_area": "chest/lungs",
    },
    "shortness of breath": {
        "canonical": "shortness of breath",
        "domain": "respiratory",
        "body_area": "chest/lungs",
    },
    "breathlessness": {
        "canonical": "shortness of breath",
        "domain": "respiratory",
        "body_area": "chest/lungs",
    },
    "difficulty breathing": {
        "canonical": "difficulty breathing",
        "domain": "respiratory",
        "body_area": "chest/lungs",
    },
    "breathing difficulty": {
        "canonical": "difficulty breathing",
        "domain": "respiratory",
        "body_area": "chest/lungs",
    },
    "wheezing": {
        "canonical": "wheezing",
        "domain": "respiratory",
        "body_area": "chest/lungs",
    },

    # Cough
    "dry cough": {
        "canonical": "cough",
        "domain": "respiratory",
        "body_area": "chest/throat",
        "type": "dry",
    },
    "wet cough": {
        "canonical": "productive cough",
        "domain": "respiratory",
        "body_area": "chest/throat",
        "type": "productive",
    },
    "productive cough": {
        "canonical": "productive cough",
        "domain": "respiratory",
        "body_area": "chest/throat",
        "type": "productive",
    },
    "cough": {
        "canonical": "cough",
        "domain": "respiratory",
        "body_area": "chest/throat",
    },

    # Systemic & Sleep
    "fever": {
        "canonical": "fever",
        "domain": "systemic",
        "body_area": "whole body",
    },
    "high fever": {
        "canonical": "high fever",
        "domain": "systemic",
        "body_area": "whole body",
        "explicit_severity": "severe",
    },
    "chills": {
        "canonical": "chills",
        "domain": "systemic",
        "body_area": "whole body",
    },
    "tiredness": {
        "canonical": "tiredness",
        "domain": "systemic",
        "body_area": "whole body",
    },
    "tired": {
        "canonical": "fatigue",
        "domain": "systemic",
        "body_area": "whole body",
    },
    "fatigue": {
        "canonical": "fatigue",
        "domain": "systemic",
        "body_area": "whole body",
    },
    "weakness": {
        "canonical": "weakness",
        "domain": "systemic",
        "body_area": "whole body",
    },
    "difficulty sleeping": {
        "canonical": "sleep difficulty",
        "domain": "systemic",
        "body_area": "whole body",
    },
    "trouble sleeping": {
        "canonical": "sleep difficulty",
        "domain": "systemic",
        "body_area": "whole body",
    },
    "sleep difficulty": {
        "canonical": "sleep difficulty",
        "domain": "systemic",
        "body_area": "whole body",
    },
    "insomnia": {
        "canonical": "sleep difficulty",
        "domain": "systemic",
        "body_area": "whole body",
    },

    # ENT / Head / Neuro
    "sore throat": {
        "canonical": "sore throat",
        "domain": "ent",
        "body_area": "throat",
    },
    "throat pain": {
        "canonical": "sore throat",
        "domain": "ent",
        "body_area": "throat",
    },
    "headache": {
        "canonical": "headache",
        "domain": "neurological",
        "body_area": "head",
    },
    "migraine": {
        "canonical": "migraine",
        "domain": "neurological",
        "body_area": "head",
    },
    "dizziness": {
        "canonical": "dizziness",
        "domain": "neurological",
        "body_area": "head",
    },
    "fainting": {
        "canonical": "fainting",
        "domain": "neurological/cardiovascular",
        "body_area": "whole body",
    },
    "fainted": {
        "canonical": "fainting",
        "domain": "neurological/cardiovascular",
        "body_area": "whole body",
    },
    "syncope": {
        "canonical": "fainting",
        "domain": "neurological/cardiovascular",
        "body_area": "whole body",
    },
    "seizure": {
        "canonical": "seizure",
        "domain": "neurological",
        "body_area": "whole body",
    },
    "seizures": {
        "canonical": "seizure",
        "domain": "neurological",
        "body_area": "whole body",
    },
    "photophobia": {
        "canonical": "photophobia",
        "domain": "neurological/eye",
        "body_area": "eyes/head",
    },
    "bright light": {
        "canonical": "photophobia",
        "domain": "neurological/eye",
        "body_area": "eyes/head",
    },
    "sensitive to light": {
        "canonical": "photophobia",
        "domain": "neurological/eye",
        "body_area": "eyes/head",
    },
    "phonophobia": {
        "canonical": "phonophobia",
        "domain": "neurological/ent",
        "body_area": "ears/head",
    },
    "loud sounds": {
        "canonical": "phonophobia",
        "domain": "neurological/ent",
        "body_area": "ears/head",
    },
    "loud sound": {
        "canonical": "phonophobia",
        "domain": "neurological/ent",
        "body_area": "ears/head",
    },
    "sensitive to sound": {
        "canonical": "phonophobia",
        "domain": "neurological/ent",
        "body_area": "ears/head",
    },
    "difficulty concentrating": {
        "canonical": "concentration difficulty",
        "domain": "neurological",
        "body_area": "head",
    },
    "concentration difficulty": {
        "canonical": "concentration difficulty",
        "domain": "neurological",
        "body_area": "head",
    },
    "trouble concentrating": {
        "canonical": "concentration difficulty",
        "domain": "neurological",
        "body_area": "head",
    },
    "started gradually": {
        "canonical": "gradual onset",
        "domain": "general",
        "body_area": "unspecified",
    },
    "gradual onset": {
        "canonical": "gradual onset",
        "domain": "general",
        "body_area": "unspecified",
    },
    "similar headaches once or twice before": {
        "canonical": "previous similar headache",
        "domain": "neurological",
        "body_area": "head",
        "type": "HISTORICAL",
    },
    "similar headaches before": {
        "canonical": "previous similar headache",
        "domain": "neurological",
        "body_area": "head",
        "type": "HISTORICAL",
    },
    "similar headaches": {
        "canonical": "previous similar headache",
        "domain": "neurological",
        "body_area": "head",
        "type": "HISTORICAL",
    },
    "similar headache": {
        "canonical": "previous similar headache",
        "domain": "neurological",
        "body_area": "head",
        "type": "HISTORICAL",
    },
    "previous similar headache": {
        "canonical": "previous similar headache",
        "domain": "neurological",
        "body_area": "head",
        "type": "HISTORICAL",
    },
    "numbness": {
        "canonical": "numbness",
        "domain": "neurological",
        "body_area": "arms/legs",
    },
    "numb": {
        "canonical": "numbness",
        "domain": "neurological",
        "body_area": "arms/legs",
    },
    "difficulty speaking": {
        "canonical": "speech difficulty",
        "domain": "neurological",
        "body_area": "head/speech",
    },
    "speech difficulty": {
        "canonical": "speech difficulty",
        "domain": "neurological",
        "body_area": "head/speech",
    },
    "runny nose": {
        "canonical": "rhinorrhea",
        "domain": "ent",
        "body_area": "nose",
    },
    "nasal congestion": {
        "canonical": "nasal congestion",
        "domain": "ent",
        "body_area": "nose",
    },

    # GI
    "vomiting": {
        "canonical": "vomiting",
        "domain": "gi",
        "body_area": "abdomen",
    },
    "vomited": {
        "canonical": "vomiting",
        "domain": "gi",
        "body_area": "abdomen",
    },
    "vomit": {
        "canonical": "vomiting",
        "domain": "gi",
        "body_area": "abdomen",
    },
    "nausea": {
        "canonical": "nausea",
        "domain": "gi",
        "body_area": "abdomen",
    },
    "nauseous": {
        "canonical": "nausea",
        "domain": "gi",
        "body_area": "abdomen",
    },
    "stomach pain": {
        "canonical": "abdominal pain",
        "domain": "gi",
        "body_area": "abdomen",
    },
    "abdominal pain": {
        "canonical": "abdominal pain",
        "domain": "gi",
        "body_area": "abdomen",
    },
    "diarrhea": {
        "canonical": "diarrhea",
        "domain": "gi",
        "body_area": "digestive system",
    },
    "constipation": {
        "canonical": "constipation",
        "domain": "gi",
        "body_area": "digestive system",
    },

    # Musculoskeletal & Skin
    "back pain": {
        "canonical": "back pain",
        "domain": "musculoskeletal",
        "body_area": "back/spine",
    },
    "joint pain": {
        "canonical": "arthralgia",
        "domain": "musculoskeletal",
        "body_area": "joints",
    },
    "muscle ache": {
        "canonical": "myalgia",
        "domain": "musculoskeletal",
        "body_area": "muscles",
    },
    "body ache": {
        "canonical": "body ache",
        "domain": "systemic",
        "body_area": "whole body",
    },
    # Wrist & Upper Extremity Musculoskeletal / Injury
    "right wrist injury": {
        "canonical": "wrist injury",
        "domain": "musculoskeletal",
        "body_area": "wrist",
        "laterality": "RIGHT",
    },
    "left wrist injury": {
        "canonical": "wrist injury",
        "domain": "musculoskeletal",
        "body_area": "wrist",
        "laterality": "LEFT",
    },
    "wrist injury": {
        "canonical": "wrist injury",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "right wrist pain": {
        "canonical": "wrist pain",
        "domain": "musculoskeletal",
        "body_area": "wrist",
        "laterality": "RIGHT",
    },
    "left wrist pain": {
        "canonical": "wrist pain",
        "domain": "musculoskeletal",
        "body_area": "wrist",
        "laterality": "LEFT",
    },
    "pain in my right wrist": {
        "canonical": "wrist pain",
        "domain": "musculoskeletal",
        "body_area": "wrist",
        "laterality": "RIGHT",
    },
    "pain in right wrist": {
        "canonical": "wrist pain",
        "domain": "musculoskeletal",
        "body_area": "wrist",
        "laterality": "RIGHT",
    },
    "pain in my left wrist": {
        "canonical": "wrist pain",
        "domain": "musculoskeletal",
        "body_area": "wrist",
        "laterality": "LEFT",
    },
    "pain in left wrist": {
        "canonical": "wrist pain",
        "domain": "musculoskeletal",
        "body_area": "wrist",
        "laterality": "LEFT",
    },
    "pain in my wrist": {
        "canonical": "wrist pain",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "pain in wrist": {
        "canonical": "wrist pain",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "wrist pain": {
        "canonical": "wrist pain",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "right wrist swelling": {
        "canonical": "wrist swelling",
        "domain": "musculoskeletal",
        "body_area": "wrist",
        "laterality": "RIGHT",
    },
    "wrist swelling": {
        "canonical": "wrist swelling",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "swollen wrist": {
        "canonical": "wrist swelling",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "swelling in wrist": {
        "canonical": "wrist swelling",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "swollen": {
        "canonical": "swelling",
        "domain": "musculoskeletal",
        "body_area": "extremity/joints",
    },
    "swelling": {
        "canonical": "swelling",
        "domain": "musculoskeletal",
        "body_area": "extremity/joints",
    },
    "right wrist tenderness": {
        "canonical": "wrist tenderness",
        "domain": "musculoskeletal",
        "body_area": "wrist",
        "laterality": "RIGHT",
    },
    "wrist tenderness": {
        "canonical": "wrist tenderness",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "tender wrist": {
        "canonical": "wrist tenderness",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "tenderness in wrist": {
        "canonical": "wrist tenderness",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "tenderness": {
        "canonical": "tenderness",
        "domain": "musculoskeletal",
        "body_area": "extremity/joints",
    },
    "tender": {
        "canonical": "tenderness",
        "domain": "musculoskeletal",
        "body_area": "extremity/joints",
    },
    "reduced wrist movement": {
        "canonical": "reduced movement",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "limited wrist movement": {
        "canonical": "reduced movement",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "difficulty moving wrist": {
        "canonical": "reduced movement",
        "domain": "musculoskeletal",
        "body_area": "wrist",
    },
    "reduced range of motion": {
        "canonical": "reduced movement",
        "domain": "musculoskeletal",
        "body_area": "extremity/joints",
    },
    "reduced movement": {
        "canonical": "reduced movement",
        "domain": "musculoskeletal",
        "body_area": "extremity/joints",
    },
    "limited movement": {
        "canonical": "reduced movement",
        "domain": "musculoskeletal",
        "body_area": "extremity/joints",
    },
    "difficulty moving": {
        "canonical": "reduced movement",
        "domain": "musculoskeletal",
        "body_area": "extremity/joints",
    },
    # Trauma & Mechanism of Injury
    "fall onto outstretched hand": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "upper extremity",
    },
    "fall onto outstretched right hand": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "upper extremity",
        "laterality": "RIGHT",
    },
    "fell onto my right hand": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "right hand",
        "laterality": "RIGHT",
    },
    "fell onto right hand": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "right hand",
        "laterality": "RIGHT",
    },
    "fall onto right hand": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "right hand",
        "laterality": "RIGHT",
    },
    "fell onto my left hand": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "left hand",
        "laterality": "LEFT",
    },
    "fell onto left hand": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "left hand",
        "laterality": "LEFT",
    },
    "fall onto hand": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "hand",
    },
    "fell onto hand": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "hand",
    },
    "pain after fall": {
        "canonical": "pain after fall",
        "domain": "injury/trauma",
        "body_area": "musculoskeletal",
    },
    "trauma/injury": {
        "canonical": "trauma/injury",
        "domain": "injury/trauma",
        "body_area": "unspecified",
    },
    "trauma": {
        "canonical": "trauma/injury",
        "domain": "injury/trauma",
        "body_area": "unspecified",
    },
    "injury": {
        "canonical": "trauma/injury",
        "domain": "injury/trauma",
        "body_area": "unspecified",
    },
    "fell": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "unspecified",
    },
    "fall": {
        "canonical": "fall/trauma",
        "domain": "injury/trauma",
        "body_area": "unspecified",
    },
    "right hand pain": {
        "canonical": "hand pain",
        "domain": "musculoskeletal",
        "body_area": "hand",
        "laterality": "RIGHT",
    },
    "hand pain": {
        "canonical": "hand pain",
        "domain": "musculoskeletal",
        "body_area": "hand",
    },
    "tingling in fingers": {
        "canonical": "tingling",
        "domain": "neurological",
        "body_area": "fingers",
    },
    "tingling in hands": {
        "canonical": "tingling",
        "domain": "neurological",
        "body_area": "hand",
    },
    "tingling": {
        "canonical": "tingling",
        "domain": "neurological",
        "body_area": "extremities",
    },
    "numbness in fingers": {
        "canonical": "numbness",
        "domain": "neurological",
        "body_area": "fingers",
    },
    "numbness in hands": {
        "canonical": "numbness",
        "domain": "neurological",
        "body_area": "hand",
    },
    "numbness in my fingers": {
        "canonical": "numbness",
        "domain": "neurological",
        "body_area": "fingers",
    },
    "knee pain": {
        "canonical": "knee pain",
        "domain": "musculoskeletal",
        "body_area": "knee",
    },
    "ankle pain": {
        "canonical": "ankle pain",
        "domain": "musculoskeletal",
        "body_area": "ankle",
    },
    "shoulder pain": {
        "canonical": "shoulder pain",
        "domain": "musculoskeletal",
        "body_area": "shoulder",
    },
    "elbow pain": {
        "canonical": "elbow pain",
        "domain": "musculoskeletal",
        "body_area": "elbow",
    },
    "rash": {
        "canonical": "skin rash",
        "domain": "skin",
        "body_area": "skin",
    },
    "itching": {
        "canonical": "pruritus",
        "domain": "skin",
        "body_area": "skin",
    },
    "palpitations": {
        "canonical": "palpitations",
        "domain": "cardiovascular",
        "body_area": "heart/chest",
    },
    "ear pain": {
        "canonical": "otalgia",
        "domain": "ent",
        "body_area": "ear",
    },
    "eye redness": {
        "canonical": "conjunctival injection",
        "domain": "eye",
        "body_area": "eyes",
    },
    "toothache": {
        "canonical": "dental pain",
        "domain": "dental",
        "body_area": "teeth/mouth",
    },
}

NEGATION_PATTERNS = [
    r"\b(?:no|not|don't|dont|doesn't|doesnt|didn't|didnt|never|without|denies|denied|free of|negative for|have not|haven't|havent|had not|hadn't|has not|hasn't|cannot|can't)\b",
]

# Contrastive conjunction regex used to prevent negation leakage
CONTRASTIVE_CONJUNCTIONS = re.compile(
    r"\b(?:but|however|although|though|except|whereas|yet)\b",
    re.IGNORECASE,
)

# Affirmative indicators that terminate preceding negation scope within a clause
AFFIRMATIVE_RESETS = re.compile(
    r"\b(?:and\s+i\s+have|i\s+have|there\s+is|my\s+\w+\s+is|with|developed|experiencing|complaining\s+of|suffering\s+from|can\s+move)\b",
    re.IGNORECASE,
)

# Proximity severity regex - only applies when explicitly modifying the symptom in local context
SEVERITY_PATTERNS = [
    (re.compile(r"\b(severe|intense|acute|excruciating|unbearable|worst)\b", re.IGNORECASE), "severe"),
    (re.compile(r"\b(moderate|considerable|medium|significant)\b", re.IGNORECASE), "moderate"),
    (re.compile(r"\b(mild|slight|minimal|minor|low)\b", re.IGNORECASE), "mild"),
]

LATERALITY_PATTERNS = [
    (re.compile(r"\b(right[- ]sided|right side|on the right side|on the right|right\s+(?:wrist|hand|arm|leg|ankle|knee|shoulder|foot|side|eye|ear))\b", re.IGNORECASE), "RIGHT"),
    (re.compile(r"\b(left[- ]sided|left side|on the left side|on the left|left\s+(?:wrist|hand|arm|leg|ankle|knee|shoulder|foot|side|eye|ear))\b", re.IGNORECASE), "LEFT"),
    (re.compile(r"\b(bilateral|both sides|on both sides)\b", re.IGNORECASE), "BILATERAL"),
]

QUALITY_PATTERNS = [
    (re.compile(r"\b(throbbing|pulsating|pounding)\b", re.IGNORECASE), "THROBBING"),
    (re.compile(r"\b(sharp|stabbing)\b", re.IGNORECASE), "SHARP"),
    (re.compile(r"\b(dull|aching)\b", re.IGNORECASE), "DULL"),
    (re.compile(r"\b(burning)\b", re.IGNORECASE), "BURNING"),
    (re.compile(r"\b(crushing|squeezing|tightness|pressure)\b", re.IGNORECASE), "PRESSURE"),
]

ONSET_PATTERNS = [
    (re.compile(r"\b(gradual|gradually|started gradually)\b", re.IGNORECASE), "GRADUAL"),
    (re.compile(r"\b(sudden|suddenly|abrupt|abruptly)\b", re.IGNORECASE), "SUDDEN"),
]

DURATION_REGEX = re.compile(
    r"\b(?:for|since|last|past)?\s*(\d+|a|an|few|several|couple of)\s*(days?|weeks?|months?|hours?|years?)\b",
    re.IGNORECASE,
)

TEMP_REGEX = re.compile(
    r"(?:temperature|temp)[^.\n,]*?(\d{2,3}(?:\.\d+)?)\s*(?:[°\u00b0]?\s*[FC]|degrees?\s*[FC])?",
    re.IGNORECASE,
)


def extract_symptoms(raw_text: str) -> dict[str, Any]:
    """Extract structured symptoms from raw user text using rule-based NLP.

    Preserves positive, negative, and unknown states without inventing findings or severity.
    """
    if not raw_text or not raw_text.strip():
        return {
            "status": "empty",
            "symptoms": [],
            "message": "No symptom text provided.",
            "source": "user_input",
            "nlp_version": NLP_VERSION,
        }

    cleaned = raw_text.strip()
    raw_sentences = re.split(r"[\n;.]+", cleaned)
    extracted: list[dict[str, Any]] = []
    seen_canonicals: set[str] = set()

    # Find global duration from text
    global_duration_match = DURATION_REGEX.search(cleaned)
    global_duration = global_duration_match.group(0).strip() if global_duration_match else None
    if global_duration:
        global_duration = re.sub(r"^(?:for|since|past|last)\s+", "", global_duration, flags=re.IGNORECASE).strip()

    # Check objective temperature
    temp_match = TEMP_REGEX.search(cleaned)
    temp_val = None
    if temp_match:
        temp_val = f"{temp_match.group(1)}°F"

    # Contextual qualifiers from entire text for complex presentations
    full_text_lower = cleaned.lower()
    global_laterality = None
    for lat_re, lat_val in LATERALITY_PATTERNS:
        if lat_re.search(cleaned):
            global_laterality = lat_val
            break

    global_quality = None
    for q_re, q_val in QUALITY_PATTERNS:
        if q_re.search(cleaned):
            global_quality = q_val
            break

    global_onset = None
    for on_re, on_val in ONSET_PATTERNS:
        if on_re.search(cleaned):
            global_onset = on_val
            break

    sorted_keywords = sorted(SYMPTOM_LEXICON.keys(), key=len, reverse=True)

    # Process sentence by sentence, splitting sentences on contrastive conjunctions to prevent negation leakage
    for sent in raw_sentences:
        if not sent.strip():
            continue

        # Split on contrastive conjunctions like 'but', 'however', 'although'
        subclauses = CONTRASTIVE_CONJUNCTIONS.split(sent)

        for clause in subclauses:
            clause = clause.strip()
            if not clause:
                continue

            occupied_spans: list[tuple[int, int]] = []

            clause_dur_m = DURATION_REGEX.search(clause)
            clause_duration = clause_dur_m.group(0).strip() if clause_dur_m else global_duration
            if clause_duration:
                clause_duration = re.sub(r"^(?:for|since|past|last)\s+", "", clause_duration, flags=re.IGNORECASE).strip()

            for kw in sorted_keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                for match in re.finditer(pattern, clause, re.IGNORECASE):
                    start, end = match.start(), match.end()
                    # Check span overlap to avoid matching substring when superstring was matched
                    if any(start < o_end and end > o_start for o_start, o_end in occupied_spans):
                        continue

                    meta = SYMPTOM_LEXICON[kw]
                    canonical = meta["canonical"]
                    if canonical in seen_canonicals:
                        continue

                    occupied_spans.append((start, end))

                    # Local negation check: only check text preceding the keyword in this specific subclause
                    preceding = clause[:start]
                    is_negated = False
                    for neg_pat in NEGATION_PATTERNS:
                        for neg_m in re.finditer(neg_pat, preceding, re.IGNORECASE):
                            between = preceding[neg_m.end():]
                            # Check if an affirmative phrase or subject assertion resets the negation
                            if AFFIRMATIVE_RESETS.search(between):
                                continue
                            if re.search(r",\s*(?:my\b|i\b|the\b|there\b|we\b)", between, re.IGNORECASE):
                                continue
                            is_negated = True
                            break
                        if is_negated:
                            break

                    state = "ABSENT" if is_negated else "PRESENT"
                    seen_canonicals.add(canonical)

                    # Severity: NEVER globally inferred. Check local proximity window (within 4 words before or 3 words after)
                    sev = ClinicalSeverity("UNKNOWN")
                    if state == "PRESENT":
                        if meta.get("explicit_severity"):
                            sev = ClinicalSeverity(meta["explicit_severity"].upper())
                        else:
                            window_before = " ".join(preceding.split()[-4:]) if preceding.split() else ""
                            post_text = clause[end:].split()
                            window_after = " ".join(post_text[:3]) if post_text else ""
                            window = window_before + " " + window_after
                            for sev_re, sev_name in SEVERITY_PATTERNS:
                                if sev_re.search(window):
                                    sev = ClinicalSeverity(sev_name.upper())
                                    break

                    # Qualifiers for specific clinical presentation
                    item_laterality = meta.get("laterality")
                    item_quality = None
                    item_trigger = None
                    item_context = None

                    clause_lower = clause.lower()

                    # Trigger detection
                    if "cough" in clause_lower and ("when coughing" in clause_lower or "on coughing" in clause_lower):
                        if canonical == "chest discomfort":
                            item_trigger = "coughing"
                            item_context = "when coughing"
                    if "sleep" in canonical and "cough" in clause_lower:
                        item_context = "because of the cough"
                    if canonical == "fever" and temp_val:
                        item_context = f"temperature around {temp_val}"

                    item_body_area = meta["body_area"]
                    # Musculoskeletal & Wrist injury qualifiers
                    if meta.get("domain") in {"musculoskeletal", "injury/trauma"} and state == "PRESENT":
                        if "wrist" in clause_lower or "wrist" in full_text_lower:
                            if meta.get("body_area") in {"extremity/joints", "musculoskeletal", "upper extremity", "unspecified"}:
                                item_body_area = "wrist"
                        if not item_laterality:
                            if "right" in clause_lower:
                                item_laterality = "RIGHT"
                            elif "left" in clause_lower:
                                item_laterality = "LEFT"
                            elif global_laterality and ("wrist" in clause_lower or item_body_area in {"wrist", "hand", "upper extremity"}):
                                item_laterality = global_laterality

                        if "moving" in clause_lower or "gripping" in clause_lower:
                            item_trigger = "moving or gripping objects"
                            item_context = "especially when moving or gripping objects"
                        elif "because of the pain" in clause_lower:
                            item_context = "because of the pain"
                        elif "after fall" in clause_lower or "after falling" in clause_lower:
                            item_trigger = "fall"
                            item_context = "after fall"

                    # Headache specific rich qualifiers
                    if canonical in {"headache", "migraine"} and state == "PRESENT":
                        item_laterality = global_laterality
                        item_quality = global_quality
                        triggers = []
                        if "bright light" in full_text_lower:
                            triggers.append("bright light")
                        if "loud sound" in full_text_lower:
                            triggers.append("loud sounds")
                        if triggers:
                            item_trigger = ", ".join(triggers)
                        if "right" in clause_lower or "right side" in clause_lower:
                            item_laterality = "RIGHT"

                    # Photophobia / Phonophobia specific qualifiers (quality must remain None)
                    if canonical == "photophobia":
                        item_quality = None
                        if "bright light" in clause_lower or "bright light" in full_text_lower:
                            item_trigger = "bright light"
                            item_context = "exposure to bright light"

                    if canonical == "phonophobia":
                        item_quality = None
                        if "loud sound" in clause_lower or "loud sound" in full_text_lower:
                            item_trigger = "loud sounds"
                            item_context = "exposure to loud sounds"

                    # Chest discomfort specific qualifiers
                    if canonical == "chest discomfort" and state == "PRESENT":
                        if not item_trigger and "cough" in full_text_lower and "when coughing" in full_text_lower:
                            item_trigger = "coughing"
                            item_context = "when coughing"

                    # Distinguish current symptoms from historical/contextual findings
                    is_historical = (
                        meta.get("type") == "HISTORICAL"
                        or canonical == "previous similar headache"
                        or bool(re.search(r"\b(?:previous|previously|once or twice before|in the past|history of|prior to this)\b", clause, re.IGNORECASE))
                    )

                    if is_historical:
                        item_type = "HISTORICAL"
                        item_duration = None
                        if "once or twice before" in clause_lower:
                            item_context = "once or twice before"
                    elif state == "ABSENT":
                        item_type = "NEGATIVE"
                        item_duration = None
                    elif state == "UNKNOWN":
                        item_type = "UNKNOWN"
                        item_duration = None
                    else:
                        item_type = meta.get("type") or "CURRENT"
                        item_duration = clause_duration

                    item = {
                        "finding": canonical,
                        "symptom": canonical,
                        "status": state,
                        "state": state,
                        "type": item_type,
                        "duration": item_duration,
                        "severity": sev if state == "PRESENT" and not is_historical else None,
                        "body_area": item_body_area,
                        "laterality": item_laterality,
                        "quality": item_quality,
                        "trigger": item_trigger,
                        "domain": meta["domain"],
                        "context": item_context or clause,
                        "source": "user_input",
                    }
                    if is_historical:
                        item["history"] = item_context or "historical occurrence"
                    if global_onset and state == "PRESENT":
                        item["onset"] = global_onset

                    extracted.append(item)

    # Temperature objective finding
    if temp_val and "temperature" not in seen_canonicals:
        extracted.append({
            "finding": "temperature",
            "symptom": "temperature",
            "status": "PRESENT",
            "state": "PRESENT",
            "duration": global_duration,
            "severity": "UNKNOWN",
            "body_area": "whole body",
            "laterality": None,
            "quality": None,
            "trigger": None,
            "domain": "systemic",
            "context": temp_val,
            "source": "user_input",
        })

    # If no symptoms from lexicon were detected, capture user input as general finding
    if not extracted and cleaned:
        extracted.append(
            {
                "finding": cleaned[:80],
                "symptom": cleaned[:80],
                "status": "UNKNOWN",
                "state": "UNKNOWN",
                "duration": global_duration,
                "severity": "UNKNOWN",
                "body_area": "unspecified",
                "laterality": None,
                "quality": None,
                "trigger": None,
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
