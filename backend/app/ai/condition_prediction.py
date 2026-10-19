
"""Machine Learning Possible-Condition Prediction Service for MediSense AI.

This service returns possible condition candidates, not a diagnosis.
Only active, present symptoms are used as positive model features.
Absent, unknown, and historical findings remain separate.
Emergency triage must run independently of this service.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from app.utils.logging import get_logger

logger = get_logger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = BACKEND_DIR / "models" / "conditions"
MODEL_PIPELINE_FILE = "condition_pipeline.joblib"
MODEL_CLASSES_FILE = "condition_classes.json"
MODEL_VERSION = "condition_pred_v1.0"

REJECTION_THRESHOLD = 0.25
MIN_CANDIDATE_SCORE = 0.04

# Explainability metadata. These profiles are not diagnostic criteria.
CONDITION_SYMPTOM_MAP: dict[str, dict[str, Any]] = {
    "Common Cold": {
        "typical_symptoms": [
            "cough", "runny nose", "rhinorrhea", "nasal congestion",
            "sore throat", "sneezing", "mild fever", "fever",
            "fatigue", "tiredness", "headache",
        ],
        "red_flags": ["severe breathing difficulty", "stridor"],
        "specialty": "General Physician / Pulmonology",
    },
    "Pneumonia": {
        "typical_symptoms": [
            "cough", "productive cough", "fever", "high fever", "chills",
            "shortness of breath", "chest pain", "difficulty breathing",
            "fatigue", "chest discomfort",
        ],
        "red_flags": [
            "severe shortness of breath", "cyanosis", "altered mental status",
        ],
        "specialty": "Pulmonology / Internal Medicine",
    },
    "Bronchial Asthma": {
        "typical_symptoms": [
            "wheezing", "shortness of breath", "cough", "chest tightness",
            "difficulty breathing", "chest discomfort",
        ],
        "red_flags": [
            "severe breathing difficulty", "silent chest",
            "inability to speak full sentences",
        ],
        "specialty": "Pulmonology / Allergy & Immunology",
    },
    "Migraine": {
        "typical_symptoms": [
            "headache", "throbbing headache", "nausea", "photophobia",
            "phonophobia", "vomiting",
        ],
        "red_flags": [
            "sudden onset", "worst headache of life", "weakness", "fainting",
        ],
        "specialty": "Neurology",
    },
    "Arthritis": {
        "typical_symptoms": [
            "joint pain", "stiffness", "swelling", "reduced movement",
            "tenderness",
        ],
        "red_flags": ["hot swollen joint with fever", "inability to bear weight"],
        "specialty": "Orthopedics / Rheumatology",
    },
    "Cervical spondylosis": {
        "typical_symptoms": [
            "neck pain", "stiffness", "numbness in fingers", "tingling",
            "shoulder pain",
        ],
        "red_flags": [
            "progressive limb weakness", "loss of bowel or bladder control",
        ],
        "specialty": "Orthopedics / Neurology",
    },
    "gastroesophageal reflux disease": {
        "typical_symptoms": [
            "heartburn", "acid reflux", "chest discomfort", "nausea",
            "stomach pain",
        ],
        "red_flags": ["difficulty swallowing", "vomiting blood", "black stool"],
        "specialty": "Gastroenterology",
    },
    "peptic ulcer disease": {
        "typical_symptoms": [
            "abdominal pain", "stomach pain", "nausea", "indigestion",
            "burning pain",
        ],
        "red_flags": [
            "blood in stool", "vomiting blood", "sudden severe abdominal pain",
        ],
        "specialty": "Gastroenterology",
    },
    "urinary tract infection": {
        "typical_symptoms": [
            "burning urination", "frequent urination", "pelvic pain",
            "cloudy urine", "fever",
        ],
        "red_flags": ["high fever with flank pain", "rigors"],
        "specialty": "Urology / Nephrology",
    },
    "Dengue": {
        "typical_symptoms": [
            "high fever", "severe headache", "joint pain", "muscle ache",
            "rash", "eye pain",
        ],
        "red_flags": ["bleeding gums", "severe abdominal pain", "persistent vomiting"],
        "specialty": "Infectious Disease / Internal Medicine",
    },
    "Typhoid": {
        "typical_symptoms": [
            "high fever", "headache", "abdominal pain", "fatigue",
            "diarrhea", "constipation",
        ],
        "red_flags": ["intestinal perforation signs", "severe confusion"],
        "specialty": "Infectious Disease / General Physician",
    },
    "Malaria": {
        "typical_symptoms": ["fever", "chills", "sweating", "headache", "nausea"],
        "red_flags": [
            "impaired consciousness", "severe anemia", "jaundice with fever",
        ],
        "specialty": "Infectious Disease",
    },
    "Chicken pox": {
        "typical_symptoms": [
            "itchy rash", "skin rash", "fever", "fatigue", "blisters",
        ],
        "red_flags": ["infected blisters", "shortness of breath"],
        "specialty": "Dermatology / Pediatrics / General Physician",
    },
    "Psoriasis": {
        "typical_symptoms": [
            "skin rash", "scaly skin", "itching", "red patches", "joint pain",
        ],
        "red_flags": ["erythrodermic rash involving >90% body area"],
        "specialty": "Dermatology",
    },
    "Acne": {
        "typical_symptoms": ["skin rash", "pimples", "blackheads", "whiteheads"],
        "red_flags": ["severe cystic infection"],
        "specialty": "Dermatology",
    },
    "Impetigo": {
        "typical_symptoms": [
            "skin rash", "sores", "honey-colored crusts", "blisters", "itching",
        ],
        "red_flags": ["rapidly spreading cellulitis"],
        "specialty": "Dermatology",
    },
    "Fungal infection": {
        "typical_symptoms": ["itching", "skin rash", "redness", "scaling"],
        "red_flags": ["systemic fungal signs"],
        "specialty": "Dermatology",
    },
    "allergy": {
        "typical_symptoms": [
            "sneezing", "runny nose", "itching", "skin rash", "watery eyes",
            "cough", "shortness of breath", "difficulty breathing",
            "wheezing", "sore throat", "fatigue", "fever",
        ],
        "red_flags": ["facial swelling", "stridor", "breathing difficulty"],
        "specialty": "Allergy & Immunology / General Physician",
    },
    "drug reaction": {
        "typical_symptoms": ["skin rash", "itching", "fever", "hives"],
        "red_flags": ["mucosal involvement", "skin peeling", "anaphylaxis"],
        "specialty": "Allergy & Clinical Immunology / Dermatology",
    },
    "diabetes": {
        "typical_symptoms": [
            "increased thirst", "frequent urination", "fatigue", "weight loss",
            "blurred vision",
        ],
        "red_flags": [
            "ketoacidosis signs", "fruity breath odor", "altered sensorium",
        ],
        "specialty": "Endocrinology",
    },
    "Hypertension": {
        "typical_symptoms": [
            "headache", "dizziness", "chest discomfort", "shortness of breath",
        ],
        "red_flags": [
            "severe chest pain", "neurological deficits", "hypertensive crisis",
        ],
        "specialty": "Cardiology / Internal Medicine",
    },
    "Jaundice": {
        "typical_symptoms": [
            "yellow skin", "yellow eyes", "dark urine", "abdominal pain",
            "fatigue",
        ],
        "red_flags": ["hepatic encephalopathy", "coagulopathy"],
        "specialty": "Gastroenterology / Hepatology",
    },
    "Varicose Veins": {
        "typical_symptoms": [
            "leg pain", "swollen veins", "aching legs", "heaviness in legs",
        ],
        "red_flags": ["sudden unilateral leg swelling with shortness of breath"],
        "specialty": "Vascular Surgery",
    },
    "Dimorphic Hemorrhoids": {
        "typical_symptoms": [
            "rectal bleeding", "pain during bowel movements", "itching",
            "swelling around anus",
        ],
        "red_flags": ["massive rectal bleeding", "severe thrombosis"],
        "specialty": "Proctology / General Surgery",
    },
}


def _normalize(value: Any) -> str:
    """Normalize text for safe phrase comparison."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_name(item: Any) -> str:
    """Extract a symptom/finding name from a string or dictionary."""
    if isinstance(item, dict):
        return _normalize(item.get("symptom") or item.get("finding") or "")
    return _normalize(item)


def _phrase_matches(profile_phrase: str, patient_phrase: str) -> bool:
    """Match whole phrases rather than arbitrary substrings."""
    profile = _normalize(profile_phrase)
    patient = _normalize(patient_phrase)

    if not profile or not patient:
        return False

    if profile == patient:
        return True

    # Allow a phrase to be embedded in a longer phrase, but only at word
    # boundaries. This avoids matching "pain" inside an unrelated word.
    return (
        re.search(r"(?<!\w)" + re.escape(profile) + r"(?!\w)", patient) is not None
        or re.search(r"(?<!\w)" + re.escape(patient) + r"(?!\w)", profile) is not None
    )


def _get_state(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("state") or "").strip().upper()


def _is_historical(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return str(item.get("type") or "").strip().upper() == "HISTORICAL"


def _match_profile(
    profile_symptoms: list[str],
    patient_symptoms: set[str],
) -> list[str]:
    return [
        profile_symptom
        for profile_symptom in profile_symptoms
        if any(_phrase_matches(profile_symptom, patient) for patient in patient_symptoms)
    ]


class ConditionPredictionService:
    """Assistive ML condition prediction service."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._pipeline: Any = None
        self._classes: list[str] = []
        self._loaded = False
        self.model_version = MODEL_VERSION

    def load_model(self) -> bool:
        """Load the trained pipeline and use its own class order."""
        pipeline_path = self.model_dir / MODEL_PIPELINE_FILE
        classes_path = self.model_dir / MODEL_CLASSES_FILE

        if not pipeline_path.is_file():
            logger.info("Condition prediction model not found at: %s", pipeline_path)
            self._pipeline = None
            self._classes = []
            self._loaded = False
            return False

        try:
            pipeline = joblib.load(pipeline_path)

            if not hasattr(pipeline, "predict_proba"):
                raise ValueError("Condition pipeline does not support predict_proba().")

            # The classifier's class order corresponds to predict_proba() columns.
            # Never replace this order with a separate JSON list.
            classifier = getattr(pipeline, "classes_", None)

            if classifier is None and hasattr(pipeline, "steps") and pipeline.steps:
                classifier = getattr(pipeline.steps[-1][1], "classes_", None)

            if classifier is None:
                raise ValueError(
                    "Cannot determine the trained model's class order. "
                    "The pipeline must expose classes_."
                )

            model_classes = [str(label) for label in classifier]

            if not model_classes:
                raise ValueError("The trained model has no classes.")

            if classes_path.is_file():
                try:
                    with classes_path.open("r", encoding="utf-8") as file:
                        metadata_classes = json.load(file)

                    if not isinstance(metadata_classes, list):
                        logger.warning(
                            "Class metadata is not a list; using model classes."
                        )
                    elif set(map(str, metadata_classes)) != set(model_classes):
                        logger.warning(
                            "Class metadata differs from trained model classes. "
                            "Using the model's authoritative class order."
                        )
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Could not read class metadata: %s", exc)

            self._pipeline = pipeline
            self._classes = model_classes
            self._loaded = True

            logger.info(
                "Loaded condition pipeline with %d model classes.", len(model_classes)
            )
            return True

        except Exception as exc:
            logger.error("Failed to load condition prediction pipeline: %s", exc)
            self._pipeline = None
            self._classes = []
            self._loaded = False
            return False

    def is_available(self) -> bool:
        if not self._loaded:
            self.load_model()
        return self._loaded

    def predict_conditions(
        self,
        symptoms_present: list[dict[str, Any]],
        symptoms_absent: list[dict[str, Any]],
        symptoms_unknown: list[dict[str, Any]],
        raw_text: Optional[str] = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Return ranked, evidence-supported candidates.

        raw_text is intentionally not used to build model features. Use the
        structured, validated symptom lists instead.
        top_k is retained for API compatibility; supported candidates are not
        truncated to the top three.
        """
        del raw_text, top_k

        safety_notice = (
            "Clinical decision-support only; these results are not a diagnosis. "
            "Emergency triage and clinician assessment take precedence."
        )

        if not self.is_available():
            return {
                "status": "unavailable",
                "model_version": None,
                "conditions": [],
                "message": (
                    "Condition prediction is unavailable because the trained model "
                    "could not be loaded. Check backend/models/conditions/."
                ),
                "is_experimental": True,
                "safety_notice": safety_notice,
            }

        present_items = symptoms_present or []
        absent_items = symptoms_absent or []
        unknown_items = symptoms_unknown or []

        # Only active PRESENT symptoms can become positive model features.
        active_present: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for item in present_items:
            if _is_historical(item):
                continue

            if isinstance(item, dict):
                state = _get_state(item)
                if state in {"ABSENT", "UNKNOWN", "HISTORICAL"}:
                    continue

                name = _extract_name(item)
                area = _normalize(item.get("body_area"))
                quality = _normalize(item.get("quality"))

                if not name:
                    continue

                key = (name, area, quality)
                if key not in seen:
                    seen.add(key)
                    normalized_item = dict(item)
                    normalized_item["symptom"] = name
                    normalized_item["state"] = "PRESENT"
                    active_present.append(normalized_item)
            else:
                name = _extract_name(item)
                if name and (name, "", "") not in seen:
                    seen.add((name, "", ""))
                    active_present.append(
                        {"symptom": name, "state": "PRESENT"}
                    )

        if not active_present:
            return {
                "status": "insufficient_evidence",
                "model_version": self.model_version,
                "conditions": [],
                "message": (
                    "There are no active present symptoms to evaluate. "
                    "Please provide the current symptoms."
                ),
                "is_experimental": True,
                "safety_notice": safety_notice,
            }

        present_names = {
            _extract_name(item)
            for item in active_present
            if _extract_name(item) and _extract_name(item) != "temperature"
        }

        absent_names = {
            _extract_name(item)
            for item in absent_items
            if _extract_name(item)
        }

        unknown_names = {
            _extract_name(item)
            for item in unknown_items
            if _extract_name(item)
        }

        # A symptom marked absent/unknown must not be treated as present,
        # even if a caller accidentally supplied it in the present list.
        absent_names.update(
            _extract_name(item)
            for item in present_items
            if _get_state(item) == "ABSENT" and _extract_name(item)
        )
        unknown_names.update(
            _extract_name(item)
            for item in present_items
            if _get_state(item) == "UNKNOWN" and _extract_name(item)
        )

        # Exclude any accidental overlap from the positive feature set.
        present_names = {
            name for name in present_names
            if name not in absent_names and name not in unknown_names
        }

        if not present_names:
            return {
                "status": "insufficient_evidence",
                "model_version": self.model_version,
                "conditions": [],
                "message": "No usable active present symptom keywords were provided.",
                "is_experimental": True,
                "safety_notice": safety_notice,
            }

        # Build ML text exclusively from validated present symptoms.
        feature_parts: list[str] = []
        for item in active_present:
            name = _extract_name(item)

            if not name or name == "temperature":
                continue
            if name not in present_names:
                continue

            feature_parts.append(name)

            quality = _normalize(item.get("quality")) if isinstance(item, dict) else ""
            if quality and quality not in name:
                feature_parts.append(f"{quality} {name}")

        # Deduplicate feature tokens without increasing their frequency.
        query_text = " ".join(dict.fromkeys(feature_parts)).strip()

        if not query_text:
            return {
                "status": "insufficient_evidence",
                "model_version": self.model_version,
                "conditions": [],
                "message": "No usable symptom keywords were found.",
                "is_experimental": True,
                "safety_notice": safety_notice,
            }

        try:
            probabilities = np.asarray(
                self._pipeline.predict_proba([query_text])[0],
                dtype=float,
            )

            # Class names and probability columns must have identical ordering.
            classes = list(self._classes)

            if len(probabilities) != len(classes):
                raise ValueError(
                    "Model probability count does not match its class count."
                )

            if not np.all(np.isfinite(probabilities)):
                raise ValueError("Model returned non-finite probabilities.")

            if np.any(probabilities < 0) or np.any(probabilities > 1):
                raise ValueError("Model returned probabilities outside [0, 1].")

            # Accommodate minor naming/capitalization differences in metadata.
            metadata_by_normalized_name = {
                _normalize(name): metadata
                for name, metadata in CONDITION_SYMPTOM_MAP.items()
            }

            ranked_indices = np.argsort(probabilities)[::-1]
            conditions: list[dict[str, Any]] = []

            for index in ranked_indices:
                condition_name = classes[int(index)]
                score = float(probabilities[int(index)])

                # Retain low scores only until a candidate has been accepted;
                # do not stop ranking before checking evidence alignment.
                if score < MIN_CANDIDATE_SCORE and conditions:
                    continue

                metadata = metadata_by_normalized_name.get(
                    _normalize(condition_name), {}
                )
                typical_symptoms = metadata.get("typical_symptoms", [])

                # Evidence is matched against structured symptoms only.
                supporting = _match_profile(typical_symptoms, present_names)
                if not supporting:
                    continue

                contradictory = _match_profile(typical_symptoms, absent_names)
                explicitly_unknown = _match_profile(
                    typical_symptoms, unknown_names
                )

                # A candidate with multiple contradictions outweighing its
                # support is excluded by this conservative heuristic.
                # This is not a clinically validated exclusion rule.
                if len(contradictory) >= 2 and len(contradictory) > len(supporting):
                    continue

                assessed_names = present_names | absent_names | unknown_names
                missing = [
                    symptom
                    for symptom in typical_symptoms
                    if not any(
                        _phrase_matches(symptom, patient_name)
                        for patient_name in assessed_names
                    )
                ][:3]

                if score >= 0.50:
                    confidence_tier = "Moderate Alignment"
                elif score >= REJECTION_THRESHOLD:
                    confidence_tier = "Possible Alignment"
                else:
                    confidence_tier = "Low Alignment / Differential"

                conditions.append({
                    "rank": len(conditions) + 1,
                    "condition": condition_name,
                    "probability_score": round(score, 4),
                    "confidence_tier": confidence_tier,
                    "supporting_symptoms": supporting,
                    "contradictory_findings": contradictory,
                    "unknown_symptoms": explicitly_unknown,
                    "unknown_or_missing_information": (
                        explicitly_unknown + missing
                    ),
                    "unassessed_symptoms": missing,
                    "suggested_specialty": metadata.get(
                        "specialty", "General Physician / Internal Medicine"
                    ),
                    "clinical_rationale": (
                        "The listed active symptoms match some features in this "
                        "condition's configured symptom profile. This is statistical "
                        "alignment, not confirmation of the condition."
                    ),
                    "disclaimer": (
                        "Possible condition for discussion with a licensed healthcare "
                        "professional; not a diagnosis or treatment recommendation."
                    ),
                })

            # Keep candidate order consistent with descending model probability.
            conditions.sort(
                key=lambda candidate: candidate["probability_score"],
                reverse=True,
            )

            for rank, candidate in enumerate(conditions, start=1):
                candidate["rank"] = rank

            if not conditions:
                return {
                    "status": "unsupported_condition",
                    "model_version": self.model_version,
                    "conditions": [],
                    "total_candidates_evaluated": len(ranked_indices),
                    "rejection_threshold": REJECTION_THRESHOLD,
                    "message": (
                        "The symptoms do not show strong alignment with any "
                        "supported condition profile. This does not rule out "
                        "a medical condition; seek professional assessment."
                    ),
                    "is_experimental": True,
                    "safety_notice": safety_notice,
                }

            # A low score is not proof that a condition is absent. Return an
            # explicit uncertainty state rather than presenting weak candidates
            # as supported results.
            if conditions[0]["probability_score"] < REJECTION_THRESHOLD:
                return {
                    "status": "unsupported_condition",
                    "model_version": self.model_version,
                    "conditions": [],
                    "total_candidates_evaluated": len(ranked_indices),
                    "rejection_threshold": REJECTION_THRESHOLD,
                    "message": (
                        "The model scores did not meet the configured acceptance "
                        "threshold. No condition candidates are being presented as "
                        "supported. A clinician can evaluate other possibilities."
                    ),
                    "is_experimental": True,
                    "safety_notice": safety_notice,
                }

            return {
                "status": "completed",
                "model_version": self.model_version,
                "conditions": conditions,
                "total_candidates_evaluated": len(ranked_indices),
                "message": (
                    f"Generated {len(conditions)} evidence-supported possible "
                    "condition candidate(s). These are not diagnoses."
                ),
                "is_experimental": True,
                "safety_notice": safety_notice,
            }

        except Exception as exc:
            logger.exception("Condition prediction inference failed: %s", exc)
            return {
                "status": "unavailable",
                "model_version": self.model_version,
                "conditions": [],
                "message": (
                    "Condition prediction could not be completed. The system should "
                    "continue to use its independent safety-triage process."
                ),
                "is_experimental": True,
                "safety_notice": safety_notice,
            }


# Singleton service instance
_condition_service: Optional[ConditionPredictionService] = None


def get_condition_prediction_service() -> ConditionPredictionService:
    global _condition_service

    if _condition_service is None:
        _condition_service = ConditionPredictionService()

    return _condition_service
