
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
MODEL_VERSION = "condition_pred_v1.1-task1"

REJECTION_THRESHOLD = 0.25
MIN_CANDIDATE_SCORE = 0.04
MAX_DISPLAY_CANDIDATES = 24

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

    # A configured profile phrase may appear inside a more specific patient
    # phrase (e.g. "chest pain" in "severe chest pain"). Do not reverse this
    # comparison: a generic patient phrase such as "pain" must not match every
    # specific profile phrase such as "chest pain" or "abdominal pain".
    return re.search(
        r"(?<!\w)" + re.escape(profile) + r"(?!\w)", patient
    ) is not None


def _get_state(item: Any) -> str:
    """Normalize symptom state aliases without treating unknown as positive."""
    if not isinstance(item, dict):
        return ""
    raw = item.get("state")
    if raw is None or str(raw).strip() == "":
        raw = item.get("status")
    state = re.sub(r"[\s-]+", "_", str(raw or "").strip().upper())

    aliases = {
        "YES": "PRESENT",
        "TRUE": "PRESENT",
        "POSITIVE": "PRESENT",
        "CURRENT": "PRESENT",
        "NO": "ABSENT",
        "FALSE": "ABSENT",
        "NEGATIVE": "ABSENT",
        "NOT_PRESENT": "ABSENT",
        "NOT_SURE": "UNKNOWN",
        "UNSURE": "UNKNOWN",
        "UNKNOWN": "UNKNOWN",
        "UNANSWERED": "UNKNOWN",
        "UNASSESSED": "UNKNOWN",
        "NOT_ASSESSED": "UNKNOWN",
        "HISTORICAL": "HISTORICAL",
        "PAST": "HISTORICAL",
    }
    return aliases.get(state, state)


def _is_historical(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = str(item.get("type") or item.get("temporal_status") or "").strip().upper()
    return item_type in {"HISTORICAL", "HISTORY", "PAST", "PREVIOUS"} or _get_state(item) == "HISTORICAL"


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
        structured, validated symptom lists instead. This classifier is
        symptom-text based; it does not directly consume lab or imaging data.
        Those sources must be integrated by the assessment/evidence layer.
        top_k is retained for API compatibility; output is capped at the model's
        supported 24-class limit rather than an arbitrary top-three list.
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
                    "could not be loaded. Check the trained pipeline, its dependencies, "
                    "and backend/models/conditions/."
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
                # Items supplied in symptoms_present may omit state for legacy
                # callers, but any explicit non-positive state must be excluded.
                if state in {"ABSENT", "UNKNOWN", "HISTORICAL", "NO", "FALSE", "NEGATIVE"}:
                    continue
                if state and state != "PRESENT":
                    # Fail closed for unrecognized explicit states: never turn
                    # an unfamiliar state into a positive model feature.
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
                    "Condition matching operates on symptom presentations; "
                    "clinical evaluation and triage rely on the available laboratory, imaging, and report evidence."
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
            and not _is_historical(item)
            and (not isinstance(item, dict) or _get_state(item) in {"", "ABSENT"})
        }

        unknown_names = {
            _extract_name(item)
            for item in unknown_items
            if _extract_name(item)
            and not _is_historical(item)
            and (not isinstance(item, dict) or _get_state(item) in {"", "UNKNOWN"})
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

            # predict_proba should return a normalized distribution. Reject
            # malformed output rather than silently ranking invalid scores.
            score_sum = float(probabilities.sum())
            if score_sum <= 0 or not np.isclose(score_sum, 1.0, atol=1e-3):
                raise ValueError(
                    f"Model probability scores are not normalized (sum={score_sum:.6f})."
                )

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

                # Apply the minimum score consistently to every candidate.
                # Never let the first candidate bypass the configured floor.
                if score < MIN_CANDIDATE_SCORE:
                    continue

                metadata = metadata_by_normalized_name.get(
                    _normalize(condition_name), {}
                )
                typical_symptoms = metadata.get("typical_symptoms", [])

                # Evidence is matched against structured symptoms only.
                supporting = _match_profile(typical_symptoms, present_names)
                if not supporting:
                    continue

                # Contradictory evidence: match against genuinely absent typical symptoms.
                # If a symptom had an explicit severity denied (e.g. 'no severe chest pain'),
                # do not treat it as a contradiction for conditions like Pneumonia where chest
                # tightness, mild discomfort, or cough-associated pleurisy are typical.
                raw_contradictory = _match_profile(typical_symptoms, absent_names)
                contradictory = []
                for contra in raw_contradictory:
                    # Check if the patient has a present related finding (e.g., chest tightness present)
                    # or if the denied finding had a restricted negated_severity.
                    has_related_present = False
                    if contra in {"chest pain", "chest discomfort"}:
                        has_related_present = any(
                            p in present_names
                            for p in {"chest tightness", "chest discomfort", "tightness"}
                        )
                    is_only_severity_denied = False
                    for item in absent_items:
                        if isinstance(item, dict) and item.get("negated_severity"):
                            item_name = _extract_name(item)
                            if _phrase_matches(contra, item_name):
                                is_only_severity_denied = True
                                break
                    if not (has_related_present or is_only_severity_denied):
                        contradictory.append(contra)

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

                # These are model score bands, not calibrated clinical
                # confidence levels. Keep the legacy field for API compatibility
                # while making that limitation explicit in the returned payload.
                if score >= 0.50:
                    confidence_tier = "Higher model score (not clinical confidence)"
                elif score >= REJECTION_THRESHOLD:
                    confidence_tier = "Possible model match (not clinical confidence)"
                else:
                    confidence_tier = "Low model score (not clinical confidence)"

                conditions.append({
                    "rank": len(conditions) + 1,
                    "condition": condition_name,
                    # Retain the legacy key used by the frontend. This is the
                    # model's predict_proba score, not a validated disease risk.
                    "probability_score": round(score, 4),
                    "model_score": round(score, 4),
                    "score_interpretation": (
                        "Unvalidated model score; not a calibrated probability "
                        "of having this condition."
                    ),
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

            # Keep candidate order consistent with descending model score and
            # cap output at the model's supported class limit (normally 24).
            conditions.sort(
                key=lambda candidate: candidate["model_score"],
                reverse=True,
            )
            conditions = conditions[: min(MAX_DISPLAY_CANDIDATES, len(classes))]

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
                "total_model_classes": len(classes),
                "total_candidates_evaluated": len(ranked_indices),
                "total_candidates_returned": len(conditions),
                "maximum_candidates_displayed": MAX_DISPLAY_CANDIDATES,
                "candidate_limit_note": (
                    f"Up to {MAX_DISPLAY_CANDIDATES} candidates are returned only when they pass the model-score "
                    "and structured symptom-evidence filters; fewer results are valid."
                ),
                "prediction_basis": "active_present_symptoms_only",
                "input_sources_used": ["validated_present_symptoms"],
                "input_sources_not_used_directly": [
                    "laboratory_reports", "radiology_report_text", "xray_images"
                ],
                "score_interpretation": (
                    "Model scores are not clinically validated probabilities and "
                    "must not be interpreted as confirmed disease likelihoods."
                ),
                "message": (
                    f"Generated {len(conditions)} possible condition candidate(s) "
                    "with matching symptom-profile evidence. These are not diagnoses."
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

    def reconcile_candidate_evidence(
        self,
        conditions: list[dict[str, Any]] | dict[str, Any],
        evidence: dict[str, Any],
        source_records: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Candidate-specific evidence reconciliation interface.

        Evaluates genuine ML condition candidates against actual assessment evidence
        (symptoms, lab/medical reports, radiology reports, eligible X-rays).
        Preserves original model scores completely untouched.
        """
        return reconcile_candidate_evidence(conditions, evidence, source_records)


# Singleton service instance

_condition_service: Optional[ConditionPredictionService] = None


def get_condition_prediction_service() -> ConditionPredictionService:
    global _condition_service

    if _condition_service is None:
        _condition_service = ConditionPredictionService()

    return _condition_service


# Documented Condition-Modality Evidence Profiles
# Maps each of the 24 trained condition classes to documented clinical relationships:
# body system, relevant regions, relevant lab tests, relevant imaging findings,
# and key missing modalities.
CONDITION_MODALITY_PROFILES: dict[str, dict[str, Any]] = {
    "Pneumonia": {
        "body_system": "Respiratory",
        "relevant_regions": ["chest", "thorax", "lungs", "pulmonary"],
        "relevant_lab_tests": [
            "wbc", "white blood cell", "white blood cells", "leukocyte", "leukocytes",
            "crp", "c reactive protein", "c-reactive protein", "procalcitonin",
            "esr", "neutrophil", "neutrophils", "sputum", "blood culture",
        ],
        "relevant_imaging_findings": [
            "consolidation", "infiltrate", "infiltrates", "opacity", "opacities",
            "air bronchogram", "pleural effusion", "effusion", "pneumonia",
            "patchy opacity", "lobar consolidation",
        ],
        "lab_interpretation_support": {"HIGH", "ABNORMAL", "POSITIVE", "ELEVATED"},
        "key_missing_modalities": ["Chest X-ray (CXR)", "Complete Blood Count / Inflammatory markers"],
    },
    "Bronchial Asthma": {
        "body_system": "Respiratory / Allergy",
        "relevant_regions": ["chest", "thorax", "lungs", "pulmonary"],
        "relevant_lab_tests": [
            "ige", "total ige", "immunoglobulin e", "eosinophil", "eosinophils",
            "absolute eosinophil count", "aec",
        ],
        "relevant_imaging_findings": [
            "hyperinflation", "bronchial wall thickening", "increased lung markings",
        ],
        "lab_interpretation_support": {"HIGH", "ABNORMAL", "ELEVATED"},
        "key_missing_modalities": ["Spirometry / Peak Flow assessment", "Chest radiograph to exclude infection"],
    },
    "Common Cold": {
        "body_system": "Respiratory / ENT",
        "relevant_regions": ["chest", "thorax", "neck", "sinuses"],
        "relevant_lab_tests": ["wbc", "lymphocyte", "crp"],
        "relevant_imaging_findings": [],
        "lab_interpretation_support": set(),
        "key_missing_modalities": ["Clinical symptom tracking"],
    },
    "Migraine": {
        "body_system": "Neurological",
        "relevant_regions": ["skull", "head", "brain"],
        "relevant_lab_tests": ["cbc", "electrolytes", "esr"],
        "relevant_imaging_findings": ["normal", "no acute intracranial abnormality"],
        "lab_interpretation_support": set(),
        "key_missing_modalities": ["Neurological clinical evaluation (neuroimaging if red flags present)"],
    },
    "Arthritis": {
        "body_system": "Musculoskeletal",
        "relevant_regions": [
            "knee", "wrist", "hand", "elbow", "ankle", "shoulder", "hip", "foot",
            "finger", "fingers", "toes", "spine", "cervical spine", "lumbar spine",
        ],
        "relevant_lab_tests": [
            "rf", "rheumatoid factor", "anti ccp", "anti-ccp", "uric acid",
            "esr", "crp", "ana",
        ],
        "relevant_imaging_findings": [
            "joint space narrowing", "osteophyte", "osteophytes", "subchondral sclerosis",
            "bone erosion", "erosions", "periarticular osteopenia", "degenerative changes",
            "osteoarthritis", "arthritis",
        ],
        "lab_interpretation_support": {"HIGH", "ABNORMAL", "POSITIVE"},
        "key_missing_modalities": ["Affected joint radiograph", "Rheumatologic panel (ESR, CRP, RF, Uric acid)"],
    },
    "Cervical spondylosis": {
        "body_system": "Musculoskeletal / Spine",
        "relevant_regions": ["cervical spine", "neck", "spine"],
        "relevant_lab_tests": ["esr", "crp"],
        "relevant_imaging_findings": [
            "disc space narrowing", "osteophyte", "osteophytes", "neural foraminal narrowing",
            "spondylosis", "degenerative disc disease",
        ],
        "lab_interpretation_support": set(),
        "key_missing_modalities": ["Cervical spine radiograph (AP and lateral views)"],
    },
    "gastroesophageal reflux disease": {
        "body_system": "Gastrointestinal",
        "relevant_regions": ["chest", "abdomen"],
        "relevant_lab_tests": ["hemoglobin", "hb"],
        "relevant_imaging_findings": ["hiatal hernia", "normal"],
        "lab_interpretation_support": set(),
        "key_missing_modalities": ["Upper GI endoscopy if symptoms persistent"],
    },
    "peptic ulcer disease": {
        "body_system": "Gastrointestinal",
        "relevant_regions": ["abdomen", "pelvis"],
        "relevant_lab_tests": [
            "h pylori", "helicobacter pylori", "stool occult blood", "fecal occult blood",
            "hemoglobin", "hb", "hematocrit",
        ],
        "relevant_imaging_findings": ["pneumoperitoneum", "free air", "normal"],
        "lab_interpretation_support": {"POSITIVE", "LOW", "ABNORMAL"},
        "key_missing_modalities": ["Upper GI endoscopy", "H. pylori testing / Stool occult blood"],
    },
    "urinary tract infection": {
        "body_system": "Renal / Urological",
        "relevant_regions": ["abdomen", "pelvis"],
        "relevant_lab_tests": [
            "urine routine", "urine microscopy", "pus cells", "leukocytes", "leukocyte esterase",
            "nitrite", "nitrites", "bacteria", "urine culture", "rbc in urine", "urinalysis",
        ],
        "relevant_imaging_findings": ["hydronephrosis", "renal calculi", "cystitis"],
        "lab_interpretation_support": {"HIGH", "ABNORMAL", "POSITIVE"},
        "key_missing_modalities": ["Urinalysis (Routine/Microscopy)", "Urine culture & sensitivity"],
    },
    "Dengue": {
        "body_system": "Infectious / Hematology",
        "relevant_regions": [],
        "relevant_lab_tests": [
            "platelet", "platelets", "platelet count", "thrombocyte",
            "hematocrit", "hct", "wbc", "ns1", "dengue ns1", "dengue igm", "dengue igg",
        ],
        "relevant_imaging_findings": ["pleural effusion", "ascites", "gallbladder wall thickening"],
        "lab_interpretation_support": {"LOW", "ABNORMAL", "POSITIVE", "HIGH"},
        "key_missing_modalities": ["Serial Complete Blood Count (Platelet and Hematocrit)", "Dengue NS1 / IgM serology"],
    },
    "Typhoid": {
        "body_system": "Infectious / Gastrointestinal",
        "relevant_regions": ["abdomen"],
        "relevant_lab_tests": [
            "widal", "widal test", "typhidot", "blood culture", "wbc", "stool culture",
        ],
        "relevant_imaging_findings": ["splenomegaly"],
        "lab_interpretation_support": {"POSITIVE", "ABNORMAL", "HIGH"},
        "key_missing_modalities": ["Blood culture", "Typhoid serology (Widal / Typhidot)"],
    },
    "Malaria": {
        "body_system": "Infectious / Hematology",
        "relevant_regions": [],
        "relevant_lab_tests": [
            "malaria", "mp", "smear for mp", "peripheral smear", "rdt", "malaria antigen",
            "platelet", "platelets", "hemoglobin", "hb",
        ],
        "relevant_imaging_findings": ["splenomegaly"],
        "lab_interpretation_support": {"POSITIVE", "ABNORMAL", "LOW"},
        "key_missing_modalities": ["Peripheral blood smear for malaria parasite", "Malaria Rapid Diagnostic Test (RDT)"],
    },
    "Chicken pox": {
        "body_system": "Dermatological / Infectious",
        "relevant_regions": ["skin"],
        "relevant_lab_tests": ["vzv", "varicella", "cbc"],
        "relevant_imaging_findings": [],
        "lab_interpretation_support": {"POSITIVE", "ABNORMAL"},
        "key_missing_modalities": ["Clinical dermatology evaluation"],
    },
    "Psoriasis": {
        "body_system": "Dermatological",
        "relevant_regions": ["skin", "knee", "elbow", "finger", "hand", "wrist"],
        "relevant_lab_tests": ["esr", "crp", "rf"],
        "relevant_imaging_findings": ["dactylitis", "erosions"],
        "lab_interpretation_support": {"HIGH", "ABNORMAL"},
        "key_missing_modalities": ["Clinical dermatology assessment"],
    },
    "Acne": {
        "body_system": "Dermatological",
        "relevant_regions": ["face", "facial bones", "neck", "chest", "skin"],
        "relevant_lab_tests": ["hormone panel", "testosterone", "dhea"],
        "relevant_imaging_findings": [],
        "lab_interpretation_support": set(),
        "key_missing_modalities": ["Clinical dermatology evaluation"],
    },
    "Impetigo": {
        "body_system": "Dermatological",
        "relevant_regions": ["face", "facial bones", "skin"],
        "relevant_lab_tests": ["wound swab", "bacterial culture", "gram stain"],
        "relevant_imaging_findings": [],
        "lab_interpretation_support": {"POSITIVE", "ABNORMAL"},
        "key_missing_modalities": ["Bacterial wound culture if refractory"],
    },
    "Fungal infection": {
        "body_system": "Dermatological",
        "relevant_regions": ["skin", "foot", "toes", "hand"],
        "relevant_lab_tests": ["koh mount", "fungal culture", "skin scraping"],
        "relevant_imaging_findings": [],
        "lab_interpretation_support": {"POSITIVE", "ABNORMAL"},
        "key_missing_modalities": ["KOH direct mount / Fungal microscopy"],
    },
    "allergy": {
        "body_system": "Allergy & Immunology",
        "relevant_regions": ["chest", "sinuses", "skin"],
        "relevant_lab_tests": [
            "ige", "total ige", "eosinophil", "eosinophils", "absolute eosinophil count", "aec",
        ],
        "relevant_imaging_findings": [],
        "lab_interpretation_support": {"HIGH", "ABNORMAL", "ELEVATED"},
        "key_missing_modalities": ["Allergen-specific IgE testing / Skin prick tests"],
    },
    "drug reaction": {
        "body_system": "Allergy & Dermatology",
        "relevant_regions": ["skin"],
        "relevant_lab_tests": ["eosinophil", "eosinophils", "aec", "lft", "creatinine", "cbc"],
        "relevant_imaging_findings": [],
        "lab_interpretation_support": {"HIGH", "ABNORMAL", "ELEVATED"},
        "key_missing_modalities": ["Medication history review / CBC with absolute eosinophil count"],
    },
    "diabetes": {
        "body_system": "Endocrinology / Metabolic",
        "relevant_regions": [],
        "relevant_lab_tests": [
            "glucose", "blood glucose", "fasting glucose", "random glucose", "postprandial glucose",
            "blood sugar", "hba1c", "glycated hemoglobin", "urine glucose", "ketone", "urine ketones",
        ],
        "relevant_imaging_findings": [],
        "lab_interpretation_support": {"HIGH", "ABNORMAL", "ELEVATED", "POSITIVE"},
        "key_missing_modalities": ["Fasting blood glucose", "HbA1c"],
    },
    "Hypertension": {
        "body_system": "Cardiology",
        "relevant_regions": ["chest", "thorax"],
        "relevant_lab_tests": [
            "creatinine", "serum creatinine", "bun", "urea", "potassium",
            "lipid profile", "cholesterol", "triglycerides", "urine protein", "albuminuria",
        ],
        "relevant_imaging_findings": [
            "cardiomegaly", "enlarged cardiac silhouette", "aortic unfolding",
        ],
        "lab_interpretation_support": {"HIGH", "ABNORMAL"},
        "key_missing_modalities": ["Serial blood pressure monitoring", "Serum creatinine & electrolytes", "Electrocardiogram (ECG)"],
    },
    "Jaundice": {
        "body_system": "Hepatobiliary",
        "relevant_regions": ["abdomen", "pelvis"],
        "relevant_lab_tests": [
            "bilirubin", "total bilirubin", "direct bilirubin", "indirect bilirubin",
            "ast", "sgot", "alt", "sgpt", "alp", "alkaline phosphatase", "ggt", "albumin",
        ],
        "relevant_imaging_findings": [
            "biliary dilatation", "gallstones", "hepatomegaly", "cholelithiasis",
        ],
        "lab_interpretation_support": {"HIGH", "ABNORMAL", "ELEVATED"},
        "key_missing_modalities": ["Comprehensive Liver Function Tests (LFT)", "Abdominal ultrasound"],
    },
    "Varicose Veins": {
        "body_system": "Vascular",
        "relevant_regions": ["femur", "lower leg", "ankle", "foot"],
        "relevant_lab_tests": ["d dimer", "d-dimer"],
        "relevant_imaging_findings": [
            "venous reflux", "incompetent perforators", "venous insufficiency",
        ],
        "lab_interpretation_support": set(),
        "key_missing_modalities": ["Lower extremity Venous Duplex Doppler ultrasound"],
    },
    "Dimorphic Hemorrhoids": {
        "body_system": "Proctology / GI",
        "relevant_regions": ["pelvis"],
        "relevant_lab_tests": ["hemoglobin", "hb", "hematocrit", "cbc", "stool occult blood"],
        "relevant_imaging_findings": [],
        "lab_interpretation_support": {"LOW", "ABNORMAL", "POSITIVE"},
        "key_missing_modalities": ["Proctoscopy / Anoscopy clinical examination", "Complete Blood Count to monitor blood loss"],
    },
}


def _matches_any_keyword(text: str, keywords: list[str]) -> bool:
    """Return True if any keyword phrase matches in the normalized text."""
    norm_text = _normalize(text)
    if not norm_text:
        return False
    for kw in keywords:
        norm_kw = _normalize(kw)
        if norm_kw and (norm_kw == norm_text or norm_kw in norm_text):
            return True
    return False


def _get_modality_profile(condition_name: str) -> dict[str, Any]:
    """Retrieve condition modality profile using case-insensitive matching."""
    norm_name = _normalize(condition_name)
    for name, profile in CONDITION_MODALITY_PROFILES.items():
        if _normalize(name) == norm_name:
            return profile
    return {
        "body_system": "General",
        "relevant_regions": [],
        "relevant_lab_tests": [],
        "relevant_imaging_findings": [],
        "lab_interpretation_support": set(),
        "key_missing_modalities": [],
    }


def _reconcile_single_candidate(
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    source_records: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one genuine candidate against actual multimodal evidence.

    Preserves original model score and rank. Categorizes findings into
    candidate-specific supporting, contradictory, reassuring, contextual,
    and missing evidence with source attribution.
    """
    reconciled = dict(candidate)
    condition_name = str(candidate.get("condition") or "").strip()
    profile = _get_modality_profile(condition_name)

    supporting_evidence: list[dict[str, Any]] = []
    contradictory_evidence: list[dict[str, Any]] = []
    reassuring_evidence: list[dict[str, Any]] = []
    contextual_evidence: list[dict[str, Any]] = []
    missing_information: list[str] = list(candidate.get("unassessed_symptoms", []))
    unassessed_information: list[str] = list(candidate.get("unknown_or_missing_information", []))
    unavailable_or_unverified: list[dict[str, Any]] = []

    seen_keys: set[tuple[str, str, str]] = set()

    # 1. Structured Symptom Evidence
    for sym_name in candidate.get("supporting_symptoms", []):
        key = ("symptom_nlp", "support", sym_name.lower())
        if key not in seen_keys:
            seen_keys.add(key)
            supporting_evidence.append({
                "source_type": "symptom_nlp",
                "source_id": f"sym_supp_{_normalize(sym_name)}",
                "source_file": None,
                "finding": sym_name,
                "value": sym_name,
                "unit": None,
                "reference_range": None,
                "finding_state": "PRESENT",
                "reason": f"Active reported symptom matches {condition_name} typical presentation profile.",
                "verification_status": "REPORTED",
                "rule_identifier": "symptom_profile_match",
            })

    for contra_name in candidate.get("contradictory_findings", []):
        key = ("symptom_nlp", "contra", contra_name.lower())
        if key not in seen_keys:
            seen_keys.add(key)
            contradictory_evidence.append({
                "source_type": "symptom_nlp",
                "source_id": f"sym_contra_{_normalize(contra_name)}",
                "source_file": None,
                "finding": contra_name,
                "value": contra_name,
                "unit": None,
                "reference_range": None,
                "finding_state": "ABSENT",
                "reason": f"Explicitly denied finding; typical presentation of {condition_name} commonly features this symptom.",
                "verification_status": "REPORTED",
                "rule_identifier": "symptom_negation_conflict",
            })

    # 2. Medical / Laboratory Reports Evidence
    report_meta = evidence.get("report", {}) if isinstance(evidence, dict) else {}
    lab_findings: list[dict[str, Any]] = []
    if isinstance(report_meta, dict):
        lab_findings.extend(report_meta.get("abnormal_findings", []) or [])
        lab_findings.extend(report_meta.get("normal_findings", []) or [])
        lab_findings.extend(report_meta.get("unknown_range_findings", []) or [])

    # Also check source_records for directly passed report findings
    if isinstance(source_records, dict):
        for med_rep in source_records.get("medical_reports", []) or []:
            if isinstance(med_rep, dict):
                structured = med_rep.get("structured_findings")
                if isinstance(structured, dict):
                    lab_findings.extend(structured.get("abnormal_findings", []) or [])
                    lab_findings.extend(structured.get("normal_findings", []) or [])

    relevant_lab_keywords = profile.get("relevant_lab_tests", [])
    lab_support_set = profile.get("lab_interpretation_support", {"HIGH", "ABNORMAL", "POSITIVE", "ELEVATED"})

    matched_relevant_labs = False
    for lab in lab_findings:
        if not isinstance(lab, dict):
            continue
        test_name = str(lab.get("test_name") or lab.get("original_test_name") or "").strip()
        if not test_name:
            continue
        val_str = f"{lab.get('value')} {lab.get('unit') or ''}".strip()
        status_raw = str(lab.get("status") or lab.get("interpretation") or "UNKNOWN").strip().upper()
        rep_id = str(lab.get("report_id") or lab.get("source_id") or lab.get("id") or "lab_report")
        src_file = lab.get("source_file") or lab.get("file_name")
        unit = lab.get("unit")
        ref_range = lab.get("reference_range")

        is_relevant = _matches_any_keyword(test_name, relevant_lab_keywords)
        if is_relevant:
            matched_relevant_labs = True
            if status_raw in lab_support_set or "ABNORMAL" in status_raw or "HIGH" in status_raw:
                key = ("laboratory_report", "support", rep_id + "_" + test_name)
                if key not in seen_keys:
                    seen_keys.add(key)
                    supporting_evidence.append({
                        "source_type": "laboratory_report",
                        "source_id": rep_id,
                        "source_file": src_file,
                        "finding": f"{test_name}: {val_str}",
                        "value": str(lab.get("value") or ""),
                        "unit": unit,
                        "reference_range": ref_range,
                        "finding_state": status_raw,
                        "reason": f"Out-of-range laboratory finding ({status_raw}) corroborating {condition_name} pathophysiology.",
                        "verification_status": "CONFIRMED_EXTRACTION",
                        "rule_identifier": "relevant_lab_elevation_support",
                    })
            elif status_raw in {"NORMAL", "NEGATIVE", "WITHIN_RANGE"}:
                key = ("laboratory_report", "reassuring", rep_id + "_" + test_name)
                if key not in seen_keys:
                    seen_keys.add(key)
                    reassuring_evidence.append({
                        "source_type": "laboratory_report",
                        "source_id": rep_id,
                        "source_file": src_file,
                        "finding": f"{test_name}: {val_str}",
                        "value": str(lab.get("value") or ""),
                        "unit": unit,
                        "reference_range": ref_range,
                        "finding_state": "NORMAL",
                        "reason": f"Laboratory parameter within standard reference limits for {condition_name}.",
                        "verification_status": "CONFIRMED_EXTRACTION",
                        "rule_identifier": "relevant_lab_normal_reassuring",
                    })
            else:
                key = ("laboratory_report", "contextual", rep_id + "_" + test_name)
                if key not in seen_keys:
                    seen_keys.add(key)
                    contextual_evidence.append({
                        "source_type": "laboratory_report",
                        "source_id": rep_id,
                        "source_file": src_file,
                        "finding": f"{test_name}: {val_str}",
                        "value": str(lab.get("value") or ""),
                        "unit": unit,
                        "reference_range": ref_range,
                        "finding_state": status_raw,
                        "reason": f"Laboratory test evaluated in context of {condition_name}.",
                        "verification_status": "CONFIRMED_EXTRACTION",
                        "rule_identifier": "relevant_lab_other_status",
                    })
        else:
            # Unrelated lab test: keep in contextual evidence without claiming support/contradiction
            key = ("laboratory_report", "contextual", rep_id + "_" + test_name)
            if key not in seen_keys:
                seen_keys.add(key)
                contextual_evidence.append({
                    "source_type": "laboratory_report",
                    "source_id": rep_id,
                    "source_file": src_file,
                    "finding": f"{test_name}: {val_str}",
                    "value": str(lab.get("value") or ""),
                    "unit": unit,
                    "reference_range": ref_range,
                    "finding_state": status_raw,
                    "reason": f"General assessment laboratory parameter; not a specific marker for {condition_name}.",
                    "verification_status": "CONFIRMED_EXTRACTION",
                    "rule_identifier": "unrelated_general_lab",
                })

    # 3. Radiology Reports Evidence
    xray_meta = evidence.get("xray", {}) if isinstance(evidence, dict) else {}
    rad_reports: list[dict[str, Any]] = []
    if isinstance(xray_meta, dict):
        rad_reports.extend(xray_meta.get("reports", []) or [])
    if isinstance(source_records, dict):
        rad_reports.extend(source_records.get("xray_reports", []) or [])

    relevant_regions = profile.get("relevant_regions", [])
    relevant_imaging_kw = profile.get("relevant_imaging_findings", [])

    matched_relevant_rad = False
    for rep in rad_reports:
        if not isinstance(rep, dict):
            continue
        rep_id = str(rep.get("id") or rep.get("report_id") or "radiology_report")
        src_file = rep.get("file_name") or rep.get("source_file")
        rep_region = str(rep.get("region") or rep.get("body_region") or rep.get("exam_region") or "").strip().lower()

        is_region_relevant = bool(relevant_regions) and any(
            _matches_any_keyword(rep_region, [r]) or _phrase_matches(r, rep_region)
            for r in relevant_regions
        )

        # Collect narrative and qualitative statements
        qual_findings = rep.get("qualitative_findings", []) or []
        narr_findings = rep.get("narrative_findings", []) or []
        finding_texts = [
            str(f.get("finding") or f.get("text") or f) if isinstance(f, dict) else str(f)
            for f in qual_findings + narr_findings
        ]

        if is_region_relevant:
            matched_relevant_rad = True
            finding_matched = False
            for ft in finding_texts:
                if relevant_imaging_kw and _matches_any_keyword(ft, relevant_imaging_kw):
                    finding_matched = True
                    key = ("radiology_report", "support", rep_id + "_" + ft)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        supporting_evidence.append({
                            "source_type": "radiology_report",
                            "source_id": rep_id,
                            "source_file": src_file,
                            "finding": f"Radiology report ({rep_region.title()}): {ft}",
                            "value": ft,
                            "unit": None,
                            "reference_range": None,
                            "finding_state": "PRESENT",
                            "reason": f"Radiology report documents radiographic findings consistent with {condition_name}.",
                            "verification_status": "CONFIRMED_REPORT",
                            "rule_identifier": "relevant_radiology_concordance",
                        })
            if not finding_matched:
                key = ("radiology_report", "contextual", rep_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    contextual_evidence.append({
                        "source_type": "radiology_report",
                        "source_id": rep_id,
                        "source_file": src_file,
                        "finding": f"Radiology report ({rep_region.title()})",
                        "value": "Documented examination without focal findings of " + condition_name,
                        "unit": None,
                        "reference_range": None,
                        "finding_state": "REPORTED",
                        "reason": f"Relevant body region ({rep_region.title()}) imaged without focal confirmation of {condition_name}.",
                        "verification_status": "CONFIRMED_REPORT",
                        "rule_identifier": "relevant_radiology_non_focal",
                    })
        else:
            # Region is unrelated
            reg_display = rep_region.title() if rep_region else "Report"
            key = ("radiology_report", "unrelated", rep_id)
            if key not in seen_keys:
                seen_keys.add(key)
                contextual_evidence.append({
                    "source_type": "radiology_report",
                    "source_id": rep_id,
                    "source_file": src_file,
                    "finding": f"Radiology report ({reg_display})",
                    "value": None,
                    "unit": None,
                    "reference_range": None,
                    "finding_state": "CONTEXTUAL",
                    "reason": f"Reported examination region ({reg_display}) is not anatomically relevant to {condition_name}.",
                    "verification_status": "CONFIRMED_REPORT",
                    "rule_identifier": "unrelated_radiology_region",
                })

    # 4. Existing X-ray Images Evidence
    xray_images: list[dict[str, Any]] = []
    if isinstance(xray_meta, dict):
        xray_images.extend(xray_meta.get("images", []) or [])
    if isinstance(source_records, dict):
        xray_images.extend(source_records.get("xray_images", []) or [])

    matched_eligible_xray = False
    for img in xray_images:
        if not isinstance(img, dict):
            continue
        img_id = str(img.get("id") or img.get("xray_result_id") or "xray_image")
        src_file = img.get("file_name")
        status = str(img.get("status") or "").strip().upper()
        is_verified = (
            bool(img.get("anatomy_independently_verified"))
            and str(img.get("anatomy_verification_status") or "").strip().lower() == "completed"
        )
        has_mismatch = bool(img.get("has_region_mismatch")) or status == "MISMATCH"
        img_region = str(img.get("region") or "").strip().lower()

        if not is_verified or status != "COMPLETED" or has_mismatch:
            key = ("xray_image", "unverified", img_id)
            if key not in seen_keys:
                seen_keys.add(key)
                reason_msg = (
                    "Anatomy has not been independently verified; output is excluded from candidate support."
                    if not is_verified
                    else ("Region mismatch detected between image and report; excluded from candidate support."
                          if has_mismatch else "Image analysis incomplete or unavailable.")
                )
                unavailable_or_unverified.append({
                    "source_type": "xray_image",
                    "source_id": img_id,
                    "source_file": src_file,
                    "finding": f"X-ray Image ({img_region.title() if img_region else 'Unspecified'})",
                    "value": None,
                    "unit": None,
                    "reference_range": None,
                    "finding_state": "ANATOMY_UNVERIFIED" if not is_verified else ("REGION_MISMATCH" if has_mismatch else status),
                    "reason": reason_msg,
                    "verification_status": "EXCLUDED",
                    "rule_identifier": "unverified_xray_excluded",
                })
        else:
            # Verified and completed X-ray
            is_region_relevant = bool(relevant_regions) and any(
                _matches_any_keyword(img_region, [r]) or _phrase_matches(r, img_region)
                for r in relevant_regions
            )
            pred_str = str(img.get("prediction") or "").strip()
            findings_raw = img.get("findings") or []
            finding_labels = [
                str(f.get("finding") or f.get("label") or f) if isinstance(f, dict) else str(f)
                for f in findings_raw
            ]

            if is_region_relevant:
                # Check for concordance with candidate imaging findings
                is_concordant = False
                for f_label in [pred_str] + finding_labels:
                    if relevant_imaging_kw and _matches_any_keyword(f_label, relevant_imaging_kw):
                        is_concordant = True
                        break

                if is_concordant:
                    matched_eligible_xray = True
                    key = ("xray_image", "support", img_id)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        supporting_evidence.append({
                            "source_type": "xray_image",
                            "source_id": img_id,
                            "source_file": src_file,
                            "finding": f"Verified X-ray ({img_region.title()}): {pred_str or 'Concordant radiographic pattern'}",
                            "value": pred_str,
                            "unit": None,
                            "reference_range": None,
                            "finding_state": "COMPLETED",
                            "reason": f"Verified image analysis identified radiographic pattern consistent with {condition_name}.",
                            "verification_status": "VERIFIED_ELIGIBLE",
                            "rule_identifier": "verified_xray_concordance",
                        })
                else:
                    key = ("xray_image", "reassuring", img_id)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        reassuring_evidence.append({
                            "source_type": "xray_image",
                            "source_id": img_id,
                            "source_file": src_file,
                            "finding": f"Verified X-ray ({img_region.title()}): No specific {condition_name} pattern identified",
                            "value": pred_str or "No focal acute pattern",
                            "unit": None,
                            "reference_range": None,
                            "finding_state": "COMPLETED",
                            "reason": f"Verified {img_region.title()} radiograph does not identify focal patterns of {condition_name}.",
                            "verification_status": "VERIFIED_ELIGIBLE",
                            "rule_identifier": "verified_xray_non_focal",
                        })
            else:
                # Verified X-ray of unrelated body region
                key = ("xray_image", "contextual", img_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    contextual_evidence.append({
                        "source_type": "xray_image",
                        "source_id": img_id,
                        "source_file": src_file,
                        "finding": f"Verified X-ray ({img_region.title()})",
                        "value": pred_str,
                        "unit": None,
                        "reference_range": None,
                        "finding_state": "COMPLETED",
                        "reason": f"Imaged region ({img_region.title()}) is not anatomically relevant to {condition_name}.",
                        "verification_status": "VERIFIED_ELIGIBLE",
                        "rule_identifier": "unrelated_xray_region",
                    })

    # 5. Key Missing Modalities
    if not matched_relevant_labs:
        for missing_mod in profile.get("key_missing_modalities", []):
            if missing_mod not in missing_information:
                missing_information.append(missing_mod)

    # 6. Dynamic Evidence Badges
    badges: list[str] = []
    has_sym_support = any(item["source_type"] == "symptom_nlp" for item in supporting_evidence)
    has_lab_support = any(item["source_type"] == "laboratory_report" for item in supporting_evidence)
    has_rad_support = any(item["source_type"] == "radiology_report" for item in supporting_evidence)
    has_xray_support = any(item["source_type"] == "xray_image" for item in supporting_evidence)

    has_sym_contra = any(item["source_type"] == "symptom_nlp" for item in contradictory_evidence)
    has_lab_contra = any(item["source_type"] == "laboratory_report" for item in contradictory_evidence)

    if has_sym_support:
        badges.append("Supported by reported symptoms")
    if has_lab_support:
        badges.append("Relevant laboratory finding available")
    if has_rad_support:
        badges.append("Relevant radiology finding available")
    if has_xray_support:
        badges.append("Existing eligible X-ray result available")

    if has_sym_contra:
        badges.append("Contradictory symptom reported")
    if has_lab_contra:
        badges.append("Contradictory laboratory finding")

    if not (has_lab_support or has_rad_support or has_xray_support):
        badges.append("No additional corroborating evidence")

    if not supporting_evidence:
        badges.append("Evidence insufficient")

    # 7. Concise Evidence Summary Sentence
    supp_counts: list[str] = []
    sym_count = sum(1 for item in supporting_evidence if item["source_type"] == "symptom_nlp")
    lab_count = sum(1 for item in supporting_evidence if item["source_type"] == "laboratory_report")
    rad_count = sum(1 for item in supporting_evidence if item["source_type"] == "radiology_report")
    xray_count = sum(1 for item in supporting_evidence if item["source_type"] == "xray_image")

    if sym_count:
        supp_counts.append(f"{sym_count} symptom(s)")
    if lab_count:
        supp_counts.append(f"{lab_count} laboratory finding(s)")
    if rad_count:
        supp_counts.append(f"{rad_count} radiology report statement(s)")
    if xray_count:
        supp_counts.append(f"{xray_count} eligible X-ray result(s)")

    if supp_counts:
        summary_text = f"Supported by {', '.join(supp_counts)}."
    else:
        summary_text = "No direct corroborating evidence identified for this candidate."

    if contradictory_evidence:
        summary_text += f" {len(contradictory_evidence)} finding(s) contradict typical presentation."
    if reassuring_evidence:
        summary_text += f" {len(reassuring_evidence)} reassuring finding(s) identified."

    reconciled["supporting_evidence"] = supporting_evidence
    reconciled["contradictory_evidence"] = contradictory_evidence
    reconciled["reassuring_evidence"] = reassuring_evidence
    reconciled["contextual_evidence"] = contextual_evidence
    reconciled["missing_information"] = missing_information[:4]
    reconciled["unassessed_information"] = unassessed_information[:4]
    reconciled["unavailable_or_unverified_information"] = unavailable_or_unverified
    reconciled["evidence_badges"] = badges
    reconciled["evidence_summary"] = summary_text
    reconciled["evidence_reconciliation_status"] = "COMPLETED"

    return reconciled


def reconcile_candidate_evidence(
    conditions: list[dict[str, Any]] | dict[str, Any],
    evidence: dict[str, Any],
    source_records: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Reconcile ML condition candidates with multimodal assessment evidence.

    Accepts a single candidate dictionary or a list of candidate dictionaries.
    Preserves original model score, rank, and class identities.
    """
    if isinstance(conditions, dict):
        return _reconcile_single_candidate(conditions, evidence, source_records)
    if isinstance(conditions, list):
        return [
            _reconcile_single_candidate(item, evidence, source_records)
            if isinstance(item, dict) else item
            for item in conditions
        ]
    return conditions

