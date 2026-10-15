"""Phase 3 Condition Prediction, Evidence Integrity, and Candidate Filtering Audit Tests.

Validates:
1. Feature purity: Raw text negations cannot contaminate ML feature tokens.
2. Evidence integrity: Every returned candidate MUST have verified supporting symptoms.
3. Temporal scoping: Historical findings are excluded from active acute condition prediction.
4. Feature normalization: Duplicate symptoms are deduplicated without frequency inflation.
5. Conflict transparency: Contradictory findings are never silently discarded.
6. Heavy contradiction disqualification: Candidates with cardinal contradictory absences are excluded.
7. Triage independence: ML prediction failure/absence never overrides deterministic triage.
8. Uncertainty state distinction: Completed, insufficient_evidence, unsupported_condition, unavailable.
"""

from __future__ import annotations

import pytest
from app.ai.condition_prediction import get_condition_prediction_service
from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.services.assessment_service import generate_assessment_feedback


def test_defect_1_raw_text_negations_cannot_contaminate_feature_tokens():
    """Negated symptoms in raw_text must never be injected as positive features into ML prediction."""
    svc = get_condition_prediction_service()
    if not svc.is_available():
        pytest.skip("Model not available on disk")

    syms_present = [{"symptom": "headache", "state": "PRESENT"}]
    syms_absent = [
        {"symptom": "fever", "state": "ABSENT"},
        {"symptom": "cough", "state": "ABSENT"},
        {"symptom": "chest pain", "state": "ABSENT"},
    ]
    raw_text = "I have a headache. No fever, no cough, no chest pain, no difficulty breathing."

    res = svc.predict_conditions(
        symptoms_present=syms_present,
        symptoms_absent=syms_absent,
        symptoms_unknown=[],
        raw_text=raw_text,
    )
    # The returned candidate conditions must NOT be conditions driven by fever/cough/chest pain
    for c in res.get("conditions", []):
        # Every candidate must be supported by the actual present symptoms (headache)
        assert any("headache" in s for s in c["supporting_symptoms"])


def test_defect_2_no_candidates_with_zero_supporting_symptoms():
    """Every returned candidate condition MUST have at least one verified supporting symptom in patient's presentation."""
    svc = get_condition_prediction_service()
    if not svc.is_available():
        pytest.skip("Model not available on disk")

    syms_present = [{"symptom": "headache", "state": "PRESENT"}]
    res = svc.predict_conditions(
        symptoms_present=syms_present,
        symptoms_absent=[],
        symptoms_unknown=[],
    )
    for c in res.get("conditions", []):
        assert len(c["supporting_symptoms"]) > 0, (
            f"Candidate '{c['condition']}' returned with zero supporting symptoms! Rationale: {c.get('clinical_rationale')}"
        )


def test_defect_3_historical_symptoms_excluded_from_condition_prediction():
    """Historical symptoms (type='HISTORICAL') must not be treated as active present symptoms for acute disease prediction."""
    svc = get_condition_prediction_service()
    if not svc.is_available():
        pytest.skip("Model not available on disk")

    # Only a historical symptom passed
    syms_present = [{"symptom": "skin rash", "state": "PRESENT", "type": "HISTORICAL"}]
    res = svc.predict_conditions(
        symptoms_present=syms_present,
        symptoms_absent=[],
        symptoms_unknown=[],
    )
    # Since there are no acute present symptoms, it must return insufficient_evidence
    assert res["status"] == "insufficient_evidence"
    assert len(res["conditions"]) == 0


def test_defect_4_duplicate_symptoms_deduplicated_without_frequency_inflation():
    """Duplicate symptoms must be deduplicated before feature construction."""
    svc = get_condition_prediction_service()
    if not svc.is_available():
        pytest.skip("Model not available on disk")

    syms_present_single = [{"symptom": "cough", "state": "PRESENT"}]
    syms_present_duplicate = [
        {"symptom": "cough", "state": "PRESENT"},
        {"symptom": "cough", "state": "PRESENT"},
        {"symptom": "cough", "state": "PRESENT"},
    ]

    res_single = svc.predict_conditions(syms_present_single, [], [])
    res_dup = svc.predict_conditions(syms_present_duplicate, [], [])

    # The probabilities and top candidate should be identical
    if res_single.get("conditions") and res_dup.get("conditions"):
        assert res_single["conditions"][0]["condition"] == res_dup["conditions"][0]["condition"]
        assert abs(res_single["conditions"][0]["probability_score"] - res_dup["conditions"][0]["probability_score"]) < 1e-4


def test_defect_5_contradictory_evidence_surfaced_in_feedback():
    """Candidate-specific contradictory findings must not be silently discarded; they must appear in feedback contradictory_evidence."""
    svc = get_condition_prediction_service()
    if not svc.is_available():
        pytest.skip("Model not available on disk")

    # Patient has cough, but shortness of breath and wheezing are explicitly ABSENT
    syms_present = [{"symptom": "cough", "state": "PRESENT", "domain": "respiratory"}]
    syms_absent = [
        {"symptom": "shortness of breath", "state": "ABSENT", "domain": "respiratory"},
        {"symptom": "wheezing", "state": "ABSENT", "domain": "respiratory"},
    ]
    evidence = combine_evidence(symptoms=syms_present + syms_absent)
    triage = apply_triage(evidence)

    condition_predictions = {
        "status": "completed",
        "conditions": [
            {
                "rank": 1,
                "condition": "Bronchial Asthma",
                "probability_score": 0.45,
                "confidence_tier": "Possible Alignment",
                "supporting_symptoms": ["cough"],
                "contradictory_findings": ["shortness of breath", "wheezing"],
                "suggested_specialty": "Pulmonology",
            }
        ],
    }

    feedback = generate_assessment_feedback(
        evidence=evidence,
        triage_result=triage,
        specialty="Pulmonology",
        condition_predictions=condition_predictions,
    )

    # Contradictory evidence MUST contain mention of the conflict
    contra = feedback.get("contradictory_evidence", [])
    assert any("Bronchial Asthma" in item and ("shortness of breath" in item or "wheezing" in item) for item in contra), (
        f"Contradictory findings silently discarded! Got contradictory_evidence: {contra}"
    )


def test_defect_6_heavy_contradictions_disqualify_candidate():
    """When a candidate's cardinal presentation is heavily contradicted by absent symptoms, it is disqualified."""
    svc = get_condition_prediction_service()
    if not svc.is_available():
        pytest.skip("Model not available on disk")

    # Patient has mild headache, but denies fever, chills, and sweats
    syms_present = [{"symptom": "headache", "state": "PRESENT"}]
    syms_absent = [
        {"symptom": "fever", "state": "ABSENT"},
        {"symptom": "chills", "state": "ABSENT"},
        {"symptom": "sweating", "state": "ABSENT"},
    ]

    res = svc.predict_conditions(
        symptoms_present=syms_present,
        symptoms_absent=syms_absent,
        symptoms_unknown=[],
    )
    # Malaria requires fever, chills, sweating, headache. With 3 cardinal symptoms absent, it should not be accepted
    cond_names = [c["condition"] for c in res.get("conditions", [])]
    assert "Malaria" not in cond_names


def test_triage_independence_maintained():
    """Emergency triage is 100% deterministic and operates independently of ML condition prediction status."""
    symptoms = [{"symptom": "chest pain at rest", "state": "PRESENT", "severity": "SEVERE"}]
    evidence = combine_evidence(symptoms=symptoms)
    triage = apply_triage(evidence)
    assert triage["pathway"] == "EMERGENCY"

    # Even if condition predictions failed / unavailable
    failed_predictions = {
        "status": "unavailable",
        "conditions": [],
        "message": "Model not loaded",
    }
    feedback = generate_assessment_feedback(
        evidence=evidence,
        triage_result=triage,
        specialty="Cardiology",
        condition_predictions=failed_predictions,
    )
    assert feedback["triage_level"]["pathway"] == "EMERGENCY"


def test_uncertainty_states_distinct():
    """System cleanly distinguishes completed, insufficient_evidence, unsupported_condition, and unavailable."""
    svc = get_condition_prediction_service()

    # Empty symptoms -> insufficient_evidence
    res_empty = svc.predict_conditions([], [], [])
    assert res_empty["status"] == "insufficient_evidence"

    # Out of domain -> unsupported_condition
    res_ood = svc.predict_conditions(
        [{"symptom": "stubbed toe", "body_area": "foot"}],
        [],
        [],
    )
    assert res_ood["status"] == "unsupported_condition"

    # In distribution -> completed
    res_valid = svc.predict_conditions(
        [{"symptom": "cough"}, {"symptom": "runny nose"}, {"symptom": "sneezing"}],
        [],
        [],
    )
    assert res_valid["status"] == "completed"
