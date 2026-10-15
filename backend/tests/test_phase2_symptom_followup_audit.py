"""Phase 2 Symptom Follow-Up Audit and Regression Tests for MediSense AI.

Validates fixes for:
- Problem 1: Follow-up duration isolation and targeting.
- Problem 2: Severity assignment isolation to relevant pain complaint.
- Problem 3: Whole-answer negation precedence and binary state accuracy.
- Downstream evidence and assessment consistency.
"""

from __future__ import annotations

import pytest
from app.ai.followup_engine import apply_followup_answer, suggest_followups
from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.services.assessment_service import generate_assessment_feedback


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 1: DURATION OVER-ASSIGNMENT & ISOLATION
# ─────────────────────────────────────────────────────────────────────────────

def test_problem_1_headache_and_nausea_duration_isolation():
    """Verify duration answer for headache does not contaminate unrelated nausea."""
    input_syms = [
        {"symptom": "headache", "state": "PRESENT", "duration": None, "domain": "neurological"},
        {"symptom": "nausea", "state": "PRESENT", "duration": None, "domain": "gi"},
    ]

    # Headache-specific duration question answer
    updated = apply_followup_answer(input_syms, "q_duration", "2 days", target_symptom="headache")
    headache_s = next(s for s in updated if s["symptom"] == "headache")
    nausea_s = next(s for s in updated if s["symptom"] == "nausea")

    assert headache_s["duration"] == "2 days"
    assert nausea_s.get("duration") is None


def test_problem_1_generic_duration_targets_primary_complaint_not_associated():
    """When target_symptom is omitted, q_duration targets primary chief complaint without contaminating secondary symptoms."""
    input_syms = [
        {"symptom": "headache", "state": "PRESENT", "duration": None, "domain": "neurological"},
        {"symptom": "nausea", "state": "PRESENT", "duration": None, "domain": "gi"},
    ]

    updated = apply_followup_answer(input_syms, "q_duration", "2 days")
    headache_s = next(s for s in updated if s["symptom"] == "headache")
    nausea_s = next(s for s in updated if s["symptom"] == "nausea")

    assert headache_s["duration"] == "2 days"
    assert nausea_s.get("duration") is None


def test_problem_1_explicit_group_scope_updates_all_when_specified():
    """When scope is explicitly specified as 'all', all present symptoms receive duration with recorded scope."""
    input_syms = [
        {"symptom": "headache", "state": "PRESENT", "duration": None},
        {"symptom": "nausea", "state": "PRESENT", "duration": None},
    ]

    updated = apply_followup_answer(input_syms, "q_duration", "2 days", scope="all")
    for s in updated:
        assert s["duration"] == "2 days"
        assert s.get("duration_scope") == "all"


def test_problem_1_single_symptom_duration_assignment():
    """Single present symptom receives duration cleanly."""
    input_syms = [{"symptom": "cough", "state": "PRESENT", "duration": None, "domain": "respiratory"}]
    updated = apply_followup_answer(input_syms, "q_duration", "5 days")
    assert updated[0]["duration"] == "5 days"


def test_problem_1_encoded_target_in_question_id():
    """Target encoded in question ID (e.g. q_duration:headache or q_duration_headache) is respected."""
    input_syms = [
        {"symptom": "headache", "state": "PRESENT", "duration": None},
        {"symptom": "nausea", "state": "PRESENT", "duration": None},
    ]

    updated = apply_followup_answer(input_syms, "q_duration:headache", "3 days")
    headache_s = next(s for s in updated if s["symptom"] == "headache")
    nausea_s = next(s for s in updated if s["symptom"] == "nausea")

    assert headache_s["duration"] == "3 days"
    assert nausea_s.get("duration") is None


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 2: SEVERITY OVER-ASSIGNMENT & ISOLATION
# ─────────────────────────────────────────────────────────────────────────────

def test_problem_2_severe_headache_with_nausea():
    """Severe headache with nausea: only headache becomes SEVERE, nausea unaffected."""
    input_syms = [
        {"symptom": "headache", "state": "PRESENT", "domain": "neurological"},
        {"symptom": "nausea", "state": "PRESENT", "domain": "gi"},
    ]

    updated = apply_followup_answer(input_syms, "q_severity", "Severe")
    headache_s = next(s for s in updated if s["symptom"] == "headache")
    nausea_s = next(s for s in updated if s["symptom"] == "nausea")

    assert headache_s.get("severity") == "SEVERE"
    assert nausea_s.get("severity") in {None, "UNKNOWN"}
    assert nausea_s.get("severity") != "SEVERE"


def test_problem_2_mild_abdominal_pain_with_vomiting():
    """Mild abdominal pain with vomiting: only abdominal pain receives selected severity."""
    input_syms = [
        {"symptom": "abdominal pain", "state": "PRESENT", "domain": "gi"},
        {"symptom": "vomiting", "state": "PRESENT", "domain": "gi"},
    ]

    updated = apply_followup_answer(input_syms, "q_severity", "Mild", target_symptom="abdominal pain")
    abdo_s = next(s for s in updated if s["symptom"] == "abdominal pain")
    vomit_s = next(s for s in updated if s["symptom"] == "vomiting")

    assert abdo_s.get("severity") == "MILD"
    assert vomit_s.get("severity") in {None, "UNKNOWN"}
    assert vomit_s.get("severity") != "MILD"


def test_problem_2_user_answers_not_sure_severity_remains_unknown():
    """User answers 'Not sure': severity is explicitly UNKNOWN, never invented or defaulted."""
    input_syms = [
        {"symptom": "headache", "state": "PRESENT", "severity": None},
    ]

    updated = apply_followup_answer(input_syms, "q_severity", "Not sure")
    assert updated[0].get("severity") == "UNKNOWN"


def test_problem_2_no_applicable_symptom_does_not_create_misleading_finding():
    """No applicable symptom exists: do not create a misleading generic severity finding."""
    # Scenario A: empty symptoms list
    updated_empty = apply_followup_answer([], "q_severity", "Severe")
    assert len(updated_empty) == 0
    assert not any(s.get("symptom") == "severity" for s in updated_empty)

    # Scenario B: only non-pain symptoms present (e.g. fatigue only)
    input_syms = [{"symptom": "fatigue", "state": "PRESENT", "domain": "general"}]
    updated_fatigue = apply_followup_answer(input_syms, "q_severity", "Severe")
    # Fatigue should not be assigned pain severity, and no dummy 'severity' symptom created
    assert not any(s.get("symptom") == "severity" for s in updated_fatigue)
    assert updated_fatigue[0].get("severity") in {None, "UNKNOWN"}


# ─────────────────────────────────────────────────────────────────────────────
# PROBLEM 3: NEGATIVE FOLLOW-UP ANSWERS & STATE ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "ans,expected_state",
    [
        ("No", "ABSENT"),
        ("No fall or trauma", "ABSENT"),
        ("I did not fall", "ABSENT"),
        ("I have not experienced this", "ABSENT"),
        ("Not sure", "UNKNOWN"),
        ("I don't know", "UNKNOWN"),
        ("Cannot tell", "UNKNOWN"),
        ("Yes, I fell", "PRESENT"),
        ("I experienced the symptom", "PRESENT"),
        ("Yes, fall or trauma", "PRESENT"),
    ],
)
def test_problem_3_injury_mechanism_answers(ans: str, expected_state: str):
    """Verify whole-answer negation precedence and state accuracy for q_injury_mechanism."""
    updated = apply_followup_answer([], "q_injury_mechanism", ans)
    assert len(updated) == 1
    item = updated[0]
    assert item["symptom"] == "fall/trauma"
    assert item["state"] == expected_state
    assert item["status"] == expected_state


@pytest.mark.parametrize(
    "ans,expected_state",
    [
        ("Normal movement", "ABSENT"),
        ("Reduced movement", "PRESENT"),
        ("Cannot move limb", "PRESENT"),
        ("Not sure", "UNKNOWN"),
        ("I can move normally without restriction", "ABSENT"),
        ("Unable to move my fingers", "PRESENT"),
    ],
)
def test_problem_3_limb_movement_answers(ans: str, expected_state: str):
    """Verify structured and natural language handling for functional limb movement."""
    updated = apply_followup_answer([], "q_limb_movement", ans)
    assert len(updated) == 1
    item = updated[0]
    assert item["symptom"] == "reduced movement"
    assert item["state"] == expected_state


def test_problem_3_ambiguous_answer_becomes_unknown_never_present():
    """Ambiguous or unsupported free text answers must become UNKNOWN, never PRESENT."""
    updated = apply_followup_answer([], "q_fever", "purple elephant")
    assert len(updated) == 1
    assert updated[0]["state"] == "UNKNOWN"
    assert updated[0]["status"] == "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# END-TO-END DOWNSTREAM PIPELINE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_severity_isolation_prevents_false_emergency_triage():
    """Verify that isolating severity to headache prevents nausea from triggering false multi-severe triage."""
    # Initial: patient has headache and nausea
    symptoms = [
        {"symptom": "headache", "state": "PRESENT", "domain": "neurological", "duration": "1 day"},
        {"symptom": "nausea", "state": "PRESENT", "domain": "gi", "duration": None},
    ]

    # Follow-up answer: headache severity is Moderate
    updated = apply_followup_answer(symptoms, "q_severity", "Moderate", target_symptom="headache")
    headache_s = next(s for s in updated if s["symptom"] == "headache")
    nausea_s = next(s for s in updated if s["symptom"] == "nausea")

    assert headache_s["severity"] == "MODERATE"
    assert nausea_s.get("severity") in {None, "UNKNOWN"}

    # Run evidence engine
    evidence = combine_evidence(symptoms=updated)
    present_list = evidence["symptoms"]["present"]
    headache_ev = next(s for s in present_list if s["symptom"] == "headache")
    nausea_ev = next(s for s in present_list if s["symptom"] == "nausea")

    assert headache_ev["severity"] == "MODERATE"
    assert nausea_ev["severity"] == "UNKNOWN"

    # Triage check: moderate headache + nausea with no red flags is CONSULTATION, never EMERGENCY
    triage = apply_triage(evidence)
    assert triage["pathway"] == "CONSULTATION"
    assert "rule_e3_severe_acute_presentation" not in triage["triggered_rules"]

    # Assessment feedback summary verification
    feedback = generate_assessment_feedback(evidence, triage, "Neurology")
    summary = feedback["assessment_summary"]
    # Headache should be described with moderate, but nausea should NOT have moderate/severe label
    assert "headache (moderate" in summary.lower()
    assert "nausea (moderate" not in summary.lower()
    assert "nausea (severe" not in summary.lower()


def test_pipeline_negated_fall_trauma_does_not_pollute_present_symptoms():
    """Answering 'No fall or trauma' must ensure fall/trauma is in absent, not present."""
    symptoms = [
        {"symptom": "wrist pain", "state": "PRESENT", "domain": "musculoskeletal", "duration": "2 days"},
    ]
    updated = apply_followup_answer(symptoms, "q_injury_mechanism", "No fall or trauma")

    evidence = combine_evidence(symptoms=updated)
    present_names = [s["symptom"] for s in evidence["symptoms"]["present"]]
    absent_names = [s["symptom"] for s in evidence["symptoms"]["absent"]]

    assert "wrist pain" in present_names
    assert "fall/trauma" not in present_names
    assert "fall/trauma" in absent_names
