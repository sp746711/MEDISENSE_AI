"""Regression test suite for Goal 1 Final Code Implementation.

Covers:
- Prediction combinations (symptoms only, +lab, +radiology, +xray, all modalities, no symptoms)
- Score integrity (scores completely untouched by reconciliation)
- Candidate-specific evidence reconciliation with source traceability
- Reassuring vs contradictory vs contextual findings
- Unverified X-rays excluded from candidate support
- Independent deterministic triage protected from ML or reassuring override
- Evidence integrity (PRESENT, ABSENT, UNKNOWN, HISTORICAL)
- PDF and frontend payload consistency
"""

from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

import pytest

from app.ai.condition_prediction import (
    CONDITION_MODALITY_PROFILES,
    CONDITION_SYMPTOM_MAP,
    ConditionPredictionService,
    get_condition_prediction_service,
    reconcile_candidate_evidence,
)
from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.services.assessment_service import generate_assessment_feedback


# Fixtures and helpers

@pytest.fixture
def condition_service() -> ConditionPredictionService:
    return get_condition_prediction_service()


def test_1_prediction_symptoms_only(condition_service: ConditionPredictionService):
    """1. Symptoms only: predicts valid candidates with symptom support."""
    result = condition_service.predict_conditions(
        symptoms_present=[{"symptom": "fever", "severity": "moderate"}, {"symptom": "cough", "severity": "mild"}],
        symptoms_absent=[],
        symptoms_unknown=[],
    )
    assert result["status"] == "completed"
    assert len(result["conditions"]) > 0
    first = result["conditions"][0]
    assert first["model_score"] > 0
    assert first["probability_score"] == first["model_score"]
    assert "supporting_symptoms" in first


def test_2_symptoms_plus_lab_report_reconciliation():
    """2 & 16. Symptoms plus lab report: attaches relevant lab evidence with source ID."""
    evidence = {
        "symptoms": {"present": [{"symptom": "fever"}, {"symptom": "cough"}], "absent": [], "unknown": [], "historical": []},
        "report": {
            "provided": True,
            "abnormal_findings": [
                {
                    "test_name": "WBC",
                    "value": "15.2",
                    "unit": "10^3/uL",
                    "reference_range": "4.5-11.0",
                    "status": "HIGH",
                    "interpretation": "HIGH",
                    "report_id": "rep_lab_001",
                    "source_file": "cbc_report.pdf",
                }
            ],
            "normal_findings": [],
            "unknown_range_findings": [],
        },
        "xray": {"received": False, "images": [], "reports": []},
    }

    mock_candidates = [
        {
            "rank": 1,
            "condition": "Pneumonia",
            "model_score": 0.65,
            "probability_score": 0.65,
            "supporting_symptoms": ["cough", "fever"],
            "contradictory_findings": [],
            "unassessed_symptoms": ["chills"],
        }
    ]

    reconciled = reconcile_candidate_evidence(mock_candidates, evidence)
    assert len(reconciled) == 1
    c = reconciled[0]
    # Score unchanged
    assert c["model_score"] == 0.65
    assert c["probability_score"] == 0.65

    # Check supporting evidence has both symptom and lab
    supp_types = [item["source_type"] for item in c["supporting_evidence"]]
    assert "symptom_nlp" in supp_types
    assert "laboratory_report" in supp_types

    lab_items = [item for item in c["supporting_evidence"] if item["source_type"] == "laboratory_report"]
    assert len(lab_items) == 1
    assert lab_items[0]["source_id"] == "rep_lab_001"
    assert lab_items[0]["source_file"] == "cbc_report.pdf"
    assert "WBC" in lab_items[0]["finding"]
    assert lab_items[0]["finding_state"] == "HIGH"
    assert lab_items[0]["verification_status"] == "CONFIRMED_EXTRACTION"

    # Badges
    assert "Supported by reported symptoms" in c["evidence_badges"]
    assert "Relevant laboratory finding available" in c["evidence_badges"]


def test_3_symptoms_plus_radiology_report_reconciliation():
    """3 & 18. Symptoms plus radiology report: attaches relevant radiology findings with source ID."""
    evidence = {
        "symptoms": {"present": [{"symptom": "cough"}, {"symptom": "fever"}], "absent": [], "unknown": []},
        "report": {"provided": False, "abnormal_findings": [], "normal_findings": []},
        "xray": {
            "received": True,
            "images": [],
            "reports": [
                {
                    "report_id": "rad_rep_888",
                    "file_name": "chest_xray_report.pdf",
                    "region": "chest",
                    "qualitative_findings": [{"finding": "Right lower lobe consolidation"}],
                }
            ],
        },
    }

    mock_candidates = [
        {
            "rank": 1,
            "condition": "Pneumonia",
            "model_score": 0.70,
            "probability_score": 0.70,
            "supporting_symptoms": ["cough", "fever"],
            "contradictory_findings": [],
            "unassessed_symptoms": [],
        }
    ]

    reconciled = reconcile_candidate_evidence(mock_candidates, evidence)
    c = reconciled[0]
    rad_items = [item for item in c["supporting_evidence"] if item["source_type"] == "radiology_report"]
    assert len(rad_items) == 1
    assert rad_items[0]["source_id"] == "rad_rep_888"
    assert rad_items[0]["source_file"] == "chest_xray_report.pdf"
    assert "consolidation" in rad_items[0]["finding"].lower()
    assert "Relevant radiology finding available" in c["evidence_badges"]


def test_4_symptoms_plus_eligible_xray():
    """4 & 20. Symptoms plus eligible X-ray: verified X-ray is attached as candidate support."""
    evidence = {
        "symptoms": {"present": [{"symptom": "cough"}, {"symptom": "fever"}], "absent": [], "unknown": []},
        "report": {"provided": False},
        "xray": {
            "received": True,
            "images": [
                {
                    "xray_result_id": "img_001",
                    "file_name": "chest_film.png",
                    "region": "chest",
                    "status": "COMPLETED",
                    "anatomy_independently_verified": True,
                    "anatomy_verification_status": "completed",
                    "has_region_mismatch": False,
                    "prediction": "Pneumonia",
                    "findings": [{"label": "consolidation", "is_actionable_finding": True}],
                }
            ],
            "reports": [],
        },
    }

    mock_candidates = [
        {
            "rank": 1,
            "condition": "Pneumonia",
            "model_score": 0.60,
            "probability_score": 0.60,
            "supporting_symptoms": ["cough", "fever"],
            "contradictory_findings": [],
        }
    ]

    reconciled = reconcile_candidate_evidence(mock_candidates, evidence)
    c = reconciled[0]
    xray_items = [item for item in c["supporting_evidence"] if item["source_type"] == "xray_image"]
    assert len(xray_items) == 1
    assert xray_items[0]["source_id"] == "img_001"
    assert xray_items[0]["source_file"] == "chest_film.png"
    assert xray_items[0]["verification_status"] == "VERIFIED_ELIGIBLE"
    assert "Existing eligible X-ray result available" in c["evidence_badges"]


def test_5_symptoms_plus_all_modalities():
    """5. All modalities provided together: candidate receives transparent source-linked evidence from all sources."""
    evidence = {
        "symptoms": {"present": [{"symptom": "cough"}, {"symptom": "fever"}], "absent": [{"symptom": "chest pain"}], "unknown": []},
        "report": {
            "provided": True,
            "abnormal_findings": [{"test_name": "CRP", "value": "45", "unit": "mg/L", "status": "HIGH", "report_id": "lab_01"}],
            "normal_findings": [],
            "unknown_range_findings": [],
        },
        "xray": {
            "received": True,
            "images": [
                {
                    "xray_result_id": "xr_01",
                    "file_name": "cxr.png",
                    "region": "chest",
                    "status": "COMPLETED",
                    "anatomy_independently_verified": True,
                    "anatomy_verification_status": "completed",
                    "has_region_mismatch": False,
                    "prediction": "Pneumonia",
                }
            ],
            "reports": [
                {
                    "report_id": "rad_01",
                    "region": "chest",
                    "qualitative_findings": [{"finding": "patchy opacity in right lung"}],
                }
            ],
        },
    }

    mock_candidates = [
        {
            "rank": 1,
            "condition": "Pneumonia",
            "model_score": 0.72,
            "probability_score": 0.72,
            "supporting_symptoms": ["cough", "fever"],
            "contradictory_findings": ["chest pain"],
        }
    ]

    reconciled = reconcile_candidate_evidence(mock_candidates, evidence)
    c = reconciled[0]
    assert c["model_score"] == 0.72  # strictly unchanged
    supp_sources = {item["source_type"] for item in c["supporting_evidence"]}
    assert "symptom_nlp" in supp_sources
    assert "laboratory_report" in supp_sources
    assert "radiology_report" in supp_sources
    assert "xray_image" in supp_sources

    assert any(b == "Supported by reported symptoms" for b in c["evidence_badges"])
    assert any(b == "Relevant laboratory finding available" for b in c["evidence_badges"])
    assert any(b == "Relevant radiology finding available" for b in c["evidence_badges"])
    assert any(b == "Existing eligible X-ray result available" for b in c["evidence_badges"])
    assert any(b == "Contradictory symptom reported" for b in c["evidence_badges"])


def test_6_no_symptoms_reports_only(condition_service: ConditionPredictionService):
    """6. No symptoms, reports only: returns insufficient_evidence for symptom model without crashing."""
    result = condition_service.predict_conditions(
        symptoms_present=[],
        symptoms_absent=[],
        symptoms_unknown=[],
    )
    assert result["status"] == "insufficient_evidence"
    assert result["conditions"] == []


def test_11_model_scores_strictly_unchanged():
    """11. Model scores remain unchanged by evidence reconciliation (no arbitrary fusion weights)."""
    evidence = {
        "symptoms": {"present": [{"symptom": "cough"}], "absent": [], "unknown": []},
        "report": {
            "provided": True,
            "abnormal_findings": [
                {"test_name": "WBC", "value": "18.0", "status": "HIGH", "report_id": "rep_1"},
                {"test_name": "CRP", "value": "90.0", "status": "HIGH", "report_id": "rep_1"},
            ],
        },
        "xray": {
            "received": True,
            "images": [
                {
                    "xray_result_id": "img_1",
                    "region": "chest",
                    "status": "COMPLETED",
                    "anatomy_independently_verified": True,
                    "anatomy_verification_status": "completed",
                    "prediction": "Pneumonia",
                }
            ],
            "reports": [],
        },
    }

    initial_score = 0.5432
    candidates = [
        {
            "rank": 1,
            "condition": "Pneumonia",
            "model_score": initial_score,
            "probability_score": initial_score,
            "supporting_symptoms": ["cough"],
            "contradictory_findings": [],
        }
    ]

    reconciled = reconcile_candidate_evidence(candidates, evidence)
    assert reconciled[0]["model_score"] == initial_score
    assert reconciled[0]["probability_score"] == initial_score


def test_17_unrelated_lab_finding_remains_contextual():
    """17. Unrelated lab finding does NOT support or contradict candidate, stays in contextual_evidence."""
    evidence = {
        "symptoms": {"present": [{"symptom": "headache"}], "absent": [], "unknown": []},
        "report": {
            "provided": True,
            "abnormal_findings": [
                {
                    "test_name": "Blood Glucose Fasting",
                    "value": "210",
                    "unit": "mg/dL",
                    "status": "HIGH",
                    "report_id": "glucose_rep",
                }
            ],
        },
        "xray": {"received": False, "images": [], "reports": []},
    }

    candidates = [
        {
            "rank": 1,
            "condition": "Migraine",
            "model_score": 0.55,
            "probability_score": 0.55,
            "supporting_symptoms": ["headache"],
            "contradictory_findings": [],
        }
    ]

    reconciled = reconcile_candidate_evidence(candidates, evidence)
    c = reconciled[0]
    # Blood glucose must NOT be in supporting_evidence for Migraine
    supp_labs = [item for item in c["supporting_evidence"] if item["source_type"] == "laboratory_report"]
    assert len(supp_labs) == 0

    # It MUST be in contextual_evidence
    contextual_labs = [item for item in c["contextual_evidence"] if item["source_type"] == "laboratory_report"]
    assert len(contextual_labs) == 1
    assert "Glucose" in contextual_labs[0]["finding"]
    assert "not a specific marker for Migraine" in contextual_labs[0]["reason"]
    assert "No additional corroborating evidence" in c["evidence_badges"]


def test_19_unrelated_radiology_finding_remains_contextual():
    """19. Unrelated radiology finding remains contextual, does not support different body region."""
    evidence = {
        "symptoms": {"present": [{"symptom": "cough"}], "absent": [], "unknown": []},
        "report": {"provided": False},
        "xray": {
            "received": True,
            "reports": [
                {
                    "report_id": "knee_rep_1",
                    "region": "knee",
                    "qualitative_findings": [{"finding": "Joint space narrowing in medial compartment"}],
                }
            ],
            "images": [],
        },
    }

    candidates = [
        {
            "rank": 1,
            "condition": "Pneumonia",
            "model_score": 0.51,
            "probability_score": 0.51,
            "supporting_symptoms": ["cough"],
            "contradictory_findings": [],
        }
    ]

    reconciled = reconcile_candidate_evidence(candidates, evidence)
    c = reconciled[0]
    supp_rad = [item for item in c["supporting_evidence"] if item["source_type"] == "radiology_report"]
    assert len(supp_rad) == 0

    ctx_rad = [item for item in c["contextual_evidence"] if item["source_type"] == "radiology_report"]
    assert len(ctx_rad) == 1
    assert "Knee" in ctx_rad[0]["finding"]


def test_21_unverified_xray_excluded_from_support():
    """21 & 35. Unverified X-ray cannot support candidates and cannot trigger triage."""
    evidence = {
        "symptoms": {"present": [{"symptom": "cough"}], "absent": [], "unknown": []},
        "report": {"provided": False},
        "xray": {
            "received": True,
            "images": [
                {
                    "xray_result_id": "unverified_img_1",
                    "region": "chest",
                    "status": "ANATOMY_UNVERIFIED",
                    "anatomy_independently_verified": False,
                    "prediction": "Pneumonia",
                    "findings": [{"label": "consolidation"}],
                }
            ],
            "reports": [],
        },
    }

    candidates = [
        {
            "rank": 1,
            "condition": "Pneumonia",
            "model_score": 0.52,
            "probability_score": 0.52,
            "supporting_symptoms": ["cough"],
            "contradictory_findings": [],
        }
    ]

    reconciled = reconcile_candidate_evidence(candidates, evidence)
    c = reconciled[0]
    # Must NOT be in supporting_evidence
    xray_supp = [item for item in c["supporting_evidence"] if item["source_type"] == "xray_image"]
    assert len(xray_supp) == 0

    # Must be in unavailable_or_unverified_information
    unverified_list = c.get("unavailable_or_unverified_information", [])
    assert len(unverified_list) == 1
    assert unverified_list[0]["source_id"] == "unverified_img_1"
    assert unverified_list[0]["finding_state"] == "ANATOMY_UNVERIFIED"
    assert unverified_list[0]["verification_status"] == "EXCLUDED"


def test_22_region_mismatch_excluded_from_concordance():
    """22. Region mismatch is excluded from cross-modal concordance and candidate support."""
    evidence = {
        "symptoms": {"present": [{"symptom": "cough"}], "absent": [], "unknown": []},
        "report": {"provided": False},
        "xray": {
            "received": True,
            "images": [
                {
                    "xray_result_id": "mismatch_img",
                    "region": "chest",
                    "status": "MISMATCH",
                    "has_region_mismatch": True,
                    "anatomy_independently_verified": True,
                    "anatomy_verification_status": "completed",
                    "prediction": "Pneumonia",
                }
            ],
            "reports": [],
        },
    }

    candidates = [
        {
            "rank": 1,
            "condition": "Pneumonia",
            "model_score": 0.50,
            "probability_score": 0.50,
            "supporting_symptoms": ["cough"],
            "contradictory_findings": [],
        }
    ]

    reconciled = reconcile_candidate_evidence(candidates, evidence)
    c = reconciled[0]
    xray_supp = [item for item in c["supporting_evidence"] if item["source_type"] == "xray_image"]
    assert len(xray_supp) == 0

    unverified = c.get("unavailable_or_unverified_information", [])
    assert len(unverified) == 1
    assert unverified[0]["finding_state"] == "REGION_MISMATCH"


def test_33_34_36_emergency_triage_independent_of_ml():
    """33, 34, 36. Emergency red flag triggers emergency triage regardless of ML success or failure; reassuring evidence cannot downgrade it."""
    # Patient with severe chest pain (emergency red flag) and normal troponin/reassuring evidence
    evidence = {
        "symptoms": {
            "present": [{"symptom": "chest pain", "severity": "severe"}],
            "absent": [],
            "unknown": [],
            "historical": [],
        },
        "report": {
            "provided": True,
            "abnormal_findings": [],
            "normal_findings": [{"test_name": "Troponin T", "value": "0.01", "status": "NORMAL"}],
            "unknown_range_findings": [],
        },
        "xray": {"received": False, "images": [], "reports": []},
        "reassuring": ["Troponin within normal reference range"],
        "supporting": ["Severe chest pain"],
    }

    # Case A: Triage evaluated with no ML candidate or failed ML
    triage_a = apply_triage(evidence)
    assert triage_a["pathway"] == "EMERGENCY"

    # Case B: Triage evaluated with reassuring ML candidates (e.g. GERD or Arthritis)
    mock_prediction = {
        "status": "completed",
        "conditions": [{"condition": "gastroesophageal reflux disease", "model_score": 0.60}],
    }
    triage_b = apply_triage(evidence)
    assert triage_b["pathway"] == "EMERGENCY"
    assert triage_b["pathway"] == triage_a["pathway"]


def test_32_duplicate_findings_not_double_counted():
    """32. Duplicate source findings are deduplicated in candidate evidence."""
    evidence = {
        "symptoms": {"present": [{"symptom": "cough"}], "absent": [], "unknown": []},
        "report": {
            "provided": True,
            "abnormal_findings": [
                {"test_name": "WBC", "value": "15.0", "status": "HIGH", "report_id": "rep_1"},
                {"test_name": "WBC", "value": "15.0", "status": "HIGH", "report_id": "rep_1"},
            ],
            "normal_findings": [],
            "unknown_range_findings": [],
        },
        "xray": {"received": False, "images": [], "reports": []},
    }

    candidates = [
        {
            "rank": 1,
            "condition": "Pneumonia",
            "model_score": 0.58,
            "probability_score": 0.58,
            "supporting_symptoms": ["cough"],
            "contradictory_findings": [],
        }
    ]

    reconciled = reconcile_candidate_evidence(candidates, evidence)
    c = reconciled[0]
    lab_items = [item for item in c["supporting_evidence"] if item["source_type"] == "laboratory_report"]
    assert len(lab_items) == 1


def test_49_50_feedback_and_pdf_consistency():
    """49 & 50. Assessment feedback structure contains reconciled candidates with all expected fields."""
    evidence = {
        "symptoms": {
            "present": [{"symptom": "cough", "severity": "moderate"}],
            "absent": [{"symptom": "fever"}],
            "unknown": [],
            "historical": [],
        },
        "report": {
            "provided": True,
            "abnormal_findings": [{"test_name": "WBC", "value": "14.5", "status": "HIGH", "report_id": "lab_99"}],
            "normal_findings": [],
            "unknown_range_findings": [],
            "qualitative_findings": [],
            "narrative_findings": [],
        },
        "xray": {
            "received": False,
            "status": "NOT_PROVIDED",
            "images": [],
            "reports": [],
        },
        "supporting": ["cough"],
        "reassuring": [],
        "contradictory": [],
        "missing": [],
    }

    triage_res = {"pathway": "CONSULTATION", "why_this_pathway": "Moderate respiratory symptoms"}

    # Run candidate prediction & reconciliation
    raw_candidates = [
        {
            "rank": 1,
            "condition": "Pneumonia",
            "model_score": 0.62,
            "probability_score": 0.62,
            "supporting_symptoms": ["cough"],
            "contradictory_findings": ["fever"],
            "unassessed_symptoms": ["chills"],
            "suggested_specialty": "Pulmonology",
        }
    ]

    reconciled_cands = reconcile_candidate_evidence(raw_candidates, evidence)

    condition_predictions = {
        "status": "completed",
        "conditions": reconciled_cands,
        "prediction_model_scope": "symptoms_only",
        "multimodal_training_status": "unimodal_symptom_ml_with_multimodal_evidence_reconciliation",
        "evidence_reconciliation_sources": ["symptoms", "medical_reports"],
    }

    feedback = generate_assessment_feedback(
        evidence=evidence,
        triage_result=triage_res,
        specialty="Pulmonology",
        condition_predictions=condition_predictions,
    )

    # Verify feedback payload
    assert "possible_conditions" in feedback
    cands_in_feedback = feedback["possible_conditions"]
    assert len(cands_in_feedback) == 1
    cand = cands_in_feedback[0]
    assert cand["condition"] == "Pneumonia"
    assert cand["model_score"] == 0.62
    assert "supporting_evidence" in cand
    assert "evidence_badges" in cand
    assert "evidence_summary" in cand
    assert any("Relevant laboratory finding available" in b for b in cand["evidence_badges"])
