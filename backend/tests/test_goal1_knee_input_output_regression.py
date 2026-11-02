"""Regression test suite for Goal 1 Right-Knee Input-to-Output verification.

Tests:
1. Symptom extraction:
   - Duration (14 days) preserved
   - Severity (6/10 -> MODERATE) preserved
   - Rich knee presentation (twisting injury, stairs/bending/squatting, morning stiffness, clicking/catching)
   - Negations preserved (fever: ABSENT, arthralgia: ABSENT)
   - Unreported numbness is NOT PRESENT (remains NOT_REPORTED)
   - No pain contradiction (never both PRESENT and ABSENT)
2. Follow-up engine:
   - No headache questions (q_sudden_onset) on knee presentations
   - Appropriate MSK/injury questions selected
3. Laboratory extraction:
   - Preserves scaled multipliers (9.8 ×10³/µL WBC, 276 ×10³/µL Platelets)
   - Extracts ESR (36 mm/hr HIGH) and CRP (12.6 mg/L HIGH)
4. Radiology report extraction:
   - Extracts fracture: ABSENT, dislocation: ABSENT
   - Extracts mild soft-tissue swelling: PRESENT
5. Assessment guidance:
   - Orthopedics recommendation, no neurologist or acute headache wording for knee cases.
"""

import pytest
from app.ai.symptom_nlp import extract_symptoms
from app.ai.followup_engine import suggest_followups
from app.ai.report_nlp import structure_report_text
from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.services.assessment_service import generate_assessment_feedback


def test_knee_symptom_extraction_regression():
    raw_symptom_text = (
        "I have right knee pain, medial knee pain, and swelling for approximately 14 days "
        "following a twisting injury while playing football. Pain severity is 6/10. "
        "Morning stiffness and pain on stairs, bending, and squatting. "
        "Intermittent clicking/catching. No fever. No other joint pain."
    )

    result = extract_symptoms(raw_symptom_text)
    symptoms = result["symptoms"]
    present_names = {s["symptom"].lower(): s for s in symptoms if s.get("state") == "PRESENT"}
    absent_names = {s["symptom"].lower(): s for s in symptoms if s.get("state") == "ABSENT"}

    # 1. Numbness must NOT be present
    assert "numbness" not in present_names, "Numbness must not be marked PRESENT"

    # 2. Knee pain checks
    assert "knee pain" in present_names
    kp = present_names["knee pain"]
    assert kp.get("laterality") == "RIGHT"
    assert kp.get("duration") == "14 days"
    assert str(kp.get("severity")) == "MODERATE"
    assert "stairs" in (kp.get("trigger") or "").lower() or "stairs" in (kp.get("context") or "").lower()

    # 3. Rich knee findings
    assert "swelling" in present_names
    assert "morning stiffness" in present_names
    assert "joint clicking/catching" in present_names
    assert "trauma/injury" in present_names

    # 4. Explicitly absent symptoms
    assert "fever" in absent_names
    assert "arthralgia" in absent_names

    # 5. No contradiction in pain
    assert "pain" not in absent_names, "Pain should not be marked ABSENT when knee pain is present"
    assert "knee pain" not in absent_names


def test_knee_followup_engine_regression():
    raw_symptom_text = (
        "I have right knee pain, medial knee pain, and swelling for approximately 14 days "
        "following a twisting injury while playing football. Pain severity is 6/10. "
        "Morning stiffness and pain on stairs, bending, and squatting. "
        "Intermittent clicking/catching. No fever. No other joint pain."
    )
    result = extract_symptoms(raw_symptom_text)
    symptoms = result["symptoms"]

    res = suggest_followups(symptoms=symptoms, raw_text=raw_symptom_text)
    questions = res.get("questions", [])
    question_ids = [q["id"] for q in questions]

    # Must NOT ask sudden onset headache question
    assert "q_sudden_onset" not in question_ids, "Sudden onset headache question must not be asked for knee complaint"

    # Must include relevant MSK/injury question
    assert any("limb" in qid or "injury" in qid or "deformity" in qid for qid in question_ids)


def test_lab_extraction_regression():
    lab_text = """
    COMPLETE BLOOD COUNT (CBC) & INFLAMMATORY MARKERS
    Test Name               Result      Unit            Reference Interval   Flag
    Total WBC count         9.8         ×10³/µL         4.5 - 11.0
    Platelet count          276         ×10³/µL         150 - 450
    C-Reactive Protein      12.6        mg/L            0.0 - 5.0            HIGH
    Erythrocyte Sedimentation Rate (ESR)  36  mm/hr     0 - 20               HIGH
    """

    res = structure_report_text(lab_text)
    assert res["status"] == "structured"

    test_map = {item["test_name"]: item for item in res["findings"]}

    # WBC
    assert any("White Blood Cell" in k or "WBC" in k for k in test_map)
    wbc_key = next(k for k in test_map if "White Blood Cell" in k or "WBC" in k)
    wbc = test_map[wbc_key]
    assert wbc["value"] == pytest.approx(9.8)
    assert "10" in wbc["unit"] or "³" in wbc["unit"] or "^3" in wbc["unit"]

    # Platelets
    assert "Platelet Count" in test_map
    plt = test_map["Platelet Count"]
    assert plt["value"] == pytest.approx(276.0)
    assert "10" in plt["unit"] or "³" in plt["unit"] or "^3" in plt["unit"]

    # CRP
    assert any("C-Reactive Protein" in k or "CRP" in k for k in test_map)
    crp_key = next(k for k in test_map if "C-Reactive Protein" in k or "CRP" in k)
    crp = test_map[crp_key]
    assert crp["value"] == pytest.approx(12.6)
    assert crp["unit"].lower() == "mg/l"
    assert crp["status"] == "HIGH"

    # ESR
    assert any("Erythrocyte Sedimentation Rate" in k or "ESR" in k for k in test_map)
    esr_key = next(k for k in test_map if "Erythrocyte Sedimentation Rate" in k or "ESR" in k)
    esr = test_map[esr_key]
    assert esr["value"] == pytest.approx(36.0)
    assert esr["unit"].lower() in ("mm/hr", "mm/h")
    assert esr["status"] == "HIGH"


def test_radiology_report_extraction_regression():
    rad_text = """
    CLINICAL HISTORY: 24-year-old male with acute twisting injury to right knee while playing football. Persistent pain and swelling.
    TECHNIQUE: Frontal and lateral radiographs of the right knee.
    FINDINGS:
    - No evidence of acute fracture or dislocation.
    - Joint spaces are preserved.
    - Mild soft-tissue swelling noted around the suprapatellar and medial compartment.
    IMPRESSION:
    1. No acute bony injury or dislocation identified.
    2. Mild periarticular soft-tissue swelling.
    3. Clinical correlation and MRI recommended if meniscal or ligamentous pathology is suspected.
    """

    res = structure_report_text(rad_text)
    assert res["status"] == "structured"
    findings_map = {f["finding"].lower(): f for f in res["qualitative_findings"]}

    assert "fracture" in findings_map
    assert findings_map["fracture"]["state"] == "ABSENT"

    assert "dislocation" in findings_map
    assert findings_map["dislocation"]["state"] == "ABSENT"

    # Mild soft-tissue swelling must be present
    has_soft_tissue = any("soft-tissue swelling" in k or "soft tissue swelling" in k for k in findings_map)
    assert has_soft_tissue, f"Mild soft-tissue swelling must be extracted, got: {list(findings_map.keys())}"
    st = next(f for k, f in findings_map.items() if "soft" in k and "swelling" in k)
    assert st["state"] == "PRESENT"


def test_assessment_service_guidance_regression():
    raw_symptom_text = (
        "I have right knee pain, medial knee pain, and swelling for approximately 14 days "
        "following a twisting injury while playing football. Pain severity is 6/10. "
        "Morning stiffness and pain on stairs, bending, and squatting. "
        "Intermittent clicking/catching. No fever. No other joint pain."
    )
    result = extract_symptoms(raw_symptom_text)
    symptoms = result["symptoms"]

    xray_meta = {
        "received": True,
        "region": "knee",
        "status": "UNAVAILABLE",
        "images": [{"region": "knee", "status": "UNAVAILABLE"}],
    }

    evidence = combine_evidence(symptoms=symptoms, xray_result=xray_meta)
    triage = apply_triage(evidence)

    feedback = generate_assessment_feedback(
        evidence=evidence,
        triage_result=triage,
        specialty="Orthopedics",
    )

    guidance = feedback.get("recommended_guidance", "")
    action = feedback.get("recommended_action", "")
    summary = feedback.get("assessment_summary", "")
    combined = f"{guidance} {action} {summary}".lower()

    # MUST NOT contain headache or neurologist text
    assert "neurologist" not in combined, "Knee case must not recommend neurologist"
    assert "headache" not in combined, "Knee case must not describe acute headache pattern"

    # Should mention orthopedics / joint / musculoskeletal
    assert "orthoped" in combined or "joint" in combined or "musculoskeletal" in combined
