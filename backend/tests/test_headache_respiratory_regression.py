"""Regression tests for Headache and Respiratory demonstration cases."""

from app.ai.symptom_nlp import extract_symptoms

HEADACHE_TEXT = """I have had a severe headache for 2 days, mainly on the right side of my head.
The pain feels throbbing and becomes worse when I am exposed to bright light or loud sounds.
I feel nauseous but I have not vomited.
I feel tired and have difficulty concentrating.
The headache started gradually yesterday morning.
I have had similar headaches once or twice before.
I do not have a fever.
I do not have weakness or numbness in my arms or legs.
I do not have difficulty speaking.
I have not fainted and I do not have a seizure."""

RESPIRATORY_TEXT = """I have had fever for 3 days. My temperature is around 101°F.
I have a dry cough, sore throat, headache and tiredness.
I also feel mild chest discomfort when coughing.
The symptoms started gradually 3 days ago.
I have mild difficulty sleeping because of the cough.
I do not have severe breathing difficulty.
I do not have chest pain when resting.
No vomiting or fainting."""


def test_headache_regression_suite():
    res = extract_symptoms(HEADACHE_TEXT)
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    # Headache qualifiers
    assert s_map["headache"]["status"] == "PRESENT"
    assert s_map["headache"]["severity"].lower() == "severe"
    assert "2 days" in s_map["headache"]["duration"]
    assert s_map["headache"]["laterality"] == "RIGHT"
    assert s_map["headache"]["quality"] == "THROBBING"

    # Associated findings
    assert s_map["photophobia"]["status"] == "PRESENT"
    assert s_map["phonophobia"]["status"] == "PRESENT"
    assert s_map["nausea"]["status"] == "PRESENT"
    assert s_map["vomiting"]["status"] == "ABSENT"
    assert s_map["fatigue"]["status"] == "PRESENT"
    assert s_map["concentration difficulty"]["status"] == "PRESENT"
    assert s_map["gradual onset"]["status"] == "PRESENT"
    assert s_map["previous similar headache"]["status"] == "PRESENT"

    # Explicit negatives
    assert s_map["fever"]["status"] == "ABSENT"
    assert s_map["weakness"]["status"] == "ABSENT"
    assert s_map["numbness"]["status"] == "ABSENT"
    assert s_map["speech difficulty"]["status"] == "ABSENT"
    assert s_map["fainting"]["status"] == "ABSENT"
    assert s_map["seizure"]["status"] == "ABSENT"

    # Zero hallucination check
    assert "shortness of breath" not in s_map or s_map["shortness of breath"]["status"] != "PRESENT"


def test_respiratory_regression_suite():
    res_resp = extract_symptoms(RESPIRATORY_TEXT)
    r_map = {s["symptom"]: s for s in res_resp["symptoms"]}

    assert r_map["fever"]["status"] == "PRESENT"
    assert "3 days" in r_map["fever"]["duration"]
    assert r_map["temperature"]["status"] == "PRESENT"
    assert "101" in r_map["temperature"]["context"]
    assert r_map["cough"]["status"] == "PRESENT"
    assert r_map["cough"]["type"] == "dry"
    assert r_map["sore throat"]["status"] == "PRESENT"
    assert r_map["headache"]["status"] == "PRESENT"
    assert r_map["headache"]["severity"] == "UNKNOWN"
    assert r_map["tiredness"]["status"] == "PRESENT"
    assert r_map["tiredness"]["severity"] == "UNKNOWN"
    assert r_map["chest discomfort"]["status"] == "PRESENT"
    assert r_map["chest discomfort"]["severity"].lower() == "mild"
    assert r_map["chest discomfort"]["trigger"] == "coughing"
    assert r_map["sleep difficulty"]["status"] == "PRESENT"
    assert r_map["sleep difficulty"]["severity"].lower() == "mild"

    # Explicit negatives
    assert r_map["severe breathing difficulty"]["status"] == "ABSENT"
    assert r_map["chest pain at rest"]["status"] == "ABSENT"
    assert r_map["vomiting"]["status"] == "ABSENT"
    assert r_map["fainting"]["status"] == "ABSENT"


def test_case_a_exact_specification_assertions():
    """Verify points 1 to 11 on the exact Case A headache input."""
    res = extract_symptoms(HEADACHE_TEXT)
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    # 1. Headache gets 2 days
    assert s_map["headache"]["duration"] == "2 days"

    # 2. Photophobia does NOT get 2 days automatically
    assert s_map["photophobia"]["duration"] is None

    # 3. Phonophobia does NOT get 2 days automatically
    assert s_map["phonophobia"]["duration"] is None

    # 4. Nausea does NOT get 2 days automatically
    assert s_map["nausea"]["duration"] is None

    # 5. Fatigue does NOT get 2 days automatically
    assert s_map["fatigue"]["duration"] is None

    # 6. Concentration difficulty does NOT get 2 days automatically
    assert s_map["concentration difficulty"]["duration"] is None

    # 7. Headache onset = GRADUAL
    assert s_map["headache"]["onset"] == "GRADUAL"

    # 8. Associated symptoms do not automatically inherit GRADUAL onset
    assert s_map["photophobia"]["onset"] is None
    assert s_map["phonophobia"]["onset"] is None
    assert s_map["nausea"]["onset"] is None
    assert s_map["fatigue"]["onset"] is None
    assert s_map["concentration difficulty"]["onset"] is None

    # 9. Previous similar headache remains HISTORICAL
    assert s_map["previous similar headache"]["type"] == "HISTORICAL"

    # 10. Historical headache duration = None/UNKNOWN
    assert s_map["previous similar headache"]["duration"] is None

    # 11. Historical headache is not included in active symptom count
    from app.ai.evidence_engine import combine_evidence
    evidence = combine_evidence(
        symptoms=res["symptoms"],
        report_findings=[],
        xray_result=None,
    )
    current_names = [s["symptom"] for s in evidence["symptoms"]["present"]]
    hist_names = [s["symptom"] for s in evidence["symptoms"]["historical"]]
    assert "previous similar headache" in hist_names
    assert "previous similar headache" not in current_names


def test_followup_semantics_and_mapping():
    """Verify points 12 to 15: followup semantics, onset mapping, nausea not answering vomiting."""
    from app.ai.followup_engine import is_target_addressed, apply_followup_answer

    # 12. Nausea does not answer vomiting
    symptoms = [{"symptom": "nausea", "state": "PRESENT", "status": "PRESENT"}]
    assert not is_target_addressed("vomiting", symptoms)
    assert not is_target_addressed("vomiting (red flag)", symptoms)

    headache_syms = [
        {"symptom": "headache", "state": "PRESENT", "status": "PRESENT", "onset": None}
    ]

    # 13. "No, gradual onset" becomes onset=GRADUAL
    updated_grad = apply_followup_answer(headache_syms, "q_sudden_onset", "No, gradual onset")
    h_grad = next(s for s in updated_grad if s["symptom"] == "headache")
    assert h_grad["onset"] == "GRADUAL"
    assert not any(s["symptom"] == "sudden onset" for s in updated_grad)

    # 14. "Yes, sudden peak" becomes onset=SUDDEN
    updated_sud = apply_followup_answer(headache_syms, "q_sudden_onset", "Yes, sudden peak")
    h_sud = next(s for s in updated_sud if s["symptom"] == "headache")
    assert h_sud["onset"] == "SUDDEN"

    # 15. "Not sure" becomes onset=UNKNOWN
    updated_unk = apply_followup_answer(headache_syms, "q_sudden_onset", "Not sure")
    h_unk = next(s for s in updated_unk if s["symptom"] == "headache")
    assert h_unk["onset"] == "UNKNOWN"


def test_complete_abnormal_labs_and_evidence_classification():
    """Verify points 16 to 21: complete 4 abnormal labs, normal WBC/CRP, contextual labs, unavailable X-ray."""
    from app.ai.evidence_engine import combine_evidence
    from app.services.assessment_service import generate_assessment_feedback

    # Case A symptoms
    sym_res = extract_symptoms(HEADACHE_TEXT)

    # Case A Report with 4 abnormal + 2 normal labs
    case_a_labs = [
        {"test_name": "WBC", "value": 12400, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "HIGH"},
        {"test_name": "ALT", "value": 68, "unit": "U/L", "reference_range": "7 - 56", "interpretation": "HIGH"},
        {"test_name": "AST", "value": 49, "unit": "U/L", "reference_range": "10 - 40", "interpretation": "HIGH"},
        {"test_name": "Total Bilirubin", "value": 1.8, "unit": "mg/dL", "reference_range": "0.2 - 1.2", "interpretation": "HIGH"},
        {"test_name": "Hemoglobin", "value": 14.6, "unit": "g/dL", "reference_range": "13.0 - 17.0", "interpretation": "NORMAL"},
        {"test_name": "Creatinine", "value": 0.9, "unit": "mg/dL", "reference_range": "0.7 - 1.3", "interpretation": "NORMAL"},
    ]

    # Case A X-ray unavailable
    xray_unavail = {
        "status": "UNAVAILABLE",
        "region": "chest",
        "prediction": None,
    }

    evidence = combine_evidence(
        symptoms=sym_res["symptoms"],
        report_findings=case_a_labs,
        xray_result=xray_unavail,
    )

    triage_result = {
        "pathway": "CONSULTATION",
        "rules_triggered": ["SYMP_SEVERITY_HIGH"],
        "why_this_pathway": "Severe headache symptom requires in-person medical consultation.",
    }
    specialty_result = {"suggested_specialty": "Neurology"}

    feedback = generate_assessment_feedback(
        evidence=evidence,
        triage_result=triage_result,
        specialty=specialty_result,
    )

    # 16 & 17. All 4 abnormal lab values appear, Bilirubin not dropped because of [:3]
    summary = feedback["assessment_summary"]
    assert "WBC" in summary
    assert "ALT" in summary
    assert "AST" in summary
    assert "Total Bilirubin" in summary
    assert "4 out-of-range" in summary

    next_steps = feedback["recommended_next_step"]["next_steps"]
    assert "Total Bilirubin" in next_steps or "ALT" in next_steps

    # 18. Normal labs are reassuring
    normal_labs = [
        {"test_name": "WBC", "value": 6500, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "NORMAL"},
        {"test_name": "CRP", "value": 1.2, "unit": "mg/L", "reference_range": "0 - 5", "interpretation": "NORMAL"},
    ]
    norm_evidence = combine_evidence(
        symptoms=sym_res["symptoms"],
        report_findings=normal_labs,
        xray_result=None,
    )
    assert any("WBC" in r and "CRP" in r for r in norm_evidence["reassuring"])
    assert not any("supports inflammation" in s.lower() for s in norm_evidence["supporting"])

    # 19. Abnormal ALT/AST/bilirubin are contextual/additional for headache
    assert any("Total Bilirubin" in s or "ALT" in s for s in evidence["separate"])
    assert not any("Total Bilirubin" in s for s in evidence["supporting"])

    # 20 & 21. Unavailable X-ray is UNKNOWN/UNASSESSED, no fake disease produced
    assert evidence["xray"]["status"] == "UNAVAILABLE"
    assert evidence["xray"]["prediction"] is None
    assert feedback["xray_findings"]["status"] == "UNAVAILABLE"
    assert feedback["xray_findings"]["prediction"] is None
    assert "pneumonia" not in str(feedback["xray_findings"]).lower()
    assert "fracture" not in str(feedback["xray_findings"]).lower()


def test_case_comparisons_and_differing_feedback():
    """Verify points 22 to 24: Case A vs Case B differences, normal vs abnormal report, valid vs unavailable X-ray."""
    from app.ai.evidence_engine import combine_evidence
    from app.ai.specialty_mapper import map_evidence_to_specialty
    from app.services.assessment_service import generate_assessment_feedback

    # Case A
    sym_a = extract_symptoms(HEADACHE_TEXT)["symptoms"]
    ev_a = combine_evidence(symptoms=sym_a, report_findings=[], xray_result=None)
    spec_a = map_evidence_to_specialty(ev_a)
    spec_a_name = spec_a.get("suggested_specialty") if isinstance(spec_a, dict) else spec_a
    assert spec_a_name == "Neurology"
    fb_a = generate_assessment_feedback(
        evidence=ev_a,
        triage_result={"pathway": "CONSULTATION", "why_this_pathway": "Headache consultation."},
        specialty=spec_a,
    )

    # Case B
    sym_b = extract_symptoms(RESPIRATORY_TEXT)["symptoms"]
    ev_b = combine_evidence(symptoms=sym_b, report_findings=[], xray_result=None)
    spec_b = map_evidence_to_specialty(ev_b)
    spec_b_name = spec_b.get("suggested_specialty") if isinstance(spec_b, dict) else spec_b
    assert spec_b_name == "Pulmonology"
    fb_b = generate_assessment_feedback(
        evidence=ev_b,
        triage_result={"pathway": "CONSULTATION", "why_this_pathway": "Respiratory consultation."},
        specialty=spec_b,
    )

    # 22. Respiratory and headache cases produce materially different feedback
    assert fb_a["assessment_summary"] != fb_b["assessment_summary"]
    assert fb_a["specialty_rationale"] != fb_b["specialty_rationale"]
    assert fb_a["recommended_next_step"]["guidance"] != fb_b["recommended_next_step"]["guidance"]
    assert fb_a["recommended_next_step"]["suggested_specialty"] == "Neurology"
    assert fb_b["recommended_next_step"]["suggested_specialty"] == "Pulmonology"

    # 23. Same symptoms + normal report vs abnormal report
    labs_abnormal = [
        {"test_name": "WBC", "value": 14000, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "HIGH"}
    ]
    labs_normal = [
        {"test_name": "WBC", "value": 6500, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "NORMAL"}
    ]
    ev_abn = combine_evidence(symptoms=sym_a, report_findings=labs_abnormal, xray_result=None)
    ev_norm = combine_evidence(symptoms=sym_a, report_findings=labs_normal, xray_result=None)
    fb_abn = generate_assessment_feedback(
        evidence=ev_abn,
        triage_result={"pathway": "CONSULTATION", "why_this_pathway": "Consultation."},
        specialty="Neurology",
    )
    fb_norm = generate_assessment_feedback(
        evidence=ev_norm,
        triage_result={"pathway": "CONSULTATION", "why_this_pathway": "Consultation."},
        specialty="Neurology",
    )
    assert fb_abn["assessment_summary"] != fb_norm["assessment_summary"]
    assert "out-of-range" in fb_abn["assessment_summary"]
    assert "within standard reference ranges" in fb_norm["assessment_summary"]

    # 24. Unavailable X-ray vs valid X-ray
    xray_valid = {"status": "COMPLETED", "region": "chest", "prediction": "Consolidation pattern"}
    xray_unavail = {"status": "UNAVAILABLE", "region": "chest", "prediction": None}
    ev_x_valid = combine_evidence(symptoms=sym_b, report_findings=[], xray_result=xray_valid)
    ev_x_unavail = combine_evidence(symptoms=sym_b, report_findings=[], xray_result=xray_unavail)
    fb_x_valid = generate_assessment_feedback(
        evidence=ev_x_valid,
        triage_result={"pathway": "CONSULTATION", "why_this_pathway": "Consultation."},
        specialty="Pulmonology",
    )
    fb_x_unavail = generate_assessment_feedback(
        evidence=ev_x_unavail,
        triage_result={"pathway": "CONSULTATION", "why_this_pathway": "Consultation."},
        specialty="Pulmonology",
    )
    assert fb_x_valid["xray_findings"]["prediction"] == "Consolidation pattern"
    assert fb_x_unavail["xray_findings"]["prediction"] is None
    assert fb_x_valid["assessment_summary"] != fb_x_unavail["assessment_summary"]
