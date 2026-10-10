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


def test_prompt_21_regression_coverage():
    """Explicitly verify all 21 regression requirements from master prompt."""
    from app.ai.evidence_engine import combine_evidence
    from app.ai.specialty_mapper import map_evidence_to_specialty
    from app.ai.followup_engine import is_target_addressed
    from app.services.assessment_service import generate_assessment_feedback

    # 1. Headache onset = GRADUAL
    res_headache = extract_symptoms(HEADACHE_TEXT)
    s_map_h = {s["symptom"]: s for s in res_headache["symptoms"]}
    assert s_map_h["headache"]["onset"] == "GRADUAL"

    # 2. Cough onset = GRADUAL
    cough_text = "The symptoms gradually started about 5 days ago. The cough is mostly dry but sometimes produces a small amount of mucus."
    res_cough = extract_symptoms(cough_text)
    s_map_c = {s["symptom"]: s for s in res_cough["symptoms"]}
    assert s_map_c["cough"]["onset"] == "GRADUAL"

    # 3. Photophobia does not inherit headache onset
    assert s_map_h["photophobia"]["onset"] is None

    # 4. Nausea does not inherit headache duration
    assert s_map_h["nausea"]["duration"] is None

    # 5. Cough "mostly dry" is preserved
    assert s_map_c["cough"].get("character") == "mostly dry" or s_map_c["cough"].get("type") == "mostly dry"

    # 6. Occasional mucus is preserved without forcing productive cough
    assert s_map_c["cough"].get("type") != "productive"
    assert "mucus" in (s_map_c["cough"].get("mucus") or s_map_c["cough"].get("qualifier") or s_map_c["cough"].get("context") or "")

    # 7. Historical headache remains HISTORICAL
    assert s_map_h["previous similar headache"]["type"] == "HISTORICAL"

    # 8. WBC/CRP abnormal is not automatically causal/supporting
    abnormal_inflam_labs = [
        {"test_name": "WBC", "value": 14000, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "HIGH"},
        {"test_name": "CRP", "value": 25.0, "unit": "mg/L", "reference_range": "0 - 5", "interpretation": "HIGH"},
    ]
    ev_resp_labs = combine_evidence(
        symptoms=res_cough["symptoms"],
        report_findings=abnormal_inflam_labs,
        xray_result=None,
    )
    assert not any("WBC" in s and "cause" in s.lower() for s in ev_resp_labs["supporting"])
    assert not any("WBC" in s for s in ev_resp_labs["supporting"])
    assert any("WBC" in s and "does not by itself establish the cause" in s for s in ev_resp_labs["separate"])

    # 9. ALT/AST/bilirubin remain contextual in headache/respiratory cases
    abnormal_liver_labs = [
        {"test_name": "ALT", "value": 75, "unit": "U/L", "reference_range": "7 - 56", "interpretation": "HIGH"},
        {"test_name": "AST", "value": 60, "unit": "U/L", "reference_range": "10 - 40", "interpretation": "HIGH"},
        {"test_name": "Total Bilirubin", "value": 2.1, "unit": "mg/dL", "reference_range": "0.2 - 1.2", "interpretation": "HIGH"},
    ]
    ev_liver = combine_evidence(
        symptoms=res_cough["symptoms"],
        report_findings=abnormal_liver_labs,
        xray_result=None,
    )
    assert not any("ALT" in s for s in ev_liver["supporting"])
    assert any("ALT" in s for s in ev_liver["separate"])

    # 10. Ankle swelling cannot support headache
    qual_ankle = [{"finding": "ankle swelling", "state": "PRESENT", "context": "bilateral"}]
    ev_headache_ankle = combine_evidence(
        symptoms=res_headache["symptoms"],
        report_findings=qual_ankle,
        xray_result=None,
    )
    assert not any("ankle swelling" in s.lower() for s in ev_headache_ankle["supporting"])
    assert any("ankle swelling" in s.lower() for s in ev_headache_ankle["separate"])

    # 11. Respiratory symptoms -> Pulmonology even with abnormal liver labs
    all_abnormal_labs = abnormal_inflam_labs + abnormal_liver_labs
    ev_resp_multimodal = combine_evidence(
        symptoms=res_cough["symptoms"],
        report_findings=all_abnormal_labs,
        xray_result=None,
    )
    spec_resp = map_evidence_to_specialty(ev_resp_multimodal)
    assert spec_resp["suggested_specialty"] == "Pulmonology"

    # 12. Headache symptoms -> Neurology
    ev_headache = combine_evidence(
        symptoms=res_headache["symptoms"],
        report_findings=abnormal_liver_labs,
        xray_result=None,
    )
    spec_headache = map_evidence_to_specialty(ev_headache)
    assert spec_headache["suggested_specialty"] == "Neurology"

    # 13. Header specialty and recommended specialty are identical
    fb_headache = generate_assessment_feedback(
        evidence=ev_headache,
        triage_result={"pathway": "CONSULTATION", "why_this_pathway": "Headache consultation."},
        specialty=spec_headache,
    )
    assert fb_headache["recommended_next_step"]["suggested_specialty"] == spec_headache["suggested_specialty"]

    # 14. All-normal labs do not generate "abnormal lab review" recommendation
    norm_labs = [
        {"test_name": "WBC", "value": 6500, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "NORMAL"},
        {"test_name": "ALT", "value": 25, "unit": "U/L", "reference_range": "7 - 56", "interpretation": "NORMAL"},
    ]
    ev_normal = combine_evidence(
        symptoms=res_headache["symptoms"],
        report_findings=norm_labs,
        xray_result=None,
    )
    fb_normal = generate_assessment_feedback(
        evidence=ev_normal,
        triage_result={"pathway": "CONSULTATION", "why_this_pathway": "Headache consultation."},
        specialty="Neurology",
    )
    # 14. All-normal labs do not generate "abnormal lab review" recommendation
    assert "abnormal laboratory findings" not in str(fb_normal["recommended_next_step"]).lower()
    assert "within the supplied reference ranges" in fb_normal["recommended_next_step"]["next_steps"]

    # 15. Abnormal labs are explicitly named when they exist
    fb_abnormal = generate_assessment_feedback(
        evidence=ev_resp_multimodal,
        triage_result={"pathway": "CONSULTATION", "why_this_pathway": "Respiratory consultation."},
        specialty="Pulmonology",
    )
    action_abn = fb_abnormal["recommended_next_step"]["next_steps"]
    assert "WBC" in action_abn
    assert "ALT" in action_abn
    assert "AST" in action_abn
    assert "Total Bilirubin" in action_abn

    # 16. Summary count equals number of listed active symptoms
    summary_text = fb_headache["assessment_summary"]
    import re
    match = re.search(r"Active symptom presentation includes (\d+) finding\(s\):\s*(.+?)\.\s*(?:Historical|Patient|$)", summary_text)
    assert match is not None
    count_reported = int(match.group(1))
    assert count_reported == len(fb_headache["current_symptoms"])
    # Confirm historical headache is excluded from active count
    assert not any(s["symptom"] == "previous similar headache" for s in fb_headache["current_symptoms"])
    assert any(h["symptom"] == "previous similar headache" for h in fb_headache.get("historical_symptoms", []))

    # 17. Generic severe bleeding does not appear as irrelevant UNKNOWN evidence
    assert not any("severe bleeding" in u.lower() for u in ev_headache.get("missing", []))
    assert not any("severe bleeding" in u.lower() for u in ev_resp_multimodal.get("missing", []))

    # 18. Nausea does not answer vomiting
    assert not is_target_addressed("vomiting", [{"symptom": "nausea", "state": "PRESENT"}])

    # 19. Chest discomfort does not answer chest pain
    assert not is_target_addressed("chest pain", [{"symptom": "chest discomfort", "state": "PRESENT"}])

    # 20. X-ray unavailable remains UNKNOWN/UNAVAILABLE
    xray_unavail = {"status": "UNAVAILABLE", "region": "chest", "prediction": None}
    ev_xray = combine_evidence(
        symptoms=res_headache["symptoms"],
        report_findings=[],
        xray_result=xray_unavail,
    )
    assert ev_xray["xray"]["status"] == "UNAVAILABLE"
    assert ev_xray["xray"]["prediction"] is None

    # 21. No X-ray disease is fabricated
    fb_xray = generate_assessment_feedback(
        evidence=ev_xray,
        triage_result={"pathway": "CONSULTATION", "why_this_pathway": "Headache consultation."},
        specialty="Neurology",
    )
    assert fb_xray["xray_findings"]["prediction"] is None
    assert "pneumonia" not in str(fb_xray["xray_findings"]).lower()
    assert "fracture" not in str(fb_xray["xray_findings"]).lower()
    assert "consolidation" not in str(fb_xray["xray_findings"]).lower()


# ─────────────────────────────────────────────────────────────
# 22-26: Dynamic Follow-Up Engine Regression Suite
# ─────────────────────────────────────────────────────────────

def test_dynamic_followup_most_important_automated_regression_state_abc():
    """Verify Section 23: State A != State B != State C dynamically recalculated."""
    from app.ai.followup_engine import suggest_followups, apply_followup_answer

    # STATE A: abdominal pain only
    syms_a = [{"symptom": "abdominal pain", "state": "PRESENT", "body_area": "abdomen", "domain": "gi"}]
    res_a = suggest_followups(symptoms=syms_a)
    assert res_a["needed"] is True
    assert len(res_a["questions"]) == 1
    q_a = res_a["questions"][0]
    assert q_a["id"] == "q_abdominal_location"
    assert "where" in q_a["question"].lower()

    # STATE B: abdominal pain + lower-right location
    syms_b = apply_followup_answer(syms_a, q_a["id"], "Lower right side")
    res_b = suggest_followups(symptoms=syms_b)
    assert res_b["needed"] is True
    assert len(res_b["questions"]) == 1
    q_b = res_b["questions"][0]
    assert q_b["id"] == "q_abdominal_onset"
    assert q_b["id"] != q_a["id"]
    assert "suddenly or gradually" in q_b["question"].lower()

    # STATE C: abdominal pain + lower-right location + sudden onset
    syms_c = apply_followup_answer(syms_b, q_b["id"], "Suddenly")
    res_c = suggest_followups(symptoms=syms_c)
    assert res_c["needed"] is True
    assert len(res_c["questions"]) == 1
    q_c = res_c["questions"][0]
    assert q_c["id"] == "q_severity"
    assert q_c["id"] not in {q_a["id"], q_b["id"]}
    assert "severe" in q_c["question"].lower()

    # STATE D: after answering severity -> sufficiency termination
    syms_d = apply_followup_answer(syms_c, q_c["id"], "Severe")
    res_d = suggest_followups(symptoms=syms_d)
    assert res_d["needed"] is False
    assert len(res_d["questions"]) == 0


def test_dynamic_followup_complete_abdominal_input():
    """Verify Section 12: Complete abdominal input asks no follow-ups."""
    from app.ai.followup_engine import suggest_followups

    complete_text = (
        "I have had abdominal pain since yesterday evening. The pain is mainly in "
        "the lower right side of my abdomen and feels sharp. It becomes worse when I "
        "walk or move. I also feel nauseous and have a reduced appetite. I have a mild "
        "fever. I have not vomited. I do not have diarrhea. I do not have blood in my "
        "stool. I have not fainted. The pain started suddenly yesterday evening."
    )
    nlp_res = extract_symptoms(complete_text)
    res = suggest_followups(symptoms=nlp_res["symptoms"], raw_text=complete_text)

    # Must NOT ask duration, onset, vomiting, diarrhea, breathing difficulty, chest pain
    q_ids = [q["id"] for q in res["questions"]]
    assert "q_duration" not in q_ids
    assert "q_abdominal_onset" not in q_ids
    assert "q_sudden_onset" not in q_ids
    assert "q_vomiting" not in q_ids
    assert "q_diarrhea" not in q_ids
    assert "q_breathing_difficulty" not in q_ids
    assert "q_chest_pain" not in q_ids

    # Sufficient: No additional questions needed
    assert res["needed"] is False
    assert len(res["questions"]) == 0


def test_dynamic_followup_same_symptom_different_inputs():
    """Verify Section 3: Same primary symptom produces different questions based on input detail."""
    from app.ai.followup_engine import suggest_followups

    # Input A: vague abdominal pain -> asks location
    syms_a = extract_symptoms("I have abdominal pain.")["symptoms"]
    res_a = suggest_followups(symptoms=syms_a)
    assert res_a["needed"] is True
    assert res_a["questions"][0]["id"] == "q_abdominal_location"

    # Input B: location and severity already given -> asks onset
    syms_b = extract_symptoms("I have severe lower-right abdominal pain.")["symptoms"]
    res_b = suggest_followups(symptoms=syms_b)
    assert res_b["needed"] is True
    assert res_b["questions"][0]["id"] == "q_abdominal_onset"
    assert res_b["questions"][0]["id"] != res_a["questions"][0]["id"]

    # Input C: location, severity, onset, duration, fever given -> asks vomiting
    syms_c = extract_symptoms("I have severe lower-right abdominal pain that started suddenly yesterday and I have a fever.")["symptoms"]
    res_c = suggest_followups(symptoms=syms_c)
    assert res_c["needed"] is True
    assert res_c["questions"][0]["id"] == "q_vomiting"
    assert res_c["questions"][0]["id"] not in {res_a["questions"][0]["id"], res_b["questions"][0]["id"]}


def test_dynamic_followup_headache_presentation():
    """Verify Section 4: Headache presentation dynamic question flow."""
    from app.ai.followup_engine import suggest_followups, apply_followup_answer

    syms = extract_symptoms("I have a severe headache.")["symptoms"]
    res = suggest_followups(symptoms=syms)
    assert res["needed"] is True
    # Asks sudden onset / thunderclap check
    assert res["questions"][0]["id"] == "q_sudden_onset"

    # Answer sudden onset
    syms_updated = apply_followup_answer(syms, "q_sudden_onset", "Yes, sudden peak")
    res_updated = suggest_followups(symptoms=syms_updated)
    assert res_updated["needed"] is True
    # Re-evaluates to neurological deficit red flags
    assert res_updated["questions"][0]["id"] == "q_neuro_weakness"
    assert res_updated["questions"][0]["id"] != "q_sudden_onset"


def test_dynamic_followup_respiratory_presentation():
    """Verify Section 5: Respiratory presentation dynamic question flow."""
    from app.ai.followup_engine import suggest_followups, apply_followup_answer

    syms = extract_symptoms("I have a cough.")["symptoms"]
    res = suggest_followups(symptoms=syms)
    assert res["needed"] is True
    # Respiratory red flag: breathing difficulty
    assert res["questions"][0]["id"] == "q_breathing_difficulty"

    # User denies difficulty breathing
    syms_1 = apply_followup_answer(syms, "q_breathing_difficulty", "No")
    res_1 = suggest_followups(symptoms=syms_1)
    assert res_1["needed"] is True
    assert res_1["questions"][0]["id"] != "q_breathing_difficulty"
    # Next missing factor: chest pain or duration
    assert res_1["questions"][0]["id"] in {"q_chest_pain", "q_duration"}

    # User provides duration
    syms_2 = apply_followup_answer(syms_1, "q_duration", "5 days")
    res_2 = suggest_followups(symptoms=syms_2)
    q_ids_2 = [q["id"] for q in res_2["questions"]]
    assert "q_duration" not in q_ids_2
    assert "q_breathing_difficulty" not in q_ids_2


def test_dynamic_followup_separations_negatives_and_unanswered():
    """Verify Sections 15, 16, 17, 18: Symptom separation, explicit negatives, no default answers."""
    from app.ai.followup_engine import is_target_addressed, apply_followup_answer

    # 1. Nausea does not answer vomiting
    assert not is_target_addressed("vomiting", [{"symptom": "nausea", "state": "PRESENT"}])

    # 2. Fatigue does not answer weakness
    assert not is_target_addressed("weakness", [{"symptom": "fatigue", "state": "PRESENT"}])

    # 3. Dizziness does not answer fainting
    assert not is_target_addressed("fainting", [{"symptom": "dizziness", "state": "PRESENT"}])

    # 4. Chest discomfort does not answer chest pain
    assert not is_target_addressed("chest pain", [{"symptom": "chest discomfort", "state": "PRESENT"}])

    # 5. Chest discomfort does not answer difficulty breathing
    assert not is_target_addressed("difficulty breathing", [{"symptom": "chest discomfort", "state": "PRESENT"}])

    # 6. Explicit negative preserved as ABSENT
    syms = [{"symptom": "abdominal pain", "state": "PRESENT"}]
    updated = apply_followup_answer(syms, "q_vomiting", "No")
    vom_item = next(s for s in updated if s["symptom"] == "vomiting")
    assert vom_item["state"] == "ABSENT"
    assert is_target_addressed("vomiting", updated)

    # 7. Unanswered question remains unchanged (no default conversion to ABSENT)
    unans_updated = apply_followup_answer(syms, "q_vomiting", "")
    assert not any(s["symptom"] == "vomiting" for s in unans_updated)

    # 8. Not sure maps to UNKNOWN, not ABSENT
    unk_updated = apply_followup_answer(syms, "q_vomiting", "Not sure")
    unk_item = next(s for s in unk_updated if s["symptom"] == "vomiting")
    assert unk_item["state"] == "UNKNOWN"


def test_dynamic_followup_no_cross_domain_irrelevant_questions():
    """Verify Section 11: Abdominal case does not receive generic respiratory or chest questions."""
    from app.ai.followup_engine import suggest_followups

    syms = [{"symptom": "abdominal pain", "state": "PRESENT", "domain": "gi", "body_area": "abdomen"}]
    res = suggest_followups(symptoms=syms)
    targets = [q["target"] for q in res["questions"]]

    assert "difficulty breathing" not in targets
    assert "chest pain" not in targets
    assert "fall/trauma" not in targets
    assert any(t in targets for t in ["abdominal location", "sudden onset", "vomiting"])


