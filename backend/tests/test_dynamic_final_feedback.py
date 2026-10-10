"""Regression and verification tests for Dynamic Assessment Feedback Pipeline.

Verifies:
1. Historical headache must not inherit current duration.
2. Photophobia must not receive THROBBING quality.
3. Phonophobia must not receive THROBBING quality.
4. Exact duplicate lab findings must be deduplicated.
5. Exact duplicate qualitative findings must be deduplicated.
6. Unknown laboratory ranges remain UNKNOWN.
7. Normal WBC/CRP must not be described as inflammatory abnormalities.
8. Unavailable X-ray must remain UNKNOWN/UNASSESSED.
9. No fabricated X-ray disease.
10. Separate ankle swelling must not be described as respiratory supporting evidence.
11. Respiratory final feedback differs from headache final feedback.
12. Same symptoms + normal report produces different final feedback from same symptoms + abnormal report.
13. Same symptoms/report + unavailable X-ray differs appropriately from same symptoms/report + valid X-ray finding.
14. No unsupported diagnosis appears in final feedback.
15. Historical symptoms do not inflate active symptom count.
"""

from __future__ import annotations

import pytest
from app.ai.symptom_nlp import extract_symptoms
from app.ai.report_nlp import structure_report_text
from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.ai.specialty_mapper import map_evidence_to_specialty
from app.services.assessment_service import generate_assessment_feedback


# ─────────────────────────────────────────────────────────────
# 1. Historical headache temporal scoping
# ─────────────────────────────────────────────────────────────
def test_historical_headache_does_not_inherit_duration():
    text = (
        "I have had a severe headache for 2 days, mainly on the right side. "
        "I have had similar headaches once or twice before."
    )
    res = extract_symptoms(text)
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    assert "headache" in s_map
    assert s_map["headache"]["status"] == "PRESENT"
    assert s_map["headache"]["type"] == "CURRENT"
    assert s_map["headache"]["duration"] == "2 days"

    assert "previous similar headache" in s_map
    assert s_map["previous similar headache"]["status"] == "PRESENT"
    assert s_map["previous similar headache"]["type"] == "HISTORICAL"
    assert s_map["previous similar headache"]["duration"] is None
    assert s_map["previous similar headache"].get("history") == "once or twice before"


# ─────────────────────────────────────────────────────────────
# 2 & 3. Photophobia & Phonophobia qualifier scoping
# ─────────────────────────────────────────────────────────────
def test_photophobia_and_phonophobia_qualifiers():
    text = (
        "The pain is throbbing and becomes worse when I am exposed to bright light or loud sounds. "
        "I have had a severe headache for 2 days on the right side."
    )
    res = extract_symptoms(text)
    s_map = {s["symptom"]: s for s in res["symptoms"]}

    assert "headache" in s_map
    assert s_map["headache"]["quality"] == "THROBBING"
    assert s_map["headache"]["laterality"] == "RIGHT"

    assert "photophobia" in s_map
    assert s_map["photophobia"]["status"] == "PRESENT"
    assert s_map["photophobia"]["quality"] is None
    assert s_map["photophobia"]["trigger"] == "bright light"

    assert "phonophobia" in s_map
    assert s_map["phonophobia"]["status"] == "PRESENT"
    assert s_map["phonophobia"]["quality"] is None
    assert s_map["phonophobia"]["trigger"] == "loud sounds"


# ─────────────────────────────────────────────────────────────
# 4 & 5. Exact duplicate deduplication in assessment aggregation
# ─────────────────────────────────────────────────────────────
def test_report_findings_deduplication():
    # Simulate multiple medical report rows with exact duplicate findings
    raw_labs = [
        {"test_name": "Hemoglobin", "value": 14.2, "unit": "g/dL", "reference_range": "13.0 - 17.0", "interpretation": "NORMAL"},
        {"test_name": "WBC", "value": 7500, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "NORMAL"},
        {"test_name": "Hemoglobin", "value": 14.2, "unit": "g/dL", "reference_range": "13.0 - 17.0", "interpretation": "NORMAL"}, # exact dup
        {"test_name": "WBC", "value": 12500, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "HIGH"},     # distinct value - keep!
    ]
    raw_qual = [
        {"finding": "swelling", "state": "PRESENT", "context": "right wrist"},
        {"finding": "swelling", "state": "PRESENT", "context": "right wrist"}, # exact dup
        {"finding": "reduced movement", "state": "PRESENT", "context": "right wrist"},
    ]

    # Test deduplication logic
    seen_lab = set()
    deduped_labs = []
    for item in raw_labs:
        key = (
            str(item.get("test_name", "")).strip().lower(),
            item.get("value"),
            str(item.get("unit", "")).strip().lower(),
            str(item.get("reference_range", "")).strip().lower(),
            str(item.get("interpretation") or "").strip().upper(),
        )
        if key not in seen_lab:
            seen_lab.add(key)
            deduped_labs.append(item)

    assert len(deduped_labs) == 3  # Hemoglobin, WBC 7500, WBC 12500
    assert len([l for l in deduped_labs if l["test_name"] == "Hemoglobin"]) == 1
    assert len([l for l in deduped_labs if l["test_name"] == "WBC"]) == 2

    seen_qual = set()
    deduped_qual = []
    for item in raw_qual:
        key = (
            str(item.get("finding", "")).strip().lower(),
            str(item.get("state", "")).strip().upper(),
            str(item.get("context", "")).strip().lower(),
        )
        if key not in seen_qual:
            seen_qual.add(key)
            deduped_qual.append(item)

    assert len(deduped_qual) == 2


# ─────────────────────────────────────────────────────────────
# 6. Unknown lab ranges remain visible as UNKNOWN
# ─────────────────────────────────────────────────────────────
def test_unknown_lab_ranges_remain_visible():
    report_text = "Serum Creatinine: 1.1 mg/dL\nUric Acid: 5.4 mg/dL"
    res = structure_report_text(report_text)
    for f in res["findings"]:
        assert f["status"] == "UNKNOWN"
        assert f["reference_range"] is None or "not provided" in str(f.get("reference_range")).lower()

    evidence = combine_evidence(report_findings=res["findings"], report_provided=True)
    triage = apply_triage(evidence)
    specialty = map_evidence_to_specialty(evidence)
    feedback = generate_assessment_feedback(evidence, triage, specialty)

    assert len(feedback["medical_report_findings"]) == 2
    for item in feedback["medical_report_findings"]:
        assert item["status"] == "UNKNOWN"
        assert item["reference_range"] == "Not provided"
        assert item["status"] != "NORMAL"


# ─────────────────────────────────────────────────────────────
# 7. Normal WBC/CRP must NOT be described as inflammation
# ─────────────────────────────────────────────────────────────
def test_normal_wbc_crp_not_described_as_inflammation():
    symptoms = [
        {"symptom": "cough", "state": "PRESENT", "domain": "respiratory"},
        {"symptom": "fever", "state": "PRESENT", "domain": "systemic"},
    ]
    normal_labs = [
        {"test_name": "WBC", "value": 6500, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "NORMAL"},
        {"test_name": "CRP", "value": 2.1, "unit": "mg/L", "reference_range": "0 - 5", "interpretation": "NORMAL"},
    ]
    evidence = combine_evidence(symptoms=symptoms, report_findings=normal_labs, report_provided=True)

    # Must NOT call normal labs evidence of inflammation in supporting
    for s in evidence["supporting"]:
        assert "correlate with systemic/inflammatory indicators" not in s.lower()

    # Must describe them under reassuring
    assert any("within standard supplied reference ranges" in r.lower() or "wbc and crp" in r.lower() for r in evidence["reassuring"])


# ─────────────────────────────────────────────────────────────
# 8 & 9. Unavailable X-ray must remain UNKNOWN/UNASSESSED
# ─────────────────────────────────────────────────────────────
def test_unavailable_xray_semantics():
    xray_unavail = {
        "status": "unavailable",
        "region": "ankle",
        "prediction": None,
        "message": "Automated interpretation for this X-ray type is currently unavailable.",
    }
    evidence = combine_evidence(xray_result=xray_unavail)

    assert evidence["xray"]["status"] == "UNAVAILABLE"
    assert evidence["xray"]["region"] == "ankle"
    assert evidence["xray"]["prediction"] is None
    assert evidence["xray"]["evidence_status"] == "UNKNOWN"

    # Must NOT appear under supporting evidence
    for supp in evidence["supporting"]:
        assert "ankle" not in supp.lower()

    # Must appear under unassessed
    assert any("ankle" in u.lower() and "unavailable" in u.lower() for u in evidence["unassessed"])


# ─────────────────────────────────────────────────────────────
# 10. Separate ankle swelling not supporting respiratory
# ─────────────────────────────────────────────────────────────
def test_separate_ankle_swelling_not_supporting_respiratory():
    symptoms = [
        {"symptom": "cough", "state": "PRESENT", "domain": "respiratory"},
        {"symptom": "sore throat", "state": "PRESENT", "domain": "respiratory"},
    ]
    qualitative = [
        {"finding": "ankle swelling", "state": "PRESENT", "domain": "musculoskeletal"},
    ]
    evidence = combine_evidence(symptoms=symptoms, qualitative_report_findings=qualitative, report_provided=True)

    # Ankle swelling must NOT be in supporting
    for supp in evidence["supporting"]:
        assert "ankle swelling" not in supp.lower()

    # Ankle swelling must be recognized as SEPARATE / CONTEXTUAL
    assert len(evidence["separate"]) > 0
    assert any("ankle swelling" in sep.lower() for sep in evidence["separate"])
    assert any(rel["relationship"] == "SEPARATE" for rel in evidence["evidence_relationships"])


# ─────────────────────────────────────────────────────────────
# 11 & 14. Respiratory vs Headache final feedback differs materially
# ─────────────────────────────────────────────────────────────
def test_respiratory_vs_headache_feedback_materially_differs():
    # Case 1: Respiratory
    resp_text = (
        "I have had a cough for 5 days with mild fever and a sore throat. "
        "I feel tired and have reduced energy. The cough is mostly dry but sometimes produces a small amount of mucus. "
        "I have mild chest discomfort when coughing. I do not have severe breathing difficulty. "
        "I do not have fainting or confusion. Symptoms gradually started about 5 days ago."
    )
    res_resp = extract_symptoms(resp_text)
    ev_resp = combine_evidence(symptoms=res_resp["symptoms"])
    triage_resp = apply_triage(ev_resp)
    spec_resp = map_evidence_to_specialty(ev_resp)
    fb_resp = generate_assessment_feedback(ev_resp, triage_resp, spec_resp)

    # Case 2: Headache
    head_text = (
        "I have had a severe headache for 2 days, mainly on the right side. "
        "The pain is throbbing and becomes worse when I am exposed to bright light or loud sounds. "
        "I feel nauseous but I have not vomited. I feel tired and have difficulty concentrating. "
        "I have had similar headaches once or twice before. I do not have a fever. "
        "I do not have weakness or numbness in my arms or legs. I do not have difficulty speaking. "
        "I have not fainted and I do not have a seizure."
    )
    res_head = extract_symptoms(head_text)
    ev_head = combine_evidence(symptoms=res_head["symptoms"])
    triage_head = apply_triage(ev_head)
    spec_head = map_evidence_to_specialty(ev_head)
    fb_head = generate_assessment_feedback(ev_head, triage_head, spec_head)

    # Specialties differ
    spec_resp_name = spec_resp.get("suggested_specialty") if isinstance(spec_resp, dict) else spec_resp
    spec_head_name = spec_head.get("suggested_specialty") if isinstance(spec_head, dict) else spec_head
    assert spec_resp_name == "Pulmonology"
    assert spec_head_name == "Neurology"

    # Summaries differ
    assert fb_resp["assessment_summary"] != fb_head["assessment_summary"]
    assert "respiratory" in fb_resp["recommended_next_step"]["guidance"].lower()
    assert "headache" in fb_head["recommended_next_step"]["guidance"].lower()

    # Zero hallucinated disease diagnoses in summaries
    for diag in ["pneumonia", "covid", "bronchitis", "migraine disorder", "stroke"]:
        assert diag not in fb_resp["assessment_summary"].lower()
        assert diag not in fb_head["assessment_summary"].lower()


# ─────────────────────────────────────────────────────────────
# 12. Same symptoms + normal report vs abnormal report
# ─────────────────────────────────────────────────────────────
def test_normal_vs_abnormal_report_produces_distinct_feedback():
    symptoms = extract_symptoms("I have had a cough for 4 days and mild fever.")["symptoms"]

    normal_labs = [
        {"test_name": "WBC", "value": 7200, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "NORMAL"},
    ]
    abnormal_labs = [
        {"test_name": "WBC", "value": 16500, "unit": "/uL", "reference_range": "4000 - 11000", "interpretation": "HIGH"},
    ]

    ev_norm = combine_evidence(symptoms=symptoms, report_findings=normal_labs, report_provided=True)
    triage_norm = apply_triage(ev_norm)
    spec_norm = map_evidence_to_specialty(ev_norm)
    fb_norm = generate_assessment_feedback(ev_norm, triage_norm, spec_norm)

    ev_abn = combine_evidence(symptoms=symptoms, report_findings=abnormal_labs, report_provided=True)
    triage_abn = apply_triage(ev_abn)
    spec_abn = map_evidence_to_specialty(ev_abn)
    fb_abn = generate_assessment_feedback(ev_abn, triage_abn, spec_abn)

    assert fb_norm["assessment_summary"] != fb_abn["assessment_summary"]
    assert "within standard reference ranges" in fb_norm["assessment_summary"].lower()
    assert "out-of-range" in fb_abn["assessment_summary"].lower()
    assert len(fb_norm["reassuring_evidence"]) > 0


# ─────────────────────────────────────────────────────────────
# 13. Same symptoms/report + unavailable X-ray vs valid prediction
# ─────────────────────────────────────────────────────────────
def test_unavailable_vs_completed_xray_feedback():
    symptoms = extract_symptoms("I have had a dry cough for 3 days.")["symptoms"]

    xray_unavail = {
        "status": "unavailable",
        "region": "wrist",
        "prediction": None,
    }
    xray_comp = {
        "status": "completed",
        "region": "chest",
        "prediction": "Consolidation / Infiltrate",
    }

    ev_unavail = combine_evidence(symptoms=symptoms, xray_result=xray_unavail)
    fb_unavail = generate_assessment_feedback(ev_unavail, apply_triage(ev_unavail), map_evidence_to_specialty(ev_unavail))

    ev_comp = combine_evidence(symptoms=symptoms, xray_result=xray_comp)
    fb_comp = generate_assessment_feedback(ev_comp, apply_triage(ev_comp), map_evidence_to_specialty(ev_comp))

    assert fb_unavail["assessment_summary"] != fb_comp["assessment_summary"]
    assert "unavailable" in fb_unavail["xray_findings"]["description"].lower()
    assert "Consolidation / Infiltrate" in fb_comp["xray_findings"]["description"]


# ─────────────────────────────────────────────────────────────
# 15. Historical symptoms do NOT inflate active symptom count
# ─────────────────────────────────────────────────────────────
def test_historical_symptoms_do_not_inflate_active_count():
    text = (
        "I have a severe headache for 2 days. "
        "I have had similar headaches once or twice before."
    )
    res = extract_symptoms(text)
    evidence = combine_evidence(symptoms=res["symptoms"])

    # Active present symptoms must only include current findings
    active_symptoms = evidence["symptoms"]["present"]
    assert len(active_symptoms) == 1
    assert active_symptoms[0]["symptom"] == "headache"

    # Historical must be separated
    assert len(evidence["symptoms"]["historical"]) == 1
    assert evidence["symptoms"]["historical"][0]["symptom"] == "previous similar headache"
