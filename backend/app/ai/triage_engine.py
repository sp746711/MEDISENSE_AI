"""Rule-based safety and triage engine for MediSense AI.

Evaluates structured clinical evidence against deterministic safety rules:
- EMERGENCY: Severe warning signs (acute chest pain, severe respiratory distress, critical labs, syncope)
- CONSULTATION: Persistent/moderate symptoms, abnormal lab tests, positive imaging findings, multi-symptom presentation
- MILD: Mild, self-limiting symptoms with normal labs and no red flags

CRITICAL PRINCIPLES:
1. LLM must NEVER decide or override triage pathways.
2. Triage is 100% deterministic, auditable, and grounded in explicit evidence.
3. Ordinary symptom counts NEVER automatically produce emergency triage.
4. UNKNOWN is never treated as SEVERE or ABSENT.
5. Records triggered_rules, supporting_evidence, ignored_rules, unknown_information, and final_triage_level.
"""

from __future__ import annotations

from typing import Any

PATHWAYS = ("EMERGENCY", "CONSULTATION", "MILD")
RULES_VERSION = "triage-rules-v2.1"

ALL_RULES = [
    "rule_e1_chest_pain_present",
    "rule_e2_dyspnea_with_chest_pain",
    "rule_e3_severe_acute_presentation",
    "rule_e4_critical_glycemic_emergency",
    "rule_e4_severe_critical_anemia",
    "rule_e4_severe_critical_thrombocytopenia",
    "rule_c1_abnormal_laboratory_findings",
    "rule_c2_positive_imaging_finding",
    "rule_c3_persistent_symptom_duration",
    "rule_c4_multi_symptom_or_moderate_severity",
    "rule_c5_specialty_consultation_indicated",
    "rule_m1_mild_self_limiting_profile",
]


def apply_triage(evidence: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic clinical safety rules to multimodal evidence."""
    evidence = evidence or {}
    state = evidence.get("evidence_state")

    symptoms_meta = evidence.get("symptoms", {})
    present_symptoms = symptoms_meta.get("present", [])
    present_names = {s.get("symptom", "").lower() for s in present_symptoms}

    report_meta = evidence.get("report", {})
    abnormal_reports = report_meta.get("abnormal_findings", [])
    normal_reports = report_meta.get("normal_findings", [])

    xray_meta = evidence.get("xray", {})
    xray_prediction = xray_meta.get("prediction")

    missing_info = evidence.get("missing", [])

    if state in {"INSUFFICIENT_EVIDENCE", None}:
        return {
            "pathway": None,
            "final_triage_level": None,
            "status": "INSUFFICIENT_EVIDENCE",
            "rules_version": RULES_VERSION,
            "rules_triggered": [],
            "triggered_rules": [],
            "ignored_rules": ALL_RULES,
            "supporting_evidence": [],
            "unknown_information": missing_info or ["No clinical inputs provided."],
            "why_this_pathway": "Available clinical evidence is insufficient to determine a triage pathway.",
            "guidance": "Please provide more detailed symptom descriptions or upload relevant health reports.",
            "next_steps": "Start a new assessment or complete missing information.",
            "clinically_validated": False,
        }

    if state == "CONFLICTING_EVIDENCE":
        return {
            "pathway": None,
            "final_triage_level": None,
            "status": "CONFLICTING_EVIDENCE",
            "rules_version": RULES_VERSION,
            "rules_triggered": ["rule_conflicting_evidence_flag"],
            "triggered_rules": ["rule_conflicting_evidence_flag"],
            "ignored_rules": [r for r in ALL_RULES if r != "rule_conflicting_evidence_flag"],
            "supporting_evidence": evidence.get("contradictory", []),
            "unknown_information": missing_info,
            "why_this_pathway": "The provided inputs contain contradictory information that requires clinical clarification.",
            "guidance": "Information provided across symptoms, reports, or imaging contains conflicting indicators.",
            "next_steps": "Review inputs or consult a physician for a direct physical examination.",
            "clinically_validated": False,
        }

    emergency_rules_triggered: list[str] = []
    consultation_rules_triggered: list[str] = []
    supporting_evidence_points: list[str] = []

    # ─────────────────────────────────────────────────────────────
    # 1. EMERGENCY RULES CHECK (Strict red-flag criteria only)
    # ─────────────────────────────────────────────────────────────
    # Rule E1: Acute / resting chest pain
    # Must NOT trigger on mild cough-triggered chest discomfort!
    has_acute_chest_pain = False
    for s in present_symptoms:
        name = s.get("symptom", "").lower()
        sev = (s.get("severity") or "").lower()
        trig = (s.get("trigger") or s.get("context") or "").lower()
        if "chest pain" in name and "cough" not in trig and sev != "mild":
            has_acute_chest_pain = True
            supporting_evidence_points.append(f"Acute chest pain reported ({s.get('context') or name}).")
    if has_acute_chest_pain:
        emergency_rules_triggered.append("rule_e1_chest_pain_present")

    # Rule E2: Shortness of breath with chest pain or severe breathing difficulty
    has_severe_dyspnea = any(
        s.get("symptom", "").lower() in {"severe breathing difficulty", "severe shortness of breath"}
        or (
            s.get("symptom", "").lower() in {"shortness of breath", "difficulty breathing", "dyspnea"}
            and (s.get("severity") or "").lower() == "severe"
        )
        for s in present_symptoms
    )
    has_sob = any(
        s.get("symptom", "").lower() in {"shortness of breath", "difficulty breathing", "dyspnea"}
        for s in present_symptoms
    )
    if has_severe_dyspnea or (has_sob and has_acute_chest_pain):
        emergency_rules_triggered.append("rule_e2_dyspnea_with_chest_pain")
        supporting_evidence_points.append("Severe respiratory distress / dyspnea identified.")

    # Rule E3: Severe acute presentation (severe headache with vomiting/dizziness, or fainting/syncope)
    # Symptom must be explicitly severe, paired with active red flags
    has_severe_headache = any(
        s.get("symptom", "").lower() == "headache" and (s.get("severity") or "").lower() == "severe"
        for s in present_symptoms
    )
    has_vomiting_or_dizziness = any(
        s.get("symptom", "").lower() in {"vomiting", "dizziness"}
        for s in present_symptoms
    )
    has_fainting = any(
        s.get("symptom", "").lower() in {"fainting", "syncope"}
        for s in present_symptoms
    )
    if (has_severe_headache and has_vomiting_or_dizziness) or has_fainting:
        emergency_rules_triggered.append("rule_e3_severe_acute_presentation")
        supporting_evidence_points.append("Severe acute neurological presentation with active red flags.")

    # Rule E4: Critical laboratory values (extreme glucose, profound anemia, severe thrombocytopenia)
    for rep in abnormal_reports:
        test = rep.get("test_name", "").lower()
        val = rep.get("value")
        if isinstance(val, (int, float)):
            if "glucose" in test and (val > 350 or val < 50):
                emergency_rules_triggered.append("rule_e4_critical_glycemic_emergency")
                supporting_evidence_points.append(f"Critical laboratory glucose value: {val} mg/dL.")
            if "hemoglobin" in test and val < 7.0:
                emergency_rules_triggered.append("rule_e4_severe_critical_anemia")
                supporting_evidence_points.append(f"Critical hemoglobin level: {val} g/dL (severe anemia).")
            if "platelet" in test and val < 30000:
                emergency_rules_triggered.append("rule_e4_severe_critical_thrombocytopenia")
                supporting_evidence_points.append(f"Critical thrombocytopenia: {val} /µL.")

    if emergency_rules_triggered:
        ignored = [r for r in ALL_RULES if r not in emergency_rules_triggered]
        return {
            "pathway": "EMERGENCY",
            "final_triage_level": "EMERGENCY",
            "status": "ok",
            "rules_version": RULES_VERSION,
            "rules_triggered": emergency_rules_triggered,
            "triggered_rules": emergency_rules_triggered,
            "ignored_rules": ignored,
            "supporting_evidence": supporting_evidence_points,
            "unknown_information": missing_info,
            "why_this_pathway": (
                "Potentially serious red-flag indicators were identified "
                f"({', '.join(emergency_rules_triggered)}). Immediate clinical evaluation is required."
            ),
            "guidance": (
                "Your symptoms or findings indicate potentially urgent health concerns. "
                "Seek emergency medical care immediately at the nearest hospital or emergency facility. "
                "Do not wait or rely on pharmacy medication first."
            ),
            "next_steps": "Go to the nearest emergency department or call emergency medical services immediately.",
            "clinically_validated": False,
        }

    # ─────────────────────────────────────────────────────────────
    # 2. CONSULTATION RULES CHECK
    # ─────────────────────────────────────────────────────────────
    # Rule C1: Any abnormal medical report finding
    if abnormal_reports:
        consultation_rules_triggered.append("rule_c1_abnormal_laboratory_findings")
        supporting_evidence_points.extend(
            [f"Abnormal laboratory parameter: {r.get('test_name')} ({r.get('value')} {r.get('unit', '')}, {r.get('interpretation')})"
             for r in abnormal_reports]
        )

    # Rule C2: Positive X-ray model prediction (only valid model findings, never unverified/unavailable)
    if (
        xray_prediction
        and str(xray_prediction).lower() not in {"normal", "clear", "no finding", "unavailable", "none"}
    ):
        consultation_rules_triggered.append("rule_c2_positive_imaging_finding")
        supporting_evidence_points.append(f"Positive radiographic deep learning finding: {xray_prediction}.")

    # Rule C3: Persistent symptoms (>= 3 days)
    for s in present_symptoms:
        dur = (s.get("duration") or "").lower()
        if any(w in dur for w in ["week", "weeks", "month", "months"]) or any(
            f"{n} day" in dur for n in range(3, 30)
        ):
            consultation_rules_triggered.append("rule_c3_persistent_symptom_duration")
            supporting_evidence_points.append(f"Persistent symptom duration reported ({dur}).")
            break

    # Rule C4: Multiple present symptoms or moderate severity
    if len(present_symptoms) >= 2 or any((s.get("severity") or "").lower() == "moderate" for s in present_symptoms):
        consultation_rules_triggered.append("rule_c4_multi_symptom_or_moderate_severity")
        supporting_evidence_points.append(f"Multiple active clinical findings identified ({len(present_symptoms)} present).")

    # Rule C5: Specific clinical symptoms requiring doctor evaluation (e.g. headache, abdominal pain, wheezing)
    consult_names = {"headache", "migraine", "photophobia", "phonophobia", "abdominal pain", "wheezing", "arthralgia", "back pain"}
    matched_consult_names = present_names.intersection(consult_names)
    if matched_consult_names:
        consultation_rules_triggered.append("rule_c5_specialty_consultation_indicated")
        supporting_evidence_points.append(f"Domain-specific symptoms requiring clinical assessment: {', '.join(sorted(matched_consult_names))}.")

    if consultation_rules_triggered:
        ignored = [r for r in ALL_RULES if r not in consultation_rules_triggered]
        return {
            "pathway": "CONSULTATION",
            "final_triage_level": "CONSULTATION",
            "status": "ok",
            "rules_version": RULES_VERSION,
            "rules_triggered": consultation_rules_triggered,
            "triggered_rules": consultation_rules_triggered,
            "ignored_rules": ignored,
            "supporting_evidence": supporting_evidence_points,
            "unknown_information": missing_info,
            "why_this_pathway": (
                "Structured findings indicate conditions requiring professional physician evaluation "
                f"({', '.join(consultation_rules_triggered)}). No emergency red-flag criteria were identified."
            ),
            "guidance": (
                "A medical consultation with a qualified healthcare professional is recommended. "
                "The doctor can examine physical findings, review lab results, and provide a definitive diagnosis."
            ),
            "next_steps": "Consult with a specialist or general physician, and take your test reports with you.",
            "clinically_validated": False,
        }

    # ─────────────────────────────────────────────────────────────
    # 3. MILD PATHWAY
    # ─────────────────────────────────────────────────────────────
    # If no emergency flags, no abnormal labs, and only mild/isolated self-limiting symptoms
    ignored = [r for r in ALL_RULES if r != "rule_m1_mild_self_limiting_profile"]
    return {
        "pathway": "MILD",
        "final_triage_level": "MILD",
        "status": "ok",
        "rules_version": RULES_VERSION,
        "rules_triggered": ["rule_m1_mild_self_limiting_profile"],
        "triggered_rules": ["rule_m1_mild_self_limiting_profile"],
        "ignored_rules": ignored,
        "supporting_evidence": ["Isolated mild finding without systemic red flags or abnormal laboratory markers."],
        "unknown_information": missing_info,
        "why_this_pathway": (
            "No emergency warning signs or abnormal lab parameters were identified. "
            "The presentation appears mild and suitable for close monitoring."
        ),
        "guidance": (
            "General health guidance and monitoring are suggested. Stay hydrated and rest. "
            "If symptoms worsen or persist beyond 3 to 5 days, seek professional medical evaluation. "
            "Local medical shops may provide approved over-the-counter supportive supplies, but do not replace a doctor."
        ),
        "next_steps": "Monitor symptoms at home. Consult a physician if warning signs develop.",
        "clinically_validated": False,
    }
