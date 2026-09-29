"""Rule-based safety and triage engine for MediSense AI.

Evaluates structured clinical evidence against deterministic safety rules:
- EMERGENCY: Severe warning signs (acute chest pain, respiratory distress, critical labs)
- CONSULTATION: Persistent/moderate symptoms, abnormal lab tests, positive imaging findings
- MILD: Mild, self-limiting symptoms with normal labs and no red flags

CRITICAL PRINCIPLE:
The LLM must NEVER independently decide or override the triage pathway.
This engine is 100% deterministic and auditable.
"""

from __future__ import annotations

from typing import Any

PATHWAYS = ("EMERGENCY", "CONSULTATION", "MILD")
RULES_VERSION = "triage-rules-v1.0"


def apply_triage(evidence: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic clinical safety rules to multimodal evidence."""
    evidence = evidence or {}
    state = evidence.get("evidence_state")

    if state in {"INSUFFICIENT_EVIDENCE", None}:
        return {
            "pathway": None,
            "status": "INSUFFICIENT_EVIDENCE",
            "rules_version": RULES_VERSION,
            "rules_triggered": [],
            "why_this_pathway": "Available clinical evidence is insufficient to determine a triage pathway.",
            "guidance": "Please provide more detailed symptom descriptions or upload relevant health reports.",
            "next_steps": "Start a new assessment or complete missing information.",
            "clinically_validated": False,
        }

    if state == "CONFLICTING_EVIDENCE":
        return {
            "pathway": None,
            "status": "CONFLICTING_EVIDENCE",
            "rules_version": RULES_VERSION,
            "rules_triggered": ["rule_conflicting_evidence_flag"],
            "why_this_pathway": "The provided inputs contain contradictory information that requires clinical clarification.",
            "guidance": "Information provided across symptoms, reports, or imaging contains conflicting indicators.",
            "next_steps": "Review inputs or consult a physician for a direct physical examination.",
            "clinically_validated": False,
        }

    symptoms_meta = evidence.get("symptoms", {})
    present_symptoms = symptoms_meta.get("present", [])
    present_names = {s.get("symptom", "").lower() for s in present_symptoms}

    report_meta = evidence.get("report", {})
    abnormal_reports = report_meta.get("abnormal_findings", [])
    normal_reports = report_meta.get("normal_findings", [])

    xray_meta = evidence.get("xray", {})
    xray_prediction = xray_meta.get("prediction")

    rules_triggered: list[str] = []

    # ─────────────────────────────────────────────────────────────
    # 1. EMERGENCY RULES CHECK
    # ─────────────────────────────────────────────────────────────
    # Rule E1: Acute chest pain
    if "chest pain" in present_names:
        rules_triggered.append("rule_e1_chest_pain_present")

    # Rule E2: Shortness of breath with chest pain or severe difficulty
    if "shortness of breath" in present_names and "chest pain" in present_names:
        rules_triggered.append("rule_e2_dyspnea_with_chest_pain")

    # Rule E3: Severe pain or acute severe headache with vomiting/dizziness
    has_severe = any(s.get("severity") == "severe" for s in present_symptoms)
    if has_severe and any(n in present_names for n in {"chest pain", "shortness of breath", "headache"}):
        rules_triggered.append("rule_e3_severe_acute_presentation")

    # Rule E4: Critical laboratory values (e.g. extreme glucose, profound anemia, severe thrombocytopenia)
    for rep in abnormal_reports:
        test = rep.get("test_name", "").lower()
        val = rep.get("value")
        if isinstance(val, (int, float)):
            if "glucose" in test and (val > 350 or val < 50):
                rules_triggered.append("rule_e4_critical_glycemic_emergency")
            if "hemoglobin" in test and val < 7.0:
                rules_triggered.append("rule_e4_severe_critical_anemia")
            if "platelet" in test and val < 30000:
                rules_triggered.append("rule_e4_severe_critical_thrombocytopenia")

    if rules_triggered:
        return {
            "pathway": "EMERGENCY",
            "status": "ok",
            "rules_version": RULES_VERSION,
            "rules_triggered": rules_triggered,
            "why_this_pathway": (
                "Potentially serious red-flag indicators were identified "
                f"({', '.join(rules_triggered)}). Immediate clinical evaluation is required."
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
        rules_triggered.append("rule_c1_abnormal_laboratory_findings")

    # Rule C2: Positive X-ray model prediction
    if xray_prediction and xray_prediction.lower() not in {"normal", "clear", "no finding"}:
        rules_triggered.append("rule_c2_positive_imaging_finding")

    # Rule C3: Persistent symptoms (e.g. > 3 days)
    for s in present_symptoms:
        dur = (s.get("duration") or "").lower()
        if any(w in dur for w in ["week", "weeks", "month", "months"]) or any(
            f"{n} day" in dur for n in range(4, 30)
        ):
            rules_triggered.append("rule_c3_persistent_symptom_duration")
            break

    # Rule C4: Multiple present symptoms or moderate severity
    if len(present_symptoms) >= 2 or any(s.get("severity") == "moderate" for s in present_symptoms):
        rules_triggered.append("rule_c4_multi_symptom_or_moderate_severity")

    # Rule C5: Specific clinical symptoms requiring doctor evaluation (e.g. abdominal pain, rash, wheezing)
    consult_names = {"abdominal pain", "wheezing", "migraine", "arthralgia", "back pain"}
    if present_names.intersection(consult_names):
        rules_triggered.append("rule_c5_specialty_consultation_indicated")

    if rules_triggered:
        return {
            "pathway": "CONSULTATION",
            "status": "ok",
            "rules_version": RULES_VERSION,
            "rules_triggered": rules_triggered,
            "why_this_pathway": (
                "Structured findings indicate conditions requiring professional physician evaluation "
                f"({', '.join(rules_triggered)})."
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
    # If no emergency flags, no abnormal labs, and only mild/isolated symptoms
    return {
        "pathway": "MILD",
        "status": "ok",
        "rules_version": RULES_VERSION,
        "rules_triggered": ["rule_m1_mild_self_limiting_profile"],
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
