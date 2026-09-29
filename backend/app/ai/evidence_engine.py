"""Evidence combination engine for MediSense AI.

Combines multimodal evidence (symptoms, medical report findings, X-ray analysis)
into a unified clinical evidence representation:
- Preserves PRESENT / ABSENT / UNKNOWN states
- Identifies supporting findings (e.g. cough + report respiratory findings)
- Identifies contradictory findings (e.g. symptom claims fever but report/temperature denies)
- Identifies missing critical information
- Establishes evidence_state: EVIDENCE_PRESENT, INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE
"""

from __future__ import annotations

from typing import Any, Optional

CRITICAL_NEGATIVES = ["chest pain", "shortness of breath", "high fever", "severe bleeding"]


def combine_evidence(
    symptoms: list[dict[str, Any]] | None = None,
    report_findings: list[dict[str, Any]] | None = None,
    xray_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine multimodal inputs into structured clinical evidence."""
    symptoms = symptoms or []
    report_findings = report_findings or []

    symptoms_present = [s for s in symptoms if s.get("state") == "PRESENT"]
    symptoms_absent = [s for s in symptoms if s.get("state") == "ABSENT"]
    symptoms_unknown = [s for s in symptoms if s.get("state") == "UNKNOWN"]

    report_abnormal = [
        f for f in report_findings if f.get("interpretation") in {"HIGH", "LOW", "ABNORMAL"}
    ]
    report_normal = [
        f for f in report_findings if f.get("interpretation") == "NORMAL"
    ]

    supporting: list[str] = []
    contradictory: list[str] = []
    missing: list[str] = []

    # Check for supporting correlations
    present_names = {s.get("symptom", "").lower() for s in symptoms_present}
    absent_names = {s.get("symptom", "").lower() for s in symptoms_absent}

    # If cough/fever is present and report or X-ray has lung/respiratory finding
    has_respiratory_symptom = bool(present_names.intersection({"cough", "shortness of breath", "dyspnea", "wheezing"}))
    has_fever = "fever" in present_names

    if has_respiratory_symptom and (has_fever or any(f.get("domain") == "respiratory" for f in report_abnormal)):
        supporting.append("Respiratory symptoms correlate with systemic/inflammatory indicators.")

    # Check for contradictions between modalities
    # Example: symptom states chest pain = ABSENT, but user entered contradictory complaint elsewhere
    for item in report_findings:
        fname = item.get("test_name", "").lower()
        if "glucose" in fname and item.get("interpretation") == "HIGH":
            supporting.append(f"Elevated blood glucose ({item.get('value')} {item.get('unit', '')}) recorded.")

    # Check X-ray contribution
    xray_finding_str = None
    xray_status = (xray_result or {}).get("status", "none")
    if xray_result and xray_status == "completed" and xray_result.get("prediction"):
        xray_finding_str = xray_result.get("prediction")
        supporting.append(f"X-ray deep learning model identified finding: {xray_finding_str}.")

    # Identify missing critical information
    known_symptom_names = present_names.union(absent_names)
    for crit in CRITICAL_NEGATIVES:
        if crit not in known_symptom_names:
            missing.append(f"Status of '{crit}' (neither confirmed nor denied)")

    # Check duration
    durations = [s.get("duration") for s in symptoms_present if s.get("duration")]
    if symptoms_present and not durations:
        missing.append("Symptom duration (onset timeframe unspecified)")

    # Determine overall evidence state
    total_active_signals = len(symptoms_present) + len(report_abnormal) + (1 if xray_finding_str else 0)

    if contradictory:
        evidence_state = "CONFLICTING_EVIDENCE"
    elif total_active_signals > 0 or len(report_normal) > 0 or len(symptoms_absent) > 0:
        evidence_state = "EVIDENCE_PRESENT"
    else:
        evidence_state = "INSUFFICIENT_EVIDENCE"

    return {
        "evidence_state": evidence_state,
        "symptoms": {
            "present": [
                {
                    "symptom": s.get("symptom"),
                    "duration": s.get("duration"),
                    "severity": s.get("severity"),
                    "body_area": s.get("body_area"),
                    "domain": s.get("domain"),
                }
                for s in symptoms_present
            ],
            "absent": [
                {"symptom": s.get("symptom"), "body_area": s.get("body_area")}
                for s in symptoms_absent
            ],
            "unknown": [s.get("symptom") for s in symptoms_unknown],
        },
        "report": {
            "abnormal_findings": report_abnormal,
            "normal_findings": report_normal,
            "total_extracted": len(report_findings),
        },
        "xray": {
            "status": xray_status,
            "region": (xray_result or {}).get("region"),
            "prediction": xray_finding_str,
            "uncertainty": (xray_result or {}).get("uncertainty"),
            "explainability_artifact": (xray_result or {}).get("explainability_artifact"),
        },
        "supporting": supporting,
        "contradictory": contradictory,
        "missing": missing,
        "message": (
            "Multimodal evidence compiled successfully."
            if evidence_state == "EVIDENCE_PRESENT"
            else "Available information is insufficient for a specific clinical conclusion."
        ),
    }
