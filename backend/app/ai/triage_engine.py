
"""Rule-based safety and triage engine for MediSense AI.

Deterministic decision support only; this module does not provide a diagnosis.
Emergency red flags take precedence over incomplete or conflicting non-emergency
information. Missing/unknown values are never silently treated as positive,
negative, or severe evidence.
"""
from __future__ import annotations

import re
from typing import Any

PATHWAYS = ("EMERGENCY", "CONSULTATION", "MILD")
RULES_VERSION = "triage-rules-v2.2"

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


def _text(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _records(value: Any) -> list[dict[str, Any]]:
    """Keep valid record dictionaries; ignore malformed items safely."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _number(value: Any) -> float | None:
    """Parse numeric lab values, including simple numeric strings."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if re.fullmatch(r"\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*", value):
            try:
                return float(value.strip())
            except ValueError:
                return None
    return None


def _result(
    *,
    pathway: str | None,
    status: str,
    triggered: list[str],
    ignored: list[str],
    supporting: list[str],
    unknown: list[Any],
    rationale: str,
    guidance: str,
    next_steps: str,
) -> dict[str, Any]:
    return {
        "pathway": pathway,
        "final_triage_level": pathway,
        "status": status,
        "rules_version": RULES_VERSION,
        "rules_triggered": triggered,
        "triggered_rules": triggered,
        "ignored_rules": ignored,
        "supporting_evidence": supporting,
        "unknown_information": unknown,
        "why_this_pathway": rationale,
        "guidance": guidance,
        "next_steps": next_steps,
        "clinically_validated": False,
    }


def apply_triage(evidence: dict[str, Any] | None) -> dict[str, Any]:
    """Apply deterministic rules without forcing an unsupported MILD outcome."""
    evidence = evidence if isinstance(evidence, dict) else {}
    state = _text(evidence.get("evidence_state")).upper()

    symptoms_meta = evidence.get("symptoms")
    symptoms_meta = symptoms_meta if isinstance(symptoms_meta, dict) else {}
    present = _records(symptoms_meta.get("present", []))
    present_names = {
        _text(s.get("symptom") or s.get("finding")) for s in present
    }
    present_names.discard("")

    report_meta = evidence.get("report")
    report_meta = report_meta if isinstance(report_meta, dict) else {}
    abnormal_reports = _records(report_meta.get("abnormal_findings", []))
    normal_reports = _records(report_meta.get("normal_findings", []))

    xray_meta = evidence.get("xray")
    xray_meta = xray_meta if isinstance(xray_meta, dict) else {}
    xray_prediction = _text(xray_meta.get("prediction"))

    missing_raw = evidence.get("missing", [])
    missing_info = (
        missing_raw
        if isinstance(missing_raw, list)
        else [str(missing_raw)] if missing_raw else []
    )

    contradictions = evidence.get("contradictory", [])
    if not isinstance(contradictions, list):
        contradictions = [str(contradictions)] if contradictions else []

    emergency: list[str] = []
    consult: list[str] = []
    supporting: list[str] = []

    # E1: concerning chest pain.
    # Unknown severity alone does not count as severe evidence.
    has_acute_chest_pain = False

    for symptom in present:
        name = _text(symptom.get("symptom") or symptom.get("finding"))
        severity = _text(symptom.get("severity"))
        context = " ".join(
            filter(
                None,
                [
                    _text(symptom.get("trigger")),
                    _text(symptom.get("context")),
                    _text(symptom.get("onset")),
                ],
            )
        )

        chest_pain = "chest pain" in name or name == "chest discomfort"
        cough_triggered = "cough" in context
        explicit_acute = any(
            term in context for term in ("acute", "at rest", "resting", "sudden")
        )
        explicitly_concerning = severity in {"moderate", "severe"} or explicit_acute

        if chest_pain and not cough_triggered and explicitly_concerning:
            has_acute_chest_pain = True
            supporting.append(
                f"Potentially concerning chest pain reported "
                f"({symptom.get('context') or name})."
            )

    if has_acute_chest_pain:
        emergency.append("rule_e1_chest_pain_present")

    # E2: severe breathing difficulty or dyspnea with concerning chest pain.
    has_severe_dyspnea = any(
        _text(s.get("symptom") or s.get("finding"))
        in {"severe breathing difficulty", "severe shortness of breath"}
        or (
            _text(s.get("symptom") or s.get("finding"))
            in {"shortness of breath", "difficulty breathing", "dyspnea"}
            and _text(s.get("severity")) == "severe"
        )
        for s in present
    )

    has_sob = any(
        _text(s.get("symptom") or s.get("finding"))
        in {"shortness of breath", "difficulty breathing", "dyspnea"}
        for s in present
    )

    if has_severe_dyspnea or (has_sob and has_acute_chest_pain):
        emergency.append("rule_e2_dyspnea_with_chest_pain")
        supporting.append(
            "Severe respiratory distress or dyspnea with concerning chest pain identified."
        )

    # E3: severe headache with vomiting/dizziness, or fainting/syncope.
    severe_headache = any(
        _text(s.get("symptom") or s.get("finding")) == "headache"
        and _text(s.get("severity")) == "severe"
        for s in present
    )

    vomiting_or_dizziness = any(
        _text(s.get("symptom") or s.get("finding"))
        in {"vomiting", "dizziness"}
        for s in present
    )

    fainting = any(
        _text(s.get("symptom") or s.get("finding"))
        in {"fainting", "syncope"}
        for s in present
    )

    if (severe_headache and vomiting_or_dizziness) or fainting:
        emergency.append("rule_e3_severe_acute_presentation")
        supporting.append(
            "Severe acute neurological presentation or fainting/syncope reported."
        )

    # E4: retain existing configured laboratory thresholds.
    # Numeric strings are parsed so they cannot silently bypass these checks.
    for report in abnormal_reports:
        test = _text(
            report.get("test_name") or report.get("original_test_name")
        )
        value = _number(report.get("value"))

        if value is None:
            continue

        if "glucose" in test and (value > 350 or value < 50):
            emergency.append("rule_e4_critical_glycemic_emergency")
            supporting.append(
                f"Potentially critical glucose value recorded: "
                f"{report.get('value')} {report.get('unit') or ''}."
            )

        if "hemoglobin" in test and value < 7.0:
            emergency.append("rule_e4_severe_critical_anemia")
            supporting.append(
                f"Potentially critical hemoglobin value recorded: "
                f"{report.get('value')} {report.get('unit') or ''}."
            )

        if "platelet" in test and value < 30000:
            emergency.append("rule_e4_severe_critical_thrombocytopenia")
            supporting.append(
                f"Potentially critical platelet value recorded: "
                f"{report.get('value')} {report.get('unit') or ''}."
            )

    emergency = list(dict.fromkeys(emergency))

    # Emergency red flags take precedence over incomplete/conflicting
    # non-emergency information.
    if emergency:
        return _result(
            pathway="EMERGENCY",
            status="ok",
            triggered=emergency,
            ignored=[r for r in ALL_RULES if r not in emergency],
            supporting=supporting,
            unknown=missing_info,
            rationale=(
                "Potentially serious red-flag indicators were identified. "
                "Immediate clinical evaluation is required."
            ),
            guidance=(
                "Your symptoms or findings may indicate an urgent health concern. "
                "Seek emergency medical care immediately at the nearest hospital "
                "or emergency facility. Do not wait for pharmacy medication."
            ),
            next_steps=(
                "Go to the nearest emergency department or call emergency "
                "medical services immediately."
            ),
        )

    # Never assign MILD when evidence is contradictory.
    if state == "CONFLICTING_EVIDENCE" or contradictions:
        return _result(
            pathway=None,
            status="CONFLICTING_EVIDENCE",
            triggered=["rule_conflicting_evidence_flag"],
            ignored=[
                r for r in ALL_RULES
                if r != "rule_conflicting_evidence_flag"
            ],
            supporting=[str(item) for item in contradictions],
            unknown=missing_info,
            rationale=(
                "The provided inputs contain contradictory information "
                "that requires clinical clarification."
            ),
            guidance=(
                "Symptoms, reports, or imaging contain conflicting indicators "
                "that need review."
            ),
            next_steps=(
                "Review the inputs and consult a physician for clarification "
                "and direct examination."
            ),
        )

    # Incomplete evidence must not fall through to MILD.
    has_any_clinical_evidence = bool(
        present or abnormal_reports or normal_reports or xray_prediction
    )

    if state == "INSUFFICIENT_EVIDENCE" or not has_any_clinical_evidence:
        unknown = missing_info or [
            "No confirmed active symptoms or interpretable clinical findings were available."
        ]
        return _result(
            pathway=None,
            status="INSUFFICIENT_EVIDENCE",
            triggered=[],
            ignored=ALL_RULES.copy(),
            supporting=[],
            unknown=unknown,
            rationale=(
                "Available clinical evidence is insufficient to determine "
                "a triage pathway."
            ),
            guidance=(
                "Please confirm current symptoms and provide relevant report "
                "details if available. Do not interpret this result as a MILD assessment."
            ),
            next_steps=(
                "Complete the missing assessment information or seek "
                "professional advice if concerned."
            ),
        )

    # C1: abnormal medical report findings.
    if abnormal_reports:
        consult.append("rule_c1_abnormal_laboratory_findings")
        supporting.extend(
            f"Abnormal reported parameter: {r.get('test_name')} "
            f"({r.get('value')} {r.get('unit') or ''}; "
            f"{r.get('interpretation') or r.get('status') or 'flagged abnormal'})."
            for r in abnormal_reports
        )

    # C2: positive imaging prediction; ignore explicit non-finding labels.
    if xray_prediction and xray_prediction not in {
        "normal", "clear", "no finding", "unavailable", "none", "unknown"
    }:
        consult.append("rule_c2_positive_imaging_finding")
        supporting.append(
            f"Radiographic model finding recorded: {xray_meta.get('prediction')}."
        )

    # C3: persistent symptoms, supporting numeric day/week/month formats.
    for symptom in present:
        duration = _text(symptom.get("duration"))
        day_match = re.search(r"\b(\d+)\s*days?\b", duration)
        day_count = int(day_match.group(1)) if day_match else None

        persistent = bool(
            re.search(r"\b(week|weeks|month|months|year|years)\b", duration)
        ) or (day_count is not None and day_count >= 3)

        if persistent:
            consult.append("rule_c3_persistent_symptom_duration")
            supporting.append(
                f"Persistent symptom duration reported ({symptom.get('duration')})."
            )
            break

    # C4: multiple confirmed active symptoms or explicitly moderate severity.
    if len(present_names) >= 2 or any(
        _text(s.get("severity")) == "moderate" for s in present
    ):
        consult.append("rule_c4_multi_symptom_or_moderate_severity")
        supporting.append(
            f"Multiple active findings or moderate severity identified "
            f"({len(present_names)} distinct symptom(s))."
        )

    # C5: symptom terms configured for professional assessment.
    consult_names = {
        "headache", "migraine", "photophobia", "phonophobia",
        "abdominal pain", "wheezing", "arthralgia", "back pain",
        "wrist pain", "wrist injury", "fall/trauma", "reduced movement",
        "swelling", "tenderness", "chest pain", "chest discomfort",
    }

    matched = present_names.intersection(consult_names)
    if matched:
        consult.append("rule_c5_specialty_consultation_indicated")
        supporting.append(
            "Symptoms requiring clinical assessment under this ruleset: "
            f"{', '.join(sorted(matched))}."
        )

    consult = list(dict.fromkeys(consult))

    if consult:
        return _result(
            pathway="CONSULTATION",
            status="ok",
            triggered=consult,
            ignored=[r for r in ALL_RULES if r not in consult],
            supporting=supporting,
            unknown=missing_info,
            rationale=(
                "Structured findings indicate the need for professional "
                "evaluation; no configured emergency rule was triggered."
            ),
            guidance=(
                "A qualified healthcare professional should review the "
                "symptoms and available medical reports."
            ),
            next_steps=(
                "Arrange a consultation with a suitable clinician and bring "
                "the available reports."
            ),
        )

    # MILD requires confirmed active symptoms and no configured
    # emergency/consultation trigger. It is not a medically validated diagnosis.
    return _result(
        pathway="MILD",
        status="ok",
        triggered=["rule_m1_mild_self_limiting_profile"],
        ignored=[
            r for r in ALL_RULES
            if r != "rule_m1_mild_self_limiting_profile"
        ],
        supporting=[
            "Confirmed active symptom(s) present; no configured emergency "
            "or consultation rule triggered."
        ],
        unknown=missing_info,
        rationale=(
            "No configured emergency or consultation rule was triggered "
            "by the available evidence. This does not rule out disease."
        ),
        guidance=(
            "Monitor symptoms and seek professional medical advice if they "
            "worsen, persist, or cause concern."
        ),
        next_steps=(
            "Monitor symptoms and consult a clinician if warning signs "
            "develop or symptoms persist."
        ),
    )
