
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
RULES_VERSION = "triage-rules-v2.7"

ALL_RULES = [
    "rule_e1_chest_pain_present",
    "rule_e2_dyspnea_with_chest_pain",
    "rule_c6_breathing_difficulty_needs_assessment",
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


def _symptom_key(value: Any) -> str:
    """Normalize common wording variants without inferring clinical state."""
    text = _text(value)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    aliases = {
        "shortness of breath": "difficulty breathing",
        "breathlessness": "difficulty breathing",
        "dyspnea": "difficulty breathing",
        "sob": "difficulty breathing",
        "chest discomfort": "chest pain",
        "syncope": "fainting",
    }
    return aliases.get(text, text)


def _is_unavailable_prediction(value: Any) -> bool:
    text = _text(value)
    return not text or text in {
        "unavailable", "none", "unknown", "not assessed", "not available",
        "processing", "failed", "error", "unsupported", "no model",
        "inference unavailable", "not run",
    }


def _is_non_finding_prediction(value: Any) -> bool:
    """Recognize explicit non-finding labels; do not infer from arbitrary text."""
    text = _text(value)
    return text in {
        "normal", "clear", "no finding", "no findings", "negative",
        "no acute finding", "no acute findings", "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary abnormalities", "no evidence of acute disease",
    }


def _records(value: Any) -> list[dict[str, Any]]:
    """Keep valid record dictionaries; ignore malformed items safely."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _anatomy_is_independently_verified(record: dict[str, Any]) -> bool:
    """Require an explicit upstream anatomy-verification gate.

    A model's predicted body region is not independent verification. Missing
    metadata is treated as unverified for safety.
    """
    return (
        _text(record.get("anatomy_verification_status")).upper() == "COMPLETED"
        and record.get("anatomy_independently_verified") is True
    )


def _xray_finding_records(xray_meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect actionable findings only from anatomy-verified X-ray outputs.

    DenseNet-style label scores alone are not positive findings. Top-level
    findings require top-level anatomy verification; image-level findings
    require that image's own verification gate. Unverified findings are not
    allowed to influence consultation or triage.
    """
    candidates: list[dict[str, Any]] = []
    if _anatomy_is_independently_verified(xray_meta):
        candidates.extend(_records(xray_meta.get("findings")))

    for image in _records(xray_meta.get("images")):
        image_status = _text(image.get("status")).upper()
        if image_status and image_status not in {"COMPLETED", "PARTIAL", "AVAILABLE"}:
            continue
        if not _anatomy_is_independently_verified(image):
            continue
        candidates.extend(_records(image.get("findings")))

    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    nonfinding_labels = {
        "normal", "clear", "no finding", "no findings", "negative",
        "no finding detected", "no acute finding", "no acute findings",
        "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary abnormalities", "no evidence of acute disease",
    }
    for item in candidates:
        label = str(item.get("label") or item.get("name") or item.get("finding") or "").strip()
        if not label or _text(label) in nonfinding_labels:
            continue

        status = _text(item.get("status") or item.get("state")).upper()
        if item.get("detected") is False or item.get("is_positive") is False:
            continue
        if status in {"ABSENT", "NEGATIVE", "NORMAL", "NOT_DETECTED", "NOT DETECTED", "FALSE"}:
            continue

        score = item.get("score", item.get("model_score", item.get("confidence")))
        score_number = _number(score)

        # Raw score-only labels are not positive evidence.
        explicitly_positive = (
            item.get("is_actionable_finding") is True
            or item.get("detected") is True
            or item.get("is_positive") is True
            or status in {"PRESENT", "POSITIVE", "DETECTED"}
        )
        if not explicitly_positive:
            continue

        key = (label.casefold(), str(score_number) if score_number is not None else "")
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(item)
        normalized["label"] = label
        if score_number is not None:
            normalized["model_score"] = score_number
        accepted.append(normalized)
    return accepted


def _xray_prediction_is_verified(xray_meta: dict[str, Any]) -> bool:
    """Only trust an aggregate prediction with explicit anatomy verification."""
    return _anatomy_is_independently_verified(xray_meta)


def _format_xray_finding(item: dict[str, Any]) -> str:
    """Format an X-ray model output without presenting it as a diagnosis."""
    label = str(item.get("label") or item.get("name") or item.get("finding") or "Unlabelled finding")
    score = item.get("model_score", item.get("score", item.get("confidence")))
    score_number = _number(score)
    if score_number is not None:
        return f"{label} (model score: {score_number:.3f})"
    return label


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
    follow_up_required: bool = False,
    critical_unknowns: list[Any] | None = None,
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
        # Unknown red-flag answers are never treated as absent. The caller/UI
        # can use these fields to request clarification before presenting a
        # non-emergency pathway as final.
        "follow_up_required": bool(follow_up_required),
        "critical_unknowns": list(critical_unknowns or []),
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
        _symptom_key(s.get("symptom") or s.get("finding")) for s in present
    }
    present_names.discard("")

    report_meta = evidence.get("report")
    report_meta = report_meta if isinstance(report_meta, dict) else {}
    abnormal_reports = _records(report_meta.get("abnormal_findings", []))
    normal_reports = _records(report_meta.get("normal_findings", []))

    xray_meta = evidence.get("xray")
    xray_meta = xray_meta if isinstance(xray_meta, dict) else {}
    xray_prediction = _text(xray_meta.get("prediction"))
    xray_findings = _xray_finding_records(xray_meta)
    # An aggregate prediction is usable only when the aggregate itself carries
    # explicit independent anatomy verification. Per-image findings are
    # separately filtered by their own verification metadata above.
    usable_xray_prediction = (
        _xray_prediction_is_verified(xray_meta)
        and not _is_unavailable_prediction(xray_prediction)
    )
    usable_xray_findings = bool(xray_findings)
    usable_xray_evidence = usable_xray_prediction or usable_xray_findings

    missing_raw = evidence.get("missing", [])
    missing_info = (
        missing_raw
        if isinstance(missing_raw, list)
        else [str(missing_raw)] if missing_raw else []
    )

    # Preserve explicitly UNKNOWN symptoms as a separate state. Do not
    # convert them to ABSENT or count them as present clinical evidence.
    unknown_symptom_records = _records(symptoms_meta.get("unknown", []))
    unknown_symptom_names = [
        _symptom_key(item.get("symptom") or item.get("finding") or item.get("question"))
        for item in unknown_symptom_records
    ]
    unknown_symptom_names = [name for name in unknown_symptom_names if name]
    # If the same safety question is explicitly answered, stale/duplicate
    # UNKNOWN metadata must not override the recorded PRESENT/ABSENT answer.
    absent_records = _records(symptoms_meta.get("absent", []))
    known_symptom_keys = set(present_names)
    known_symptom_keys.update(
        _symptom_key(item.get("symptom") or item.get("finding"))
        for item in absent_records
    )

    # Identify unresolved red-flag questions by their wording. This is only a
    # follow-up signal; unknown answers do NOT trigger emergency rules by
    # themselves. Emergency rules based on confirmed findings run first below.
    critical_terms = (
        "difficulty breathing", "breathing difficulty", "shortness of breath",
        "dyspnea", "chest pain", "chest discomfort", "high fever",
        "severe fever", "fainting", "syncope",
    )
    critical_unknowns: list[Any] = []
    for item in [*missing_info, *unknown_symptom_records, *unknown_symptom_names]:
        if isinstance(item, dict):
            label = " ".join(
                str(item.get(key) or "")
                for key in ("symptom", "finding", "question", "question_text", "label", "detail", "message")
            ).strip()
        else:
            label = str(item or "").strip()
        lowered = label.lower()
        normalized_label = _symptom_key(label)
        if normalized_label in known_symptom_keys:
            continue
        if label and any(term in lowered for term in critical_terms):
            if label not in critical_unknowns:
                critical_unknowns.append(label)

    follow_up_required = bool(critical_unknowns)

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
        name = _symptom_key(symptom.get("symptom") or symptom.get("finding"))
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
        _symptom_key(s.get("symptom") or s.get("finding"))
        in {"severe breathing difficulty", "severe shortness of breath"}
        or (
            _symptom_key(s.get("symptom") or s.get("finding"))
            in {"shortness of breath", "difficulty breathing", "dyspnea"}
            and _text(s.get("severity")) == "severe"
        )
        for s in present
    )

    has_sob = any(
        _symptom_key(s.get("symptom") or s.get("finding"))
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
        _symptom_key(s.get("symptom") or s.get("finding")) == "headache"
        and _text(s.get("severity")) == "severe"
        for s in present
    )

    vomiting_or_dizziness = any(
        _symptom_key(s.get("symptom") or s.get("finding"))
        in {"vomiting", "dizziness"}
        for s in present
    )

    fainting = any(
        _symptom_key(s.get("symptom") or s.get("finding"))
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
            follow_up_required=follow_up_required,
            critical_unknowns=critical_unknowns,
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
            follow_up_required=follow_up_required,
            critical_unknowns=critical_unknowns,
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
    # Normal reports or an unavailable image model are not, by themselves,
    # enough to classify the user's current condition as MILD.
    has_any_clinical_evidence = bool(
        present or abnormal_reports or normal_reports or usable_xray_evidence
    )
    has_actionable_evidence = bool(present or abnormal_reports or usable_xray_evidence)

    if (
        state == "INSUFFICIENT_EVIDENCE"
        or not has_any_clinical_evidence
        or not has_actionable_evidence
    ):
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
            follow_up_required=follow_up_required,
            critical_unknowns=critical_unknowns,
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

    # If a potentially critical safety question remains unanswered, do not
    # finalize a non-emergency pathway (including MILD/CONSULTATION) yet.
    # This branch is reached only after confirmed emergency rules have run.
    # It asks for clarification without assuming the unknown symptom is present.
    if follow_up_required:
        return _result(
            pathway=None,
            status="INSUFFICIENT_EVIDENCE",
            triggered=[],
            ignored=ALL_RULES.copy(),
            supporting=[],
            unknown=missing_info or critical_unknowns,
            critical_unknowns=critical_unknowns,
            follow_up_required=True,
            rationale=(
                "One or more safety-critical symptom questions remain unanswered. "
                "The system cannot safely finalize a non-emergency pathway from the available information."
            ),
            guidance=(
                "Please answer the highlighted safety questions. An unknown answer does not mean the symptom is absent. "
                "If you currently have severe breathing difficulty, severe or persistent chest pain, fainting, or rapidly worsening symptoms, "
                "seek emergency medical care now rather than waiting for this assessment."
            ),
            next_steps=(
                "Complete the unanswered safety questions before the non-emergency assessment is finalized. "
                "If you cannot answer or are concerned, contact a qualified healthcare professional."
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

    # C2: positive imaging output. Raw DenseNet121 scores across all labels
    # are not positive findings. Only explicit actionable/positive findings
    # from upstream may trigger this rule; legacy ResNet prediction remains
    # supported for compatibility. Outputs are not confirmed diagnoses.
    positive_prediction = (
        usable_xray_prediction and not _is_non_finding_prediction(xray_prediction)
    )
    positive_findings = [
        item for item in xray_findings
        if not _is_non_finding_prediction(item.get("label"))
    ]
    if positive_prediction or positive_findings:
        consult.append("rule_c2_positive_imaging_finding")
        if positive_prediction:
            supporting.append(
                f"Radiographic model output for clinician review: {xray_meta.get('prediction')} "
                "(not a confirmed diagnosis)."
            )
        if positive_findings:
            displayed = [_format_xray_finding(item) for item in positive_findings[:18]]
            extra_count = len(positive_findings) - len(displayed)
            summary = "; ".join(displayed)
            if extra_count > 0:
                summary += f"; and {extra_count} additional model finding(s)"
            supporting.append(
                "X-ray model findings for clinician review: "
                f"{summary}. Model scores are not confirmed diagnoses or calibrated probabilities."
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

    # C5: breathing difficulty that is not already an emergency still needs
    # prompt professional assessment. Do not let an unknown severity fall
    # through to MILD. Severe cases were handled by the emergency rule above.
    if has_sob:
        consult.append("rule_c6_breathing_difficulty_needs_assessment")
        supporting.append(
            "Breathing difficulty was reported; severity and cause require clinical assessment."
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
                "symptoms and available medical reports. This rules-based pathway "
                "is decision support, not a diagnosis."
            ),
            next_steps=(
                "Arrange a consultation with a suitable clinician and bring "
                "the available reports."
            ),
        )

    # MILD requires at least one confirmed active symptom and no configured
    # emergency/consultation trigger. Normal reports alone do not establish MILD.
    if not present:
        return _result(
            pathway=None,
            status="INSUFFICIENT_EVIDENCE",
            triggered=[],
            ignored=ALL_RULES.copy(),
            supporting=[],
            unknown=missing_info or [
                "No current active symptoms were confirmed; normal findings alone do not establish a mild condition."
            ],
            rationale=(
                "The available information does not support assigning a MILD pathway "
                "without confirmed current symptoms or another actionable finding."
            ),
            guidance=(
                "Confirm the current symptoms and review the submitted reports. "
                "Seek professional advice if you are concerned."
            ),
            next_steps=(
                "Provide the missing current-symptom information or consult a clinician."
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
