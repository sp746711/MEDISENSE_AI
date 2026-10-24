"""Evidence combination engine for MediSense AI.

Combines multimodal evidence (symptoms, medical report findings, X-ray analysis)
into a unified clinical evidence representation:
- Preserves PRESENT / ABSENT / UNKNOWN states without losing qualifiers
- Preserves laterality, quality, duration, severity, triggers, and onset
- Identifies supporting findings (e.g. cough + report respiratory findings, normal lab confirmations)
- Identifies contradictory findings (e.g. fever denied vs high objective temp)
- Identifies missing critical information accurately without re-asking answered questions
- Truthful X-ray representation (received vs interpreted vs unavailable vs not provided)
- Establishes evidence_state: EVIDENCE_PRESENT, INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE
"""

from __future__ import annotations

from typing import Any, Optional

CRITICAL_NEGATIVES = ["chest pain", "shortness of breath", "high fever", "severe bleeding"]


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    """Return only dictionary records from an optional list-like input."""
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalize_xray_findings(value: Any) -> list[dict[str, Any]]:
    """Normalize multi-label model outputs without treating scores as diagnoses."""
    findings: list[dict[str, Any]] = []
    if not isinstance(value, (list, tuple)):
        return findings
    for item in value:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("finding") or item.get("name") or "").strip()
        if not label:
            continue
        normalized = dict(item)
        normalized["label"] = label
        # Keep the original model score field and value; it is not a calibrated
        # probability and must not be presented as diagnostic certainty.
        findings.append(normalized)
    return findings


def _xray_finding_label(item: dict[str, Any]) -> str:
    return str(item.get("label") or item.get("finding") or item.get("name") or "").strip()


def _xray_finding_is_actionable(item: dict[str, Any]) -> bool:
    """Only explicit, model/service-designated positives can support clinical evidence.

    DenseNet multi-label scores are raw model outputs unless an upstream validated
    thresholding/interpretation layer explicitly marks them actionable.
    """
    if not isinstance(item, dict):
        return False
    if item.get("is_actionable_finding") is True:
        return True
    if item.get("detected") is True or item.get("is_positive") is True:
        return True
    state = str(item.get("state") or item.get("status") or "").strip().upper()
    return state in {"PRESENT", "POSITIVE", "DETECTED"}


def _state(record: dict[str, Any]) -> str:
    """Normalize symptom state without converting unanswered findings to negatives."""
    raw = str(record.get("state") or record.get("status") or "").strip().upper()
    aliases = {
        "YES": "PRESENT", "TRUE": "PRESENT", "POSITIVE": "PRESENT",
        "NO": "ABSENT", "FALSE": "ABSENT", "NEGATIVE": "ABSENT",
        "NOT_SURE": "UNKNOWN", "NOT SURE": "UNKNOWN", "UNSURE": "UNKNOWN",
        "UNANSWERED": "UNKNOWN", "UNASSESSED": "UNKNOWN", "": "UNKNOWN",
    }
    return aliases.get(raw, raw if raw in {"PRESENT", "ABSENT", "UNKNOWN"} else "UNKNOWN")


def _status(record: dict[str, Any]) -> str:
    """Normalize lab status, prioritizing explicit interpretation/flag over workflow status."""
    # Some records use `status=RECORDED` for workflow state while the actual
    # clinical flag is in `interpretation` or `explicit_flag`.
    for key in ("explicit_flag", "interpretation", "calculated_status", "status"):
        value = str(record.get(key) or "").strip().upper()
        if value in {"HIGH", "LOW", "ABNORMAL", "NORMAL", "UNKNOWN"}:
            return value
    return "RECORDED" if any(record.get(k) for k in ("value", "test_name")) else "UNKNOWN"


def _lookup_symptom_domain(symptom_name: str | None) -> str:
    if not symptom_name:
        return "general"
    try:
        from app.ai.symptom_nlp import SYMPTOM_LEXICON
    except ImportError:
        # Keep evidence compilation usable in isolated tests/tools; the full
        # application supplies the lexicon and should be used in production.
        return "general"

    key = str(symptom_name).strip().lower()
    if key in SYMPTOM_LEXICON:
        return SYMPTOM_LEXICON[key].get("domain", "general")
    for k, v in SYMPTOM_LEXICON.items():
        if k in key or key in k:
            return v.get("domain", "general")
    return "general"


def combine_evidence(
    symptoms: list[dict[str, Any]] | None = None,
    report_findings: list[dict[str, Any]] | None = None,
    xray_result: dict[str, Any] | None = None,
    report_provided: bool | None = None,
    qualitative_report_findings: list[dict[str, Any]] | None = None,
    narrative_report_findings: list[dict[str, Any]] | None = None,
    xray_results: list[dict[str, Any]] | None = None,
    xray_reports: list[dict[str, Any]] | None = None,
    medical_reports: list[dict[str, Any]] | None = None,
    medical_report_provided: bool | None = None,
    radiology_report_provided: bool | None = None,
) -> dict[str, Any]:
    """Combine multimodal inputs into structured clinical evidence."""
    # Normalize inputs at the boundary so malformed records do not crash the
    # entire assessment. Copy symptom dictionaries before adding inferred onset.
    symptoms = [dict(item) for item in _as_dict_list(symptoms)]
    report_findings = _as_dict_list(report_findings)
    qualitative_report_findings = _as_dict_list(qualitative_report_findings)
    narrative_report_findings = _as_dict_list(narrative_report_findings)
    xray_images = _as_dict_list(xray_results) if xray_results is not None else ([dict(xray_result)] if isinstance(xray_result, dict) else [])
    xray_reports = _as_dict_list(xray_reports)
    medical_reports = _as_dict_list(medical_reports)

    # If qualitative findings are passed inside report_findings, normalize them
    for f in list(report_findings):
        if "finding" in f and "test_name" not in f and f not in qualitative_report_findings:
            qualitative_report_findings.append(dict(f))

    # Keep report categories separate. The legacy report_provided argument
    # remains supported as an overall report-availability signal.
    is_medical_report_provided = (
        medical_report_provided
        if medical_report_provided is not None
        else bool(
            report_findings
            or qualitative_report_findings
            or narrative_report_findings
            or medical_reports
        )
    )
    is_radiology_report_provided = (
        radiology_report_provided
        if radiology_report_provided is not None
        else bool(xray_reports)
    )
    is_report_provided = (
        report_provided
        if report_provided is not None
        else (is_medical_report_provided or is_radiology_report_provided)
    )

    current_present = [
        s for s in symptoms
        if _state(s) == "PRESENT"
        and str(s.get("type") or "").strip().upper() != "HISTORICAL"
        and str(s.get("symptom") or "").strip().lower() not in {"previous similar headache", "gradual onset", "sudden onset"}
    ]
    historical = [
        s for s in symptoms
        if _state(s) == "PRESENT"
        and (str(s.get("type") or "").strip().upper() == "HISTORICAL" or str(s.get("symptom") or "").strip().lower() == "previous similar headache")
    ]
    # Propagate onset attribute from gradual onset / sudden onset pseudo-symptoms if not already set
    has_gradual = any(s.get("symptom") == "gradual onset" for s in symptoms)
    has_sudden = any(s.get("symptom") == "sudden onset" for s in symptoms)
    if has_gradual or has_sudden:
        target_onset = "GRADUAL" if has_gradual else "SUDDEN"
        for s in current_present:
            if not s.get("onset"):
                if s.get("symptom") in {"headache", "migraine", "cough", "chest pain"}:
                    s["onset"] = target_onset
                    break

    symptoms_present = current_present
    symptoms_absent = [s for s in symptoms if _state(s) == "ABSENT"]
    symptoms_unknown = [s for s in symptoms if _state(s) == "UNKNOWN"]

    report_abnormal = [f for f in report_findings if _status(f) in {"HIGH", "LOW", "ABNORMAL"}]
    report_normal = [f for f in report_findings if _status(f) == "NORMAL"]
    report_unknown_range = [f for f in report_findings if _status(f) in {"UNKNOWN", "RECORDED", ""}]

    supporting: list[str] = []
    reassuring: list[str] = []
    separate: list[str] = []
    contradictory: list[str] = []
    missing: list[str] = []
    unassessed: list[str] = []
    evidence_relationships: list[dict[str, Any]] = []

    present_names = {str(s.get("symptom") or "").strip().lower() for s in current_present} - {""}
    absent_names = {str(s.get("symptom") or "").strip().lower() for s in symptoms_absent} - {""}
    all_known_names = present_names.union(absent_names).union({str(s.get("symptom") or "").strip().lower() for s in historical} - {""})

    # 1. Contradiction Detection
    has_fever_absent = "fever" in absent_names
    for s in current_present:
        if s.get("symptom") == "temperature":
            val_str = str(s.get("context") or "")
            import re
            m = re.search(r"(\d{2,3}(?:\.\d+)?)", val_str)
            if m:
                try:
                    t_val = float(m.group(1))
                    if t_val >= 100.4 and has_fever_absent:
                        contradictory.append(
                            f"Contradiction: Patient reported absence of fever, but objective temperature is {t_val}°F (febrile range)."
                        )
                except ValueError:
                    pass

    # Laboratory explicit flag vs reference range conflict detection
    for item in report_findings:
        if item.get("conflict"):
            fname = item.get("test_name", "Laboratory parameter")
            ex_flag = item.get("explicit_flag")
            calc_st = item.get("calculated_status")
            val = item.get("value")
            unit = item.get("unit") or ""
            ref = item.get("reference_range") or "unspecified"
            contradictory.append(
                f"Contradictory laboratory evidence for {fname}: Explicit report flag is '{ex_flag}', but measured value {val} {unit} evaluates to '{calc_st}' against reference range ({ref})."
            )
            evidence_relationships.append({
                "source_a": "report_flag",
                "source_b": "report_reference_range",
                "finding": fname,
                "relationship": "CONFLICTING",
                "reason": f"Explicit flag '{ex_flag}' conflicts with calculated status '{calc_st}' for {fname}.",
            })

    # 2. Supporting Clinical Correlations & Reassuring Evidence
    has_respiratory_symptom = bool(
        present_names.intersection({"cough", "dry cough", "productive cough", "shortness of breath", "difficulty breathing", "breathing difficulty", "dyspnea", "wheezing", "chest discomfort", "chest pain"})
    )
    has_fever = "fever" in present_names or "temperature" in present_names

    normal_lab_tests = {f.get("test_name", "").lower() for f in report_normal}
    has_normal_wbc = any(k in t for t in normal_lab_tests for k in ["wbc", "white blood cell"])
    has_normal_crp = any(k in t for t in normal_lab_tests for k in ["crp", "c-reactive"])
    has_abnormal_wbc = any(
        _status(f) in {"HIGH", "ABNORMAL"}
        and any(k in f.get("test_name", "").lower() for k in ["wbc", "white blood cell"])
        for f in report_abnormal
    )
    has_abnormal_crp = any(
        _status(f) in {"HIGH", "ABNORMAL"}
        and any(k in f.get("test_name", "").lower() for k in ["crp", "c-reactive"])
        for f in report_abnormal
    )

    if has_respiratory_symptom:
        reassuring_lab_parts = []
        if has_normal_wbc:
            reassuring_lab_parts.append("WBC")
        if has_normal_crp:
            reassuring_lab_parts.append("CRP")
        if reassuring_lab_parts:
            reassuring.append(f"Available {' and '.join(reassuring_lab_parts)} value(s) are within the supplied reference ranges.")
        elif has_fever and not (has_normal_wbc or has_normal_crp):
            supporting.append("Respiratory symptoms correlate with systemic febrile presentation.")

    if "headache" in present_names:
        headache_sym = next((s for s in current_present if s.get("symptom") == "headache"), None)
        features = []
        if headache_sym and headache_sym.get("laterality"):
            features.append(f"{headache_sym.get('laterality').lower()}-sided")
        if headache_sym and headache_sym.get("quality"):
            features.append(f"{headache_sym.get('quality').lower()}")
        if "photophobia" in present_names:
            features.append("photophobia")
        if "phonophobia" in present_names:
            features.append("phonophobia")
        if "nausea" in present_names:
            features.append("nausea")
        if features:
            supporting.append(f"Headache presentation characterized by {', '.join(features)}.")

    # Musculoskeletal correlations
    if any(n in present_names for n in {"wrist pain", "wrist injury", "fall/trauma", "swelling", "tenderness"}):
        msk_features = []
        for s in current_present:
            if s.get("body_area") == "wrist" or s.get("symptom") in {"wrist pain", "wrist injury", "fall/trauma"}:
                lat = f"{s.get('laterality').lower()} " if s.get("laterality") else ""
                msk_features.append(f"{lat}{s.get('symptom')}")
        if msk_features:
            supporting.append(f"Musculoskeletal trauma indicators: {', '.join(list(dict.fromkeys(msk_features)))}.")

    # Abnormal laboratory parameters: contextual laboratory evidence (never causal proof)
    for item in report_abnormal:
        fname = item.get("test_name", "")
        val = item.get("value")
        unit = item.get("unit") or ""
        interp = item.get("interpretation") or "ABNORMAL"
        ref = item.get("reference_range")
        ref_text = f" (reference: {ref})" if ref else ""

        separate.append(f"Additional abnormal laboratory finding: {fname} recorded outside reference range ({val} {unit}{ref_text}, {interp}); requires clinical interpretation and does not by itself establish the cause of the presenting complaint.")
        evidence_relationships.append({
            "source_a": "symptoms",
            "source_b": "medical_report",
            "finding": fname,
            "relationship": "SEPARATE",
            "reason": f"{fname} is outside standard reference limits but does not directly establish the cause of the presenting complaints."
        })

    if report_abnormal and "headache" in present_names:
        separate.append("Additional abnormal laboratory findings were identified and require clinical interpretation; they do not by themselves establish the cause of the headache.")
    elif report_abnormal and has_respiratory_symptom:
        separate.append("Additional abnormal laboratory findings were identified and require clinical interpretation; they do not by themselves establish the cause of the respiratory symptoms.")

    # Normal laboratory parameters (reassuring evidence)
    if report_normal:
        normal_names = [f.get("test_name") for f in report_normal if f.get("test_name")]
        reassuring.append(f"Available laboratory values ({', '.join(normal_names)}) are within standard supplied reference ranges.")

    # Cross-modal relevance: Aligned vs Separate/Contextual qualitative findings
    for qf in qualitative_report_findings:
        f_name = qf.get("finding", "")
        f_state = str(qf.get("state") or "PRESENT").strip().upper()
        f_lower = str(f_name or "").lower()

        is_relevant = False
        if has_respiratory_symptom:
            if any(k in f_lower for k in ["lung", "breath", "cough", "throat", "chest", "wheez", "rales", "rhonchi", "bronch", "pharynx"]):
                is_relevant = True
        elif "headache" in present_names or any(s.get("domain") == "neurological" for s in current_present):
            if any(k in f_lower for k in ["head", "neuro", "pupil", "cranial", "mening", "vision", "speech", "reflex"]):
                is_relevant = True
        elif any(s.get("domain") in {"musculoskeletal", "injury/trauma"} for s in current_present):
            if any(k in f_lower for k in ["wrist", "hand", "joint", "bone", "swelling", "fracture", "tendon", "ligament", "sprain", "strain"]):
                is_relevant = True

        if not is_relevant:
            separate.append(f"Qualitative finding of {f_name} does not directly account for the primary presenting clinical presentation.")
            evidence_relationships.append({
                "source_a": "symptoms",
                "source_b": "medical_report",
                "finding": f_name,
                "relationship": "SEPARATE",
                "reason": f"{f_name.capitalize()} does not directly explain the primary symptom presentation."
            })
        else:
            if f_state == "PRESENT":
                supporting.append(f"Medical report qualitative finding confirmed: {f_name}.")
            elif f_state == "ABSENT":
                reassuring.append(f"Medical report explicitly confirms absence of: {f_name}.")

    import re
    synthetic_instruction_re = re.compile(
        r"\b(?:synthetic|demonstration|demo\s+only|sample\s+report|test\s+case|no\s+actual\s+clinical\s+findings|instructions?|placeholder|dummy\s+data|simulated|example\s+report)\b",
        re.IGNORECASE,
    )
    for nf in narrative_report_findings:
        sec = str(nf.get("section") or "").strip()
        cnt = str(nf.get("content") or "").strip()
        # Narrative text may be history, a disclaimer, or imperfect OCR. Do not
        # automatically treat every narrative sentence as clinical support.
        # Preserve it as contextual information; structured findings retain states.
        if cnt and not synthetic_instruction_re.search(cnt) and not synthetic_instruction_re.search(sec):
            short_cnt = cnt[:240] + "..." if len(cnt) > 240 else cnt
            separate.append(f"Report narrative for review ({sec or 'section unspecified'}): {short_cnt}")

    # 3. Truthful X-ray Status and Interpretation (Multi-Image & Multi-Report Aware)
    processed_images: list[dict[str, Any]] = []
    xray_finding_str = None
    xray_received = bool(xray_images)
    xray_status_display = "NOT_PROVIDED"
    xray_interpretation = "NOT_PROVIDED"
    xray_evidence_status = "NOT_ASSESSED"
    xray_region = "not declared"

    if xray_images:
        completed_preds = []
        any_unavailable = False
        all_regions = set()

        for idx, img in enumerate(xray_images):
            img_status = str(img.get("status") or "unavailable").strip().lower()
            reg_raw = img.get("region")
            if isinstance(reg_raw, (list, tuple)):
                reg_raw = reg_raw[0] if reg_raw else ""
            img_reg = str(reg_raw or "").strip().lower() or "not declared"
            all_regions.add(img_reg)
            img_fname = img.get("file_name") or img.get("source_file") or f"xray_image_{idx+1}"
            img_label = f"{img_reg.capitalize()} X-ray ({img_fname})" if img_reg not in {"not declared", "unknown"} else f"X-ray ({img_fname})"
            pred = img.get("prediction")
            findings = _normalize_xray_findings(img.get("findings"))
            has_model_output = bool(pred) or bool(findings)

            p_entry = {
                "id": img.get("id") or img.get("xray_result_id"),
                "file_name": img_fname,
                "region": img_reg,
                "status": img_status.upper(),
                "prediction": pred,
                "findings": findings,
                "model_version": img.get("model_version") or "N/A",
                "confidence": img.get("confidence_if_valid"),
                "uncertainty": img.get("uncertainty"),
                "message": img.get("message"),
                "source": "xray_image",
                "source_type": "XRAY_IMAGE",
                "source_file": img_fname,
            }
            processed_images.append(p_entry)

            # Check anatomical alignment with primary complaints.
            is_anatomically_aligned = True
            if has_respiratory_symptom and img_reg in {"wrist", "hand", "ankle", "knee", "extremity"}:
                is_anatomically_aligned = False
            elif any(s.get("domain") in {"musculoskeletal", "injury/trauma"} for s in current_present) and not has_respiratory_symptom and img_reg == "chest":
                is_anatomically_aligned = False

            if img_status == "completed" and has_model_output:
                # A legacy single prediction is retained for compatibility.
                # Multi-label score records are NOT clinical findings unless the
                # model/service explicitly marks them actionable.
                actionable_findings = [f for f in findings if _xray_finding_is_actionable(f)]
                labels = [str(pred).strip()] if pred else [
                    _xray_finding_label(f) for f in actionable_findings
                ]
                labels = [label for label in labels if label]
                raw_score_labels = [
                    _xray_finding_label(f)
                    for f in findings
                    if not _xray_finding_is_actionable(f) and _xray_finding_label(f)
                ]

                if labels:
                    completed_preds.extend(labels)
                    if is_anatomically_aligned:
                        supporting.append(
                            f"{img_label} model/service returned explicitly actionable candidate finding(s): "
                            f"{', '.join(labels[:18])}."
                            + (f" And {len(labels) - 18} more." if len(labels) > 18 else "")
                            + " This is not a confirmed diagnosis and requires clinical interpretation."
                        )
                    else:
                        separate.append(
                            f"{img_label} actionable model output ({', '.join(labels[:18])}) is anatomically separate from the primary clinical complaint."
                        )
                        evidence_relationships.append({
                            "source_a": "symptoms",
                            "source_b": "xray_image",
                            "finding": f"{img_label}: {', '.join(labels[:18])}",
                            "relationship": "SEPARATE",
                            "reason": f"Anatomical region ({img_reg}) does not directly explain the primary presenting symptoms."
                        })

                if raw_score_labels:
                    # Preserve raw scores in processed_images for transparent display,
                    # but never promote them to clinical support, diagnosis, or triage.
                    score_preview = ", ".join(raw_score_labels[:18])
                    unassessed.append(
                        f"{img_label} inference completed and returned raw, unthresholded model scores "
                        f"({score_preview}{'; and more' if len(raw_score_labels) > 18 else ''}). "
                        "These scores are not calibrated diagnostic probabilities and are not counted as actionable clinical evidence."
                    )

                if not labels and not raw_score_labels:
                    any_unavailable = True
                    unassessed.append(
                        f"{img_label} was marked completed but contained no usable prediction or findings; interpretation cannot be verified."
                    )
            elif img_status == "unavailable":
                any_unavailable = True
                unassessed.append(f"{img_label} received; automated interpretation unavailable for this region. No disease finding fabricated.")
            else:
                any_unavailable = True
                unassessed.append(f"{img_label} status: {img_status.upper()}. Automated interpretation unavailable.")

        xray_region = ", ".join(sorted(all_regions)) if all_regions else "not declared"
        if completed_preds:
            xray_finding_str = "; ".join(completed_preds)
            xray_interpretation = "AVAILABLE"
            aligned_prediction_added = any(
                "explicitly actionable candidate finding" in item
                for item in supporting
            )
            xray_evidence_status = "SUPPORTING" if aligned_prediction_added else "RECORDED"
            xray_status_display = "PARTIAL" if any_unavailable else "COMPLETED"
            if any_unavailable:
                unassessed.append(
                    "Some uploaded X-ray images were interpreted, while one or more images remain unavailable or unprocessed."
                )
        elif any(
            str(img.get("status") or "").strip().lower() == "completed"
            and (img.get("prediction") or _normalize_xray_findings(img.get("findings")))
            for img in xray_images
        ):
            # Inference completed, but only raw non-actionable scores were returned.
            # This is recorded as model output, not as an interpreted clinical finding.
            xray_interpretation = "RECORDED_UNINTERPRETED"
            xray_evidence_status = "RECORDED"
            xray_status_display = "PARTIAL" if any_unavailable else "COMPLETED"
        elif any_unavailable:
            xray_interpretation = "UNAVAILABLE"
            xray_evidence_status = "UNKNOWN"
            xray_status_display = "UNAVAILABLE"
        else:
            xray_interpretation = "UNAVAILABLE"
            xray_evidence_status = "UNKNOWN"
            xray_status_display = "RECEIVED"

    # Process X-ray / Radiology Reports
    processed_xray_reports: list[dict[str, Any]] = []
    for idx, xr in enumerate(xray_reports):
        xr_fname = xr.get("file_name") or f"xray_report_{idx+1}.pdf"
        struct = xr.get("structured_findings") or {}
        xr_qual = struct.get("qualitative_findings", []) if isinstance(struct, dict) else []
        xr_narr = struct.get("narrative_findings", []) if isinstance(struct, dict) else []
        xr_entry = {
            "id": xr.get("id") or xr.get("report_id"),
            "file_name": xr_fname,
            "source_type": "XRAY_REPORT",
            "source_file": xr_fname,
            "status": "EXTRACTED" if (xr.get("extracted_text") or struct) else "UNREADABLE",
            "structured_findings": struct,
            "qualitative_findings": xr_qual,
            "narrative_findings": xr_narr,
            "created_at": xr.get("created_at"),
            "is_historical": xr.get("is_historical", False),
        }
        processed_xray_reports.append(xr_entry)

        # Keep source statements in processed_xray_reports. Do not automatically
        # promote each statement to supporting/reassuring evidence here.
        # Classify radiology findings across all reports below, so conflicts
        # between separate reports remain visible.

        # Check for body-region mismatch between X-ray images and radiology reports
        if processed_images:
            rep_text = str(xr.get("extracted_text") or "").lower() + " " + str(xr_fname).lower()
            for img in processed_images:
                img_reg = img.get("region", "").lower()
                if img_reg and img_reg not in {"not declared", "unknown"}:
                    comp_regions = ["chest", "knee", "wrist", "hand", "elbow", "shoulder", "spine", "pelvis", "hip", "ankle", "foot"]
                    for other_r in comp_regions:
                        if other_r != img_reg and other_r in rep_text and img_reg not in rep_text:
                            contradictory.append(
                                f"Conflicting evidence / body region mismatch: Image region is '{img_reg.capitalize()}', but radiology report '{xr_fname}' indicates '{other_r.capitalize()}'. Evidence kept unlinked."
                            )
                            evidence_relationships.append({
                                "source_a": f"xray_image ({img.get('file_name')})",
                                "source_b": f"xray_report ({xr_fname})",
                                "finding": "body_region_mismatch",
                                "relationship": "CONFLICTING",
                                "reason": f"Image anatomical region '{img_reg}' differs from radiology report '{other_r}'.",
                            })
                            break

    # Problem 1: classify radiology report findings across all reports before
    # allowing them to support or reassure. Keep source statements intact.
    radiology_groups: dict[str, list[dict[str, Any]]] = {}
    radiology_conflicts: list[dict[str, Any]] = []
    radiology_uncertain: list[dict[str, Any]] = []

    def _radiology_key(value: Any) -> str:
        import re
        text = str(value or "").lower().strip()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    for rep in processed_xray_reports:
        for qf in rep.get("qualitative_findings", []):
            if not isinstance(qf, dict):
                continue
            finding_name = str(qf.get("finding") or "").strip()
            if not finding_name:
                continue
            state = str(qf.get("state") or qf.get("status") or "UNKNOWN").strip().upper()
            if state not in {"PRESENT", "ABSENT", "UNCERTAIN", "CONFLICTING", "UNKNOWN", "HISTORICAL"}:
                state = "UNKNOWN"
            statement = {
                **qf,
                "finding": finding_name,
                "state": state,
                "source": "radiology_report",
                "source_file": rep.get("file_name"),
                "source_report_id": rep.get("id"),
            }
            key = _radiology_key(finding_name)
            if key:
                radiology_groups.setdefault(key, []).append(statement)

    for key, statements in radiology_groups.items():
        states = {item["state"] for item in statements}
        display_name = statements[0]["finding"]
        is_conflicting = "CONFLICTING" in states or ("PRESENT" in states and "ABSENT" in states)
        if is_conflicting:
            radiology_conflicts.append({
                "finding": display_name,
                "state": "CONFLICTING",
                "source": "radiology_report",
                "source_statements": statements,
                "message": "Conflicting radiology statements; finding remains unresolved and is not confirmed.",
            })
            contradictory.append(
                f"Conflicting radiology evidence for '{display_name}': "
                + " | ".join(f"{item['state']}: {item['finding']} (source: {item.get('source_file') or 'radiology report'})" for item in statements)
                + ". Finding remains unresolved and is excluded from confirmed supporting evidence."
            )
            evidence_relationships.append({
                "source_a": statements[0].get("source_file") or "radiology_report",
                "source_b": statements[-1].get("source_file") or "radiology_report",
                "finding": display_name,
                "relationship": "CONFLICTING",
                "reason": "Radiology sources contain both PRESENT and ABSENT statements, or an explicit CONFLICTING state.",
                "source_statements": statements,
            })
            continue

        if "UNCERTAIN" in states:
            radiology_uncertain.append({
                "finding": display_name,
                "state": "UNCERTAIN",
                "source": "radiology_report",
                "source_statements": statements,
                "message": "Uncertain radiology finding; not treated as confirmed supporting evidence.",
            })
            separate.append(
                f"Radiology report finding remains UNCERTAIN: {display_name}; not counted as confirmed supporting evidence."
            )
            evidence_relationships.append({
                "source_a": statements[0].get("source_file") or "radiology_report",
                "source_b": "assessment_evidence",
                "finding": display_name,
                "relationship": "UNCERTAIN",
                "reason": "Radiology wording/state is uncertain; excluded from confirmed supporting evidence.",
                "source_statements": statements,
            })
            continue

        if states == {"PRESENT"}:
            supporting.append(
                f"Radiology report finding: {display_name} (PRESENT; source: {statements[0].get('source_file') or 'radiology report'})."
            )
        elif states == {"ABSENT"}:
            reassuring.append(
                f"Radiology report explicitly records absence of: {display_name} (source: {statements[0].get('source_file') or 'radiology report'})."
            )
        else:
            separate.append(
                f"Radiology report finding '{display_name}' has state {', '.join(sorted(states)) or 'UNKNOWN'}; not used as confirmed evidence."
            )

    # Cross-modal concordance must not promote uncertain or conflicting report
    # text into confirmed image/report support.
    cross_modal_concordance: list[dict[str, Any]] = []
    if processed_images and processed_xray_reports:
        for img in processed_images:
            img_pred = img.get("prediction")
            img_findings = _normalize_xray_findings(img.get("findings"))
            img_labels = [str(img_pred).strip()] if img_pred else [
                _xray_finding_label(f)
                for f in img_findings
                if _xray_finding_label(f) and _xray_finding_is_actionable(f)
            ]
            img_status = (img.get("status") or "").lower()
            img_file = img.get("file_name") or "xray_image"
            for rep in processed_xray_reports:
                rep_file = rep.get("file_name") or "radiology_report.pdf"
                rep_records = [q for q in rep.get("qualitative_findings", []) if isinstance(q, dict)]
                rep_findings = [
                    q.get("finding", "").lower() for q in rep_records
                    if str(q.get("state") or q.get("status") or "UNKNOWN").upper() == "PRESENT"
                ]
                rep_uncertain_or_conflicting = any(
                    str(q.get("state") or q.get("status") or "UNKNOWN").upper() in {"UNCERTAIN", "CONFLICTING"}
                    for q in rep_records
                )
                rep_narrative = " ".join([n.get("content", "").lower() for n in rep.get("narrative_findings", []) if isinstance(n, dict)])

                if img_status == "completed" and img_labels:
                    labels_lower = [label.lower() for label in img_labels]
                    matches = [
                        label for label in labels_lower
                        if any(label in rf for rf in rep_findings) or label in rep_narrative
                    ]
                    absent_matches = [
                        label for label in labels_lower
                        if any(f"no {label}" in rf or f"without {label}" in rf for rf in rep_findings)
                        or f"no {label}" in rep_narrative
                    ]
                    if rep_uncertain_or_conflicting and not matches:
                        status_str = "UNRESOLVED"
                        reason = f"Radiology report '{rep_file}' contains uncertain or conflicting findings; no confirmed report corroboration is asserted."
                    elif matches:
                        status_str = "PARTIALLY_CONCORDANT" if len(matches) < len(labels_lower) else "CONCORDANT"
                        reason = (
                            f"Model candidate pattern(s) {', '.join(matches)} overlap with radiology report '{rep_file}'. "
                            "This is text-level overlap, not independent diagnostic confirmation."
                        )
                        supporting.append(f"Cross-modal text-level overlap: model candidate pattern(s) {', '.join(matches)} also appear in radiology report '{rep_file}'; this does not establish a diagnosis.")
                    elif absent_matches:
                        status_str = "DISCORDANT"
                        reason = f"Model candidate pattern(s) {', '.join(absent_matches)} contrast with negative/absent wording in radiology report '{rep_file}'. Manual clinical review is needed."
                        contradictory.append(f"Cross-modal discordance: X-ray model candidate(s) {', '.join(absent_matches)} contrast with absence wording in radiology report '{rep_file}'; neither source is treated as definitive.")
                    else:
                        status_str = "INDEPENDENT"
                        reason = f"Model candidate pattern(s) {', '.join(img_labels[:18])} are not explicitly confirmed or ruled out by radiology report '{rep_file}'."
                    cross_modal_concordance.append({
                        "image_file": img_file,
                        "report_file": rep_file,
                        "image_prediction": img_pred,
                        "image_findings": img_findings,
                        "concordance_status": status_str,
                        "details": reason,
                    })
                else:
                    cross_modal_concordance.append({
                        "image_file": img_file,
                        "report_file": rep_file,
                        "image_prediction": None,
                        "image_findings": img_findings,
                        "concordance_status": "NOT_APPLICABLE",
                        "details": f"Automated image model interpretation is unavailable for '{img_file}'. Relying on extracted radiology report findings from '{rep_file}'.",
                    })

    # 4. Missing critical information (context-sensitive, never generic across unrelated domains)
    relevant_critical_checks: list[str] = []
    # Use independent domain checks rather than an elif chain so a patient with
    # symptoms from more than one domain does not lose relevant safety checks.
    if has_respiratory_symptom or present_names.intersection({"chest pain", "chest discomfort", "difficulty breathing", "shortness of breath", "dyspnea"}):
        relevant_critical_checks.extend(["difficulty breathing", "chest pain", "high fever"])
    if "headache" in present_names or any(s.get("domain") == "neurological" for s in current_present):
        relevant_critical_checks.extend(["sudden onset", "weakness", "speech difficulty", "fever"])
    if any(s.get("domain") in {"musculoskeletal", "injury/trauma"} for s in current_present):
        relevant_critical_checks.extend(["severe swelling/deformity", "numbness", "severe bleeding"])
    if any(s.get("domain") in {"gi", "gastrointestinal", "abdominal"} for s in current_present) or any("abdominal" in str(s.get("symptom") or "").lower() for s in current_present):
        relevant_critical_checks.extend(["sudden onset", "vomiting", "blood in stool", "fever"])
    relevant_critical_checks = list(dict.fromkeys(relevant_critical_checks))

    for crit in relevant_critical_checks:
        addressed = False
        if crit in all_known_names:
            addressed = True
        elif crit == "chest pain" and any("chest pain" in n for n in all_known_names):
            addressed = True
        elif crit in {"difficulty breathing", "shortness of breath"} and any("breath" in n or "dyspnea" in n for n in all_known_names):
            addressed = True
        elif crit in {"high fever", "fever"} and ("fever" in all_known_names or "temperature" in all_known_names):
            addressed = True
        elif crit == "sudden onset" and (any(s.get("onset") for s in current_present) or any("onset" in n for n in all_known_names)):
            addressed = True
        elif crit == "weakness" and any("weakness" in n or "numb" in n for n in all_known_names):
            addressed = True
        elif crit == "speech difficulty" and any("speech" in n for n in all_known_names):
            addressed = True
        elif crit == "vomiting" and any("vomit" in n for n in all_known_names):
            addressed = True
        elif crit == "blood in stool" and any("blood in stool" in n for n in all_known_names):
            addressed = True
        elif crit == "severe bleeding" and any("bleed" in n or "hemorrhage" in n for n in all_known_names):
            addressed = True
        elif crit == "severe swelling/deformity" and any("deform" in n or "swelling" in n for n in all_known_names):
            addressed = True

        if not addressed:
            missing.append(f"Status of '{crit}' (neither confirmed nor denied)")

    durations = [s.get("duration") for s in current_present if s.get("duration")]
    if current_present and not durations:
        missing.append("Symptom duration (onset timeframe unspecified)")

    if not is_medical_report_provided:
        missing.append("Medical laboratory report (biochemical parameters unassessed)")

    if not xray_received:
        missing.append("Radiographic imaging (radiographic status unassessed)")

    # 5. Determine overall evidence state
    definite_qualitative_signals = sum(
        1 for item in qualitative_report_findings
        if isinstance(item, dict)
        and str(item.get("state") or item.get("status") or "").strip().upper() in {"PRESENT", "ABSENT"}
    )
    # Radiology PDFs are a distinct input stream; their extracted findings must
    # also count as evidence, without treating UNKNOWN/UNCERTAIN as confirmed.
    definite_qualitative_signals += sum(
        1
        for report in processed_xray_reports
        for item in report.get("qualitative_findings", [])
        if isinstance(item, dict)
        and str(item.get("state") or item.get("status") or "UNKNOWN").strip().upper() in {"PRESENT", "ABSENT"}
    )
    total_active_signals = (
        len(current_present)
        + len(report_abnormal)
        + definite_qualitative_signals
        + (1 if xray_finding_str else 0)
    )

    if contradictory:
        evidence_state = "CONFLICTING_EVIDENCE"
    elif (
        total_active_signals > 0
        or len(report_normal) > 0
        or len(symptoms_absent) > 0
        or len(historical) > 0
        or is_report_provided
        or (xray_received and xray_interpretation == "UNAVAILABLE")
    ):
        evidence_state = "EVIDENCE_PRESENT"
    else:
        evidence_state = "INSUFFICIENT_EVIDENCE"

    has_definite_report_evidence = bool(
        report_abnormal
        or report_normal
        or any(
            isinstance(item, dict)
            and str(item.get("state") or item.get("status") or "").strip().upper() in {"PRESENT", "ABSENT"}
            for item in qualitative_report_findings
        )
    )
    report_evidence_status = (
        "SUPPORTING" if has_definite_report_evidence
        else ("NOT_ASSESSED" if not is_medical_report_provided else "RECORDED")
    )

    current_present_list = [
        {
            "finding": s.get("symptom"),
            "symptom": s.get("symptom"),
            "status": "PRESENT",
            "state": "PRESENT",
            "duration": s.get("duration"),
            "severity": (s.get("severity") or "UNKNOWN").upper(),
            "body_area": s.get("body_area"),
            "laterality": s.get("laterality"),
            "quality": s.get("quality"),
            "domain": s.get("domain") or _lookup_symptom_domain(s.get("symptom")),
            "context": s.get("context"),
            "trigger": s.get("trigger"),
            "type": s.get("type") or "CURRENT",
            "onset": s.get("onset"),
            "source": "user_input",
            "provenance": "symptom_nlp",
        }
        for s in current_present
    ]

    historical_list = [
        {
            "finding": s.get("symptom"),
            "symptom": s.get("symptom"),
            "status": "PRESENT",
            "state": "PRESENT",
            "duration": None,
            "severity": (s.get("severity") or "UNKNOWN").upper(),
            "body_area": s.get("body_area"),
            "laterality": s.get("laterality"),
            "quality": s.get("quality"),
            "domain": s.get("domain") or _lookup_symptom_domain(s.get("symptom")),
            "context": s.get("context"),
            "trigger": s.get("trigger"),
            "type": "HISTORICAL",
            "history": s.get("history") or s.get("context") or "historical occurrence",
            "source": "user_input",
            "provenance": "symptom_nlp",
        }
        for s in historical
    ]

    return {
        "evidence_state": evidence_state,
        "symptoms": {
            "current_present": current_present_list,
            "historical": historical_list,
            "present": current_present_list,
            "all_present": current_present_list + historical_list,
            "absent": [
                {
                    "finding": s.get("symptom"),
                    "symptom": s.get("symptom"),
                    "status": "ABSENT",
                    "state": "ABSENT",
                    "type": "NEGATIVE",
                    "body_area": s.get("body_area"),
                    "context": s.get("context"),
                    "domain": s.get("domain") or _lookup_symptom_domain(s.get("symptom")),
                    "source": "user_input",
                    "provenance": "symptom_nlp",
                }
                for s in symptoms_absent
            ],
            "unknown": [
                {
                    "finding": s.get("symptom") if isinstance(s, dict) else str(s),
                    "symptom": s.get("symptom") if isinstance(s, dict) else str(s),
                    "status": "UNKNOWN",
                    "state": "UNKNOWN",
                    "type": "UNCERTAIN",
                    "body_area": s.get("body_area") if isinstance(s, dict) else None,
                    "laterality": s.get("laterality") if isinstance(s, dict) else None,
                    "quality": s.get("quality") if isinstance(s, dict) else None,
                    "duration": s.get("duration") if isinstance(s, dict) else None,
                    "severity": (s.get("severity") or "UNKNOWN").upper() if isinstance(s, dict) else "UNKNOWN",
                    "onset": s.get("onset") if isinstance(s, dict) else None,
                    "trigger": s.get("trigger") if isinstance(s, dict) else None,
                    "context": s.get("context") if isinstance(s, dict) else None,
                    "domain": (s.get("domain") or _lookup_symptom_domain(s.get("symptom") if isinstance(s, dict) else str(s))) if isinstance(s, dict) else _lookup_symptom_domain(str(s)),
                    "source": "user_input",
                    "provenance": "symptom_nlp",
                }
                for s in symptoms_unknown
            ],
        },
        "report": {
            # `provided` now means a medical/laboratory report was provided.
            "provided": is_medical_report_provided,
            "medical_report_provided": is_medical_report_provided,
            "radiology_report_provided": is_radiology_report_provided,
            "any_report_provided": is_report_provided,
            "evidence_status": report_evidence_status,
            "medical_reports": medical_reports,
            "xray_reports": processed_xray_reports,
            "abnormal_findings": [
                {
                    **f,
                    "status": f.get("status") or f.get("interpretation") or "ABNORMAL",
                    "interpretation": f.get("interpretation") or "ABNORMAL",
                    "source": "uploaded_report",
                    "provenance": "report_ocr_nlp",
                }
                for f in report_abnormal
            ],
            "normal_findings": [
                {
                    **f,
                    "status": f.get("status") or "NORMAL",
                    "interpretation": "NORMAL",
                    "source": "uploaded_report",
                    "provenance": "report_ocr_nlp",
                }
                for f in report_normal
            ],
            "qualitative_findings": qualitative_report_findings,
            "narrative_findings": narrative_report_findings,
            "radiology_conflicts": radiology_conflicts,
            "radiology_uncertain_findings": radiology_uncertain,
            "unknown_range_findings": report_unknown_range,
            "total_extracted": (
                len(report_findings)
                + len(qualitative_report_findings)
                + len(narrative_report_findings)
                + sum(
                    len(rep.get("qualitative_findings", [])) + len(rep.get("narrative_findings", []))
                    for rep in processed_xray_reports
                )
            ),
        },
        "xray": {
            "received": xray_received,
            "interpretation": xray_interpretation,
            "evidence_status": xray_evidence_status,
            "status": xray_status_display,
            "region": xray_region,
            "prediction": xray_finding_str,
            "images": processed_images,
            "reports": processed_xray_reports,
            "cross_modal_concordance": cross_modal_concordance,
            "model_version": (xray_images[0] if xray_images else (xray_result or {})).get("model_version") or "N/A",
            "uncertainty": (xray_images[0] if xray_images else (xray_result or {})).get("uncertainty"),
            "explainability_artifact": (xray_images[0] if xray_images else (xray_result or {})).get("explainability_artifact"),
            "message": (xray_images[0] if xray_images else (xray_result or {})).get("message"),
            "provenance": "xray_deep_learning" if xray_received else "none",
        },
        "supporting": supporting,
        "reassuring": reassuring,
        "separate": separate,
        "separate_or_contextual": separate,
        "unassessed": unassessed,
        "contradictory": contradictory,
        "missing": missing,
        # These are unanswered safety checks, not confirmed negative symptoms.
        # Downstream triage/follow-up layers can consume them without parsing prose.
        "critical_unknowns": [
            item for item in missing
            if any(term in item.lower() for term in (
                "difficulty breathing", "chest pain", "high fever", "severe bleeding",
                "sudden onset", "speech difficulty", "weakness", "numbness",
                "severe swelling/deformity", "vomiting", "blood in stool", "fever",
            ))
        ],
        "follow_up_required": any(
            any(term in item.lower() for term in (
                "difficulty breathing", "chest pain", "high fever", "severe bleeding",
                "sudden onset", "speech difficulty", "weakness", "numbness",
                "severe swelling/deformity", "vomiting", "blood in stool", "fever",
            ))
            for item in missing
        ),
        "evidence_relationships": evidence_relationships,
        "message": (
            "Multimodal evidence compiled successfully."
            if evidence_state == "EVIDENCE_PRESENT"
            else "Available information is insufficient for a specific clinical conclusion."
        ),
    }
