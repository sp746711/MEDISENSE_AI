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


def _lookup_symptom_domain(symptom_name: str | None) -> str:
    if not symptom_name:
        return "general"
    from app.ai.symptom_nlp import SYMPTOM_LEXICON

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
) -> dict[str, Any]:
    """Combine multimodal inputs into structured clinical evidence."""
    symptoms = symptoms or []
    report_findings = report_findings or []
    qualitative_report_findings = qualitative_report_findings or []
    narrative_report_findings = narrative_report_findings or []
    xray_images = list(xray_results) if xray_results is not None else ([xray_result] if xray_result else [])
    xray_reports = list(xray_reports or [])
    medical_reports = list(medical_reports or [])

    # If qualitative findings are passed inside report_findings, normalize them
    for f in list(report_findings):
        if "finding" in f and "test_name" not in f and f not in qualitative_report_findings:
            qualitative_report_findings.append(f)

    is_report_provided = (
        report_provided
        if report_provided is not None
        else bool(report_findings or qualitative_report_findings or narrative_report_findings or medical_reports or xray_reports)
    )

    current_present = [
        s for s in symptoms
        if s.get("state") == "PRESENT"
        and s.get("type") != "HISTORICAL"
        and s.get("symptom") not in {"previous similar headache", "gradual onset", "sudden onset"}
    ]
    historical = [
        s for s in symptoms
        if s.get("state") == "PRESENT"
        and (s.get("type") == "HISTORICAL" or s.get("symptom") == "previous similar headache")
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
    symptoms_absent = [s for s in symptoms if s.get("state") == "ABSENT"]
    symptoms_unknown = [s for s in symptoms if s.get("state") == "UNKNOWN"]

    report_abnormal = [
        f for f in report_findings if f.get("interpretation") in {"HIGH", "LOW", "ABNORMAL"}
    ]
    report_normal = [
        f for f in report_findings if f.get("interpretation") == "NORMAL"
    ]
    report_unknown_range = [
        f for f in report_findings if f.get("interpretation") in {"UNKNOWN", "RECORDED"}
    ]

    supporting: list[str] = []
    reassuring: list[str] = []
    separate: list[str] = []
    contradictory: list[str] = []
    missing: list[str] = []
    unassessed: list[str] = []
    evidence_relationships: list[dict[str, Any]] = []

    present_names = {s.get("symptom", "").lower() for s in current_present}
    absent_names = {s.get("symptom", "").lower() for s in symptoms_absent}
    all_known_names = present_names.union(absent_names).union({s.get("symptom", "").lower() for s in historical})

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

    # 2. Supporting Clinical Correlations & Reassuring Evidence
    has_respiratory_symptom = bool(
        present_names.intersection({"cough", "dry cough", "productive cough", "shortness of breath", "dyspnea", "wheezing", "chest discomfort"})
    )
    has_fever = "fever" in present_names or "temperature" in present_names

    normal_lab_tests = {f.get("test_name", "").lower() for f in report_normal}
    has_normal_wbc_or_crp = any(
        k in t for t in normal_lab_tests for k in ["wbc", "white blood cell", "crp", "c-reactive"]
    )
    has_abnormal_wbc_or_crp = any(
        f.get("interpretation") in {"HIGH", "ABNORMAL"}
        and any(k in f.get("test_name", "").lower() for k in ["wbc", "crp", "white blood cell", "c-reactive", "esr"])
        for f in report_abnormal
    )

    if has_respiratory_symptom:
        if has_normal_wbc_or_crp:
            reassuring.append("Available WBC and CRP values are within the supplied reference ranges.")
        elif has_fever and not has_normal_wbc_or_crp:
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
        f_state = qf.get("state", "PRESENT")
        f_lower = f_name.lower()

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

    for nf in narrative_report_findings:
        sec = nf.get("section", "")
        cnt = nf.get("content", "")
        if cnt:
            short_cnt = cnt[:100] + "..." if len(cnt) > 100 else cnt
            supporting.append(f"Report narrative ({sec}): {short_cnt}")

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
            img_status = (img.get("status") or "unavailable").lower()
            img_reg = (img.get("region") or "").strip().lower() or "not declared"
            all_regions.add(img_reg)
            img_fname = img.get("file_name") or img.get("source_file") or f"xray_image_{idx+1}"
            img_label = f"{img_reg.capitalize()} X-ray ({img_fname})" if img_reg not in {"not declared", "unknown"} else f"X-ray ({img_fname})"
            pred = img.get("prediction")

            p_entry = {
                "id": img.get("id") or img.get("xray_result_id"),
                "file_name": img_fname,
                "region": img_reg,
                "status": img_status.upper(),
                "prediction": pred,
                "model_version": img.get("model_version") or "N/A",
                "confidence": img.get("confidence_if_valid"),
                "uncertainty": img.get("uncertainty"),
                "message": img.get("message"),
                "source": "xray_image",
                "source_type": "XRAY_IMAGE",
                "source_file": img_fname,
            }
            processed_images.append(p_entry)

            if img_status == "completed" and pred:
                completed_preds.append(pred)
                supporting.append(f"{img_label} deep learning model identified radiographic pattern: {pred}.")
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
            xray_evidence_status = "SUPPORTING"
            xray_status_display = "COMPLETED"
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
        xr_entry = {
            "id": xr.get("id") or xr.get("report_id"),
            "file_name": xr_fname,
            "source_type": "XRAY_REPORT",
            "source_file": xr_fname,
            "status": "EXTRACTED" if xr.get("extracted_text") else "UNREADABLE",
            "structured_findings": struct,
            "created_at": xr.get("created_at"),
            "is_historical": xr.get("is_historical", False),
        }
        processed_xray_reports.append(xr_entry)

        for qf in xr_qual:
            q_name = qf.get("finding", "")
            q_lower = q_name.lower()
            q_state = qf.get("state", "PRESENT")
            if any(neg in q_lower for neg in ["no fracture", "no acute fracture", "no dislocation", "intact", "unremarkable", "normal"]):
                reassuring.append(f"Radiology report ({xr_fname}) explicitly confirms: {q_name}.")
            else:
                supporting.append(f"Radiology report ({xr_fname}) clinical finding: {q_name} ({q_state}).")

        # Check for body-region mismatch between X-ray images and radiology reports
        if processed_images:
            rep_text = (xr.get("extracted_text") or "").lower() + " " + xr_fname.lower()
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

    # 4. Missing critical information (context-sensitive, never generic across unrelated domains)
    relevant_critical_checks: list[str] = []
    if has_respiratory_symptom:
        relevant_critical_checks.extend(["difficulty breathing", "chest pain", "high fever"])
    elif "headache" in present_names or any(s.get("domain") == "neurological" for s in current_present):
        relevant_critical_checks.extend(["sudden onset", "weakness", "speech difficulty", "fever", "chest pain"])
    elif any(s.get("domain") in {"musculoskeletal", "injury/trauma"} for s in current_present):
        relevant_critical_checks.extend(["severe swelling/deformity", "numbness", "severe bleeding"])
    elif any(s.get("domain") in {"gi", "gastrointestinal", "abdominal"} for s in current_present) or any("abdominal" in s.get("symptom", "").lower() for s in current_present):
        relevant_critical_checks.extend(["sudden onset", "vomiting", "blood in stool", "fever"])

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

    if not is_report_provided:
        missing.append("Medical laboratory report (biochemical parameters unassessed)")

    if not xray_received:
        missing.append("Radiographic imaging (radiographic status unassessed)")

    # 5. Determine overall evidence state
    total_active_signals = (
        len(current_present)
        + len(report_abnormal)
        + len(qualitative_report_findings)
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

    report_evidence_status = (
        "SUPPORTING"
        if (report_abnormal or report_normal or qualitative_report_findings or narrative_report_findings)
        else ("NOT_ASSESSED" if not is_report_provided else "RECORDED")
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
            "unknown": [s.get("symptom") for s in symptoms_unknown],
        },
        "report": {
            "provided": is_report_provided,
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
            "unknown_range_findings": report_unknown_range,
            "total_extracted": len(report_findings) + len(qualitative_report_findings) + len(narrative_report_findings),
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
        "evidence_relationships": evidence_relationships,
        "message": (
            "Multimodal evidence compiled successfully."
            if evidence_state == "EVIDENCE_PRESENT"
            else "Available information is insufficient for a specific clinical conclusion."
        ),
    }
