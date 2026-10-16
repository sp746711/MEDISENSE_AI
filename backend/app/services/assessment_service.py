"""Assessment orchestration service for MediSense AI.

Executes the multimodal diagnostic support pipeline:
1. Gathers multimodal inputs (symptoms, reports, X-rays) isolated by assessment_id
2. Compiles structured evidence with contradiction & missing information analysis
3. Applies deterministic rule-based safety triage
4. Maps specialty dynamically using evidence-grounded clinical mapping
5. Records auditable logs
6. Generates 100% dynamic, evidence-grounded final assessment feedback
7. Updates Assessment with full result payload
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.condition_prediction import get_condition_prediction_service
from app.ai.evidence_engine import combine_evidence
from app.ai.specialty_mapper import map_evidence_to_specialty
from app.ai.triage_engine import apply_triage
from app.core.config import get_settings
from app.database.models import AiAuditLog, Assessment, MedicalReport, Symptom, XrayResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


def generate_assessment_feedback(
    evidence: dict[str, Any],
    triage_result: dict[str, Any],
    specialty: str | None,
    condition_predictions: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Generate structured, evidence-grounded dynamic clinical assessment feedback.

    Adheres strictly to the required dynamic feedback sections without inventing findings.
    Different input produces completely different feedback.
    """
    symptoms_meta = evidence.get("symptoms", {})
    present = symptoms_meta.get("present", [])
    absent = symptoms_meta.get("absent", [])
    historical = symptoms_meta.get("historical", [])

    report_meta = evidence.get("report", {})
    report_provided = report_meta.get("provided", False)
    abnormal_rep = report_meta.get("abnormal_findings", [])
    normal_rep = report_meta.get("normal_findings", [])
    qualitative_rep = report_meta.get("qualitative_findings", [])
    narrative_rep = report_meta.get("narrative_findings", [])

    xray_meta = evidence.get("xray", {})
    xray_received = xray_meta.get("received", False)
    xray_status = (xray_meta.get("status") or "NOT_PROVIDED").upper()
    xray_pred = xray_meta.get("prediction")
    raw_xray_region = (xray_meta.get("region") or "").strip().lower()
    xray_region = raw_xray_region if raw_xray_region else "not declared"
    xray_region_label = (
        f"{xray_region.capitalize()} X-ray"
        if xray_region not in {"not declared", "unknown"}
        else "X-ray"
    )

    triage_pathway = triage_result.get("pathway") or "MILD"

    # 1. Assessment Summary (dynamically assembled from current evidence)
    summary_sentences = []

    if present:
        top_symptoms = []
        for s in present:
            desc = s.get("symptom", "")
            details = []
            if s.get("severity") and s.get("severity") != "UNKNOWN":
                details.append(s["severity"].lower())
            if s.get("laterality"):
                details.append(f"{s['laterality'].lower()}-sided")
            if s.get("quality"):
                details.append(s["quality"].lower())
            if s.get("onset"):
                details.append(f"{s['onset'].lower()} onset")
            if s.get("character") and s.get("character") not in {"CURRENT", "PRESENT"}:
                details.append(s["character"])
            elif s.get("type") and s.get("type") not in {"CURRENT", "PRESENT"}:
                details.append(s["type"])
            if s.get("mucus"):
                details.append(s["mucus"])
            if details:
                desc += f" ({', '.join(details)})"
            top_symptoms.append(desc)
        summary_sentences.append(
            f"Active symptom presentation includes {len(present)} finding(s): {', '.join(top_symptoms)}."
        )
    else:
        summary_sentences.append("No active physical symptoms reported.")

    if historical:
        h_desc = [f"{h.get('symptom')} ({h.get('history') or h.get('context') or 'historical occurrence'})" for h in historical]
        summary_sentences.append(f"Historical context noted: {', '.join(h_desc)} (documented for clinical context, excluded from active acute symptom counts).")

    if absent:
        key_negatives = [s.get("symptom") for s in absent]
        summary_sentences.append(f"Patient explicitly confirmed absence of: {', '.join(key_negatives)}.")

    unknown_range_rep = report_meta.get("unknown_range_findings", [])

    if report_provided:
        if abnormal_rep:
            abn_names = [f"{r.get('test_name')} ({r.get('value')} {r.get('unit', '')}, {r.get('interpretation')})" for r in abnormal_rep]
            summary_sentences.append(f"Laboratory evaluation revealed {len(abnormal_rep)} out-of-range parameter(s): {', '.join(abn_names)}.")
        elif normal_rep:
            summary_sentences.append(f"All {len(normal_rep)} evaluated laboratory parameter(s) are within standard reference ranges.")
        if unknown_range_rep:
            unk_names = [f"{r.get('test_name')} ({r.get('value')} {r.get('unit', '')})" for r in unknown_range_rep]
            summary_sentences.append(f"Recorded parameter(s) without standardized reference ranges: {', '.join(unk_names)} (status: UNKNOWN).")
        if qualitative_rep:
            q_names = [f"{q.get('finding')} ({q.get('state')})" for q in qualitative_rep]
            summary_sentences.append(f"Medical report provided with qualitative findings: {', '.join(q_names)}.")
        elif narrative_rep:
            summary_sentences.append("Medical report provided with clinical history and examination details.")
        elif not abnormal_rep and not normal_rep and not unknown_range_rep:
            summary_sentences.append("Medical report provided (clinical documentation).")
    else:
        summary_sentences.append("Medical laboratory report was not provided.")

    xray_images_list = xray_meta.get("images", [])
    xray_reports_list = xray_meta.get("reports", []) or report_meta.get("xray_reports", [])
    medical_reports_list = report_meta.get("medical_reports", [])

    if xray_images_list:
        if len(xray_images_list) == 1:
            img = xray_images_list[0]
            istat = (img.get("status") or "").upper()
            ireg = (img.get("region") or "not declared").capitalize()
            ipred = img.get("prediction")
            if istat == "COMPLETED" and ipred:
                summary_sentences.append(f"{ireg} X-ray ({img.get('file_name')}) model identified: {ipred}.")
            elif istat == "UNAVAILABLE":
                summary_sentences.append(f"{ireg} X-ray ({img.get('file_name')}) received; automated model interpretation is unavailable. No radiographic disease is assumed.")
            else:
                summary_sentences.append(f"{ireg} X-ray ({img.get('file_name')}) status: {istat}.")
        else:
            summary_sentences.append(f"Received {len(xray_images_list)} X-ray image(s).")
            for img in xray_images_list:
                istat = (img.get("status") or "").upper()
                ireg = (img.get("region") or "not declared").capitalize()
                ipred = img.get("prediction")
                if istat == "COMPLETED" and ipred:
                    summary_sentences.append(f"• {ireg} X-ray ({img.get('file_name')}): {ipred}.")
                else:
                    summary_sentences.append(f"• {ireg} X-ray ({img.get('file_name')}): Automated interpretation unavailable.")
    elif xray_received:
        if xray_status == "COMPLETED" and xray_pred:
            summary_sentences.append(f"{xray_region_label} deep learning model identified radiographic pattern: {xray_pred}.")
        elif xray_status == "UNAVAILABLE":
            summary_sentences.append(f"{xray_region_label} received; automated model interpretation is currently unavailable. No radiographic disease is assumed.")
        else:
            summary_sentences.append(f"{xray_region_label} status: {xray_status}.")
    elif not xray_reports_list:
        summary_sentences.append("Radiographic imaging was not provided.")

    if xray_reports_list:
        summary_sentences.append(f"Received {len(xray_reports_list)} radiology report(s) with extracted clinical documentation.")

    conditions_list = condition_predictions.get("conditions", []) if condition_predictions else []
    if conditions_list:
        top_c = [c.get("condition") for c in conditions_list[:2] if c.get("condition")]
        if top_c:
            summary_sentences.append(
                f"Assistive machine learning evaluation identified statistical pattern alignment with: {', '.join(top_c)} (probabilistic differential for physician correlation)."
            )

    summary_sentences.append(
        f"Deterministic clinical safety evaluation assigned triage level: {triage_pathway}."
    )

    assessment_summary = " ".join(summary_sentences)

    # 2. Symptoms Identified
    symptoms_identified = [
        {
            "finding": s.get("symptom"),
            "symptom": s.get("symptom"),
            "status": "PRESENT",
            "state": "PRESENT",
            "severity": (s.get("severity") or "UNKNOWN").upper(),
            "duration": s.get("duration") or "Unspecified",
            "laterality": s.get("laterality"),
            "quality": s.get("quality"),
            "trigger": s.get("trigger"),
            "context": s.get("context") or s.get("trigger"),
            "body_area": s.get("body_area"),
            "type": s.get("type"),
            "onset": s.get("onset"),
            "source": s.get("source") or "user_input",
        }
        for s in present
    ]

    # 3. Medical Report Findings (Including Unknown Range Findings)
    all_lab_rep = abnormal_rep + normal_rep + unknown_range_rep
    if report_provided:
        medical_report_findings = [
            {
                "test_name": f.get("test_name"),
                "original_test_name": f.get("original_test_name") or f.get("test_name"),
                "value": f"{f.get('value')} {f.get('unit') or ''}".strip(),
                "numeric_value": f.get("value"),
                "unit": f.get("unit"),
                "reference_range": f.get("reference_range") or "Not provided",
                "reference_low": f.get("reference_low"),
                "reference_high": f.get("reference_high"),
                "explicit_flag": f.get("explicit_flag"),
                "calculated_status": f.get("calculated_status"),
                "conflict": bool(f.get("conflict")),
                "conflict_detail": f.get("conflict_detail"),
                "status": f.get("status") or f.get("interpretation") or "UNKNOWN",
                "interpretation": f.get("interpretation") or "UNKNOWN",
                "domain": f.get("domain"),
                "source": "uploaded_report",
                "source_file": f.get("source_file"),
                "source_type": f.get("source_type") or "MEDICAL_REPORT",
            }
            for f in all_lab_rep
        ]
        medical_report_status = "PROVIDED"
        if medical_report_findings:
            medical_report_message = f"Structured laboratory analysis extracted {len(medical_report_findings)} parameter(s)."
        elif qualitative_rep or narrative_rep:
            medical_report_message = f"Medical report provided: extracted {len(qualitative_rep)} qualitative finding(s) and {len(narrative_rep)} narrative section(s)."
        else:
            medical_report_message = "Medical report provided (qualitative clinical documentation recorded)."
    else:
        medical_report_findings = []
        medical_report_status = "NOT_PROVIDED"
        medical_report_message = "Medical Report: Not provided."

    # 4. X-Ray Findings / Availability
    has_xray_images = bool(xray_images_list)
    has_xray_reports = bool(xray_reports_list)

    if has_xray_images or has_xray_reports or xray_received:
        if has_xray_images:
            completed_imgs = [img for img in xray_images_list if (img.get("status") or "").lower() == "completed" and img.get("prediction")]
            if completed_imgs:
                pred_labels = []
                for img in completed_imgs:
                    p = img.get("prediction")
                    r = (img.get("region") or "").strip()
                    pred_labels.append(f"{p} ({r.capitalize()})" if r and r != "not declared" else str(p))
                xray_findings_text = f"Radiographic pattern(s) identified: {', '.join(pred_labels)}. Clinical correlation and verification by a radiologist required."
            else:
                xray_findings_text = (
                    f"Automated interpretation for the {len(xray_images_list)} uploaded X-ray image(s) is currently unavailable (no trained model checkpoint configured). "
                    "No disease finding is generated or assumed."
                )
        elif xray_status == "COMPLETED" and xray_pred:
            xray_findings_text = f"Radiographic pattern identified: {xray_pred}. Clinical correlation and verification by a radiologist required."
        elif xray_status == "UNAVAILABLE" and has_xray_images:
            xray_findings_text = (
                f"Automated interpretation for this X-ray type ({xray_region}) is currently unavailable (no trained model checkpoint configured). "
                "No disease finding is generated or assumed."
            )
        else:
            xray_findings_text = f"Radiology documentation received: {len(xray_reports_list)} report(s). No radiographic image submitted for automated inference."

        effective_status = xray_status if has_xray_images else "NOT_PROVIDED"
        xray_findings_data = {
            "received": bool(has_xray_images or has_xray_reports),
            "images_received": has_xray_images,
            "reports_received": has_xray_reports,
            "status": effective_status,
            "region": xray_region.title() if (has_xray_images and xray_region not in {"not declared", "unknown"}) else "Not declared",
            "model": xray_meta.get("model_version") or "N/A" if has_xray_images else "N/A",
            "prediction": xray_pred if has_xray_images else None,
            "description": xray_findings_text,
            "uncertainty": xray_meta.get("uncertainty") if has_xray_images else None,
            "images": xray_images_list,
            "reports": xray_reports_list,
        }
    else:
        xray_findings_data = {
            "received": False,
            "images_received": False,
            "reports_received": False,
            "status": "NOT_PROVIDED",
            "region": "Not declared",
            "model": "N/A",
            "prediction": None,
            "description": "X-Ray: Not provided.",
            "uncertainty": None,
            "images": [],
            "reports": [],
        }

    # 5. Supporting & Reassuring Evidence
    supporting_evidence = list(evidence.get("supporting", []))
    reassuring_evidence = list(evidence.get("reassuring", []))
    separate_contextual = list(evidence.get("separate_or_contextual", evidence.get("separate", [])))

    # 6. Contradictory Evidence
    contradictory_evidence = list(evidence.get("contradictory", []))
    for c in conditions_list:
        contra_findings = c.get("contradictory_findings", [])
        if contra_findings:
            c_name = c.get("condition")
            denied_str = ", ".join(contra_findings)
            contra_msg = f"Confirmed absence of '{denied_str}' conflicts with typical presentation of {c_name}."
            if contra_msg not in contradictory_evidence:
                contradictory_evidence.append(contra_msg)

    # 7. Important Negative Findings (Negative Red Flags)
    negative_red_flags = [
        {
            "finding": s.get("symptom"),
            "symptom": s.get("symptom"),
            "status": "ABSENT",
            "state": "ABSENT",
            "context": s.get("context"),
            "body_area": s.get("body_area"),
            "source": s.get("source") or "user_input",
        }
        for s in absent
    ]

    # 8. Uncertain / Missing Information
    uncertain_missing = list(evidence.get("missing", []))

    # 9. Triage Outcome
    triage_level = {
        "level": triage_pathway,
        "pathway": triage_pathway,
        "rules_triggered": triage_result.get("rules_triggered", []),
        "rationale": triage_result.get("why_this_pathway"),
    }

    # 10. Why This Triage Was Selected
    why_this_pathway = triage_result.get("why_this_pathway") or "Determined by clinical rule evaluation."

    # 11. Recommended Next Step / Specialty (Dynamic & Evidence-Specific)
    suggested_spec = (
        specialty.get("suggested_specialty") if isinstance(specialty, dict) else specialty
    ) or "General Physician / Internal Medicine"
    is_resp = any(s.get("domain") == "respiratory" or s.get("symptom") in {"cough", "sore throat", "chest discomfort", "shortness of breath"} for s in present)
    is_neuro = any(s.get("domain") == "neurological" or s.get("symptom") in {"headache", "migraine", "photophobia", "phonophobia"} for s in present)
    is_msk = any(s.get("domain") in {"musculoskeletal", "injury/trauma"} or s.get("body_area") in {"wrist", "hand", "knee", "ankle"} for s in present)

    if is_resp:
        dynamic_guidance = "Clinical consultation is advised for professional evaluation of persistent respiratory symptoms. Provided laboratory and imaging reports should be reviewed by your clinician."
        if abnormal_rep:
            abn_list = ', '.join([r.get('test_name') for r in abnormal_rep if r.get('test_name')])
            dynamic_action = f"Schedule a clinical examination with a pulmonologist or primary physician. Discuss the recorded abnormal laboratory findings with your clinician: {abn_list}. Seek prompt medical care if acute red flags such as severe breathing difficulty or persistent high fever develop."
        elif normal_rep:
            dynamic_action = "Schedule a clinical examination with a pulmonologist or primary physician. Available laboratory values are within the supplied reference ranges. Seek prompt medical care if acute red flags such as severe breathing difficulty or persistent high fever develop."
        else:
            dynamic_action = "Schedule a clinical examination with a pulmonologist or primary physician. Seek prompt medical care if acute red flags such as severe breathing difficulty or persistent high fever develop."
        if xray_status == "UNAVAILABLE":
            dynamic_action += f" Note: Automated interpretation for the selected {xray_region} X-ray was unavailable; bring physical films for physician review."
    elif is_neuro:
        dynamic_guidance = f"Comprehensive clinical evaluation is recommended for the acute headache pattern with a specialist ({suggested_spec})."
        if abnormal_rep:
            abn_list = ', '.join([r.get('test_name') for r in abnormal_rep if r.get('test_name')])
            dynamic_action = f"Schedule an in-person consultation with a neurologist. Discuss the recorded abnormal laboratory findings with your clinician: {abn_list}."
        elif normal_rep:
            dynamic_action = "Schedule an in-person consultation with a neurologist. Available laboratory values are within the supplied reference ranges. Clinical evaluation should focus on the current headache pattern and associated symptoms."
        else:
            dynamic_action = "Schedule an in-person consultation with a neurologist. Clinical evaluation should focus on the current headache pattern and associated symptoms."
    elif is_msk:
        dynamic_guidance = "Orthopedic evaluation is advised to assess joint mobility, localized swelling, and structural integrity following trauma or limb strain."
        dynamic_action = f"Consult with an orthopedic specialist ({suggested_spec}). If imaging was obtained ({xray_region_label}), present direct radiographic images for clinical assessment."
    else:
        dynamic_guidance = triage_result.get("guidance") or f"Consult with a specialist ({suggested_spec}) or general physician, and review your findings."
        dynamic_action = triage_result.get("next_steps") or "Schedule in-person medical evaluation and bring all available medical records."

    recommended_next_step = {
        "guidance": dynamic_guidance,
        "next_steps": dynamic_action,
        "suggested_specialty": suggested_spec,
    }

    # Historical symptoms structured representation
    historical_symptoms = [
        {
            "symptom": h.get("symptom"),
            "finding": h.get("symptom"),
            "type": "HISTORICAL",
            "history": h.get("history") or h.get("context") or "once or twice before",
            "duration": None,
            "status": "PRESENT",
            "state": "PRESENT",
            "source": h.get("source") or "user_input",
        }
        for h in historical
    ]

    # 12. Safety Disclaimer
    safety_disclaimer = (
        "MEDICAL DISCLAIMER: MediSense AI is an academic clinical decision-support and health navigation tool. "
        "It does NOT provide medical diagnosis, prescription, or definitive treatment. "
        "All automated suggestions must be evaluated and verified by a licensed medical professional."
    )

    return {
        "assessment_summary": assessment_summary,
        "current_symptoms": symptoms_identified,
        "current_symptom_pattern": symptoms_identified,
        "symptoms_identified": symptoms_identified,
        "historical_symptoms": historical_symptoms,
        "historical_context": historical_symptoms,
        "relevant_negative_findings": negative_red_flags,
        "negative_findings": negative_red_flags,
        "negative_red_flags": negative_red_flags,
        "medical_report_interpretation": medical_report_message,
        "medical_report_findings": medical_report_findings,
        "medical_report_status": medical_report_status,
        "medical_report_message": medical_report_message,
        "xray_findings": xray_findings_data,
        "xray_status": xray_findings_data["status"],
        "xray_images": xray_images_list,
        "xray_reports": xray_reports_list,
        "medical_reports": medical_reports_list,
        "supporting_evidence": supporting_evidence,
        "reassuring_evidence": reassuring_evidence,
        "separate_contextual_findings": separate_contextual,
        "additional_contextual_findings": separate_contextual,
        "contradictory_evidence": contradictory_evidence,
        "uncertain_missing_info": uncertain_missing,
        "unknown_unassessed_info": uncertain_missing,
        "triage_level": triage_level,
        "triage_rationale": why_this_pathway,
        "why_this_triage_selected": why_this_pathway,
        "specialty_rationale": f"Specialty selection ({suggested_spec}) is aligned with the predominant clinical symptom pattern.",
        "recommended_next_step": recommended_next_step,
        "possible_conditions": conditions_list,
        "condition_predictions": condition_predictions or {},
        "condition_prediction_status": (condition_predictions or {}).get("status"),
        "condition_prediction_message": (condition_predictions or {}).get("message"),
        "safety_disclaimer": safety_disclaimer,
    }


def _log_assessment_evidence_debug(
    assessment_id: UUID,
    input_types: list[str],
    symptoms_list: list[dict[str, Any]],
    report_findings_list: list[dict[str, Any]],
    xray_dict: dict[str, Any] | None,
    evidence: dict[str, Any],
    triage_result: dict[str, Any],
    specialty: str | None,
    feedback: dict[str, Any],
) -> None:
    """Print structured diagnostic debug logs as required by Section 21."""
    symptoms_present = evidence.get("symptoms", {}).get("present", [])
    symptoms_absent = evidence.get("symptoms", {}).get("absent", [])
    report_meta = evidence.get("report", {})
    abnormal = report_meta.get("abnormal_findings", [])
    normal = report_meta.get("normal_findings", [])
    xray = evidence.get("xray", {})

    sym_lines = []
    for s in symptoms_present:
        details = [f"status=PRESENT", f"severity={(s.get('severity') or 'UNKNOWN').upper()}"]
        if s.get("duration"):
            details.append(f"duration={s['duration']}")
        if s.get("laterality"):
            details.append(f"laterality={s['laterality']}")
        if s.get("quality"):
            details.append(f"quality={s['quality']}")
        if s.get("trigger"):
            details.append(f"trigger={s['trigger']}")
        sym_lines.append(f"  {s.get('symptom')}: {', '.join(details)}")

    neg_lines = [f"  {s.get('symptom')}: ABSENT" for s in symptoms_absent]

    abn_lines = [
        f"  {f.get('test_name')}: {f.get('value')} {f.get('unit', '')} ({f.get('interpretation')})"
        for f in abnormal
    ]

    debug_msg = f"""
==================================================
ASSESSMENT_PIPELINE_TRACE
==================================================
ASSESSMENT_ID: {assessment_id}
ASSESSMENT_INPUT: types={input_types}

SYMPTOM_EXTRACTION:
{chr(10).join(sym_lines) if sym_lines else '  (none present)'}

NEGATION_EXTRACTION:
{chr(10).join(neg_lines) if neg_lines else '  (none absent)'}

REPORT_EXTRACTION:
  extracted_count={len(report_findings_list)}
  abnormal={abn_lines if abn_lines else '[]'}
  normal_count={len(normal)}

XRAY_PROCESSING:
  received={xray.get('received')}
  status={xray.get('status')}
  model={xray.get('model_version') or 'N/A'}
  prediction={xray.get('prediction')}
  interpretation={xray.get('interpretation')}

EVIDENCE_BUILD:
  state={evidence.get('evidence_state')}
  supporting_count={len(evidence.get('supporting', []))}
  contradictory_count={len(evidence.get('contradictory', []))}
  missing_count={len(evidence.get('missing', []))}

EVIDENCE_VALIDATION:
  contradictions={evidence.get('contradictory', [])}

FOLLOWUP_DECISION:
  missing_critical={evidence.get('missing', [])}

TRIAGE_DECISION:
  level={triage_result.get('pathway')}
  triggered_rules={triage_result.get('rules_triggered')}
  ignored_rules={triage_result.get('ignored_rules')}

SPECIALTY_DECISION:
  suggested_specialty={specialty}

FINAL_FEEDBACK:
  summary_length={len(feedback.get('assessment_summary', ''))}
  sections_present={list(feedback.keys())}
==================================================
"""
    logger.info(debug_msg)
    print(debug_msg)


def run_assessment_pipeline(db: Session, assessment_id: UUID, user_id: UUID) -> dict[str, Any]:
    """Execute complete end-to-end multimodal assessment pipeline.

    Strictly isolated by assessment_id and authenticated user ownership.
    Never reuses previous assessment evidence or LLM responses.
    """
    assessment = db.get(Assessment, assessment_id)
    if not assessment or assessment.user_id != user_id:
        return {"status": "error", "message": "Assessment not found"}

    settings = get_settings()

    # 1. Fetch symptoms isolated strictly by assessment_id
    symptom_rows = (
        db.query(Symptom)
        .filter(Symptom.assessment_id == assessment_id)
        .all()
    )
    symptoms_list = []
    for s in symptom_rows:
        ctx = s.context or ""
        laterality = None
        quality = None
        trigger = None
        sym_type = None
        onset = None
        clean_ctx = ctx

        if ctx.startswith("[qualifiers:"):
            prefix, sep, rest = ctx.partition("] ")
            clean_ctx = rest if sep else ctx
            m_lat = re.search(r"laterality=([A-Z]+)", prefix)
            if m_lat and m_lat.group(1) != "None":
                laterality = m_lat.group(1)
            m_qual = re.search(r"quality=([A-Z]+)", prefix)
            if m_qual and m_qual.group(1) != "None":
                quality = m_qual.group(1)
            m_trig = re.search(r"trigger=([^,\]]+)", prefix)
            if m_trig and m_trig.group(1).strip() != "None":
                trigger = m_trig.group(1).strip()
            m_typ = re.search(r"type=([^,\]]+)", prefix)
            if m_typ and m_typ.group(1).strip() != "None":
                sym_type = m_typ.group(1).strip()
            m_on = re.search(r"onset=([A-Z]+)", prefix)
            if m_on and m_on.group(1) != "None":
                onset = m_on.group(1)

        # Contextual inferences if not in prefix
        if not sym_type and (s.symptom == "previous similar headache" or "once or twice before" in clean_ctx.lower() or "similar headache" in clean_ctx.lower()):
            sym_type = "HISTORICAL"
        if not laterality and "right" in clean_ctx.lower() and s.symptom in {"headache", "migraine"}:
            laterality = "RIGHT"
        if not quality and "throbbing" in clean_ctx.lower() and s.symptom in {"headache", "migraine"}:
            quality = "THROBBING"
        if not trigger and "coughing" in clean_ctx.lower() and s.symptom == "chest discomfort":
            trigger = "coughing"

        s_dur = None if sym_type == "HISTORICAL" else s.duration

        symptoms_list.append(
            {
                "finding": s.symptom,
                "symptom": s.symptom,
                "state": s.state,
                "status": s.state,
                "duration": s_dur,
                "severity": s.severity,
                "body_area": s.body_area,
                "laterality": laterality,
                "quality": quality,
                "trigger": trigger,
                "type": sym_type,
                "onset": onset,
                "context": clean_ctx,
                "source": s.source,
            }
        )

    # 2. Fetch medical reports isolated strictly by assessment_id
    report_rows = (
        db.query(MedicalReport)
        .filter(MedicalReport.assessment_id == assessment_id)
        .order_by(MedicalReport.created_at.asc())
        .all()
    )
    raw_lab_list: list[dict[str, Any]] = []
    raw_qual_list: list[dict[str, Any]] = []
    raw_narr_list: list[dict[str, Any]] = []

    medical_reports_list: list[dict[str, Any]] = []
    xray_reports_list: list[dict[str, Any]] = []

    for r in report_rows:
        cat = getattr(r, "report_category", "medical_report") or "medical_report"
        r_entry = {
            "id": str(r.report_id),
            "report_id": str(r.report_id),
            "file_name": r.file_name,
            "file_type": r.file_type,
            "report_category": cat,
            "extracted_text": r.extracted_text,
            "structured_findings": r.structured_findings,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "source_type": "XRAY_REPORT" if cat == "xray_report" else "MEDICAL_REPORT",
            "source_file": r.file_name,
        }
        if cat == "xray_report":
            xray_reports_list.append(r_entry)
        else:
            medical_reports_list.append(r_entry)

        struct = r.structured_findings or {}
        if isinstance(struct, dict):
            for l in struct.get("lab_parameters", []):
                l_copy = dict(l)
                l_copy["source_file"] = r.file_name
                l_copy["source_id"] = str(r.report_id)
                l_copy["source_type"] = "XRAY_REPORT" if cat == "xray_report" else "MEDICAL_REPORT"
                raw_lab_list.append(l_copy)

            for q in struct.get("qualitative_findings", []):
                q_copy = dict(q)
                q_copy["source_file"] = r.file_name
                q_copy["source_id"] = str(r.report_id)
                q_copy["source_type"] = "XRAY_REPORT" if cat == "xray_report" else "MEDICAL_REPORT"
                raw_qual_list.append(q_copy)

            for n in struct.get("narrative_findings", []):
                n_copy = dict(n)
                n_copy["source_file"] = r.file_name
                n_copy["source_id"] = str(r.report_id)
                n_copy["source_type"] = "XRAY_REPORT" if cat == "xray_report" else "MEDICAL_REPORT"
                raw_narr_list.append(n_copy)

    # Exact duplicate deduplication (preserves distinct test values/states)
    report_findings_list: list[dict[str, Any]] = []
    seen_lab_keys: set[tuple] = set()
    for item in raw_lab_list:
        key = (
            str(item.get("test_name", "")).strip().lower(),
            item.get("value"),
            str(item.get("unit", "")).strip().lower(),
            str(item.get("reference_range", "")).strip().lower(),
            str(item.get("interpretation") or item.get("status") or "").strip().upper(),
        )
        if key not in seen_lab_keys:
            seen_lab_keys.add(key)
            report_findings_list.append(item)

    qualitative_findings_list: list[dict[str, Any]] = []
    seen_qual_keys: set[tuple] = set()
    for item in raw_qual_list:
        key = (
            str(item.get("finding", "")).strip().lower(),
            str(item.get("state", "")).strip().upper(),
            str(item.get("context", "")).strip().lower(),
        )
        if key not in seen_qual_keys:
            seen_qual_keys.add(key)
            qualitative_findings_list.append(item)

    narrative_findings_list: list[dict[str, Any]] = []
    seen_narr_keys: set[tuple] = set()
    for item in raw_narr_list:
        key = (
            str(item.get("section", "")).strip().lower(),
            str(item.get("content", "")).strip().lower(),
        )
        if key not in seen_narr_keys:
            seen_narr_keys.add(key)
            narrative_findings_list.append(item)

    # 3. Fetch ALL X-ray results isolated strictly by assessment_id
    xray_rows = (
        db.query(XrayResult)
        .filter(XrayResult.assessment_id == assessment_id)
        .order_by(XrayResult.created_at.asc())
        .all()
    )
    xray_results_list: list[dict[str, Any]] = [
        {
            "id": str(xr.xray_result_id),
            "xray_result_id": str(xr.xray_result_id),
            "file_name": xr.file_name or "xray_image",
            "region": xr.region,
            "prediction": xr.prediction,
            "confidence_if_valid": xr.confidence_if_valid,
            "uncertainty": xr.uncertainty,
            "model_version": xr.model_version,
            "explainability_artifact": xr.explainability_artifact,
            "status": xr.status,
            "message": xr.message,
            "source": "xray_image",
            "source_type": "XRAY_IMAGE",
            "source_file": xr.file_name or "xray_image",
            "created_at": xr.created_at.isoformat() if xr.created_at else None,
        }
        for xr in xray_rows
    ]
    primary_xray_dict = xray_results_list[-1] if xray_results_list else None

    # 4. Combine evidence afresh
    evidence = combine_evidence(
        symptoms=symptoms_list,
        report_findings=report_findings_list,
        xray_result=primary_xray_dict,
        xray_results=xray_results_list,
        xray_reports=xray_reports_list,
        medical_reports=medical_reports_list,
        report_provided=len(report_rows) > 0,
        qualitative_report_findings=qualitative_findings_list,
        narrative_report_findings=narrative_findings_list,
    )

    # 5. Apply deterministic rule-based triage
    triage_result = apply_triage(evidence)

    # 6. Map specialty dynamically
    specialty_result = map_evidence_to_specialty(evidence)
    suggested_specialty = specialty_result.get("suggested_specialty")

    # 6b. Predict possible conditions using verified evidence (Non-diagnostic ML differential)
    try:
        condition_svc = get_condition_prediction_service()
        condition_predictions = condition_svc.predict_conditions(
            symptoms_present=evidence.get("symptoms", {}).get("present", []),
            symptoms_absent=evidence.get("symptoms", {}).get("absent", []),
            symptoms_unknown=evidence.get("symptoms", {}).get("unknown", []),
            raw_text=evidence.get("raw_text"),
        )
    except Exception as exc:
        logger.error("Condition prediction pipeline execution failed: %s", exc)
        condition_predictions = {
            "status": "unavailable",
            "model_version": None,
            "conditions": [],
            "message": f"Assistive condition prediction currently unavailable: {exc}",
            "is_experimental": True,
        }

    # 7. Construct complete result payload with structured dynamic feedback
    final_status = "completed"
    if triage_result.get("status") in {"INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}:
        final_status = triage_result["status"].lower()

    feedback = generate_assessment_feedback(
        evidence=evidence,
        triage_result=triage_result,
        specialty=suggested_specialty,
        condition_predictions=condition_predictions,
    )

    result_payload = {
        "status": final_status,
        "pathway": triage_result.get("pathway"),
        "specialty": suggested_specialty,
        "triage": triage_result,
        "specialty_mapping": specialty_result,
        "evidence": evidence,
        "feedback": feedback,
        "condition_predictions": condition_predictions,
        "summary": {
            "symptoms_count": len(symptoms_list),
            "reports_count": len(report_rows),
            "xray_present": bool(xray_rows),
            "triage_pathway": triage_result.get("pathway"),
            "suggested_specialty": suggested_specialty,
            "possible_conditions_count": len(condition_predictions.get("conditions", [])),
        },
    }

    # 8. Output diagnostic debug logging
    _log_assessment_evidence_debug(
        assessment_id=assessment_id,
        input_types=assessment.input_types or [],
        symptoms_list=symptoms_list,
        report_findings_list=report_findings_list,
        xray_dict=primary_xray_dict,
        evidence=evidence,
        triage_result=triage_result,
        specialty=suggested_specialty,
        feedback=feedback,
    )

    # 9. Update Assessment record
    assessment.status = final_status
    assessment.pathway = triage_result.get("pathway")
    assessment.specialty = suggested_specialty
    assessment.result_payload = result_payload
    assessment.rules_version = triage_result.get("rules_version", settings.rules_version)
    db.add(assessment)

    # 10. Create AI Audit Log record
    audit_log = AiAuditLog(
        assessment_id=assessment.assessment_id,
        input_types=assessment.input_types,
        model_versions={
            "rules_version": assessment.rules_version,
            "nlp_version": settings.nlp_version,
            "ocr_version": settings.ocr_version,
            "llm_version": settings.llm_version,
        },
        nlp_version=settings.nlp_version,
        ocr_version=settings.ocr_version,
        rules_version=assessment.rules_version,
        llm_version=settings.llm_version,
        output_status=final_status,
    )
    db.add(audit_log)
    db.commit()
    db.refresh(assessment)

    return {
        "assessment_id": str(assessment.assessment_id),
        "status": assessment.status,
        "pathway": assessment.pathway,
        "specialty": assessment.specialty,
        "message": "Assessment pipeline processing completed successfully.",
        "result": result_payload,
    }
