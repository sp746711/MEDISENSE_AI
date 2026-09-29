"""Assessment orchestration service for MediSense AI.

Executes the multimodal diagnostic support pipeline:
1. Gathers multimodal inputs (symptoms, reports, X-rays)
2. Compiles structured evidence with contradiction & missing check
3. Applies deterministic rule-based triage
4. Maps specialty using controlled clinical mapping
5. Records auditable log in ai_audit_logs
6. Updates Assessment with full result payload
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.evidence_engine import combine_evidence
from app.ai.specialty_mapper import map_evidence_to_specialty
from app.ai.triage_engine import apply_triage
from app.core.config import get_settings
from app.database.models import AiAuditLog, Assessment, MedicalReport, Symptom, XrayResult


def run_assessment_pipeline(db: Session, assessment_id: UUID, user_id: UUID) -> dict[str, Any]:
    """Execute complete end-to-end multimodal assessment pipeline."""
    assessment = db.get(Assessment, assessment_id)
    if not assessment or assessment.user_id != user_id:
        return {"status": "error", "message": "Assessment not found"}

    settings = get_settings()

    # 1. Fetch symptoms
    symptom_rows = (
        db.query(Symptom)
        .filter(Symptom.assessment_id == assessment_id)
        .all()
    )
    symptoms_list = [
        {
            "symptom": s.symptom,
            "state": s.state,
            "duration": s.duration,
            "severity": s.severity,
            "body_area": s.body_area,
            "source": s.source,
        }
        for s in symptom_rows
    ]

    # 2. Fetch medical reports
    report_rows = (
        db.query(MedicalReport)
        .filter(MedicalReport.assessment_id == assessment_id)
        .all()
    )
    report_findings_list: list[dict[str, Any]] = []
    for r in report_rows:
        struct = r.structured_findings or {}
        if isinstance(struct, dict):
            report_findings_list.extend(struct.get("lab_parameters", []))

    # 3. Fetch X-ray result
    xray_row = (
        db.query(XrayResult)
        .filter(XrayResult.assessment_id == assessment_id)
        .order_by(XrayResult.created_at.desc())
        .first()
    )
    xray_dict = None
    if xray_row:
        xray_dict = {
            "status": xray_row.status,
            "region": xray_row.region,
            "prediction": xray_row.prediction,
            "model_version": xray_row.model_version,
            "uncertainty": xray_row.uncertainty,
            "explainability_artifact": xray_row.explainability_artifact,
            "message": xray_row.message,
        }

    # 4. Combine evidence
    evidence = combine_evidence(
        symptoms=symptoms_list,
        report_findings=report_findings_list,
        xray_result=xray_dict,
    )

    # 5. Apply deterministic rule-based triage
    triage_result = apply_triage(evidence)

    # 6. Map specialty
    specialty_result = map_evidence_to_specialty(evidence)
    suggested_specialty = specialty_result.get("suggested_specialty")

    # 7. Construct complete result payload
    final_status = "completed"
    if triage_result.get("status") in {"INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"}:
        final_status = triage_result["status"].lower()

    result_payload = {
        "status": final_status,
        "pathway": triage_result.get("pathway"),
        "specialty": suggested_specialty,
        "triage": triage_result,
        "specialty_mapping": specialty_result,
        "evidence": evidence,
        "summary": {
            "symptoms_count": len(symptoms_list),
            "reports_count": len(report_rows),
            "xray_present": bool(xray_row),
            "triage_pathway": triage_result.get("pathway"),
            "suggested_specialty": suggested_specialty,
        },
    }

    # 8. Update Assessment record
    assessment.status = final_status
    assessment.pathway = triage_result.get("pathway")
    assessment.specialty = suggested_specialty
    assessment.result_payload = result_payload
    assessment.rules_version = triage_result.get("rules_version", settings.rules_version)
    db.add(assessment)

    # 9. Create AI Audit Log record
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
