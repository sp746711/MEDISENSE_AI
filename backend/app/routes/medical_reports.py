"""Medical report upload routes — OCR/NLP in Stage 6."""

import uuid
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment, MedicalReport
from app.utils.file_validation import validate_report_file

router = APIRouter(prefix="/assessments", tags=["Medical Reports"])


@router.post("/{assessment_id}/reports")
async def upload_report(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
    file: UploadFile = File(...),
) -> dict:
    assessment = _owned(db, assessment_id, current_user.user_id)
    if "medical_report" not in (assessment.input_types or []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This assessment was not configured for medical report input.",
        )

    settings = get_settings()
    content = await file.read()
    validation = validate_report_file(file.filename or "", content, settings.max_upload_size_mb)
    if not validation["ok"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=validation["message"])

    reports_dir = settings.upload_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{validation['extension']}"
    dest = reports_dir / stored_name
    dest.write_bytes(content)

    from app.ai.report_ocr import extract_text_from_file
    from app.ai.report_nlp import structure_report_text

    ocr_result = extract_text_from_file(str(dest))
    extracted_text = ocr_result.get("extracted_text")

    nlp_result = structure_report_text(extracted_text)
    structured_findings = nlp_result.get("findings", [])
    qualitative_findings = nlp_result.get("qualitative_findings", [])
    combined_findings = {
        "lab_parameters": structured_findings,
        "qualitative_findings": qualitative_findings,
        "total_extracted": len(structured_findings) + len(qualitative_findings),
    }

    report = MedicalReport(
        assessment_id=assessment.assessment_id,
        file_name=Path(file.filename or "report").name,
        stored_name=stored_name,
        file_type=validation["file_type"],
        extracted_text=extracted_text,
        structured_findings=combined_findings,
        reference_ranges={f["test_name"]: f.get("reference_range") for f in structured_findings if f.get("reference_range")},
        ocr_metadata={
            "ocr_status": ocr_result.get("status"),
            "ocr_version": ocr_result.get("ocr_version"),
            "nlp_status": nlp_result.get("status"),
            "pages": ocr_result.get("pages", 1),
            "message": ocr_result.get("message"),
        },
    )
    db.add(report)
    if assessment.status == "draft":
        assessment.status = "inputs_received"
    db.add(assessment)
    db.commit()
    db.refresh(report)

    return {
        "status": "extracted" if extracted_text else "unreadable",
        "report_id": str(report.report_id),
        "file_name": report.file_name,
        "message": nlp_result.get("message") or ocr_result.get("message"),
        "structured_findings": combined_findings,
        "extracted_text_snippet": (extracted_text[:200] + "...") if extracted_text else None,
    }


@router.get("/{assessment_id}/reports")
def list_reports(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    _owned(db, assessment_id, current_user.user_id)
    rows = (
        db.query(MedicalReport)
        .filter(MedicalReport.assessment_id == assessment_id)
        .all()
    )
    return {
        "assessment_id": str(assessment_id),
        "reports": [
            {
                "report_id": str(r.report_id),
                "file_name": r.file_name,
                "file_type": r.file_type,
                "created_at": r.created_at.isoformat(),
                "has_extracted_text": bool(r.extracted_text),
                "structured_findings": r.structured_findings,
            }
            for r in rows
        ],
    }


def _owned(db: DbSession, assessment_id: UUID, user_id: UUID) -> Assessment:
    assessment = db.get(Assessment, assessment_id)
    if assessment is None or assessment.user_id != user_id:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment
