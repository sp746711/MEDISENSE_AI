"""Assessment routes — framework endpoints (processing logic in later stages)."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment
from app.schemas.assessment import AssessmentCreateRequest, AssessmentResponse

router = APIRouter(prefix="/assessments", tags=["Assessments"])

ALLOWED_INPUT_TYPES = {"symptoms", "medical_report", "xray"}


@router.post("", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
def create_assessment(
    payload: AssessmentCreateRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> AssessmentResponse:
    types = [t.strip().lower() for t in payload.input_types]
    if not types or any(t not in ALLOWED_INPUT_TYPES for t in types):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="input_types must be a non-empty subset of: symptoms, medical_report, xray",
        )
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_types: list[str] = []
    for t in types:
        if t not in seen:
            seen.add(t)
            unique_types.append(t)

    settings = get_settings()
    assessment = Assessment(
        user_id=current_user.user_id,
        input_types=unique_types,
        status="draft",
        rules_version=settings.rules_version,
        model_versions={},
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return AssessmentResponse.model_validate(assessment)


@router.get("", response_model=list[AssessmentResponse])
def list_assessments(
    current_user: CurrentUser,
    db: DbSession,
) -> list[AssessmentResponse]:
    rows = (
        db.query(Assessment)
        .filter(Assessment.user_id == current_user.user_id)
        .order_by(Assessment.created_at.desc())
        .all()
    )
    return [AssessmentResponse.model_validate(r) for r in rows]


@router.get("/{assessment_id}", response_model=AssessmentResponse)
def get_assessment(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> AssessmentResponse:
    assessment = _owned_assessment(db, assessment_id, current_user.user_id)
    return AssessmentResponse.model_validate(assessment)


@router.get("/{assessment_id}/result")
def get_result(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    assessment = _owned_assessment(db, assessment_id, current_user.user_id)
    if assessment.status not in {"completed", "insufficient_evidence", "conflicting_evidence"}:
        return {
            "assessment_id": str(assessment.assessment_id),
            "status": assessment.status,
            "message": "Assessment result is not ready yet.",
        }
    return {
        "assessment_id": str(assessment.assessment_id),
        "status": assessment.status,
        "pathway": assessment.pathway,
        "specialty": assessment.specialty,
        "result": assessment.result_payload,
        "input_types": assessment.input_types,
        "created_at": assessment.created_at.isoformat(),
        "rules_version": assessment.rules_version,
        "model_versions": assessment.model_versions,
    }


@router.post("/{assessment_id}/process")
def process_assessment(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Execute complete multimodal assessment pipeline."""
    from app.services.assessment_service import run_assessment_pipeline

    _owned_assessment(db, assessment_id, current_user.user_id)
    return run_assessment_pipeline(db, assessment_id, current_user.user_id)


@router.get("/{assessment_id}/report.pdf")
def download_report_pdf(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
):
    from fastapi.responses import FileResponse
    from app.services.pdf_service import generate_assessment_pdf

    assessment = _owned_assessment(db, assessment_id, current_user.user_id)
    pdf_res = generate_assessment_pdf(assessment.assessment_id, db)
    if pdf_res.get("status") != "ok" or not pdf_res.get("path"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=pdf_res.get("message") or "Failed to generate PDF report",
        )

    return FileResponse(
        path=pdf_res["path"],
        media_type="application/pdf",
        filename=pdf_res.get("filename", f"MediSense_Report_{assessment.assessment_id}.pdf"),
    )


def _owned_assessment(db: DbSession, assessment_id: UUID, user_id: UUID) -> Assessment:
    assessment = db.get(Assessment, assessment_id)
    if assessment is None or assessment.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found",
        )
    return assessment
