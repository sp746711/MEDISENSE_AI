"""Assessment routes for creating, processing, and retrieving assessments."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment
from app.schemas.assessment import AssessmentCreateRequest, AssessmentResponse

router = APIRouter(prefix="/assessments", tags=["Assessments"])

# Keep these values aligned with AssessmentCreateRequest and the frontend.
# Lab/CBC and radiology-report PDFs are represented by medical_report;
# image-based X-ray inference is represented by xray.
ALLOWED_INPUT_TYPES = {"symptoms", "medical_report", "xray"}
RESULT_READY_STATUSES = {
    "completed",
    "insufficient_evidence",
    "conflicting_evidence",
}


def _normalize_input_types(input_types: list[str] | None) -> list[str]:
    """Validate, normalize, and deduplicate input types without changing order."""
    if not input_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "input_types must be a non-empty subset of: "
                "symptoms, medical_report, xray"
            ),
        )

    normalized: list[str] = []
    seen: set[str] = set()

    for item in input_types:
        if not isinstance(item, str):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Each input_type must be a string.",
            )

        input_type = item.strip().lower()
        if not input_type or input_type not in ALLOWED_INPUT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "input_types must be a non-empty subset of: "
                    "symptoms, medical_report, xray"
                ),
            )

        if input_type not in seen:
            seen.add(input_type)
            normalized.append(input_type)

    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one valid input_type is required.",
        )

    return normalized


@router.post("", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
def create_assessment(
    payload: AssessmentCreateRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> AssessmentResponse:
    unique_types = _normalize_input_types(payload.input_types)

    settings = get_settings()
    assessment = Assessment(
        user_id=current_user.user_id,
        input_types=unique_types,
        status="draft",
        rules_version=settings.rules_version,
        model_versions={},
    )

    try:
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
    except Exception:
        db.rollback()
        # Do not expose database exception details to the client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create assessment. Please try again.",
        ) from None

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
    return [AssessmentResponse.model_validate(row) for row in rows]


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

    if assessment.status not in RESULT_READY_STATUSES:
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
        "created_at": (
            assessment.created_at.isoformat()
            if assessment.created_at is not None
            else None
        ),
        "rules_version": assessment.rules_version,
        "model_versions": assessment.model_versions or {},
    }


@router.post("/{assessment_id}/process")
def process_assessment(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Execute the complete multimodal assessment pipeline."""
    from app.services.assessment_service import run_assessment_pipeline

    # Verify ownership before invoking any processing logic.
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
        filename=pdf_res.get(
            "filename",
            f"MediSense_Report_{assessment.assessment_id}.pdf",
        ),
    )


def _owned_assessment(
    db: DbSession,
    assessment_id: UUID,
    user_id: UUID,
) -> Assessment:
    """Return an assessment only when it belongs to the authenticated user."""
    assessment = db.get(Assessment, assessment_id)
    if assessment is None or assessment.user_id != user_id:
        # Avoid revealing whether another user's assessment exists.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found",
        )
    return assessment
