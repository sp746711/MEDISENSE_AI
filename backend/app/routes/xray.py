"""X-ray upload routes — model inference in Stage 7."""

import uuid
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.ai.xray_model import XRayModelService
from app.core.config import get_settings
from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment, XrayResult
from app.utils.file_validation import validate_image_file

router = APIRouter(prefix="/assessments", tags=["X-Ray"])


@router.post("/{assessment_id}/xray")
async def upload_xray(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
    file: UploadFile = File(...),
) -> dict:
    assessment = _owned(db, assessment_id, current_user.user_id)
    if "xray" not in (assessment.input_types or []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This assessment was not configured for X-ray input.",
        )

    settings = get_settings()
    content = await file.read()
    validation = validate_image_file(file.filename or "", content, settings.max_upload_size_mb)
    if not validation["ok"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=validation["message"])

    xray_dir = settings.upload_path / "xrays"
    xray_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}{validation['extension']}"
    dest = xray_dir / stored_name
    dest.write_bytes(content)

    # Adapter: never invent findings if model unavailable
    model_service = XRayModelService()
    analysis = model_service.analyze(str(dest))

    result = XrayResult(
        assessment_id=assessment.assessment_id,
        region=analysis.get("region"),
        model_version=analysis.get("model_version"),
        prediction=analysis.get("prediction"),
        confidence_if_valid=analysis.get("confidence_if_valid"),
        uncertainty=analysis.get("uncertainty"),
        explainability_artifact=analysis.get("explainability_artifact"),
        status=analysis.get("status", "unavailable"),
        message=analysis.get("message"),
        stored_name=stored_name,
    )
    db.add(result)
    if assessment.status == "draft":
        assessment.status = "inputs_received"
    db.add(assessment)
    db.commit()
    db.refresh(result)

    return {
        "status": result.status,
        "xray_result_id": str(result.xray_result_id),
        "message": result.message,
        "region": result.region,
        "model_version": result.model_version,
        "prediction": result.prediction,
        "uncertainty": result.uncertainty,
        "source": "trained_model" if result.prediction else "unavailable",
    }


@router.get("/{assessment_id}/xray")
def get_xray(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    _owned(db, assessment_id, current_user.user_id)
    rows = db.query(XrayResult).filter(XrayResult.assessment_id == assessment_id).all()
    return {
        "assessment_id": str(assessment_id),
        "results": [
            {
                "xray_result_id": str(r.xray_result_id),
                "region": r.region,
                "model_version": r.model_version,
                "prediction": r.prediction,
                "confidence_if_valid": r.confidence_if_valid,
                "uncertainty": r.uncertainty,
                "status": r.status,
                "message": r.message,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
    }


def _owned(db: DbSession, assessment_id: UUID, user_id: UUID) -> Assessment:
    assessment = db.get(Assessment, assessment_id)
    if assessment is None or assessment.user_id != user_id:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment
