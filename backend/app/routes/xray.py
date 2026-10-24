"""X-ray upload routes — model inference in Stage 7.

Task 1 safeguards:
- enforce the upload size limit while streaming, before buffering an unlimited body;
- do not pass an unrecognized user-supplied region directly to model inference;
- clean up the stored image if routing, inference, or persistence fails;
- keep model-unavailable outcomes explicit rather than inventing findings;
- preserve assessment ownership checks and source/status metadata.
"""

import uuid
import logging
from pathlib import Path
from typing import Any, Optional
import math
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError

from app.ai.xray_model import get_xray_model_service
from app.ai.xray_router import route_xray
from app.core.config import get_settings
from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment, XrayResult
from app.utils.file_validation import validate_image_file

router = APIRouter(prefix="/assessments", tags=["X-Ray"])
logger = logging.getLogger(__name__)

_READ_CHUNK_SIZE = 1024 * 1024


@router.get("/xray/catalogue")
def get_xray_catalogue() -> dict:
    """Return supported body regions and their current model availability."""
    from app.ai.xray_catalogue import (
        get_backend_body_regions,
        get_xray_catalogue_with_model_status,
    )

    regions = get_backend_body_regions()
    catalogue = get_xray_catalogue_with_model_status()
    return {
        "regions": regions,
        "total_regions": len(regions),
        "catalogue": catalogue,
    }


@router.post("/{assessment_id}/xray")
async def upload_xray(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
    file: UploadFile = File(...),
    region: Optional[str] = Form(None),
) -> dict:
    """Validate, store, route and analyse one X-ray image for an owned assessment."""
    assessment = _owned(db, assessment_id, current_user.user_id)
    if "xray" not in (assessment.input_types or []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This assessment was not configured for X-ray input.",
        )

    settings = get_settings()
    max_bytes = int(settings.max_upload_size_mb * 1024 * 1024)
    try:
        content = await _read_limited_upload(file, max_bytes)
    finally:
        await file.close()

    original_name = Path(file.filename or "xray_image").name
    validation = validate_image_file(
        original_name,
        content,
        settings.max_upload_size_mb,
    )
    if not validation.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation.get("message") or "The uploaded image is invalid.",
        )

    extension = validation.get("extension")
    if not extension or not str(extension).startswith("."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image has an unsupported file extension.",
        )

    xray_dir = settings.upload_path / "xrays"
    try:
        xray_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The X-ray upload directory is unavailable.",
        ) from exc

    stored_name = f"{uuid.uuid4().hex}{extension}"
    dest = xray_dir / stored_name
    clean_file_name = validation.get("filename") or original_name

    try:
        dest.write_bytes(content)

        # Route first so unrecognized/unsupported regions are never interpreted
        # as if they were supported. The registry's availability diagnostic is
        # also retained instead of being hidden by a generic model message.
        routing = route_xray(str(dest), declared_region=region)
        resolved_region = routing.get("region")
        if not isinstance(resolved_region, str) or not resolved_region.strip():
            resolved_region = "unknown"
        else:
            resolved_region = resolved_region.strip()

        route_supported = bool(
            routing.get("is_supported", routing.get("supported", False))
        )
        route_available = bool(routing.get("is_available", False))

        if not route_supported or not route_available:
            # Registry says this region/checkpoint is not usable. Do not run
            # inference anyway; persist an explicit unavailable result and its
            # reason so the feedback page can explain the real limitation.
            analysis = {
                "status": "unavailable",
                "reason_code": routing.get("reason_code")
                or ("unsupported_region" if not route_supported else "checkpoint_unavailable"),
                "region": resolved_region,
                "model_version": None,
                "prediction": None,
                "confidence_if_valid": None,
                "uncertainty": routing.get("diagnostic_message")
                or routing.get("message")
                or "The selected X-ray model is not available.",
                "explainability_artifact": None,
                "message": routing.get("message")
                or "Automated X-ray interpretation is unavailable for this region.",
                "diagnostic_message": routing.get("diagnostic_message"),
            }
        else:
            model_service = get_xray_model_service()
            analysis = model_service.analyze(str(dest), region=resolved_region)

            # DenseNet121 is multi-label: it returns a list of finding scores
            # and intentionally leaves `prediction=None`. Legacy ResNet models
            # may still return a single prediction. Validate either output form.
            if not isinstance(analysis, dict):
                analysis = {
                    "status": "unavailable",
                    "message": "The X-ray model returned an invalid response.",
                    "reason_code": "invalid_model_response",
                }

            findings = _validated_findings(analysis.get("findings"))
            raw_prediction = analysis.get("prediction")
            # A prediction must be a non-empty label string. Do not treat arbitrary
            # objects, numbers, or booleans as valid model output.
            prediction_is_valid = isinstance(raw_prediction, str) and bool(raw_prediction.strip())
            if prediction_is_valid:
                raw_prediction = raw_prediction.strip()

            # Prevent a model response for a different body region from being
            # stored as a valid result for the region selected/routed by the user.
            model_region = analysis.get("region")
            region_matches = (
                not isinstance(model_region, str)
                or not model_region.strip()
                or model_region.strip().casefold() == resolved_region.casefold()
            )
            has_multilabel_findings = bool(findings)
            valid_output = prediction_is_valid or has_multilabel_findings
            if (
                analysis.get("status") != "completed"
                or not valid_output
                or not region_matches
            ):
                analysis["status"] = "unavailable"
                analysis["prediction"] = None
                analysis["confidence_if_valid"] = None
                analysis["findings"] = []
                if not region_matches:
                    analysis["reason_code"] = "model_region_mismatch"
                    analysis["message"] = (
                        "The model response did not match the selected X-ray body region. "
                        "No prediction was saved."
                    )
            else:
                # Use the normalized/validated representation from this point on.
                analysis["prediction"] = raw_prediction if prediction_is_valid else None
                analysis["findings"] = findings

            # Confidence is optional metadata. Reject malformed or non-finite values
            # rather than returning misleading confidence information.
            raw_confidence = analysis.get("confidence_if_valid")
            try:
                confidence_num = float(raw_confidence) if raw_confidence is not None else None
            except (TypeError, ValueError, OverflowError):
                confidence_num = None
            if confidence_num is not None and (
                not math.isfinite(confidence_num) or confidence_num < 0.0 or confidence_num > 1.0
            ):
                confidence_num = None
            analysis["confidence_if_valid"] = confidence_num

            # If the model service returns a generic unavailable message, keep
            # the registry's specific checkpoint/routing diagnosis visible.
            if analysis.get("status") == "unavailable":
                analysis.setdefault("reason_code", routing.get("reason_code"))
                analysis.setdefault(
                    "diagnostic_message",
                    routing.get("diagnostic_message") or routing.get("message"),
                )
                if routing.get("reason_code") and routing.get("message"):
                    analysis["message"] = (
                        f"{analysis.get('message') or 'Automated X-ray interpretation is unavailable.'} "
                        f"Routing diagnostic: {routing['message']}"
                    )

        analysis_status = analysis.get("status") or "unavailable"
        prediction = analysis.get("prediction")
        if not isinstance(prediction, str) or not prediction.strip():
            prediction = None
        else:
            prediction = prediction.strip()
        findings = _validated_findings(analysis.get("findings"))
        successful_inference = analysis_status == "completed" and (
            prediction is not None or bool(findings)
        )
        if not successful_inference:
            prediction = None
            findings = []
            confidence = None
            source = "unavailable"
            analysis_status = "unavailable"
        else:
            raw_confidence = analysis.get("confidence_if_valid")
            try:
                confidence = float(raw_confidence) if raw_confidence is not None else None
            except (TypeError, ValueError, OverflowError):
                confidence = None
            if confidence is not None and (
                not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0
            ):
                confidence = None
            source = "trained_model"

        # Do not silently discard DenseNet121's output. The DB model must expose
        # a JSON `findings` (or `findings_json`) column for durable storage.
        findings_column = None
        if hasattr(XrayResult, "findings"):
            findings_column = "findings"
        elif hasattr(XrayResult, "findings_json"):
            findings_column = "findings_json"
        if successful_inference and findings and findings_column is None:
            # Do not mark the result completed if the current schema cannot save
            # the actual multi-label output used by downstream assessment code.
            prediction = None
            findings = []
            confidence = None
            source = "unavailable"
            analysis_status = "unavailable"
            analysis["message"] = (
                "The X-ray model produced findings, but the database schema does "
                "not yet support saving multi-label findings. Update XrayResult "
                "with a JSON findings column and apply the database migration."
            )

        result = XrayResult(
            assessment_id=assessment.assessment_id,
            file_name=clean_file_name,
            region=analysis.get("region") or resolved_region,
            model_version=analysis.get("model_version"),
            prediction=prediction,
            confidence_if_valid=confidence,
            uncertainty=analysis.get("uncertainty"),
            explainability_artifact=analysis.get("explainability_artifact"),
            status=analysis_status,
            message=analysis.get("message") or routing.get("message"),
            stored_name=stored_name,
            **({findings_column: findings} if findings_column and findings else {}),
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
            "file_name": result.file_name,
            "message": result.message,
            "region": result.region,
            "model_version": result.model_version,
            "prediction": result.prediction,
            "findings": getattr(result, findings_column, []) if findings_column else [],
            "confidence_if_valid": result.confidence_if_valid,
            "uncertainty": result.uncertainty,
            "source": source,
            "reason_code": analysis.get("reason_code"),
            "diagnostic_message": analysis.get("diagnostic_message"),
        }
    except HTTPException:
        _remove_file(dest)
        raise
    except SQLAlchemyError as exc:
        # Keep detailed exception information in the backend log only. Do not
        # expose database internals to the browser response.
        logger.exception(
            "X-ray database save failed for assessment_id=%s",
            assessment_id,
        )
        db.rollback()
        _remove_file(dest)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The X-ray result could not be saved. Please retry the upload.",
        ) from exc
    except Exception as exc:
        # Log the original processing exception so it can be diagnosed without
        # exposing filesystem/model details to the client.
        logger.exception(
            "X-ray processing failed for assessment_id=%s",
            assessment_id,
        )
        db.rollback()
        _remove_file(dest)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="X-ray processing failed. No result was saved; please retry.",
        ) from exc
    finally:
        await file.close()


@router.get("/{assessment_id}/xray")
def get_xray(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """List saved X-ray results for an assessment owned by the current user."""
    _owned(db, assessment_id, current_user.user_id)
    rows = (
        db.query(XrayResult)
        .filter(XrayResult.assessment_id == assessment_id)
        .order_by(XrayResult.created_at.asc())
        .all()
    )
    return {
        "assessment_id": str(assessment_id),
        "results": [
            {
                "xray_result_id": str(row.xray_result_id),
                "file_name": row.file_name or "xray_image",
                "region": row.region,
                "model_version": row.model_version,
                "prediction": row.prediction,
                "findings": getattr(row, "findings", getattr(row, "findings_json", [])) or [],
                "confidence_if_valid": row.confidence_if_valid,
                "uncertainty": row.uncertainty,
                "status": row.status,
                "message": row.message,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
    }


def _validated_findings(value: Any) -> list[dict]:
    """Validate multi-label model output while preserving safety metadata.

    TorchXRayVision/DenseNet services may emit ``model_score`` or
    ``unthresholded_model_score`` instead of ``score``. Preserve those raw
    scores for display/audit, but do not promote them to actionable clinical
    findings. A score is not a diagnosis or a calibrated probability.
    """
    if not isinstance(value, list):
        return []

    clean: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        label = item.get("label") or item.get("name") or item.get("finding")
        if not isinstance(label, str) or not label.strip():
            continue

        # Keep the source score field so different model versions remain
        # compatible. The normalized `score` is retained for older consumers.
        raw_score = item.get("score")
        if raw_score is None:
            raw_score = item.get("probability")
        if raw_score is None:
            raw_score = item.get("model_score")
        if raw_score is None:
            raw_score = item.get("unthresholded_model_score")

        try:
            score_num = float(raw_score)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(score_num):
            continue

        record: dict[str, Any] = {
            "label": label.strip(),
            "score": score_num,
        }

        # Preserve model output and provenance metadata needed by downstream
        # evidence/triage logic. Never infer actionability from a numeric score.
        metadata_keys = (
            "model_score",
            "unthresholded_model_score",
            "probability",
            "is_actionable_finding",
            "detected",
            "is_positive",
            "status",
            "source",
            "threshold",
            "threshold_applied",
            "model_version",
            "region",
            "notes",
        )
        for key in metadata_keys:
            if key in item:
                record[key] = item[key]

        # Raw DenseNet label scores must stay non-actionable unless the model
        # explicitly supplies an actionability signal. A score alone is never
        # sufficient evidence for triage.
        if (
            "is_actionable_finding" not in record
            and "detected" not in record
            and "is_positive" not in record
            and "status" not in record
        ):
            record["is_actionable_finding"] = False

        clean.append(record)

    return clean


async def _read_limited_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in bounded chunks and reject it as soon as it exceeds limit."""
    if max_bytes <= 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The configured upload size limit is invalid.",
        )

    chunks = []
    total = 0
    while True:
        chunk = await file.read(min(_READ_CHUNK_SIZE, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds the {max_bytes // (1024 * 1024)} MB upload limit.",
            )
        chunks.append(chunk)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )
    return b"".join(chunks)


def _remove_file(path: Path) -> None:
    """Best-effort cleanup of an upload that did not produce a saved result."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Do not hide the primary routing, inference or database failure.
        pass


def _owned(db: DbSession, assessment_id: UUID, user_id: UUID) -> Assessment:
    assessment = db.get(Assessment, assessment_id)
    if assessment is None or assessment.user_id != user_id:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment
