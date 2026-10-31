"""X-ray upload routes — model inference in Stage 7.

Task 1 safeguards:
- enforce the upload size limit while streaming, before buffering an unlimited body;
- do not pass an unrecognized user-supplied region directly to model inference;
- clean up the stored image if routing, inference, or persistence fails;
- keep model-unavailable outcomes explicit rather than inventing findings;
- preserve assessment ownership checks and source/status metadata;
- persist anatomy-classifier metadata separately from disease-model output;
- do not combine radiology-report and image evidence without explicit association and region compatibility.
"""

import hashlib
import uuid
import logging
from pathlib import Path
from typing import Any, Literal
import math
from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError

from app.ai.xray_model import get_xray_model_service
from app.ai.xray_router import normalize_region_name, route_xray
from app.core.config import get_settings
from app.core.dependencies import CurrentUser, DbSession
from app.database.models import (
    Assessment,
    MedicalReport,
    XrayReportAssociation,
    XrayResult,
)
from app.utils.file_validation import validate_image_file

router = APIRouter(prefix="/assessments", tags=["X-Ray"])
logger = logging.getLogger(__name__)

_READ_CHUNK_SIZE = 1024 * 1024


class XrayReportAssociationRequest(BaseModel):
    """Body for an explicit user decision about an image/report pair."""

    status: Literal["confirmed", "rejected"]


class XrayAnatomyConfirmationRequest(BaseModel):
    """Specific body region selected/confirmed by the user after image upload.

    This confirms the anatomical routing choice only. It does not verify a
    clinical finding or replace radiologist review.
    """

    region: str


_ALL_33_ANATOMY_CONFIRMATION_LABELS = {
    "ABDOMEN", "ANKLE", "ARM", "CERVICAL_SPINE", "CHEST", "DORSAL_SPINE",
    "ELBOW", "EXTREMITY", "FEMUR", "FINGER", "FOOT", "FOREARM", "HAND",
    "HEAD", "HEEL", "HIP", "KNEE", "KUB", "LEG", "LOWER_EXTREMITY",
    "LUMBAR_SPINE", "NASOPHARYNX", "NECK", "PELVIS", "PNS", "SHOULDER",
    "SI_JOINT", "SKULL", "SPINE", "TEMPORAL_BONE", "TIBIA",
    "UPPER_EXTREMITY", "WRIST",
}
_SPECIFIC_ANATOMY_CONFIRMATION_LABELS = {
    "ABDOMEN", "ANKLE", "CERVICAL_SPINE", "CHEST", "DORSAL_SPINE", "ELBOW",
    "FEMUR", "FINGER", "FOOT", "FOREARM", "HAND", "HEEL", "HIP", "KNEE",
    "KUB", "LUMBAR_SPINE", "NASOPHARYNX", "NECK", "PELVIS", "PNS", "SHOULDER",
    "SI_JOINT", "SKULL", "TEMPORAL_BONE", "TIBIA", "WRIST",
}
_BROAD_ANATOMY_CONFIRMATION_LABELS = {
    "EXTREMITY", "UPPER_EXTREMITY", "LOWER_EXTREMITY", "ARM", "LEG", "HEAD", "SPINE",
}


def _canonical_user_confirmed_region(raw_region: str) -> str:
    """Validate a user-confirmed anatomy label against the 33-class catalogue.

    Accepts any of the 33 required anatomical classes. If a broad or unsupported
    region is confirmed, routing will safely record the confirmed anatomy and
    disclose that automated abnormality interpretation is unsupported.
    """
    token = str(raw_region or "").strip().upper().replace("-", "_")
    token = "_".join(token.split())
    if token not in _ALL_33_ANATOMY_CONFIRMATION_LABELS:
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Choose a valid region from the 33 supported anatomical classes.",
        )
    canonical, recognized = normalize_region_name(token)
    if not recognized or not canonical or canonical == "unknown":
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="The selected anatomical region could not be normalized safely.",
        )
    return canonical


def _safe_json_value(value: Any, *, depth: int = 0) -> Any:
    """Convert classifier metadata to bounded, JSON-safe primitive values."""
    if depth > 8:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in list(value.items())[:100]:
            if isinstance(key, (str, int, float, bool)):
                safe[str(key)[:100]] = _safe_json_value(item, depth=depth + 1)
        return safe
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item, depth=depth + 1) for item in value[:100]]
    # Support numpy scalar values without making numpy a route dependency.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item = item_method()
        except Exception:
            return None
        if item is not value:
            return _safe_json_value(item, depth=depth + 1)
    return str(value)[:500]


def _build_anatomy_metadata(
    *,
    routing: dict[str, Any],
    resolved_region: str,
    region_source: str,
    anatomy_status: str,
    body_part_confidence: float | None,
) -> dict[str, Any]:
    """Build durable anatomy-classifier metadata, separate from disease findings."""
    detection = routing.get("anatomy_detection")
    if not isinstance(detection, dict):
        detection = {
            "candidate_body_part": routing.get("classifier_candidate"),
            "status": routing.get("classifier_status"),
            "confidence": routing.get("classifier_confidence"),
            "top_predictions": routing.get("classifier_top_predictions")
            or routing.get("top_predictions"),
            "classifier_version": routing.get("classifier_version"),
            "model_version": routing.get("classifier_model_version"),
            "model_name": routing.get("classifier_model_name"),
            "architecture": routing.get("classifier_architecture"),
            "confidence_threshold_validated": routing.get(
                "confidence_threshold_validated", False
            ),
        }
        if not any(value is not None for value in detection.values()):
            detection = {}
    candidate_region = (
        routing.get("candidate_region")
        or routing.get("classifier_candidate")
    )
    if not isinstance(candidate_region, str) or not candidate_region.strip():
        candidate_region = detection.get("candidate_body_part") or detection.get("candidate_region")
    if isinstance(candidate_region, str):
        candidate_region = candidate_region.strip().lower() or None
    else:
        candidate_region = None

    classifier_version = (
        detection.get("classifier_version")
        or detection.get("model_version")
        or detection.get("model_name")
        or detection.get("architecture")
        or routing.get("classifier_version")
    )
    top_candidates = (
        detection.get("top_candidates")
        or detection.get("candidates")
        or detection.get("top_predictions")
    )

    return {
        "status": str(anatomy_status or "unverified").upper(),
        "classification_status": str(
            routing.get("classification_status")
            or routing.get("classifier_status")
            or (detection.get("status") if isinstance(detection, dict) else None)
            or "unknown"
        ).lower(),
        "anatomy_classifier_accepted": routing.get("anatomy_classifier_accepted") is True,
        "classifier_gate_passed": bool(
            routing.get("classifier_gate_passed") or routing.get("anatomy_classifier_accepted")
        ),
        "broad_region_classifier_accepted": routing.get("broad_region_classifier_accepted") is True,
        "is_broad_region": routing.get("is_broad_region") is True,
        "specific_region_confirmation_required": routing.get("specific_region_confirmation_required") is True,
        "regional_model_configured": bool(routing.get("regional_model_configured", False)),
        "regional_model_available": bool(routing.get("regional_model_available", False)),
        "regional_model_name": routing.get("regional_model_name"),
        "regional_model_status": routing.get("regional_model_status"),
        "anatomy_user_confirmed": bool(
            region_source == "user_confirmed"
            or routing.get("anatomy_user_confirmed") is True
            or routing.get("user_confirmation_status") == "confirmed"
        ),
        "anatomy_verification_method": _safe_json_value(routing.get("anatomy_verification_method")),
        "source": str(region_source or "unknown"),
        "detected_region": (
            None if resolved_region == "unknown" or routing.get("is_broad_region") is True
            else resolved_region
        ),
        "candidate_region": candidate_region,
        "body_part_confidence": body_part_confidence,
        "confidence_threshold_validated": bool(
            detection.get("confidence_threshold_validated", False)
        ),
        "preprocessing_id": _safe_json_value(detection.get("preprocessing_id") or detection.get("preprocessing_mode")),
        "preprocessing_mode": _safe_json_value(detection.get("preprocessing_mode") or detection.get("preprocessing_id")),
        "preprocessing_verified_against_training": bool(
            detection.get("preprocessing_verified_against_training", False)
        ),
        "model_artifact_sha256": _safe_json_value(detection.get("model_sha256")),
        "labels_sha256": _safe_json_value(detection.get("labels_sha256")),
        "model_size_bytes": _safe_json_value(detection.get("model_size_bytes")),
        "model_input_shape": _safe_json_value(detection.get("model_input_shape")),
        "model_output_shape": _safe_json_value(detection.get("model_output_shape")),
        "output_score_semantics": _safe_json_value(detection.get("output_score_semantics")),
        "classifier_version": _safe_json_value(classifier_version),
        # Keep the source classifier payload so later diagnostics can inspect the
        # actual candidate labels and scores rather than reconstructing them.
        "classifier_output": _safe_json_value(detection),
        "two_model_comparison": _safe_json_value(
            routing.get("two_model_comparison")
            or detection.get("two_model_comparison")
        ),
        "final_candidate": _safe_json_value(
            routing.get("final_candidate")
            or detection.get("final_candidate")
            or candidate_region
        ),
        "selection_source": _safe_json_value(
            routing.get("selection_source")
            or detection.get("selection_source")
            or "PRIMARY MODEL"
        ),
    }


def _extract_report_region(report: MedicalReport) -> tuple[str | None, bool]:
    """Return a canonical report region and whether its label is recognized."""
    metadata = report.ocr_metadata if isinstance(report.ocr_metadata, dict) else {}
    report_region = metadata.get("report_region")
    if not isinstance(report_region, str) or not report_region.strip():
        verification = metadata.get("region_verification")
        report_region = verification.get("report_region") if isinstance(verification, dict) else None
    if not isinstance(report_region, str) or not report_region.strip():
        return None, False
    canonical_region, recognized = normalize_region_name(report_region)
    return (canonical_region if recognized else report_region.strip().lower()), bool(recognized)


def _build_association_safety(
    *, status_value: str, image_region: str | None, report_region: str | None, region_recognized: bool
) -> dict[str, Any]:
    """Pairing can support evidence combination only after explicit confirmation and region match."""
    image = str(image_region or "unknown").strip().lower()
    report = str(report_region or "unknown").strip().lower()
    if image == "unknown" or report == "unknown" or not region_recognized:
        region_check = "UNVERIFIED"
        compatible = False
    elif image == report:
        region_check = "MATCH"
        compatible = True
    else:
        region_check = "MISMATCH"
        compatible = False

    confirmed = status_value == "confirmed"
    can_combine = confirmed and compatible
    if region_check == "MISMATCH":
        message = "The selected image and report have different detected regions. Keep their evidence separate even though the pairing decision was recorded."
    elif region_check == "UNVERIFIED":
        message = "The image/report region match could not be verified. Keep their evidence separate."
    elif confirmed:
        message = "The image/report pair was explicitly confirmed and the detected regions match. This does not validate the clinical findings themselves."
    elif status_value == "rejected":
        message = "The image/report pair was rejected and must remain separate."
    else:
        message = "The image/report pair is not explicitly confirmed and must remain separate."

    return {
        "status": status_value.upper(),
        "association_explicitly_confirmed": confirmed,
        "region_check": region_check,
        "region_compatible": compatible,
        "can_combine": can_combine,
        "message": message,
    }


def _build_report_region_verification(
    *,
    resolved_region: str,
    report_regions: list[str],
    unrecognized_report_region_count: int,
    report_count: int,
    existing_image_count: int,
) -> dict[str, Any]:
    """Compare regions only when one report can be compared with one image.

    Reports and images are stored per assessment, not explicitly linked to each
    other in the current schema. Comparing every report against every image creates
    false mismatches for assessments containing multiple images or reports. When
    the association cannot be uniquely inferred, preserve it as UNVERIFIED and do
    not authorize evidence fusion.
    """
    image_region = str(resolved_region or "unknown").strip().lower() or "unknown"
    normalized_report_regions = [
        str(region).strip().lower()
        for region in report_regions
        if isinstance(region, str) and region.strip()
    ]
    unique_pair_can_be_compared = (
        image_region != "unknown"
        and report_count == 1
        and existing_image_count == 0
        and len(normalized_report_regions) == 1
        and unrecognized_report_region_count == 0
    )

    mismatch = False
    match: bool | None = None
    if unique_pair_can_be_compared:
        mismatch = normalized_report_regions[0] != image_region
        match = not mismatch

    if mismatch:
        status_value = "MISMATCH"
        message = (
            "The radiology report region conflicts with the automatically detected image type. "
            "Keep the image and report findings separate until the mismatch is reviewed."
        )
    elif match is True:
        status_value = "MATCH"
        message = (
            "The sole radiology report and image have compatible detected body regions. "
            "This is a region-consistency check, not confirmation of clinical findings."
        )
    else:
        status_value = "UNVERIFIED"
        message = (
            "The image/report relationship could not be uniquely verified. Keep report findings "
            "separate from image-model predictions until the correct pair is confirmed."
        )

    return {
        "status": status_value,
        "detected_image_region": None if image_region == "unknown" else image_region,
        "report_regions": normalized_report_regions,
        "unrecognized_report_region_count": max(0, int(unrecognized_report_region_count)),
        "association_scope": (
            "SINGLE_IMAGE_SINGLE_REPORT" if unique_pair_can_be_compared else "AMBIGUOUS_OR_INCOMPLETE"
        ),
        "association_explicitly_confirmed": False,
        "region_compatible": match is True,
        # The current schema has no explicit user-confirmed report/image link.
        # Matching region labels are necessary but not sufficient to fuse evidence.
        "can_combine": False,
        "message": message,
    }


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
) -> dict:
    """Automatically identify, route and analyse an X-ray image for an owned assessment.

    The body region is derived from image pixels; no user-selected region is required.
    """
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
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "X-ray upload stream read failed for assessment_id=%s",
            assessment_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded X-ray image could not be read. Please retry the upload.",
        ) from exc
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

        # Route strictly from image pixels; never use a user-selected region.
        routing = route_xray(str(dest))
        resolved_region = routing.get("region")
        if not isinstance(resolved_region, str) or not resolved_region.strip():
            resolved_region = "unknown"
        else:
            resolved_region = resolved_region.strip().lower()

        region_source = routing.get("region_source") or "unknown"
        anatomy_status = str(
            routing.get("anatomy_verification_status") or "unverified"
        ).strip().lower()

        # A classifier output is a candidate, not user confirmation. Persist a
        # canonical region only after explicit confirmation (or a separately
        # established independent verification). Keep the model's candidate and
        # scores in anatomy_metadata so the UI can ask the user to confirm it.
        # This also prevents an unconfirmed candidate from being compared with an
        # X-ray report as though the image region were already established.
        anatomy_independently_verified_for_storage = (
            routing.get("anatomy_independently_verified") is True
        )
        anatomy_user_confirmed_for_storage = (
            str(region_source).strip().lower() == "user_confirmed_region"
            or routing.get("anatomy_user_confirmed") is True
        )
        confirmation_pending = (
            routing.get("specific_region_confirmation_required") is True
            or anatomy_status in {"confirmation_required", "unverified"}
        )
        if (
            routing.get("is_broad_region") is True
            or (
                str(region_source).strip().lower() == "image_classifier"
                and confirmation_pending
                and not anatomy_independently_verified_for_storage
                and not anatomy_user_confirmed_for_storage
            )
        ):
            resolved_region = "unknown"
        anatomy_detection = routing.get("anatomy_detection")
        if not isinstance(anatomy_detection, dict):
            # Older/current router responses may expose classifier values as
            # flat fields rather than an `anatomy_detection` object. Normalize
            # them here so API responses and durable metadata do not lose the
            # actual classifier candidate/status/confidence.
            anatomy_detection = {
                "candidate_body_part": routing.get("classifier_candidate")
                or routing.get("candidate_region"),
                "status": routing.get("classifier_status")
                or routing.get("anatomy_verification_status"),
                "confidence": routing.get("classifier_confidence")
                if routing.get("classifier_confidence") is not None
                else routing.get("body_part_confidence"),
                "top_predictions": routing.get("classifier_top_predictions")
                or routing.get("top_predictions"),
                "classifier_version": routing.get("classifier_version"),
                "model_version": routing.get("classifier_model_version"),
                "model_name": routing.get("classifier_model_name"),
                "architecture": routing.get("classifier_architecture"),
                "confidence_threshold_validated": routing.get(
                    "confidence_threshold_validated", False
                ),
            }
            # Do not emit an empty synthetic object when the router has no
            # classifier metadata at all.
            if not any(value is not None for value in anatomy_detection.values()):
                anatomy_detection = None
        body_part_confidence = routing.get("body_part_confidence")
        if body_part_confidence is None and isinstance(anatomy_detection, dict):
            body_part_confidence = anatomy_detection.get("confidence")
        if body_part_confidence is not None:
            try:
                body_part_confidence = float(body_part_confidence)
            except (TypeError, ValueError, OverflowError):
                body_part_confidence = None
            if body_part_confidence is not None and (
                not math.isfinite(body_part_confidence)
                or body_part_confidence < 0.0
                or body_part_confidence > 1.0
            ):
                body_part_confidence = None

        # If a report already exists, compare its extracted region against the
        # automatic image classification. A mismatch prevents later combination,
        # but does not discard otherwise valid image-only findings.
        saved_xray_reports = (
            db.query(MedicalReport)
            .filter(
                MedicalReport.assessment_id == assessment.assessment_id,
                MedicalReport.report_category == "xray_report",
            )
            .all()
        )
        # The current schema stores reports and images against an assessment but
        # does not store an explicit report_id <-> xray_result_id association.
        # Therefore, region comparison is only meaningful when this is the first
        # image in the assessment and exactly one report is present.
        existing_xray_count = (
            db.query(XrayResult)
            .filter(XrayResult.assessment_id == assessment.assessment_id)
            .count()
        )
        report_regions: list[str] = []
        unrecognized_report_regions = 0
        for saved_report in saved_xray_reports:
            canonical_report_region, report_region_recognized = _extract_report_region(saved_report)
            if canonical_report_region is None:
                continue
            if report_region_recognized:
                report_regions.append(canonical_report_region)
            else:
                unrecognized_report_regions += 1

        report_region_verification = _build_report_region_verification(
            resolved_region=resolved_region,
            report_regions=report_regions,
            unrecognized_report_region_count=unrecognized_report_regions,
            report_count=len(saved_xray_reports),
            existing_image_count=existing_xray_count,
        )
        report_region_mismatch = report_region_verification["status"] == "MISMATCH"

        route_supported = bool(
            routing.get("is_supported", routing.get("supported", False))
        )
        route_available = bool(routing.get("is_available", False))
        # `anatomy_classifier_accepted` means the classifier passed its confidence
        # and ambiguity gates. A broad label may pass that gate, but the status stays
        # confirmation_required and therefore cannot authorize regional inference.
        anatomy_independently_verified = (
            routing.get("anatomy_independently_verified") is True
        )
        anatomy_classifier_accepted = routing.get("anatomy_classifier_accepted") is True
        anatomy_verified_for_inference = (
            anatomy_status == "completed"
            and (anatomy_independently_verified or anatomy_classifier_accepted)
        )

        if not anatomy_verified_for_inference or not route_supported or not route_available:
            if not anatomy_verified_for_inference:
                if routing.get("broad_region_classifier_accepted") is True:
                    safe_reason_code = "broad_anatomy_label"
                    safe_message = routing.get("message") or (
                        "The classifier accepted a broad body region, but did not identify a specific site. "
                        "Choose a specific body region before regional interpretation. No regional model was run."
                    )
                elif anatomy_status == "confirmation_required":
                    safe_reason_code = "anatomy_confirmation_required"
                    safe_message = (
                        "The image classifier suggested a body region, but a specific body region "
                        "still needs confirmation. No regional disease model was run."
                    )
                elif anatomy_status == "unavailable":
                    safe_reason_code = "anatomy_classifier_unavailable"
                    safe_message = (
                        "The anatomy classifier is unavailable. No regional disease model was run."
                    )
                else:
                    safe_reason_code = "anatomy_unverified"
                    safe_message = (
                        "The image type could not be verified. No regional disease model was run."
                    )
            else:
                safe_reason_code = routing.get("reason_code") or (
                    "unsupported_region" if not route_supported else "checkpoint_unavailable"
                )
                safe_message = routing.get("message") or (
                    "Automated X-ray interpretation is unavailable for this region."
                )

            # Persist an explicit unavailable outcome instead of allowing a
            # recognized-but-unverified anatomy to reach model inference.
            analysis = {
                "status": "unavailable",
                "reason_code": safe_reason_code,
                "region": resolved_region,
                "model_version": None,
                "prediction": None,
                "confidence_if_valid": None,
                "uncertainty": routing.get("diagnostic_message")
                or routing.get("message")
                or safe_message,
                "explainability_artifact": None,
                "message": safe_message,
                "diagnostic_message": routing.get("diagnostic_message") or safe_message,
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

        anatomy_confirmation_required = bool(
            routing.get("specific_region_confirmation_required") is True
            or (
                not anatomy_classifier_accepted
                and not anatomy_independently_verified
                and anatomy_status in {
                    "confirmation_required", "unverified", "unavailable", "failed"
                }
            )
        )

        image_anatomy_verification = {
            "status": anatomy_status.upper(),
            "confirmation_required": anatomy_confirmation_required,
            "next_action": (
                "confirm_or_select_body_region"
                if anatomy_confirmation_required else None
            ),
            "classification_status": str(
                routing.get("classification_status")
                or routing.get("classifier_status")
                or (
                    anatomy_detection.get("status")
                    if isinstance(anatomy_detection, dict)
                    else None
                )
                or "unknown"
            ).lower(),
            "anatomy_independently_verified": anatomy_independently_verified,
            "anatomy_classifier_accepted": anatomy_classifier_accepted,
            "classifier_gate_passed": bool(routing.get("classifier_gate_passed") or anatomy_classifier_accepted),
            "broad_region_classifier_accepted": routing.get("broad_region_classifier_accepted") is True,
            "is_broad_region": routing.get("is_broad_region") is True,
            "specific_region_confirmation_required": anatomy_confirmation_required,
            "regional_model_configured": bool(routing.get("regional_model_configured", False)),
            "regional_model_available": bool(routing.get("regional_model_available", False)),
            "regional_model_name": routing.get("regional_model_name"),
            "regional_model_status": routing.get("regional_model_status"),
            "verification_method": routing.get("anatomy_verification_method"),
            "source": region_source,
            "detected_region": (
                None if resolved_region == "unknown" or routing.get("is_broad_region") is True
                else resolved_region
            ),
            "candidate_region": (
                routing.get("candidate_region")
                or routing.get("classifier_candidate")
                or (
                    anatomy_detection.get("candidate_body_part")
                    if isinstance(anatomy_detection, dict)
                    else None
                )
            ),
            "body_part_confidence": body_part_confidence,
            "confidence_threshold_validated": (
                anatomy_detection.get("confidence_threshold_validated", False)
                if isinstance(anatomy_detection, dict) else False
            ),
            "message": routing.get("message") or (
                "The image type could not be verified automatically. No regional model was selected."
            ),
        }
        base_result_message = analysis.get("message") or routing.get("message")
        verification_message = image_anatomy_verification["message"]
        persisted_parts = [part for part in (base_result_message, verification_message) if part]
        if report_region_mismatch:
            persisted_parts.append(report_region_verification["message"])
        persisted_message = " ".join(dict.fromkeys(persisted_parts))

        image_sha256 = hashlib.sha256(content).hexdigest()
        anatomy_metadata = _build_anatomy_metadata(
            routing=routing,
            resolved_region=resolved_region,
            region_source=str(region_source),
            anatomy_status=anatomy_status,
            body_part_confidence=body_part_confidence,
        )
        anatomy_metadata["image_sha256"] = image_sha256
        result_kwargs = {}
        if hasattr(XrayResult, "anatomy_metadata"):
            result_kwargs["anatomy_metadata"] = anatomy_metadata

        result = XrayResult(
            assessment_id=assessment.assessment_id,
            file_name=clean_file_name,
            # The router is the source of the validated canonical anatomy.
            # Never let a model adapter override it with a different alias/casing.
            region=resolved_region,
            model_version=analysis.get("model_version"),
            prediction=prediction,
            confidence_if_valid=confidence,
            uncertainty=analysis.get("uncertainty"),
            explainability_artifact=analysis.get("explainability_artifact"),
            status=analysis_status,
            message=persisted_message,
            stored_name=stored_name,
            **({findings_column: findings} if findings_column and findings else {}),
            **result_kwargs,
        )
        db.add(result)
        if assessment.status == "draft":
            assessment.status = "inputs_received"
        db.add(assessment)
        db.commit()
        db.refresh(result)

        saved_findings = (
            getattr(result, findings_column, []) if findings_column else []
        ) or []
        
        is_unresolved_mismatch = (
            routing.get("two_model_comparison", {}).get("is_unresolved") is True
            or routing.get("two_model_comparison", {}).get("decision_case") == "CASE_D_UNRESOLVED"
        )
        if result.status == "failed" or routing.get("anatomy_verification_status") == "failed":
            decision_state = "UNABLE_TO_CLASSIFY"
        elif (
            routing.get("classifier_gate_passed")
            and not routing.get("is_broad_region")
            and not is_unresolved_mismatch
            and body_part_confidence is not None
            and body_part_confidence >= 0.60
        ):
            decision_state = "HIGH_CONFIDENCE_CANDIDATE"
        else:
            decision_state = "NEEDS_CONFIRMATION"

        return {
            "status": result.status,
            "decision_state": decision_state,
            "xray_result_id": str(result.xray_result_id),
            "file_name": result.file_name,
            "message": result.message,
            "region": result.region,
            "model_version": result.model_version,
            "prediction": result.prediction,
            "findings": saved_findings,
            "confidence_if_valid": result.confidence_if_valid,
            "uncertainty": result.uncertainty,
            "source": source,
            "reason_code": analysis.get("reason_code"),
            "diagnostic_message": analysis.get("diagnostic_message"),
            "image_anatomy_verification": image_anatomy_verification,
            "image_type_detection": anatomy_detection,
            "anatomy_metadata_persisted": hasattr(XrayResult, "anatomy_metadata"),
            "anatomy_metadata": anatomy_metadata,
            "detected_region": (
                None if resolved_region == "unknown" or routing.get("is_broad_region") is True
                else resolved_region
            ),
            "candidate_region": routing.get("candidate_region") or routing.get("classifier_candidate"),
            "region_source": region_source,
            "is_broad_region": routing.get("is_broad_region") is True,
            "broad_region_classifier_accepted": routing.get("broad_region_classifier_accepted") is True,
            "specific_region_confirmation_required": anatomy_confirmation_required,
            "anatomy_verification_status": anatomy_status,
            "classification_status": str(
                routing.get("classification_status")
                or routing.get("classifier_status")
                or (
                    anatomy_detection.get("status")
                    if isinstance(anatomy_detection, dict)
                    else None
                )
                or "unknown"
            ).lower(),
            "anatomy_independently_verified": anatomy_independently_verified,
            "anatomy_classifier_accepted": anatomy_classifier_accepted,
            "classifier_gate_passed": bool(routing.get("classifier_gate_passed") or anatomy_classifier_accepted),
            "regional_model_configured": bool(routing.get("regional_model_configured", False)),
            "regional_model_available": bool(routing.get("regional_model_available", False)),
            "regional_model_name": routing.get("regional_model_name"),
            "regional_model_status": routing.get("regional_model_status"),
            "anatomy_verification_method": routing.get("anatomy_verification_method"),
            "anatomy_confirmation_required": anatomy_confirmation_required,
            "next_action": (
                "confirm_or_select_body_region"
                if anatomy_confirmation_required else None
            ),
            "body_part_confidence": body_part_confidence,
            "report_region_verification": report_region_verification,
            "primary_prediction": (
                {
                    "model_name": anatomy_detection.get("primary_classifier_result", {}).get("model_name", "convnext_tiny_onnx_33class"),
                    "candidate_body_part": anatomy_detection.get("primary_classifier_result", {}).get("candidate_body_part"),
                    "confidence": anatomy_detection.get("primary_classifier_result", {}).get("confidence"),
                    "body_part_confidence": anatomy_detection.get("primary_classifier_result", {}).get("confidence"),
                    "top_two_margin": anatomy_detection.get("primary_classifier_result", {}).get("top_two_margin"),
                    "top_predictions": anatomy_detection.get("primary_classifier_result", {}).get("top_predictions", []),
                    "gate_passed": anatomy_detection.get("primary_classifier_result", {}).get("gate_passed", False),
                    "is_broad_region": anatomy_detection.get("primary_classifier_result", {}).get("is_broad_region", False),
                }
                if isinstance(anatomy_detection, dict) and anatomy_detection.get("primary_classifier_result")
                else None
            ),
            "specialist_prediction": (
                {
                    "model_name": anatomy_detection.get("upper_extremity_refinement", {}).get("model_name"),
                    "candidate_body_part": anatomy_detection.get("upper_extremity_refinement", {}).get("candidate_body_part"),
                    "confidence": anatomy_detection.get("upper_extremity_refinement", {}).get("confidence"),
                    "body_part_confidence": anatomy_detection.get("upper_extremity_refinement", {}).get("confidence"),
                    "top_two_margin": anatomy_detection.get("upper_extremity_refinement", {}).get("top_two_margin"),
                    "top_predictions": anatomy_detection.get("upper_extremity_refinement", {}).get("top_predictions", []),
                    "eligibility": anatomy_detection.get("upper_extremity_refinement", {}).get("eligibility"),
                    "status": anatomy_detection.get("upper_extremity_refinement", {}).get("status"),
                    "reason_code": anatomy_detection.get("upper_extremity_refinement", {}).get("reason_code"),
                    "message": anatomy_detection.get("upper_extremity_refinement", {}).get("message"),
                }
                if isinstance(anatomy_detection, dict) and anatomy_detection.get("upper_extremity_refinement")
                else None
            ),
            "two_model_comparison": (
                routing.get("two_model_comparison")
                or (anatomy_detection.get("two_model_comparison") if isinstance(anatomy_detection, dict) else None)
            ),
            "final_candidate": (
                routing.get("final_candidate")
                or (anatomy_detection.get("final_candidate") if isinstance(anatomy_detection, dict) else None)
                or routing.get("candidate_region")
            ),
            "selection_source": (
                routing.get("selection_source")
                or (anatomy_detection.get("selection_source") if isinstance(anatomy_detection, dict) else None)
                or "PRIMARY MODEL"
            ),
            "preprocessing_id": (
                anatomy_metadata.get("preprocessing_id")
                or (anatomy_detection.get("preprocessing_id") if isinstance(anatomy_detection, dict) else None)
                or (anatomy_detection.get("preprocessing_mode") if isinstance(anatomy_detection, dict) else None)
            ),
            "preprocessing_mode": (
                anatomy_metadata.get("preprocessing_mode")
                or (anatomy_detection.get("preprocessing_mode") if isinstance(anatomy_detection, dict) else None)
                or (anatomy_detection.get("preprocessing_id") if isinstance(anatomy_detection, dict) else None)
            ),
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


@router.put("/{assessment_id}/xray/{xray_result_id}/anatomy-verification")
def confirm_xray_anatomy(
    assessment_id: UUID,
    xray_result_id: UUID,
    payload: XrayAnatomyConfirmationRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Confirm/correct a candidate region after upload and retry matching inference.

    The file is never renamed, moved, deleted, or re-uploaded. Explicit user
    choice resolves the current confirmation-required dead end, while the API
    keeps anatomy confirmation separate from disease-model availability/output.
    """
    _owned(db, assessment_id, current_user.user_id)
    result = (
        db.query(XrayResult)
        .filter(
            XrayResult.assessment_id == assessment_id,
            XrayResult.xray_result_id == xray_result_id,
        )
        .first()
    )
    if result is None:
        raise HTTPException(status_code=404, detail="X-ray image not found")

    selected_region = _canonical_user_confirmed_region(payload.region)
    stored_name = result.stored_name
    if not isinstance(stored_name, str) or not stored_name or Path(stored_name).name != stored_name:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The saved X-ray filename is unavailable or invalid; the original image was not modified.",
        )

    xray_dir = (get_settings().upload_path / "xrays").resolve()
    stored_path = (xray_dir / stored_name).resolve()
    if stored_path.parent != xray_dir or not stored_path.is_file() or stored_path.stat().st_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The saved X-ray image could not be found. Upload the image again before retrying.",
        )

    # The user-confirmed region is checked against the same registry used by
    # automatic routing. Unsupported/unavailable regions are still persisted as
    # confirmed anatomy, but never receive fabricated pathology output.
    routing = route_xray(str(stored_path), declared_region=selected_region)
    if routing.get("is_recognized") is False:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=routing.get("message") or "The selected region is not recognized.",
        )

    if routing.get("is_supported") and routing.get("is_available"):
        try:
            analysis = get_xray_model_service().analyze(
                str(stored_path), region=selected_region
            )
        except Exception as exc:
            logger.exception(
                "X-ray inference retry failed after anatomy confirmation: assessment_id=%s result_id=%s",
                assessment_id,
                xray_result_id,
            )
            analysis = {
                "status": "unavailable",
                "reason_code": "inference_failed",
                "region": selected_region,
                "message": "The selected region was confirmed, but automated interpretation failed. No finding was saved.",
                "diagnostic_message": f"{type(exc).__name__}: {exc}",
                "uncertainty": "Inference failed after the user confirmed the anatomical region.",
            }
    else:
        analysis = {
            "status": "unavailable",
            "reason_code": routing.get("reason_code") or "unsupported_region",
            "region": selected_region,
            "model_version": None,
            "prediction": None,
            "findings": [],
            "confidence_if_valid": None,
            "uncertainty": routing.get("diagnostic_message") or routing.get("message"),
            "message": routing.get("message") or "The region was confirmed, but no compatible interpretation model is available.",
            "diagnostic_message": routing.get("diagnostic_message"),
        }

    if not isinstance(analysis, dict):
        analysis = {
            "status": "unavailable",
            "reason_code": "invalid_model_response",
            "region": selected_region,
            "message": "The X-ray model returned an invalid response. No finding was saved.",
        }

    findings = _validated_findings(analysis.get("findings"))
    raw_prediction = analysis.get("prediction")
    prediction = raw_prediction.strip() if isinstance(raw_prediction, str) and raw_prediction.strip() else None
    model_region = analysis.get("region")
    model_region_matches = True
    if isinstance(model_region, str) and model_region.strip():
        normalized_model_region, model_region_recognized = normalize_region_name(model_region)
        model_region_matches = model_region_recognized and normalized_model_region == selected_region

    successful_inference = (
        str(analysis.get("status") or "").lower() == "completed"
        and (prediction is not None or bool(findings))
        and model_region_matches
    )
    if not model_region_matches:
        analysis["reason_code"] = "model_region_mismatch"
        analysis["message"] = "The model response did not match the user-confirmed region. No prediction was saved."

    analysis_status = "completed" if successful_inference else "unavailable"
    if not successful_inference:
        prediction = None
        findings = []

    raw_confidence = analysis.get("confidence_if_valid") if successful_inference else None
    try:
        confidence = float(raw_confidence) if raw_confidence is not None else None
    except (TypeError, ValueError, OverflowError):
        confidence = None
    if confidence is not None and (not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0):
        confidence = None

    if findings and not (hasattr(XrayResult, "findings") or hasattr(XrayResult, "findings_json")):
        findings = []
        prediction = None
        confidence = None
        analysis_status = "unavailable"
        analysis["reason_code"] = "findings_persistence_unavailable"
        analysis["message"] = "The model returned multi-label scores, but this database schema cannot persist them. No findings were saved."

    metadata = result.anatomy_metadata if isinstance(getattr(result, "anatomy_metadata", None), dict) else {}
    classifier_output = metadata.get("classifier_output") if isinstance(metadata.get("classifier_output"), dict) else {}
    original_candidate = (
        metadata.get("candidate_region")
        or classifier_output.get("candidate_body_part")
        or classifier_output.get("candidate_region")
    )
    original_candidate_region = None
    if isinstance(original_candidate, str) and original_candidate.strip():
        candidate_canonical, candidate_recognized = normalize_region_name(original_candidate)
        original_candidate_region = candidate_canonical if candidate_recognized else original_candidate.strip().lower()
    region_was_corrected = original_candidate_region is not None and original_candidate_region != selected_region
    selected_region_matches_candidate = (
        original_candidate_region is not None
        and original_candidate_region == selected_region
    )
    score_for_selected_region = (
        metadata.get("body_part_confidence")
        if selected_region_matches_candidate else None
    )

    message_parts = [
        f"You confirmed the X-ray body region as {selected_region.replace('_', ' ').title()}. "
        "This confirms the routing region only; it does not confirm any medical finding."
    ]
    result_message = analysis.get("message") or routing.get("message")
    if result_message:
        message_parts.append(str(result_message))
    if not successful_inference:
        message_parts.append("The image was saved, but automated abnormality interpretation is unavailable; no pathology finding was inferred.")
    persisted_message = " ".join(dict.fromkeys(message_parts))
    if "image_sha256" not in metadata and stored_path.is_file():
        metadata["image_sha256"] = hashlib.sha256(stored_path.read_bytes()).hexdigest()

    metadata.update({
        "status": "CONFIRMED",
        "classification_status": "completed",
        "detected_region": selected_region,
        "user_confirmed_region": selected_region,
        "user_confirmation_status": "confirmed",
        "verification_source": "explicit_user_region_confirmation",
        "source": "user_confirmed_region",
        "anatomy_user_confirmed": True,
        # A user confirmation is not represented as independent clinical verification.
        "anatomy_independently_verified": False,
        "region_was_corrected": bool(region_was_corrected),
        "original_classifier_candidate": original_candidate_region,
        "interpretation_status": analysis_status,
        "interpretation_reason_code": analysis.get("reason_code") or routing.get("reason_code"),
        "interpretation_model_version": analysis.get("model_version") if successful_inference else None,
        "message": persisted_message,
    })

    result.region = selected_region
    result.model_version = analysis.get("model_version") if successful_inference else None
    result.prediction = prediction
    if hasattr(result, "findings"):
        result.findings = findings
    elif hasattr(result, "findings_json"):
        result.findings_json = findings
    result.confidence_if_valid = confidence
    result.uncertainty = analysis.get("uncertainty") or (routing.get("message") if not successful_inference else None)
    result.explainability_artifact = analysis.get("explainability_artifact") if successful_inference else None
    result.status = analysis_status
    result.message = persisted_message
    if hasattr(result, "anatomy_metadata"):
        result.anatomy_metadata = metadata

    try:
        db.add(result)
        db.commit()
        db.refresh(result)
    except SQLAlchemyError as exc:
        logger.exception(
            "X-ray anatomy confirmation persistence failed: assessment_id=%s result_id=%s",
            assessment_id,
            xray_result_id,
        )
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The region choice could not be saved. The original image file remains unchanged.",
        ) from exc

    return {
        "status": result.status,
        "decision_state": "USER_CORRECTED" if region_was_corrected else "USER_CONFIRMED",
        "xray_result_id": str(result.xray_result_id),
        "file_name": result.file_name,
        "message": result.message,
        "region": result.region,
        "detected_region": selected_region,
        "candidate_region": original_candidate_region,
        "primary_prediction": metadata.get("classifier_output", {}).get("primary_classifier_result"),
        "specialist_prediction": metadata.get("classifier_output", {}).get("upper_extremity_refinement"),
        "two_model_comparison": metadata.get("two_model_comparison") or metadata.get("classifier_output", {}).get("two_model_comparison"),
        "final_candidate": metadata.get("final_candidate") or metadata.get("classifier_output", {}).get("final_candidate"),
        "selection_source": metadata.get("selection_source") or metadata.get("classifier_output", {}).get("selection_source"),
        "model_version": result.model_version,
        "prediction": result.prediction,
        "findings": getattr(result, "findings", getattr(result, "findings_json", [])) or [],
        "confidence_if_valid": result.confidence_if_valid,
        "uncertainty": result.uncertainty,
        "source": "trained_model" if successful_inference else "unavailable",
        "reason_code": analysis.get("reason_code") or routing.get("reason_code"),
        "diagnostic_message": analysis.get("diagnostic_message") or routing.get("diagnostic_message"),
        "region_source": "user_confirmed_region",
        "anatomy_verification_status": "confirmed",
        "classification_status": "completed",
        "anatomy_independently_verified": False,
        "anatomy_user_confirmed": True,
        "anatomy_classifier_accepted": bool(metadata.get("anatomy_classifier_accepted", True)),
        "classifier_gate_passed": bool(metadata.get("classifier_gate_passed", True)),
        "regional_model_configured": bool(routing.get("is_supported", False)),
        "regional_model_available": bool(routing.get("is_available", False)),
        "regional_model_name": routing.get("registry_entry", {}).get("display_name") if isinstance(routing.get("registry_entry"), dict) else None,
        "regional_model_status": "completed" if successful_inference else (analysis.get("reason_code") or routing.get("reason_code") or "unavailable"),
        "anatomy_confirmation_required": False,
        "specific_region_confirmation_required": False,
        "next_action": "review_interpretation_status" if not successful_inference else None,
        "body_part_confidence": score_for_selected_region,
        "image_anatomy_verification": {
            "status": "CONFIRMED",
            "classification_status": "completed",
            "anatomy_user_confirmed": True,
            "anatomy_independently_verified": False,
            "source": "user_confirmed_region",
            "detected_region": selected_region,
            "candidate_region": original_candidate_region,
            "body_part_confidence": score_for_selected_region,
            "region_was_corrected": bool(region_was_corrected),
            "message": "The anatomical region was confirmed by the user. This does not confirm pathology findings.",
        },
        "image_type_detection": classifier_output,
        "anatomy_metadata_persisted": hasattr(result, "anatomy_metadata"),
        "anatomy_metadata": metadata,
    }


@router.delete("/{assessment_id}/xray/{xray_result_id}")
def delete_xray(
    assessment_id: UUID,
    xray_result_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Delete an uploaded X-ray result and its stored image for an owned assessment.

    This endpoint is used by the Goal 1 upload page's Remove action. It removes
    the persistent database row as well as the corresponding file, so the UI
    does not merely hide a result that would still be included in assessment.
    """
    _owned(db, assessment_id, current_user.user_id)
    result = (
        db.query(XrayResult)
        .filter(
            XrayResult.assessment_id == assessment_id,
            XrayResult.xray_result_id == xray_result_id,
        )
        .first()
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="X-ray result not found")

    stored_name = result.stored_name
    result_id = str(result.xray_result_id)
    try:
        db.delete(result)
        db.commit()
    except SQLAlchemyError as exc:
        logger.exception(
            "X-ray delete failed for assessment_id=%s, xray_result_id=%s",
            assessment_id,
            xray_result_id,
        )
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The X-ray result could not be removed. Please try again.",
        ) from exc

    # stored_name is created by the server as a UUID filename. Never allow a
    # database value containing a path to escape the X-ray upload directory.
    if isinstance(stored_name, str) and stored_name and Path(stored_name).name == stored_name:
        try:
            xray_dir = (get_settings().upload_path / "xrays").resolve()
            stored_path = (xray_dir / stored_name).resolve()
            if stored_path.parent == xray_dir:
                _remove_file(stored_path)
            else:
                logger.warning("Refusing to delete X-ray path outside upload directory: %s", stored_name)
        except OSError:
            logger.exception(
                "X-ray result row deleted but its image file cleanup failed for xray_result_id=%s",
                xray_result_id,
            )

    return {
        "status": "deleted",
        "assessment_id": str(assessment_id),
        "xray_result_id": result_id,
        "message": "X-ray image and saved result removed from this assessment.",
    }


@router.put("/{assessment_id}/xray/{xray_result_id}/reports/{report_id}/association")
def set_xray_report_association(
    assessment_id: UUID,
    xray_result_id: UUID,
    report_id: UUID,
    payload: XrayReportAssociationRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Record the user's explicit decision about a radiology-report/image pair.

    Region agreement is a safety check, not proof that a report belongs to an
    image. Even after confirmation, evidence may only be combined when regions
    are recognized and compatible.
    """
    _owned(db, assessment_id, current_user.user_id)
    image = (
        db.query(XrayResult)
        .filter(
            XrayResult.assessment_id == assessment_id,
            XrayResult.xray_result_id == xray_result_id,
        )
        .first()
    )
    if image is None:
        raise HTTPException(status_code=404, detail="X-ray image not found")

    report = (
        db.query(MedicalReport)
        .filter(
            MedicalReport.assessment_id == assessment_id,
            MedicalReport.report_id == report_id,
            MedicalReport.report_category == "xray_report",
        )
        .first()
    )
    if report is None:
        raise HTTPException(status_code=404, detail="Radiology report not found for this assessment")

    report_region, region_recognized = _extract_report_region(report)
    safety = _build_association_safety(
        status_value=payload.status,
        image_region=image.region,
        report_region=report_region,
        region_recognized=region_recognized,
    )
    association = (
        db.query(XrayReportAssociation)
        .filter(
            XrayReportAssociation.report_id == report_id,
            XrayReportAssociation.xray_result_id == xray_result_id,
        )
        .first()
    )
    now = datetime.now(timezone.utc)
    if association is None:
        association = XrayReportAssociation(
            report_id=report_id,
            xray_result_id=xray_result_id,
            status=payload.status,
            confirmed_at=now if payload.status == "confirmed" else None,
        )
        db.add(association)
    else:
        association.status = payload.status
        association.confirmed_at = now if payload.status == "confirmed" else None
        db.add(association)

    try:
        db.commit()
        db.refresh(association)
    except SQLAlchemyError as exc:
        logger.exception(
            "X-ray/report association save failed for assessment_id=%s, xray_result_id=%s, report_id=%s",
            assessment_id,
            xray_result_id,
            report_id,
        )
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="The X-ray/report association could not be saved. Please retry.",
        ) from exc

    return {
        "assessment_id": str(assessment_id),
        "xray_result_id": str(xray_result_id),
        "report_id": str(report_id),
        "association_id": str(association.association_id),
        **safety,
        "confirmed_at": association.confirmed_at.isoformat() if association.confirmed_at else None,
    }


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
    results = []
    for row in rows:
        meta = row.anatomy_metadata if isinstance(getattr(row, "anatomy_metadata", None), dict) else {}
        is_user_confirmed = bool(meta.get("anatomy_user_confirmed", False) or meta.get("user_confirmation_status") == "confirmed")
        conf_req = not is_user_confirmed
        det_reg = meta.get("detected_region") or (row.region if row.region and row.region != "unknown" else None)
        cand_reg = meta.get("candidate_region") or meta.get("original_classifier_candidate") or (meta.get("classifier_output", {}).get("candidate_body_part") if isinstance(meta.get("classifier_output"), dict) else None)
        classifier_accepted = bool(meta.get("anatomy_classifier_accepted", False) or meta.get("classifier_gate_passed", False))
        clf_out = meta.get("classifier_output") if isinstance(meta.get("classifier_output"), dict) else {}
        primary_pred = clf_out.get("primary_classifier_result")
        spec_pred = clf_out.get("upper_extremity_refinement")

        if is_user_confirmed:
            decision_state = "USER_CORRECTED" if (meta.get("region_was_corrected") or (det_reg and cand_reg and det_reg != cand_reg)) else "USER_CONFIRMED"
        elif row.status == "failed":
            decision_state = "UNABLE_TO_CLASSIFY"
        elif (
            classifier_accepted
            and not bool(meta.get("is_broad_region", False))
            and meta.get("two_model_comparison", {}).get("decision_case") != "CASE_D_UNRESOLVED"
            and (meta.get("body_part_confidence") or 0) >= 0.60
        ):
            decision_state = "HIGH_CONFIDENCE_CANDIDATE"
        else:
            decision_state = "NEEDS_CONFIRMATION"

        results.append({
            "xray_result_id": str(row.xray_result_id),
            "file_name": row.file_name or "xray_image",
            "decision_state": decision_state,
            "region": row.region,
            "detected_region": det_reg,
            "candidate_region": cand_reg,
            "body_part_confidence": meta.get("body_part_confidence"),
            "primary_prediction": (
                {
                    "model_name": primary_pred.get("model_name", "convnext_tiny_onnx_33class"),
                    "candidate_body_part": primary_pred.get("candidate_body_part"),
                    "confidence": primary_pred.get("confidence"),
                    "top_two_margin": primary_pred.get("top_two_margin"),
                    "top_predictions": primary_pred.get("top_predictions", []),
                    "gate_passed": primary_pred.get("gate_passed", False),
                    "is_broad_region": primary_pred.get("is_broad_region", False),
                }
                if isinstance(primary_pred, dict) and primary_pred.get("candidate_body_part")
                else None
            ),
            "specialist_prediction": (
                {
                    "model_name": spec_pred.get("model_name"),
                    "candidate_body_part": spec_pred.get("candidate_body_part"),
                    "confidence": spec_pred.get("confidence"),
                    "top_two_margin": spec_pred.get("top_two_margin"),
                    "top_predictions": spec_pred.get("top_predictions", []),
                    "eligibility": spec_pred.get("eligibility"),
                    "status": spec_pred.get("status"),
                    "reason_code": spec_pred.get("reason_code"),
                    "message": spec_pred.get("message"),
                }
                if isinstance(spec_pred, dict) and (spec_pred.get("candidate_body_part") or spec_pred.get("status"))
                else None
            ),
            "two_model_comparison": (
                meta.get("two_model_comparison")
                or clf_out.get("two_model_comparison")
            ),
            "final_candidate": (
                meta.get("final_candidate")
                or clf_out.get("final_candidate")
                or cand_reg
            ),
            "selection_source": (
                meta.get("selection_source")
                or clf_out.get("selection_source")
                or "PRIMARY MODEL"
            ),
            "model_version": row.model_version,
            "prediction": row.prediction,
            "findings": getattr(row, "findings", getattr(row, "findings_json", [])) or [],
            "confidence_if_valid": row.confidence_if_valid,
            "uncertainty": row.uncertainty,
            "status": row.status,
            "message": row.message,
            "classification_status": str(meta.get("classification_status") or "completed").lower(),
            "anatomy_verification_status": str(meta.get("status") or ("confirmed" if is_user_confirmed else "unverified")).lower(),
            "anatomy_user_confirmed": is_user_confirmed,
            "anatomy_independently_verified": bool(meta.get("anatomy_independently_verified", False)),
            "anatomy_classifier_accepted": classifier_accepted,
            "classifier_gate_passed": classifier_accepted,
            "is_broad_region": bool(meta.get("is_broad_region", False)),
            "broad_region_classifier_accepted": bool(meta.get("broad_region_classifier_accepted", False)),
            "specific_region_confirmation_required": conf_req,
            "anatomy_confirmation_required": conf_req,
            "regional_model_configured": bool(meta.get("regional_model_configured", False)),
            "regional_model_available": bool(meta.get("regional_model_available", False)),
            "regional_model_name": meta.get("regional_model_name"),
            "regional_model_status": meta.get("regional_model_status"),
            "next_action": ("confirm_or_select_body_region" if conf_req else None),
            "image_type_detection": meta.get("classifier_output") if isinstance(meta.get("classifier_output"), dict) else None,
            "anatomy_metadata": meta,
            "image_anatomy_verification": (
                meta if meta else {
                    "status": "NOT_PERSISTED",
                    "detected_region": row.region if str(row.region or "").strip().lower() not in {"", "unknown"} else None,
                    "body_part_confidence": None,
                    "candidate_region": None,
                    "metadata_persistence": "LIMITED_BY_CURRENT_XRAYRESULT_SCHEMA",
                    "message": "Anatomy classifier confidence/candidate data is unavailable in this legacy saved row.",
                }
            ),
            "report_associations": [
                {
                    "association_id": str(association.association_id),
                    "report_id": str(association.report_id),
                    "report_file_name": association.medical_report.file_name if association.medical_report else None,
                    **_build_association_safety(
                        status_value=str(association.status or "pending").lower(),
                        image_region=row.region,
                        report_region=_extract_report_region(association.medical_report)[0] if association.medical_report else None,
                        region_recognized=_extract_report_region(association.medical_report)[1] if association.medical_report else False,
                    ),
                    "confirmed_at": association.confirmed_at.isoformat() if association.confirmed_at else None,
                }
                for association in (getattr(row, "report_associations", None) or [])
            ],
            "created_at": row.created_at.isoformat() if row.created_at else None,
        })

    return {
        "assessment_id": str(assessment_id),
        "results": results,
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
