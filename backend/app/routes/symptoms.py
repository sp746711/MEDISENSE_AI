"""Symptoms routes — NLP integration in Stage 5."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment, Symptom
from app.schemas.assessment import SymptomSubmitRequest

router = APIRouter(prefix="/assessments", tags=["Symptoms"])


@router.post("/{assessment_id}/symptoms")
def submit_symptoms(
    assessment_id: UUID,
    payload: SymptomSubmitRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    assessment = _owned(db, assessment_id, current_user.user_id)
    if "symptoms" not in (assessment.input_types or []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This assessment was not configured for symptoms input.",
        )

    from app.ai.symptom_nlp import extract_symptoms

    nlp_result = extract_symptoms(payload.raw_text)
    extracted_symptoms = nlp_result.get("symptoms", [])

    # Clear prior symptoms for this assessment if re-submitting
    db.query(Symptom).filter(Symptom.assessment_id == assessment.assessment_id).delete()

    created_rows = []
    for item in extracted_symptoms:
        row = Symptom(
            assessment_id=assessment.assessment_id,
            symptom=item["symptom"],
            state=item["state"],
            duration=item.get("duration"),
            severity=item.get("severity"),
            body_area=item.get("body_area"),
            context=item.get("context"),
            source="user_input",
        )
        db.add(row)
        created_rows.append(row)

    if not created_rows:
        fallback_row = Symptom(
            assessment_id=assessment.assessment_id,
            symptom="unspecified symptom",
            state="UNKNOWN",
            context=payload.raw_text,
            source="user_input",
        )
        db.add(fallback_row)
        created_rows.append(fallback_row)

    if assessment.status == "draft":
        assessment.status = "inputs_received"
    db.add(assessment)
    db.commit()

    return {
        "status": "extracted",
        "message": f"Successfully extracted {len(created_rows)} symptom finding(s).",
        "raw_text_stored": True,
        "extracted": [
            {
                "symptom": r.symptom,
                "state": r.state,
                "duration": r.duration,
                "severity": r.severity,
                "body_area": r.body_area,
            }
            for r in created_rows
        ],
    }


@router.get("/{assessment_id}/symptoms")
def list_symptoms(
    assessment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    _owned(db, assessment_id, current_user.user_id)
    rows = db.query(Symptom).filter(Symptom.assessment_id == assessment_id).all()
    return {
        "assessment_id": str(assessment_id),
        "symptoms": [
            {
                "symptom_id": str(s.symptom_id),
                "symptom": s.symptom,
                "state": s.state,
                "duration": s.duration,
                "severity": s.severity,
                "body_area": s.body_area,
                "context": s.context,
                "source": s.source,
            }
            for s in rows
        ],
    }


def _owned(db: DbSession, assessment_id: UUID, user_id: UUID) -> Assessment:
    assessment = db.get(Assessment, assessment_id)
    if assessment is None or assessment.user_id != user_id:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment
