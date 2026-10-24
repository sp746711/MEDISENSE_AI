"""Symptoms routes — NLP integration and follow-up answer persistence.

Task 1 safeguards:
- Apply follow-up answers consistently in submit and preview endpoints.
- Preserve PRESENT / ABSENT / UNKNOWN without treating uncertainty as a negative.
- Use explicit symptom targets when supplied; otherwise defer mapping to the follow-up engine.
- Rebuild follow-up suggestions from the updated symptom state.
- Replace symptom rows transactionally for the current assessment only.
"""
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment, Symptom
from app.schemas.assessment import SymptomSubmitRequest

router = APIRouter(prefix="/assessments", tags=["Symptoms"])


def _normalize_state(value: Any) -> str:
    """Normalize an explicit clinical state while preserving uncertainty."""
    normalized = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "YES": "PRESENT",
        "TRUE": "PRESENT",
        "POSITIVE": "PRESENT",
        "PRESENT": "PRESENT",
        "NO": "ABSENT",
        "FALSE": "ABSENT",
        "NEGATIVE": "ABSENT",
        "ABSENT": "ABSENT",
        "NOT_SURE": "UNKNOWN",
        "UNSURE": "UNKNOWN",
        "UNKNOWN": "UNKNOWN",
        "UNANSWERED": "UNKNOWN",
        "UNASSESSED": "UNKNOWN",
        "": "UNKNOWN",
    }
    return aliases.get(normalized, "UNKNOWN")


def _answer_text(answer_payload: Any) -> str:
    """Extract the user's selected answer from supported payload shapes."""
    if isinstance(answer_payload, dict):
        for key in ("answer", "value", "state"):
            value = answer_payload.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""
    if answer_payload is None:
        return ""
    return str(answer_payload).strip()


def _find_explicit_target(extracted_symptoms: list[dict], target: Any) -> dict | None:
    """Find a symptom only when the payload names an actual extracted symptom."""
    if not isinstance(target, str) or not target.strip():
        return None
    wanted = target.strip().casefold().replace("_", " ").replace("-", " ")
    for item in extracted_symptoms:
        name = str(item.get("symptom") or "").strip().casefold().replace("_", " ").replace("-", " ")
        if name and (name == wanted or name in wanted or wanted in name):
            return item
    return None


def _apply_answers(
    extracted_symptoms: list[dict], followup_answers: Any
) -> tuple[list[dict], int]:
    """Apply answers to extracted findings; return the number of usable answers.

    Explicit targets are applied directly only when they match an extracted finding.
    Otherwise the follow-up engine remains responsible for question-ID mapping.
    Unknown/Not sure values are never converted to ABSENT.
    """
    if not isinstance(followup_answers, dict) or not followup_answers:
        return extracted_symptoms, 0

    from app.ai.followup_engine import apply_followup_answer

    applied = 0
    for question_id, answer_payload in followup_answers.items():
        answer = _answer_text(answer_payload)
        if not answer:
            continue

        target_symptom = None
        scope = None
        if isinstance(answer_payload, dict):
            target_symptom = (
                answer_payload.get("target_symptom")
                or answer_payload.get("symptom")
                or answer_payload.get("symptom_name")
            )
            scope = answer_payload.get("scope")

        explicit_target = _find_explicit_target(extracted_symptoms, target_symptom)
        normalized_state = _normalize_state(
            answer_payload.get("state") if isinstance(answer_payload, dict) and answer_payload.get("state")
            else answer
        )

        if explicit_target is not None:
            explicit_target["state"] = normalized_state
            explicit_target["source"] = explicit_target.get("source") or "followup_answer"
            prior_context = str(explicit_target.get("context") or "").strip()
            answer_context = f"Follow-up answer: {answer}"
            if answer_context not in prior_context:
                explicit_target["context"] = f"{prior_context}; {answer_context}".strip("; ")
            applied += 1
            continue

        # Do not pass a question ID as a symptom target. The engine can map the
        # question ID itself, while explicit target metadata is only used if valid.
        extracted_symptoms = apply_followup_answer(
            extracted_symptoms,
            str(question_id),
            answer,
            target_symptom=target_symptom,
            scope=scope,
        )
        applied += 1

    return extracted_symptoms, applied


def _serialize_symptom(item: dict) -> dict:
    state = _normalize_state(item.get("state", "UNKNOWN"))
    return {
        "symptom": item.get("symptom") or "unspecified symptom",
        "state": state,
        "duration": item.get("duration"),
        "severity": str(item.get("severity")) if item.get("severity") else None,
        "body_area": item.get("body_area"),
        "context": item.get("context") or "",
        "source": item.get("source") or "user_input",
    }


def _store_qualifiers(item: dict, context: str) -> str:
    qual_parts = []
    for key in ("laterality", "quality", "trigger", "type", "onset"):
        value = item.get(key)
        if value:
            qual_parts.append(f"{key}={value}")
    return f"[qualifiers: {', '.join(qual_parts)}] {context}" if qual_parts else context


def _owned(db: DbSession, assessment_id: UUID, user_id: UUID) -> Assessment:
    assessment = db.get(Assessment, assessment_id)
    if assessment is None or assessment.user_id != user_id:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment


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
    from app.ai.followup_engine import suggest_followups

    raw_text = (payload.raw_text or "").strip()
    nlp_result = extract_symptoms(raw_text)
    extracted_symptoms = nlp_result.get("symptoms", [])
    extracted_symptoms, answers_applied = _apply_answers(
        extracted_symptoms, getattr(payload, "followup_answers", None)
    )

    if not extracted_symptoms:
        extracted_symptoms = [{
            "symptom": "unspecified symptom",
            "state": "UNKNOWN",
            "context": raw_text,
            "source": "user_input",
        }]

    try:
        db.query(Symptom).filter(
            Symptom.assessment_id == assessment.assessment_id
        ).delete(synchronize_session=False)

        created_rows = []
        for item in extracted_symptoms:
            normalized = _serialize_symptom(item)
            row = Symptom(
                assessment_id=assessment.assessment_id,
                symptom=normalized["symptom"],
                state=normalized["state"],
                duration=normalized["duration"],
                severity=normalized["severity"],
                body_area=normalized["body_area"],
                context=_store_qualifiers(item, normalized["context"]),
                source=normalized["source"],
            )
            db.add(row)
            created_rows.append(row)

        followup_res = suggest_followups(symptoms=extracted_symptoms)
        if assessment.status == "draft":
            assessment.status = "inputs_received"
        db.add(assessment)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "status": "extracted",
        "message": f"Successfully extracted {len(created_rows)} symptom finding(s).",
        "raw_text_stored": True,
        "followup_answers_applied": answers_applied,
        "followups": followup_res,
        "extracted": [
            {
                "symptom": row.symptom,
                "state": row.state,
                "duration": row.duration,
                "severity": row.severity,
                "body_area": row.body_area,
                "context": row.context,
                "source": row.source,
            }
            for row in created_rows
        ],
    }


@router.post("/{assessment_id}/symptoms/analyze")
def analyze_symptoms_preview(
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
    from app.ai.followup_engine import suggest_followups

    nlp_result = extract_symptoms((payload.raw_text or "").strip())
    extracted = nlp_result.get("symptoms", [])
    extracted, answers_applied = _apply_answers(
        extracted, getattr(payload, "followup_answers", None)
    )
    followup_res = suggest_followups(symptoms=extracted)
    return {
        "status": "analyzed",
        "extracted": extracted,
        "followup_answers_applied": answers_applied,
        "followups": followup_res,
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
                "symptom_id": str(row.symptom_id),
                "symptom": row.symptom,
                "state": row.state,
                "duration": row.duration,
                "severity": row.severity,
                "body_area": row.body_area,
                "context": row.context,
                "source": row.source,
            }
            for row in rows
        ],
    }
