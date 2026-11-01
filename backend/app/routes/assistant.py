"""AI Assistant routes — full RAG/LLM wiring in Stage 15."""

from fastapi import APIRouter

from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment, AssistantSession
from app.llm.llm_service import LLMService
from app.schemas.assessment import AssistantChatRequest

router = APIRouter(prefix="/assistant", tags=["AI Assistant"])


def _build_assessment_context(assessment: Assessment) -> dict:
    result = assessment.result_payload or {}
    evidence = result.get("evidence") or {}
    feedback = result.get("feedback") or {}
    condition_preds = result.get("condition_predictions") or {}
    report_meta = evidence.get("report") or {}
    xray_meta = evidence.get("xray") or {}

    symptoms_present = [
        s.get("symptom") or s.get("finding")
        for s in evidence.get("symptoms", {}).get("present", [])
        if s.get("symptom") or s.get("finding")
    ]
    symptoms_absent = [
        s.get("symptom") or s.get("finding")
        for s in evidence.get("symptoms", {}).get("absent", [])
        if s.get("symptom") or s.get("finding")
    ]
    abnormal_labs = [
        f"{f.get('test_name')}: {f.get('value')} {f.get('unit') or ''} ({f.get('interpretation') or f.get('status') or 'abnormal'})"
        for f in report_meta.get("abnormal_findings", [])
        if f.get("test_name")
    ]
    xray_summary = xray_meta.get("prediction") or (
        [f.get("label") for f in xray_meta.get("findings", []) if f.get("label")]
        if xray_meta.get("findings")
        else None
    )

    return {
        "assessment_id": str(assessment.assessment_id),
        "pathway": assessment.pathway,
        "specialty": assessment.specialty,
        "input_types": assessment.input_types,
        "status": assessment.status,
        "summary": feedback.get("assessment_summary"),
        "symptoms": symptoms_present,
        "absent_symptoms": symptoms_absent,
        "report_findings": abnormal_labs,
        "xray": xray_summary,
        "possible_conditions": [
            c.get("condition")
            for c in condition_preds.get("conditions", [])
            if c.get("condition")
        ],
        "triage_rationale": feedback.get("why_this_triage_selected"),
        "recommended_next_step": feedback.get("recommended_next_step"),
    }


@router.post("/chat")
def chat(
    payload: AssistantChatRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    assessment_ctx = None
    if payload.assessment_id:
        assessment = db.get(Assessment, payload.assessment_id)
        if assessment is None or assessment.user_id != current_user.user_id:
            return {
                "status": "error",
                "message": "Assessment not found for this user.",
            }
        assessment_ctx = _build_assessment_context(assessment)
    else:
        latest = (
            db.query(Assessment)
            .filter(Assessment.user_id == current_user.user_id)
            .order_by(Assessment.created_at.desc())
            .first()
        )
        if latest:
            assessment_ctx = _build_assessment_context(latest)

    session = None
    if payload.session_id:
        session = db.get(AssistantSession, payload.session_id)
        if session is None or session.user_id != current_user.user_id:
            session = None
    if session is None:
        session = AssistantSession(
            user_id=current_user.user_id,
            assessment_id=assessment_ctx["assessment_id"] if assessment_ctx else None,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    llm = LLMService()
    response = llm.generate_assistant_reply(
        user_message=payload.message,
        assessment_context=assessment_ctx,
    )

    return {
        "session_id": str(session.session_id),
        "assessment_context": assessment_ctx,
        **response,
    }


@router.get("/sessions")
def list_sessions(current_user: CurrentUser, db: DbSession) -> dict:
    rows = (
        db.query(AssistantSession)
        .filter(AssistantSession.user_id == current_user.user_id)
        .order_by(AssistantSession.created_at.desc())
        .all()
    )
    return {
        "sessions": [
            {
                "session_id": str(s.session_id),
                "assessment_id": str(s.assessment_id) if s.assessment_id else None,
                "created_at": s.created_at.isoformat(),
            }
            for s in rows
        ]
    }


@router.get("/sessions/{session_id}")
def get_session(session_id: str, current_user: CurrentUser, db: DbSession) -> dict:
    from uuid import UUID

    try:
        sid = UUID(session_id)
    except ValueError:
        return {"status": "error", "message": "Invalid session id"}
    session = db.get(AssistantSession, sid)
    if session is None or session.user_id != current_user.user_id:
        return {"status": "error", "message": "Session not found"}
    return {
        "session_id": str(session.session_id),
        "assessment_id": str(session.assessment_id) if session.assessment_id else None,
        "created_at": session.created_at.isoformat(),
    }
