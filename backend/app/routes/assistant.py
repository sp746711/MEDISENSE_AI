"""AI Assistant routes — full RAG/LLM wiring in Stage 15."""

from fastapi import APIRouter

from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment, AssistantSession
from app.llm.llm_service import LLMService
from app.schemas.assessment import AssistantChatRequest

router = APIRouter(prefix="/assistant", tags=["AI Assistant"])


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
        assessment_ctx = {
            "assessment_id": str(assessment.assessment_id),
            "pathway": assessment.pathway,
            "specialty": assessment.specialty,
            "input_types": assessment.input_types,
            "status": assessment.status,
        }
    else:
        latest = (
            db.query(Assessment)
            .filter(Assessment.user_id == current_user.user_id)
            .order_by(Assessment.created_at.desc())
            .first()
        )
        if latest:
            assessment_ctx = {
                "assessment_id": str(latest.assessment_id),
                "pathway": latest.pathway,
                "specialty": latest.specialty,
                "input_types": latest.input_types,
                "status": latest.status,
            }

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
