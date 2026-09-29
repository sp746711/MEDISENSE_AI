"""Assessment history routes."""

from fastapi import APIRouter

from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Assessment

router = APIRouter(prefix="/history", tags=["History"])


@router.get("")
def get_history(current_user: CurrentUser, db: DbSession) -> dict:
    rows = (
        db.query(Assessment)
        .filter(Assessment.user_id == current_user.user_id)
        .order_by(Assessment.created_at.desc())
        .all()
    )
    return {
        "history": [
            {
                "assessment_id": str(a.assessment_id),
                "created_at": a.created_at.isoformat(),
                "input_types": a.input_types or [],
                "pathway": a.pathway,
                "specialty": a.specialty,
                "status": a.status,
                "model_versions": a.model_versions,
                "rules_version": a.rules_version,
            }
            for a in rows
        ]
    }
