"""Dashboard summary route — returns real DB aggregates (zeros for new users)."""

from fastapi import APIRouter
from sqlalchemy import func

from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Appointment, Assessment, MedicalReport, XrayResult

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary")
def dashboard_summary(current_user: CurrentUser, db: DbSession) -> dict:
    user_id = current_user.user_id

    total_assessments = (
        db.query(func.count(Assessment.assessment_id))
        .filter(Assessment.user_id == user_id)
        .scalar()
        or 0
    )

    xrays_processed = (
        db.query(func.count(XrayResult.xray_result_id))
        .join(Assessment, XrayResult.assessment_id == Assessment.assessment_id)
        .filter(Assessment.user_id == user_id)
        .scalar()
        or 0
    )

    reports_processed = (
        db.query(func.count(MedicalReport.report_id))
        .join(Assessment, MedicalReport.assessment_id == Assessment.assessment_id)
        .filter(Assessment.user_id == user_id)
        .scalar()
        or 0
    )

    appointments_count = (
        db.query(func.count(Appointment.appointment_id))
        .filter(Appointment.user_id == user_id)
        .scalar()
        or 0
    )

    recent = (
        db.query(Assessment)
        .filter(Assessment.user_id == user_id)
        .order_by(Assessment.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "user_name": current_user.name,
        "total_assessments": total_assessments,
        "xrays_processed": xrays_processed,
        "reports_processed": reports_processed,
        "appointments": appointments_count,
        "recent_assessments": [
            {
                "assessment_id": str(a.assessment_id),
                "created_at": a.created_at.isoformat(),
                "input_types": a.input_types or [],
                "pathway": a.pathway,
                "status": a.status,
                "specialty": a.specialty,
            }
            for a in recent
        ],
    }
