"""Demo appointment routes — in-application demo only."""

from datetime import date, datetime, time
from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.core.dependencies import CurrentUser, DbSession
from app.database.models import Appointment, Doctor
from app.schemas.assessment import AppointmentCreateRequest

router = APIRouter(prefix="/appointments", tags=["Appointments"])

DEMO_DISCLAIMER = (
    "Demo appointment confirmed. This does not confirm an actual appointment "
    "with the healthcare provider."
)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_appointment(
    payload: AppointmentCreateRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    doctor = db.get(Doctor, payload.doctor_id)
    if doctor is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Selected provider is unavailable. Demo appointments require a "
                "provider record from a legitimate data source."
            ),
        )

    try:
        appt_date = date.fromisoformat(payload.appointment_date)
        parts = payload.appointment_time.split(":")
        appt_time = time(int(parts[0]), int(parts[1]))
    except (ValueError, IndexError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid appointment_date or appointment_time format",
        ) from exc

    appt = Appointment(
        user_id=current_user.user_id,
        doctor_id=doctor.doctor_id,
        appointment_date=appt_date,
        appointment_time=appt_time,
        appointment_type=payload.appointment_type or "demo",
        status="DEMO_CONFIRMED",
    )
    db.add(appt)
    db.commit()
    db.refresh(appt)

    return {
        "appointment_id": str(appt.appointment_id),
        "status": appt.status,
        "appointment_date": appt.appointment_date.isoformat(),
        "appointment_time": appt.appointment_time.isoformat(),
        "doctor_id": str(appt.doctor_id),
        "source": "application_demo_database",
        "disclaimer": DEMO_DISCLAIMER,
    }


@router.get("")
def list_appointments(current_user: CurrentUser, db: DbSession) -> dict:
    rows = (
        db.query(Appointment)
        .filter(Appointment.user_id == current_user.user_id)
        .order_by(Appointment.created_at.desc())
        .all()
    )
    return {
        "appointments": [
            {
                "appointment_id": str(a.appointment_id),
                "doctor_id": str(a.doctor_id),
                "appointment_date": a.appointment_date.isoformat(),
                "appointment_time": a.appointment_time.isoformat(),
                "appointment_type": a.appointment_type,
                "status": a.status,
                "created_at": a.created_at.isoformat(),
                "source": "application_demo_database",
                "disclaimer": DEMO_DISCLAIMER,
            }
            for a in rows
        ],
        "disclaimer": DEMO_DISCLAIMER,
    }


@router.get("/{appointment_id}")
def get_appointment(
    appointment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    appt = db.get(Appointment, appointment_id)
    if appt is None or appt.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return {
        "appointment_id": str(appt.appointment_id),
        "doctor_id": str(appt.doctor_id),
        "appointment_date": appt.appointment_date.isoformat(),
        "appointment_time": appt.appointment_time.isoformat(),
        "appointment_type": appt.appointment_type,
        "status": appt.status,
        "created_at": appt.created_at.isoformat(),
        "source": "application_demo_database",
        "disclaimer": DEMO_DISCLAIMER,
    }


@router.patch("/{appointment_id}/cancel")
def cancel_appointment(
    appointment_id: UUID,
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    appt = db.get(Appointment, appointment_id)
    if appt is None or appt.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Appointment not found")
    appt.status = "CANCELLED"
    db.add(appt)
    db.commit()
    return {
        "appointment_id": str(appt.appointment_id),
        "status": appt.status,
        "message": "Demo appointment cancelled.",
        "source": "application_demo_database",
    }
