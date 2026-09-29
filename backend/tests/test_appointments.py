"""Unit tests for Demo Appointment System."""

from datetime import date, time
from uuid import uuid4

import pytest

from app.database.database import SessionLocal, init_db
from app.database.models import Appointment, Doctor, User


@pytest.fixture(scope="module")
def db_session():
    init_db()
    session = SessionLocal()
    yield session
    session.close()


def test_demo_appointment_flow(db_session):
    # Create doctor
    doctor = Doctor(
        name="Dr. Verified Specialist",
        specialization="Pulmonology",
        state="Karnataka",
        district="Bengaluru Urban",
        city="Bengaluru",
        source="State Medical Council Registry",
    )
    db_session.add(doctor)

    # Create user
    user = User(
        name="Appointment Patient",
        email=f"patient_{uuid4().hex[:8]}@example.com",
        password_hash="fakehash",
        state="Karnataka",
        district="Bengaluru Urban",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(doctor)
    db_session.refresh(user)

    # Book demo appointment
    appt = Appointment(
        user_id=user.user_id,
        doctor_id=doctor.doctor_id,
        appointment_date=date.today(),
        appointment_time=time(10, 30),
        status="DEMO_CONFIRMED",
    )
    db_session.add(appt)
    db_session.commit()
    db_session.refresh(appt)

    assert appt.status == "DEMO_CONFIRMED"
    assert appt.appointment_type == "demo"
    assert appt.doctor_id == doctor.doctor_id
