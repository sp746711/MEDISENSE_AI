"""SQLAlchemy ORM models for MediSense AI."""

from __future__ import annotations

import uuid
from datetime import date, datetime, time, timezone
from typing import Any, Optional

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    Time,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Use JSONB on PostgreSQL; JSON for broader dialect compatibility in tests.
JSONType = JSON().with_variant(JSONB(), "postgresql")


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    state: Mapped[str] = mapped_column(String(100), nullable=False)
    district: Mapped[str] = mapped_column(String(100), nullable=False)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )

    assessments: Mapped[list["Assessment"]] = relationship(back_populates="user")
    appointments: Mapped[list["Appointment"]] = relationship(back_populates="user")
    assistant_sessions: Mapped[list["AssistantSession"]] = relationship(
        back_populates="user"
    )


class Assessment(Base):
    __tablename__ = "assessments"

    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )
    # JSON list e.g. ["symptoms", "medical_report", "xray"]
    input_types: Mapped[Any] = mapped_column(JSONType, nullable=False, default=list)
    pathway: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    specialty: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="draft",
    )
    model_versions: Mapped[Any] = mapped_column(JSONType, nullable=True)
    rules_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    result_payload: Mapped[Any] = mapped_column(JSONType, nullable=True)

    user: Mapped["User"] = relationship(back_populates="assessments")
    symptoms: Mapped[list["Symptom"]] = relationship(back_populates="assessment")
    medical_reports: Mapped[list["MedicalReport"]] = relationship(
        back_populates="assessment"
    )
    xray_results: Mapped[list["XrayResult"]] = relationship(back_populates="assessment")
    audit_logs: Mapped[list["AiAuditLog"]] = relationship(back_populates="assessment")


class Symptom(Base):
    __tablename__ = "symptoms"

    symptom_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("assessments.assessment_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symptom: Mapped[str] = mapped_column(String(200), nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False)  # PRESENT/ABSENT/UNKNOWN
    duration: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    severity: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    body_area: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="user_input")

    assessment: Mapped["Assessment"] = relationship(back_populates="symptoms")


class MedicalReport(Base):
    __tablename__ = "medical_reports"

    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("assessments.assessment_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    structured_findings: Mapped[Any] = mapped_column(JSONType, nullable=True)
    reference_ranges: Mapped[Any] = mapped_column(JSONType, nullable=True)
    ocr_metadata: Mapped[Any] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )

    assessment: Mapped["Assessment"] = relationship(back_populates="medical_reports")


class XrayResult(Base):
    __tablename__ = "xray_results"

    xray_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("assessments.assessment_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    region: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    prediction: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    confidence_if_valid: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    uncertainty: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    explainability_artifact: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stored_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )

    assessment: Mapped["Assessment"] = relationship(back_populates="xray_results")


class Doctor(Base):
    __tablename__ = "doctors"

    doctor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    specialization: Mapped[str] = mapped_column(String(150), nullable=False)
    qualification: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    facility: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    state: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    district: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    registration_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    registration_council: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    contact: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source: Mapped[str] = mapped_column(String(150), nullable=False)
    last_verified: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    appointments: Mapped[list["Appointment"]] = relationship(back_populates="doctor")


class Facility(Base):
    __tablename__ = "facilities"

    facility_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    district: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    contact: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    emergency_available: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    source: Mapped[str] = mapped_column(String(150), nullable=False)
    last_verified: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class MedicalShop(Base):
    __tablename__ = "medical_shops"

    shop_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    state: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    district: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    contact: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source: Mapped[str] = mapped_column(String(150), nullable=False)
    last_verified: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class Appointment(Base):
    __tablename__ = "appointments"

    appointment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    doctor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("doctors.doctor_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    appointment_date: Mapped[date] = mapped_column(Date, nullable=False)
    appointment_time: Mapped[time] = mapped_column(Time, nullable=False)
    appointment_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="demo",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="DEMO_REQUESTED",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="appointments")
    doctor: Mapped["Doctor"] = relationship(back_populates="appointments")


class AssistantSession(Base):
    __tablename__ = "assistant_sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    assessment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("assessments.assessment_id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="assistant_sessions")


class AiAuditLog(Base):
    __tablename__ = "ai_audit_logs"
    __table_args__ = (UniqueConstraint("audit_id"),)

    audit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    assessment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("assessments.assessment_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )
    input_types: Mapped[Any] = mapped_column(JSONType, nullable=True)
    model_versions: Mapped[Any] = mapped_column(JSONType, nullable=True)
    nlp_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    ocr_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    rules_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    llm_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    knowledge_source_ids: Mapped[Any] = mapped_column(JSONType, nullable=True)
    output_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    assessment: Mapped[Optional["Assessment"]] = relationship(back_populates="audit_logs")
