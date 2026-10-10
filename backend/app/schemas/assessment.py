"""Placeholder schemas — expanded in later stages."""

from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class AssessmentCreateRequest(BaseModel):
    input_types: List[str] = Field(
        ...,
        min_length=1,
        description="Subset of: symptoms, medical_report, xray",
    )


class AssessmentResponse(BaseModel):
    assessment_id: UUID
    user_id: UUID
    created_at: datetime
    input_types: List[str]
    pathway: Optional[str] = None
    specialty: Optional[str] = None
    status: str
    model_versions: Optional[dict[str, Any]] = None
    rules_version: Optional[str] = None

    model_config = {"from_attributes": True}


class SymptomSubmitRequest(BaseModel):
    raw_text: str = Field(..., min_length=1, max_length=5000)
    followup_answers: Optional[dict[str, Any]] = None


class AssistantChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    assessment_id: Optional[UUID] = None
    session_id: Optional[UUID] = None


class AppointmentCreateRequest(BaseModel):
    doctor_id: UUID
    appointment_date: str
    appointment_time: str
    appointment_type: str = "demo"
