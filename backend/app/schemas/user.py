"""Pydantic schemas for user profile."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserResponse(BaseModel):
    user_id: UUID
    name: str
    email: EmailStr
    state: str
    district: str
    city: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=150)
    state: Optional[str] = Field(default=None, min_length=2, max_length=100)
    district: Optional[str] = Field(default=None, min_length=2, max_length=100)
    city: Optional[str] = Field(default=None, max_length=100)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)
