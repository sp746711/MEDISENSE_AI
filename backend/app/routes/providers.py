"""Provider / doctor routes — legitimate data sources only."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query

from app.core.dependencies import CurrentUser
from app.providers.provider_repository import ProviderRepository

router = APIRouter(prefix="/doctors", tags=["Doctors"])


@router.get("")
def search_doctors(
    current_user: CurrentUser,
    specialization: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    district: Optional[str] = Query(default=None),
    city: Optional[str] = Query(default=None),
    pin: Optional[str] = Query(default=None),
    latitude: Optional[float] = Query(default=None),
    longitude: Optional[float] = Query(default=None),
    radius_km: Optional[float] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    assessment_id: Optional[str] = Query(default=None),
    page_token: Optional[str] = Query(default=None),
) -> dict:
    repo = ProviderRepository()
    return repo.search_doctors(
        specialization=specialization,
        state=state,
        district=district,
        city=city,
        pin=pin,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        page=page,
        page_size=page_size,
        assessment_id=assessment_id,
        page_token=page_token,
    )


@router.get("/{doctor_id}")
def get_doctor(doctor_id: str, current_user: CurrentUser) -> dict:
    repo = ProviderRepository()
    result = repo.get_doctor(doctor_id)
    if result is None:
        return {
            "status": "unavailable",
            "message": "Healthcare provider information is currently unavailable.",
        }
    return result
