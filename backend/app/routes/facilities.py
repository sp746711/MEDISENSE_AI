"""Facility search routes."""

from typing import Optional
from fastapi import APIRouter, Query

from app.core.dependencies import CurrentUser
from app.providers.provider_repository import ProviderRepository

router = APIRouter(prefix="/facilities", tags=["Facilities"])


@router.get("")
def search_facilities(
    current_user: CurrentUser,
    state: Optional[str] = Query(default=None),
    district: Optional[str] = Query(default=None),
    city: Optional[str] = Query(default=None),
    pin: Optional[str] = Query(default=None),
    latitude: Optional[float] = Query(default=None),
    longitude: Optional[float] = Query(default=None),
    radius_km: Optional[float] = Query(default=None),
    emergency_only: bool = Query(default=False),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    assessment_id: Optional[str] = Query(default=None),
) -> dict:
    repo = ProviderRepository()
    return repo.search_facilities(
        state=state,
        district=district,
        city=city,
        pin=pin,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        emergency_only=emergency_only,
        page=page,
        page_size=page_size,
        assessment_id=assessment_id,
    )
