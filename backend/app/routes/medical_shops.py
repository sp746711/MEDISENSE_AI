"""Medical shop search routes."""

from typing import Optional
from fastapi import APIRouter, Query

from app.core.dependencies import CurrentUser
from app.providers.provider_repository import ProviderRepository

router = APIRouter(prefix="/medical-shops", tags=["Medical Shops"])


@router.get("")
def search_shops(
    current_user: CurrentUser,
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
) -> dict:
    repo = ProviderRepository()
    return repo.search_medical_shops(
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
    )
