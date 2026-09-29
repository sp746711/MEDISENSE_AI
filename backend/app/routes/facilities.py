"""Facility search routes."""

from fastapi import APIRouter, Query

from app.core.dependencies import CurrentUser
from app.providers.provider_repository import ProviderRepository

router = APIRouter(prefix="/facilities", tags=["Facilities"])


@router.get("")
def search_facilities(
    current_user: CurrentUser,
    state: str = Query(..., min_length=2),
    district: str = Query(..., min_length=2),
    city: str | None = Query(default=None),
    emergency_only: bool = Query(default=False),
) -> dict:
    repo = ProviderRepository()
    return repo.search_facilities(
        state=state,
        district=district,
        city=city,
        emergency_only=emergency_only,
    )
