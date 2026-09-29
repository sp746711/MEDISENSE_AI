"""Provider / doctor routes — legitimate data sources only."""

from uuid import UUID

from fastapi import APIRouter, Query

from app.core.dependencies import CurrentUser
from app.providers.provider_repository import ProviderRepository

router = APIRouter(prefix="/doctors", tags=["Doctors"])


@router.get("")
def search_doctors(
    current_user: CurrentUser,
    state: str = Query(..., min_length=2),
    district: str = Query(..., min_length=2),
    city: str | None = Query(default=None),
    pin: str | None = Query(default=None),
    specialization: str | None = Query(default=None),
) -> dict:
    repo = ProviderRepository()
    return repo.search_doctors(
        state=state,
        district=district,
        city=city,
        pin=pin,
        specialization=specialization,
    )


@router.get("/{doctor_id}")
def get_doctor(doctor_id: UUID, current_user: CurrentUser) -> dict:
    repo = ProviderRepository()
    result = repo.get_doctor(doctor_id)
    if result is None:
        return {
            "status": "unavailable",
            "message": "Healthcare provider information is currently unavailable.",
        }
    return result
