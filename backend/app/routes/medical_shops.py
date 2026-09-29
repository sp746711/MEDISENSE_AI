"""Medical shop search routes."""

from fastapi import APIRouter, Query

from app.core.dependencies import CurrentUser
from app.providers.provider_repository import ProviderRepository

router = APIRouter(prefix="/medical-shops", tags=["Medical Shops"])


@router.get("")
def search_shops(
    current_user: CurrentUser,
    state: str = Query(..., min_length=2),
    district: str = Query(..., min_length=2),
    city: str | None = Query(default=None),
) -> dict:
    repo = ProviderRepository()
    return repo.search_medical_shops(state=state, district=district, city=city)
