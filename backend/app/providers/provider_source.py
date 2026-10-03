"""External provider source abstraction interface (authorized registries/APIs)."""

from typing import Any, Optional


class ProviderSource:
    """Base interface for legitimate healthcare provider, facility, and medical shop data sources."""

    def is_configured(self) -> bool:
        """Return True if an authorized source is configured, False otherwise."""
        return False

    def fetch_doctors(
        self,
        specialization: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        city: Optional[str] = None,
        pin: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_km: Optional[float] = None,
        page: int = 1,
        page_size: int = 10,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch doctors from legitimate external source. Returns normalized response or unavailable state."""
        return {
            "status": "unavailable",
            "message": "Healthcare provider information is currently unavailable because no authorized provider data source is configured.",
            "data": [],
            "page": page,
            "page_size": page_size,
            "has_next": False,
            "has_previous": False,
            "source_metadata": {
                "source": None,
                "configured": False,
            },
        }

    def fetch_facilities(
        self,
        state: Optional[str] = None,
        district: Optional[str] = None,
        city: Optional[str] = None,
        pin: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_km: Optional[float] = None,
        emergency_only: bool = False,
        page: int = 1,
        page_size: int = 10,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch facilities from legitimate external source. Returns normalized response or unavailable state."""
        return {
            "status": "unavailable",
            "message": "Healthcare facility information is currently unavailable because no authorized facility data source is configured.",
            "data": [],
            "page": page,
            "page_size": page_size,
            "has_next": False,
            "has_previous": False,
            "source_metadata": {
                "source": None,
                "configured": False,
            },
        }

    def fetch_medical_shops(
        self,
        state: Optional[str] = None,
        district: Optional[str] = None,
        city: Optional[str] = None,
        pin: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_km: Optional[float] = None,
        page: int = 1,
        page_size: int = 10,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch medical shops from legitimate external source. Returns normalized response or unavailable state."""
        return {
            "status": "unavailable",
            "message": "Medical shop information is currently unavailable because no authorized medical shop data source is configured.",
            "data": [],
            "page": page,
            "page_size": page_size,
            "has_next": False,
            "has_previous": False,
            "source_metadata": {
                "source": None,
                "configured": False,
            },
        }
