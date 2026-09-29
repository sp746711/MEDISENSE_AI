"""External provider source adapters (authorized registries/APIs)."""

from typing import Any


class ProviderSource:
    """Base adapter for legitimate healthcare provider data sources."""

    def is_configured(self) -> bool:
        return False

    def fetch_doctors(self, **_filters: Any) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "message": "No authorized provider data source is configured.",
            "doctors": [],
        }
