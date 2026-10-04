"""Authorized provider source adapter.

Connects strictly to configured authorized provider APIs / registries.
Does NOT fabricate any data, does NOT generate dummy rows.
If not configured or if API calls fail/rate-limit, returns explicit status.
"""

from typing import Any, Optional
import httpx
import logging

from app.core.config import get_settings
from app.providers.geoapify_places import GeoapifyPlacesClient
from app.providers.provider_source import ProviderSource

logger = logging.getLogger(__name__)


class AuthorizedProviderSource(ProviderSource):
    """Adapter for authorized external healthcare provider sources."""

    def __init__(self) -> None:
        settings = get_settings()
        self.provider_source_name = settings.provider_data_source
        self.provider_base_url = settings.provider_api_base_url
        self.provider_api_key = settings.provider_api_key

        self.facility_source_name = settings.facility_data_source
        self.facility_base_url = settings.facility_api_base_url
        self.facility_api_key = settings.facility_api_key

        self.shop_source_name = settings.medical_shop_data_source
        self.shop_base_url = settings.medical_shop_api_base_url
        self.shop_api_key = settings.medical_shop_api_key

        self.geoapify_client = GeoapifyPlacesClient()

    def is_configured(self) -> bool:
        """True only if an authorized provider data source and base URL are configured."""
        return bool(self.provider_source_name and self.provider_base_url)

    def is_facility_configured(self) -> bool:
        """True only if an authorized facility data source is configured."""
        if self.facility_source_name and self.facility_source_name.strip().lower() == "geoapify":
            return bool(
                self.geoapify_client.api_key
                and self.geoapify_client.api_key.strip()
                and self.geoapify_client.base_url
                and self.geoapify_client.base_url.strip()
            )
        return bool(self.facility_source_name and self.facility_base_url)

    def is_shop_configured(self) -> bool:
        """True only if an authorized medical shop data source is configured."""
        if self.shop_source_name and self.shop_source_name.strip().lower() == "geoapify":
            return bool(
                self.geoapify_client.api_key
                and self.geoapify_client.api_key.strip()
                and self.geoapify_client.base_url
                and self.geoapify_client.base_url.strip()
            )
        return bool(self.shop_source_name and self.shop_base_url)

    def _get_headers(self, api_key: Optional[str] = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "MediSenseAI-ProviderClient/1.0",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-Key"] = api_key
        return headers

    def normalize_doctor(self, raw: dict[str, Any], default_source: Optional[str] = None) -> dict[str, Any]:
        """Normalize raw external doctor record into canonical structure. Missing fields become None."""
        return {
            "external_id": raw.get("external_id") or raw.get("id"),
            "name": raw.get("name"),
            "specialization": raw.get("specialization") or raw.get("specialty"),
            "qualification": raw.get("qualification"),
            "registration_number": raw.get("registration_number") or raw.get("reg_no"),
            "registration_council": raw.get("registration_council") or raw.get("council"),
            "facility": raw.get("facility") or raw.get("hospital") or raw.get("clinic"),
            "state": raw.get("state"),
            "district": raw.get("district"),
            "city": raw.get("city"),
            "address": raw.get("address"),
            "latitude": float(raw["latitude"]) if raw.get("latitude") is not None else None,
            "longitude": float(raw["longitude"]) if raw.get("longitude") is not None else None,
            "contact": raw.get("contact") or raw.get("phone"),
            "source": raw.get("source") or default_source,
            "source_url": raw.get("source_url"),
            "last_verified": raw.get("last_verified"),
        }

    def normalize_facility(self, raw: dict[str, Any], default_source: Optional[str] = None) -> dict[str, Any]:
        """Normalize raw external facility record into canonical structure."""
        return {
            "external_id": raw.get("external_id") or raw.get("id"),
            "name": raw.get("name"),
            "facility_type": raw.get("facility_type") or raw.get("type"),
            "state": raw.get("state"),
            "district": raw.get("district"),
            "city": raw.get("city"),
            "address": raw.get("address"),
            "pin": raw.get("pin") or raw.get("pincode"),
            "latitude": float(raw["latitude"]) if raw.get("latitude") is not None else None,
            "longitude": float(raw["longitude"]) if raw.get("longitude") is not None else None,
            "has_emergency": bool(raw.get("has_emergency")),
            "contact": raw.get("contact") or raw.get("phone"),
            "source": raw.get("source") or default_source,
            "source_url": raw.get("source_url"),
            "last_verified": raw.get("last_verified"),
        }

    def normalize_medical_shop(self, raw: dict[str, Any], default_source: Optional[str] = None) -> dict[str, Any]:
        """Normalize raw external medical shop record into canonical structure."""
        return {
            "external_id": raw.get("external_id") or raw.get("id"),
            "name": raw.get("name"),
            "state": raw.get("state"),
            "district": raw.get("district"),
            "city": raw.get("city"),
            "address": raw.get("address"),
            "pin": raw.get("pin") or raw.get("pincode"),
            "latitude": float(raw["latitude"]) if raw.get("latitude") is not None else None,
            "longitude": float(raw["longitude"]) if raw.get("longitude") is not None else None,
            "contact": raw.get("contact") or raw.get("phone"),
            "source": raw.get("source") or default_source,
            "source_url": raw.get("source_url"),
            "last_verified": raw.get("last_verified"),
        }

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
        if not self.is_configured():
            return {
                "status": "unavailable",
                "message": "Healthcare provider information is currently unavailable because no authorized provider data source is configured.",
                "data": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 1,
                "source_metadata": {
                    "source": None,
                    "configured": False,
                },
            }

        params: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
        }
        if specialization:
            params["specialization"] = specialization
        if state:
            params["state"] = state
        if district:
            params["district"] = district
        if city:
            params["city"] = city
        if pin:
            params["pin"] = pin
        if latitude is not None and longitude is not None:
            params["latitude"] = latitude
            params["longitude"] = longitude
        if radius_km is not None:
            params["radius_km"] = radius_km

        url = f"{self.provider_base_url.rstrip('/')}/doctors"

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params, headers=self._get_headers(self.provider_api_key))
                
                if response.status_code == 429:
                    return {
                        "status": "rate_limited",
                        "message": "Healthcare provider data source is temporarily unavailable due to rate limits. Please try again shortly.",
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "has_next": False,
                        "has_previous": page > 1,
                        "source_metadata": {
                            "source": self.provider_source_name,
                            "configured": True,
                        },
                    }

                if response.status_code != 200:
                    logger.warning("External provider source returned HTTP %s", response.status_code)
                    return {
                        "status": "error",
                        "message": f"External provider source error (HTTP {response.status_code}).",
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "has_next": False,
                        "has_previous": page > 1,
                        "source_metadata": {
                            "source": self.provider_source_name,
                            "configured": True,
                        },
                    }

                payload = response.json()
                raw_items = payload.get("data") or payload.get("doctors") or []
                normalized = [self.normalize_doctor(item, self.provider_source_name) for item in raw_items]

                has_next = payload.get("has_next")
                if has_next is None:
                    has_next = len(normalized) == page_size

                res: dict[str, Any] = {
                    "status": "success",
                    "data": normalized,
                    "page": page,
                    "page_size": page_size,
                    "has_next": bool(has_next),
                    "has_previous": page > 1,
                    "source_metadata": {
                        "source": self.provider_source_name,
                        "configured": True,
                        "source_url": payload.get("source_url") or self.provider_base_url,
                    },
                }
                if "total_count" in payload:
                    res["total_count"] = payload["total_count"]
                return res

        except Exception as exc:
            logger.error("Failed to query authorized provider source: %s", exc)
            return {
                "status": "unavailable",
                "message": "Healthcare provider source is currently unreachable.",
                "data": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 1,
                "source_metadata": {
                    "source": self.provider_source_name,
                    "configured": True,
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
        if not self.is_facility_configured():
            return {
                "status": "unavailable",
                "message": "Healthcare facility information is currently unavailable because no authorized facility data source is configured.",
                "data": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 1,
                "source_metadata": {
                    "source": None,
                    "configured": False,
                },
            }

        # Check if delegated to dedicated Geoapify Places client
        if self.facility_source_name and self.facility_source_name.strip().lower() == "geoapify":
            if latitude is not None and longitude is not None:
                return self.geoapify_client.search_hospitals(
                    latitude=latitude,
                    longitude=longitude,
                    radius_km=radius_km,
                    emergency_only=emergency_only,
                    page=page,
                    page_size=page_size,
                )
            return {
                "status": "unavailable",
                "message": "Healthcare facility search via Geoapify requires GPS or map-selected coordinates.",
                "data": [],
                "facilities": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 1,
                "source_metadata": {
                    "source": "Geoapify / OpenStreetMap",
                    "configured": self.geoapify_client.is_configured(),
                },
            }

        params: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
        }
        if emergency_only:
            params["emergency_only"] = "true"
        if state:
            params["state"] = state
        if district:
            params["district"] = district
        if city:
            params["city"] = city
        if pin:
            params["pin"] = pin
        if latitude is not None and longitude is not None:
            params["latitude"] = latitude
            params["longitude"] = longitude
        if radius_km is not None:
            params["radius_km"] = radius_km

        url = f"{self.facility_base_url.rstrip('/')}/facilities"

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params, headers=self._get_headers(self.facility_api_key))
                
                if response.status_code == 429:
                    return {
                        "status": "rate_limited",
                        "message": "Healthcare facility data source is temporarily unavailable due to rate limits. Please try again shortly.",
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "has_next": False,
                        "has_previous": page > 1,
                        "source_metadata": {
                            "source": self.facility_source_name,
                            "configured": True,
                        },
                    }
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "message": f"External facility source error (HTTP {response.status_code}).",
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "has_next": False,
                        "has_previous": page > 1,
                        "source_metadata": {
                            "source": self.facility_source_name,
                            "configured": True,
                        },
                    }

                payload = response.json()
                raw_items = payload.get("data") or payload.get("facilities") or []
                normalized = [self.normalize_facility(item, self.facility_source_name) for item in raw_items]

                has_next = payload.get("has_next")
                if has_next is None:
                    has_next = len(normalized) == page_size

                res = {
                    "status": "success",
                    "data": normalized,
                    "page": page,
                    "page_size": page_size,
                    "has_next": bool(has_next),
                    "has_previous": page > 1,
                    "source_metadata": {
                        "source": self.facility_source_name,
                        "configured": True,
                        "source_url": payload.get("source_url") or self.facility_base_url,
                    },
                }
                if "total_count" in payload:
                    res["total_count"] = payload["total_count"]
                return res
        except Exception as exc:
            logger.error("Failed to query authorized facility source: %s", exc)
            return {
                "status": "unavailable",
                "message": "Healthcare facility source is currently unreachable.",
                "data": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 1,
                "source_metadata": {
                    "source": self.facility_source_name,
                    "configured": True,
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
        if not self.is_shop_configured():
            return {
                "status": "unavailable",
                "message": "Medical shop information is currently unavailable because no authorized medical shop data source is configured.",
                "data": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 1,
                "source_metadata": {
                    "source": None,
                    "configured": False,
                },
            }

        # Check if delegated to dedicated Geoapify Places client
        if self.shop_source_name and self.shop_source_name.strip().lower() == "geoapify":
            if latitude is not None and longitude is not None:
                return self.geoapify_client.search_pharmacies(
                    latitude=latitude,
                    longitude=longitude,
                    radius_km=radius_km,
                    page=page,
                    page_size=page_size,
                )
            return {
                "status": "unavailable",
                "message": "Medical shop search via Geoapify requires GPS or map-selected coordinates.",
                "data": [],
                "medical_shops": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 1,
                "disclaimer": "Medical shop listings do not include medication prescribing or dosage advice.",
                "source_metadata": {
                    "source": "Geoapify / OpenStreetMap",
                    "configured": self.geoapify_client.is_configured(),
                },
            }

        params: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
        }
        if state:
            params["state"] = state
        if district:
            params["district"] = district
        if city:
            params["city"] = city
        if pin:
            params["pin"] = pin
        if latitude is not None and longitude is not None:
            params["latitude"] = latitude
            params["longitude"] = longitude
        if radius_km is not None:
            params["radius_km"] = radius_km

        url = f"{self.shop_base_url.rstrip('/')}/medical-shops"

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params, headers=self._get_headers(self.shop_api_key))
                
                if response.status_code == 429:
                    return {
                        "status": "rate_limited",
                        "message": "Medical shop data source is temporarily unavailable due to rate limits. Please try again shortly.",
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "has_next": False,
                        "has_previous": page > 1,
                        "source_metadata": {
                            "source": self.shop_source_name,
                            "configured": True,
                        },
                    }

                if response.status_code != 200:
                    return {
                        "status": "error",
                        "message": f"External medical shop source error (HTTP {response.status_code}).",
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "has_next": False,
                        "has_previous": page > 1,
                        "source_metadata": {
                            "source": self.shop_source_name,
                            "configured": True,
                        },
                    }

                payload = response.json()
                raw_items = payload.get("data") or payload.get("medical_shops") or []
                normalized = [self.normalize_medical_shop(item, self.shop_source_name) for item in raw_items]
                if latitude is not None and longitude is not None:
                    from app.providers.location_utils import calculate_distance_km
                    for item in normalized:
                        if item.get("distance_km") is None and item.get("latitude") is not None and item.get("longitude") is not None:
                            item["distance_km"] = calculate_distance_km(latitude, longitude, item["latitude"], item["longitude"])
                    normalized.sort(
                        key=lambda s: s["distance_km"] if s.get("distance_km") is not None else float("inf")
                    )

                has_next = payload.get("has_next")
                if has_next is None:
                    has_next = len(normalized) == page_size

                res = {
                    "status": "success",
                    "data": normalized,
                    "page": page,
                    "page_size": page_size,
                    "has_next": bool(has_next),
                    "has_previous": page > 1,
                    "source_metadata": {
                        "source": self.shop_source_name,
                        "configured": True,
                        "source_url": payload.get("source_url") or self.shop_base_url,
                    },
                }
                if "total_count" in payload:
                    res["total_count"] = payload["total_count"]
                return res
        except Exception as exc:
            logger.error("Failed to query authorized medical shop source: %s", exc)
            return {
                "status": "unavailable",
                "message": "Medical shop source is currently unreachable.",
                "data": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 1,
                "source_metadata": {
                    "source": self.shop_source_name,
                    "configured": True,
                },
            }
