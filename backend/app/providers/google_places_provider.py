"""Google Places API (New) provider discovery adapter.

Strictly integrates with Google Places API (New) for real provider and clinic discovery.
NEVER fabricates doctor names, qualifications, registration numbers, or councils.
All errors (401, 403, 429, 400, 500+, timeout, network, malformed) return explicit status codes
with zero synthetic fallback records.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from typing import Any, Optional

import httpx

from app.core.config import get_settings
from app.providers.location_utils import calculate_distance_km

logger = logging.getLogger(__name__)

# Controlled specialty-search mapping (Requirement 7 & 17)
# Prevents dynamic hallucination of provider search categories.
SPECIALTY_SEARCH_TERMS: dict[str, str] = {
    "general physician": "General Physician",
    "general practice": "General Physician",
    "general medicine": "General Physician",
    "internal medicine": "General Physician",
    "cardiology": "Cardiologist",
    "cardiologist": "Cardiologist",
    "dermatology": "Dermatologist",
    "dermatologist": "Dermatologist",
    "pulmonology": "Pulmonologist",
    "pulmonologist": "Pulmonologist",
    "neurology": "Neurologist",
    "neurologist": "Neurologist",
    "orthopedics": "Orthopedic doctor",
    "orthopedic": "Orthopedic doctor",
    "orthopedic doctor": "Orthopedic doctor",
    "orthopedist": "Orthopedic doctor",
    "ent": "ENT doctor",
    "ent doctor": "ENT doctor",
    "otolaryngology": "ENT doctor",
    "ophthalmology": "Ophthalmologist",
    "ophthalmologist": "Ophthalmologist",
    "gastroenterology": "Gastroenterologist",
    "gastroenterologist": "Gastroenterologist",
    "endocrinology": "Endocrinologist",
    "endocrinologist": "Endocrinologist",
    "dentistry": "Dentist",
    "dentist": "Dentist",
    "pediatrics": "Pediatrician",
    "pediatrician": "Pediatrician",
    "gynecology": "Gynecologist",
    "gynecologist": "Gynecologist",
    "oncology": "Oncologist",
    "oncologist": "Oncologist",
    "psychiatry": "Psychiatrist",
    "psychiatrist": "Psychiatrist",
    "urology": "Urologist",
    "urologist": "Urologist",
    "nephrology": "Nephrologist",
    "nephrologist": "Nephrologist",
}

# Supported Google Places healthcare types (Requirement 1 & 8)
HEALTHCARE_PLACE_TYPES = [
    "doctor",
    "medical_clinic",
    "medical_center",
    "hospital",
]


def normalize_specialty_term(specialty: Optional[str]) -> Optional[str]:
    """Map raw specialty string to controlled search keyword."""
    if not specialty or not specialty.strip():
        return None
    cleaned = re.sub(r"\s+", " ", specialty.strip().lower())
    # Exact lookup
    if cleaned in SPECIALTY_SEARCH_TERMS:
        return SPECIALTY_SEARCH_TERMS[cleaned]
    # Check if a known key is contained within the string (e.g. "Cardiology / Heart")
    for key, term in SPECIALTY_SEARCH_TERMS.items():
        if key in cleaned:
            return term
    # Controlled fallback: preserve user string trimmed without dynamic expansion
    return specialty.strip()


def determine_entity_type(types: Optional[list[str]], name: str = "") -> str:
    """Classify entity as doctor, clinic, medical_center, or hospital.

    Requirement 8:
    A hospital listing is not automatically a doctor.
    If the returned entity is a clinic rather than an individual doctor,
    mark entity_type accordingly. Do not call a clinic an individual doctor without evidence.
    """
    types_set = set(t.lower() for t in (types or []))
    name_lower = (name or "").lower()

    # Hospital takes precedence if explicitly typed or named as hospital
    if "hospital" in types_set or "hospital" in name_lower:
        return "hospital"

    # Clinic / dispensary / polyclinic
    if "medical_clinic" in types_set or any(
        w in name_lower for w in ["clinic", "polyclinic", "dispensary"]
    ):
        return "clinic"

    # Medical center
    if "medical_center" in types_set or any(
        w in name_lower for w in ["medical center", "medical centre", "health centre", "nursing home"]
    ):
        return "medical_center"

    # Doctor
    if (
        "doctor" in types_set
        or name_lower.startswith("dr.")
        or name_lower.startswith("dr ")
        or "physician" in name_lower
    ):
        return "doctor"

    if "physiotherapist" in types_set or "dentist" in types_set:
        return "doctor"

    # If ambiguous but contains general health, default based on clinic keywords
    if any(w in name_lower for w in ["center", "centre"]):
        return "medical_center"

    return "doctor"


class GooglePlacesClient:
    """Client for Google Places API (New) with rigorous provenance and error handling."""

    SEARCH_FIELD_MASK = (
        "places.id,places.displayName,places.formattedAddress,"
        "places.location,places.nationalPhoneNumber,places.internationalPhoneNumber,"
        "places.websiteUri,places.types,places.businessStatus,places.regularOpeningHours,"
        "places.googleMapsUri,nextPageToken"
    )

    DETAILS_FIELD_MASK = (
        "id,displayName,formattedAddress,location,"
        "nationalPhoneNumber,internationalPhoneNumber,websiteUri,"
        "types,businessStatus,regularOpeningHours,googleMapsUri"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        page_size: Optional[int] = None,
        radius_km: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.google_places_api_key
        self.base_url = (base_url or settings.google_places_base_url or "https://places.googleapis.com/v1").rstrip("/")
        self.page_size = page_size or settings.google_places_page_size or 10
        self.radius_km = radius_km or settings.google_places_radius_km or 25
        self.timeout_seconds = 10.0

    def is_configured(self) -> bool:
        """True only if non-empty API key is present."""
        return bool(self.api_key and self.api_key.strip())

    def _headers(self, field_mask: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key or "",
            "X-Goog-FieldMask": field_mask,
        }

    def _safe_log_error(self, message: str, exc: Optional[Exception] = None) -> None:
        """Log message ensuring API key is never leaked."""
        safe_msg = message
        if self.api_key and self.api_key in safe_msg:
            safe_msg = safe_msg.replace(self.api_key, "[REDACTED_API_KEY]")
        if exc:
            exc_str = str(exc)
            if self.api_key and self.api_key in exc_str:
                exc_str = exc_str.replace(self.api_key, "[REDACTED_API_KEY]")
            logger.error("%s: %s", safe_msg, exc_str)
        else:
            logger.error("%s", safe_msg)

    def normalize_place(
        self,
        place: dict[str, Any],
        query_specialty: Optional[str] = None,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
    ) -> dict[str, Any]:
        """Normalize a Google Place (New) object into the canonical Doctor API structure.

        Requirement 10:
        Only populate values actually returned by the source.
        Missing information must remain unavailable/null.
        """
        place_id = place.get("id") or ""

        # Deterministic UUID for database mapping and demo appointment foreign key integrity
        deterministic_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"google_places:{place_id}"))

        display_name = place.get("displayName")
        name = ""
        if isinstance(display_name, dict):
            name = display_name.get("text", "")
        elif isinstance(display_name, str):
            name = display_name

        location = place.get("location") or {}
        p_lat = float(location["latitude"]) if location.get("latitude") is not None else None
        p_lon = float(location["longitude"]) if location.get("longitude") is not None else None

        types = place.get("types") or []
        entity_type = determine_entity_type(types, name)

        # Contact: national or international phone
        contact = place.get("nationalPhoneNumber") or place.get("internationalPhoneNumber") or None
        website = place.get("websiteUri") or None

        # Source URL
        source_url = (
            place.get("googleMapsUri")
            or (f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else None)
        )

        # Distance calculation via straight-line Haversine
        dist_km: Optional[float] = None
        if user_lat is not None and user_lon is not None and p_lat is not None and p_lon is not None:
            dist_km = calculate_distance_km(user_lat, user_lon, p_lat, p_lon)

        # Specialty
        specialization = query_specialty or ("General Physician" if entity_type == "doctor" else None)

        return {
            "doctor_id": deterministic_uuid,
            "external_id": place_id,
            "name": name,
            "specialization": specialization,
            "entity_type": entity_type,
            "facility": None,
            "address": place.get("formattedAddress"),
            "city": None,
            "district": None,
            "state": None,
            "contact": contact,
            "website": website,
            "latitude": p_lat,
            "longitude": p_lon,
            "distance_km": dist_km,
            "source": "Google Places",
            "source_url": source_url,
            "verification_status": "not_verified",
            "qualification": None,  # NEVER invent MBBS/MD
            "registration_number": None,  # NEVER invent registration number
            "registration_council": None,  # NEVER invent council
            "last_verified": None,
        }

    def search_text(
        self,
        *,
        query: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_km: Optional[float] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        specialization: Optional[str] = None,
        page: int = 1,
    ) -> dict[str, Any]:
        """Execute Text Search via Google Places API (New).

        POST /v1/places:searchText
        """
        if not self.is_configured():
            return {
                "status": "configuration_missing",
                "message": (
                    "Google Places provider discovery is not configured. "
                    "Please set GOOGLE_PLACES_API_KEY in the backend environment."
                ),
                "doctors": [],
                "data": [],
                "page": page,
                "page_size": page_size or self.page_size,
                "has_next": False,
                "has_previous": page > 1,
                "total_count": None,
                "source_metadata": {
                    "source": "Google Places",
                    "configured": False,
                },
            }

        url = f"{self.base_url}/places:searchText"
        limit = min(page_size or self.page_size, 20)

        body: dict[str, Any] = {
            "textQuery": query.strip(),
            "pageSize": limit,
        }

        if page_token:
            body["pageToken"] = page_token

        # Geographic restriction / bias when coordinates exist
        if latitude is not None and longitude is not None:
            eff_radius_km = radius_km or self.radius_km
            # Google Places max circle radius is 50,000 meters
            radius_meters = min(float(eff_radius_km * 1000.0), 50000.0)
            body["locationRestriction"] = {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude,
                    },
                    "radius": radius_meters,
                }
            }

        headers = self._headers(self.SEARCH_FIELD_MASK)

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, json=body, headers=headers)

                # 401 Authorization Error
                if response.status_code == 401:
                    self._safe_log_error("Google Places API authorization error (401)")
                    return {
                        "status": "authorization_error",
                        "message": "Google Places API authorization failed. Check your API key.",
                        "doctors": [],
                        "data": [],
                        "page": page,
                        "page_size": limit,
                        "has_next": False,
                        "has_previous": page > 1,
                        "total_count": None,
                        "source_metadata": {"source": "Google Places", "configured": True},
                    }

                # 403 Forbidden / Project Denied
                if response.status_code == 403:
                    self._safe_log_error("Google Places API access denied (403)")
                    return {
                        "status": "authorization_error",
                        "message": "Google Places API access denied or Places API (New) not enabled for this key.",
                        "doctors": [],
                        "data": [],
                        "page": page,
                        "page_size": limit,
                        "has_next": False,
                        "has_previous": page > 1,
                        "total_count": None,
                        "source_metadata": {"source": "Google Places", "configured": True},
                    }

                # 429 Rate Limited
                if response.status_code == 429:
                    self._safe_log_error("Google Places API rate limit exceeded (429)")
                    return {
                        "status": "rate_limited",
                        "message": (
                            "Healthcare provider data source is temporarily unavailable due to rate limits. "
                            "Please try again shortly."
                        ),
                        "doctors": [],
                        "data": [],
                        "page": page,
                        "page_size": limit,
                        "has_next": False,
                        "has_previous": page > 1,
                        "total_count": None,
                        "source_metadata": {"source": "Google Places", "configured": True},
                    }

                # 400 Bad Request
                if response.status_code == 400:
                    self._safe_log_error(f"Google Places API bad request (400): {response.text}")
                    return {
                        "status": "source_error",
                        "message": "Google Places API request was invalid.",
                        "doctors": [],
                        "data": [],
                        "page": page,
                        "page_size": limit,
                        "has_next": False,
                        "has_previous": page > 1,
                        "total_count": None,
                        "source_metadata": {"source": "Google Places", "configured": True},
                    }

                # 500+ Server Error
                if response.status_code >= 500:
                    self._safe_log_error(f"Google Places API server error ({response.status_code})")
                    return {
                        "status": "unavailable",
                        "message": "Google Places API is currently unavailable. Please try again later.",
                        "doctors": [],
                        "data": [],
                        "page": page,
                        "page_size": limit,
                        "has_next": False,
                        "has_previous": page > 1,
                        "total_count": None,
                        "source_metadata": {"source": "Google Places", "configured": True},
                    }

                # Parse JSON
                try:
                    payload = response.json()
                except Exception as json_err:
                    self._safe_log_error("Failed to parse JSON response from Google Places", json_err)
                    return {
                        "status": "source_error",
                        "message": "Healthcare provider discovery returned a malformed response.",
                        "doctors": [],
                        "data": [],
                        "page": page,
                        "page_size": limit,
                        "has_next": False,
                        "has_previous": page > 1,
                        "total_count": None,
                        "source_metadata": {"source": "Google Places", "configured": True},
                    }

                raw_places = payload.get("places") or []
                next_token = payload.get("nextPageToken")

                doctors = [
                    self.normalize_place(
                        p,
                        query_specialty=specialization,
                        user_lat=latitude,
                        user_lon=longitude,
                    )
                    for p in raw_places
                ]

                # Sort by distance when user coordinates are provided
                if latitude is not None and longitude is not None:
                    doctors.sort(
                        key=lambda d: d["distance_km"] if d["distance_km"] is not None else float("inf")
                    )

                if not doctors:
                    return {
                        "status": "success",
                        "message": "No matching healthcare providers found for this location and specialty.",
                        "doctors": [],
                        "data": [],
                        "page": page,
                        "page_size": limit,
                        "has_next": False,
                        "has_previous": page > 1,
                        "total_count": None,
                        "source_metadata": {
                            "source": "Google Places",
                            "configured": True,
                            "source_url": "https://places.googleapis.com/v1",
                        },
                    }

                return {
                    "status": "success",
                    "message": f"Found {len(doctors)} matching provider(s).",
                    "doctors": doctors,
                    "data": doctors,
                    "page": page,
                    "page_size": limit,
                    "has_next": bool(next_token),
                    "next_page_token": next_token,
                    "has_previous": page > 1,
                    "total_count": None,  # Do NOT fabricate total_count
                    "expanded": False,
                    "expansion_message": None,
                    "source_metadata": {
                        "source": "Google Places",
                        "configured": True,
                        "source_url": "https://places.googleapis.com/v1",
                    },
                }

        except httpx.TimeoutException as exc:
            self._safe_log_error("Google Places request timed out", exc)
            return {
                "status": "unavailable",
                "message": "Healthcare provider discovery service timed out. Please try again.",
                "doctors": [],
                "data": [],
                "page": page,
                "page_size": limit,
                "has_next": False,
                "has_previous": page > 1,
                "total_count": None,
                "source_metadata": {"source": "Google Places", "configured": True},
            }
        except httpx.RequestError as exc:
            self._safe_log_error("Google Places network request error", exc)
            return {
                "status": "unavailable",
                "message": "Unable to connect to healthcare provider discovery service.",
                "doctors": [],
                "data": [],
                "page": page,
                "page_size": limit,
                "has_next": False,
                "has_previous": page > 1,
                "total_count": None,
                "source_metadata": {"source": "Google Places", "configured": True},
            }

    def search_nearby(
        self,
        *,
        latitude: float,
        longitude: float,
        radius_km: Optional[float] = None,
        included_types: Optional[list[str]] = None,
        page_size: Optional[int] = None,
        specialization: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute Nearby Search via Google Places API (New).

        POST /v1/places:searchNearby
        """
        if not self.is_configured():
            return {
                "status": "configuration_missing",
                "message": (
                    "Google Places provider discovery is not configured. "
                    "Please set GOOGLE_PLACES_API_KEY in the backend environment."
                ),
                "doctors": [],
                "data": [],
                "page": 1,
                "page_size": page_size or self.page_size,
                "has_next": False,
                "has_previous": False,
                "total_count": None,
                "source_metadata": {"source": "Google Places", "configured": False},
            }

        url = f"{self.base_url}/places:searchNearby"
        eff_radius_km = radius_km or self.radius_km
        radius_meters = min(float(eff_radius_km * 1000.0), 50000.0)
        limit = min(page_size or self.page_size, 20)

        types = included_types or ["doctor", "medical_clinic"]

        body: dict[str, Any] = {
            "includedTypes": types,
            "maxResultCount": limit,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude,
                    },
                    "radius": radius_meters,
                }
            },
        }

        headers = self._headers(self.SEARCH_FIELD_MASK)

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, json=body, headers=headers)

                if response.status_code == 401 or response.status_code == 403:
                    return {
                        "status": "authorization_error",
                        "message": "Google Places API authorization failed. Check your API key.",
                        "doctors": [],
                        "data": [],
                        "page": 1,
                        "page_size": limit,
                        "has_next": False,
                        "has_previous": False,
                        "total_count": None,
                        "source_metadata": {"source": "Google Places", "configured": True},
                    }

                if response.status_code == 429:
                    return {
                        "status": "rate_limited",
                        "message": "Healthcare provider data source is temporarily unavailable due to rate limits.",
                        "doctors": [],
                        "data": [],
                        "page": 1,
                        "page_size": limit,
                        "has_next": False,
                        "has_previous": False,
                        "total_count": None,
                        "source_metadata": {"source": "Google Places", "configured": True},
                    }

                if response.status_code >= 400:
                    return {
                        "status": "source_error" if response.status_code < 500 else "unavailable",
                        "message": "Healthcare provider discovery failed.",
                        "doctors": [],
                        "data": [],
                        "page": 1,
                        "page_size": limit,
                        "has_next": False,
                        "has_previous": False,
                        "total_count": None,
                        "source_metadata": {"source": "Google Places", "configured": True},
                    }

                payload = response.json()
                raw_places = payload.get("places") or []

                doctors = [
                    self.normalize_place(
                        p,
                        query_specialty=specialization,
                        user_lat=latitude,
                        user_lon=longitude,
                    )
                    for p in raw_places
                ]

                doctors.sort(
                    key=lambda d: d["distance_km"] if d["distance_km"] is not None else float("inf")
                )

                return {
                    "status": "success",
                    "message": f"Found {len(doctors)} matching provider(s).",
                    "doctors": doctors,
                    "data": doctors,
                    "page": 1,
                    "page_size": limit,
                    "has_next": False,
                    "has_previous": False,
                    "total_count": None,
                    "source_metadata": {
                        "source": "Google Places",
                        "configured": True,
                        "source_url": "https://places.googleapis.com/v1",
                    },
                }

        except httpx.TimeoutException:
            return {
                "status": "unavailable",
                "message": "Healthcare provider discovery service timed out. Please try again.",
                "doctors": [],
                "data": [],
                "page": 1,
                "page_size": limit,
                "has_next": False,
                "has_previous": False,
                "total_count": None,
                "source_metadata": {"source": "Google Places", "configured": True},
            }
        except httpx.RequestError:
            return {
                "status": "unavailable",
                "message": "Unable to connect to healthcare provider discovery service.",
                "doctors": [],
                "data": [],
                "page": 1,
                "page_size": limit,
                "has_next": False,
                "has_previous": False,
                "total_count": None,
                "source_metadata": {"source": "Google Places", "configured": True},
            }

    def get_place_details(self, place_id: str) -> dict[str, Any]:
        """Fetch Place Details via GET /v1/places/{place_id}.

        Requirement 9:
        Only retrieves necessary fields.
        """
        if not self.is_configured():
            return {
                "status": "configuration_missing",
                "message": "Google Places provider discovery is not configured.",
                "doctor": None,
            }

        if not place_id or not place_id.strip():
            return {
                "status": "source_error",
                "message": "Place ID must not be empty.",
                "doctor": None,
            }

        url = f"{self.base_url}/places/{place_id.strip()}"
        headers = self._headers(self.DETAILS_FIELD_MASK)

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.get(url, headers=headers)

                if response.status_code == 401 or response.status_code == 403:
                    return {
                        "status": "authorization_error",
                        "message": "Google Places API authorization failed.",
                        "doctor": None,
                    }
                if response.status_code == 404:
                    return {
                        "status": "unavailable",
                        "message": "Healthcare provider information is currently unavailable.",
                        "doctor": None,
                    }
                if response.status_code == 429:
                    return {
                        "status": "rate_limited",
                        "message": "Healthcare provider data source is temporarily unavailable due to rate limits.",
                        "doctor": None,
                    }
                if response.status_code >= 400:
                    return {
                        "status": "unavailable",
                        "message": "Healthcare provider information is currently unavailable.",
                        "doctor": None,
                    }

                place_data = response.json()
                normalized = self.normalize_place(place_data)
                return {
                    "status": "success",
                    "doctor": normalized,
                    **normalized,
                }

        except httpx.TimeoutException:
            return {
                "status": "unavailable",
                "message": "Healthcare provider discovery service timed out.",
                "doctor": None,
            }
        except httpx.RequestError:
            return {
                "status": "unavailable",
                "message": "Unable to connect to healthcare provider discovery service.",
                "doctor": None,
            }
