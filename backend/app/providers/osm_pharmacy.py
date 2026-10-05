"""Direct OpenStreetMap / Overpass pharmacy discovery.

This provider is intentionally independent from Geoapify. It is used only to
increase real-world pharmacy/chemist coverage when Geoapify does not return a
nearby place that is present in OpenStreetMap.

No synthetic records are generated. Missing source fields remain None.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from app.providers.location_utils import calculate_distance_km

logger = logging.getLogger(__name__)


class OSMPharmacyProvider:
    """Search real pharmacy/chemist objects from OpenStreetMap via Overpass."""

    DEFAULT_ENDPOINTS = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    )

    def __init__(self, timeout_seconds: int = 30) -> None:
        configured = os.getenv("OSM_OVERPASS_URL", "").strip()
        self.endpoints = (configured,) if configured else self.DEFAULT_ENDPOINTS

        # The Overpass query itself can take longer than Geoapify.  Keep a
        # separate minimum timeout so a 10-second Geoapify timeout does not
        # immediately kill the OSM request.
        self.timeout_seconds = max(int(timeout_seconds), 30)

    @staticmethod
    def _build_query(
        latitude: float,
        longitude: float,
        radius_meters: int,
    ) -> str:
        """Build one compact Overpass query for pharmacy/chemist tags."""
        lat = float(latitude)
        lon = float(longitude)
        radius = int(radius_meters)

        return f"""
[out:json][timeout:25];
(
  nwr(around:{radius},{lat},{lon})["amenity"="pharmacy"];
  nwr(around:{radius},{lat},{lon})["healthcare"="pharmacy"];
  nwr(around:{radius},{lat},{lon})["shop"~"^(chemist|pharmacy|drugstore)$"];
);
out center tags;
"""

    @staticmethod
    def _coordinates(
        element: dict[str, Any],
    ) -> tuple[float, float] | None:
        try:
            if element.get("type") == "node":
                lat = element.get("lat")
                lon = element.get("lon")
            else:
                center = element.get("center") or {}
                lat = center.get("lat")
                lon = center.get("lon")

            if lat is None or lon is None:
                return None

            lat_f = float(lat)
            lon_f = float(lon)

            if not (-90.0 <= lat_f <= 90.0):
                return None
            if not (-180.0 <= lon_f <= 180.0):
                return None

            return lat_f, lon_f
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _address(tags: dict[str, Any]) -> str | None:
        parts: list[str] = []

        house = str(tags.get("addr:housenumber") or "").strip()
        street = str(tags.get("addr:street") or "").strip()
        place = str(tags.get("addr:place") or "").strip()

        first = " ".join(p for p in (house, street) if p)
        if first:
            parts.append(first)
        elif place:
            parts.append(place)

        for key in (
            "addr:suburb",
            "addr:neighbourhood",
            "addr:city",
            "addr:town",
            "addr:village",
            "addr:state",
            "addr:postcode",
        ):
            value = str(tags.get(key) or "").strip()
            if value and value not in parts:
                parts.append(value)

        return ", ".join(parts) if parts else None

    @staticmethod
    def _city(tags: dict[str, Any]) -> str | None:
        for key in ("addr:city", "addr:town", "addr:village", "addr:suburb"):
            value = str(tags.get(key) or "").strip()
            if value:
                return value
        return None

    @staticmethod
    def _state(tags: dict[str, Any]) -> str | None:
        value = str(tags.get("addr:state") or "").strip()
        return value or None

    @staticmethod
    def _district(tags: dict[str, Any]) -> str | None:
        for key in ("addr:district", "addr:county", "is_in:district"):
            value = str(tags.get(key) or "").strip()
            if value:
                return value
        return None

    @staticmethod
    def _contact(tags: dict[str, Any]) -> str | None:
        for key in ("phone", "contact:phone"):
            value = str(tags.get(key) or "").strip()
            if value:
                return value
        return None

    @staticmethod
    def _website(tags: dict[str, Any]) -> str | None:
        for key in ("website", "contact:website"):
            value = str(tags.get(key) or "").strip()
            if value:
                return value
        return None

    @staticmethod
    def _osm_url(element_type: str, element_id: Any) -> str | None:
        if not element_type or element_id is None:
            return None
        return f"https://www.openstreetmap.org/{element_type}/{element_id}"

    @staticmethod
    def _is_pharmacy(tags: dict[str, Any]) -> bool:
        return (
            tags.get("amenity") == "pharmacy"
            or tags.get("healthcare") == "pharmacy"
            or tags.get("shop") in {"chemist", "pharmacy", "drugstore"}
        )

    def _request(self, query: str) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "MediSenseAI/1.0 (medical shop discovery)",
        }

        timeout = httpx.Timeout(
            timeout=float(self.timeout_seconds),
            connect=8.0,
        )

        last_error: Exception | None = None

        for endpoint in self.endpoints:
            try:
                with httpx.Client(
                    timeout=timeout,
                    follow_redirects=True,
                ) as client:
                    response = client.post(
                        endpoint,
                        data={"data": query},
                        headers=headers,
                    )

                    response.raise_for_status()
                    payload = response.json()

                if not isinstance(payload, dict):
                    raise ValueError("Overpass returned malformed JSON.")

                return payload

            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "OSM Overpass request failed: endpoint=%s error=%s",
                    endpoint,
                    exc,
                )

        if last_error is not None:
            raise last_error

        raise RuntimeError("No Overpass endpoint is configured.")

    def search_pharmacies(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
    ) -> dict[str, Any]:
        """Search real OSM pharmacies/chemists around the confirmed coordinates."""

        try:
            lat = float(latitude)
            lon = float(longitude)
            radius_meters = int(float(radius_km) * 1000)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": "Invalid coordinates or search radius.",
                "data": [],
                "raw_count": 0,
            }

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {
                "status": "error",
                "message": "Invalid coordinates.",
                "data": [],
                "raw_count": 0,
            }

        if radius_meters <= 0:
            return {
                "status": "error",
                "message": "Search radius must be greater than zero.",
                "data": [],
                "raw_count": 0,
            }

        query = self._build_query(lat, lon, radius_meters)

        try:
            payload = self._request(query)
        except Exception as exc:
            logger.warning("OpenStreetMap search unavailable: %s", exc)
            return {
                "status": "error",
                "message": "OpenStreetMap search is temporarily unavailable.",
                "data": [],
                "raw_count": 0,
            }

        elements = payload.get("elements") or []
        items: list[dict[str, Any]] = []
        seen: set[str] = set()

        for element in elements:
            if not isinstance(element, dict):
                continue

            tags = element.get("tags") or {}
            if not isinstance(tags, dict) or not self._is_pharmacy(tags):
                continue

            # Keep only source-backed named locations because the UI needs a
            # real shop name.  We never invent a name for an unnamed object.
            name = str(tags.get("name") or "").strip()
            if not name:
                continue

            coords = self._coordinates(element)
            if coords is None:
                continue

            item_lat, item_lon = coords

            element_type = str(element.get("type") or "").strip()
            element_id = element.get("id")
            if not element_type or element_id is None:
                continue

            osm_id = f"osm:{element_type}:{element_id}"
            if osm_id in seen:
                continue
            seen.add(osm_id)

            distance_km = calculate_distance_km(
                lat,
                lon,
                item_lat,
                item_lon,
            )

            address = self._address(tags)

            items.append(
                {
                    "shop_id": osm_id,
                    "external_id": osm_id,
                    "name": name,
                    "location": address or self._city(tags),
                    "state": self._state(tags),
                    "district": self._district(tags),
                    "city": self._city(tags),
                    "address": address,
                    "latitude": item_lat,
                    "longitude": item_lon,
                    "contact": self._contact(tags),
                    "source": "OpenStreetMap",
                    "source_url": (
                        self._website(tags)
                        or self._osm_url(element_type, element_id)
                    ),
                    "last_verified": None,
                    "distance_km": distance_km,
                }
            )

        items.sort(
            key=lambda item: (
                item["distance_km"]
                if item.get("distance_km") is not None
                else float("inf")
            )
        )

        logger.info(
            "OSM medical shop search: lat=%s lon=%s radius=%skm raw_elements=%d valid_items=%d",
            lat,
            lon,
            radius_km,
            len(elements),
            len(items),
        )

        return {
            "status": "success",
            "message": f"Found {len(items)} OpenStreetMap medical shop candidate(s).",
            "data": items,
            "count": len(items),
            "raw_count": len(elements),
            "source": "OpenStreetMap / Overpass",
        }
