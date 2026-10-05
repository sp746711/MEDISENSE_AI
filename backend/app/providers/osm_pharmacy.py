"""
OpenStreetMap / Overpass provider for nearby medical shops.

Uses only real OSM elements returned by Overpass.
No coordinates, names, phone numbers, addresses, or distances are invented.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

import socket

# Ensure IPv4 addresses are prioritized over IPv6 to avoid 21s Windows TCP SYN timeouts
_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_preferred_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    res = _orig_getaddrinfo(host, port, family, type, proto, flags)
    return sorted(res, key=lambda x: 0 if x[0] == socket.AF_INET else 1)

if socket.getaddrinfo is not _ipv4_preferred_getaddrinfo:
    socket.getaddrinfo = _ipv4_preferred_getaddrinfo

from app.providers.location_utils import calculate_distance_km

logger = logging.getLogger(__name__)


class OSMPharmacyProvider:
    """Search pharmacies / medical shops directly in OpenStreetMap via Overpass."""

    DEFAULT_ENDPOINTS = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    )

    def __init__(self, timeout_seconds: float = 1.5) -> None:
        configured = os.getenv("OSM_OVERPASS_URL", "").strip()

        if configured:
            self.endpoints = (configured,)
        else:
            self.endpoints = self.DEFAULT_ENDPOINTS

        # Bounded timeout: maximum 2.0 seconds so OSM never hangs progressive searches
        self.timeout_seconds = min(max(float(timeout_seconds), 1.0), 2.0) if timeout_seconds else 1.5

    @staticmethod
    def _build_query(
        latitude: float,
        longitude: float,
        radius_meters: int,
    ) -> str:
        # Keep the Overpass query bounded and fast. Use nw (nodes and ways) instead of nwr
        # because pharmacies are never relations, avoiding severe 504 server timeouts.
        radius = int(radius_meters)
        return f"""
[out:json][timeout:10];
(
  nw(around:{radius},{float(latitude)},{float(longitude)})["amenity"="pharmacy"];
  nw(around:{radius},{float(latitude)},{float(longitude)})["healthcare"="pharmacy"];
  nw(around:{radius},{float(latitude)},{float(longitude)})["shop"="chemist"];
  nw(around:{radius},{float(latitude)},{float(longitude)})["shop"="pharmacy"];
  nw(around:{radius},{float(latitude)},{float(longitude)})["shop"="drugstore"];
  nw(around:{radius},{float(latitude)},{float(longitude)})["shop"="medical_supply"];
  nw(around:{radius},{float(latitude)},{float(longitude)})["dispensing"="yes"];
);
out center tags;
"""

    @staticmethod
    def _element_coordinates(
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
    def _format_address(tags: dict[str, Any]) -> str | None:
        parts: list[str] = []

        house = str(tags.get("addr:housenumber") or "").strip()
        street = str(tags.get("addr:street") or "").strip()
        place = str(tags.get("addr:place") or "").strip()

        line1 = " ".join(
            part for part in (house, street) if part
        )

        if line1:
            parts.append(line1)
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
        for key in (
            "addr:city",
            "addr:town",
            "addr:village",
            "addr:suburb",
        ):
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
        for key in (
            "addr:district",
            "addr:county",
            "is_in:district",
        ):
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
    def _source_object_url(
        element_type: str,
        element_id: Any,
    ) -> str | None:
        if not element_type or element_id is None:
            return None

        return (
            f"https://www.openstreetmap.org/"
            f"{element_type}/{element_id}"
        )

    @staticmethod
    def _is_medical_shop(tags: dict[str, Any]) -> bool:
        return (
            tags.get("amenity") == "pharmacy"
            or tags.get("healthcare") == "pharmacy"
            or tags.get("shop")
            in {"chemist", "pharmacy", "drugstore", "medical_supply"}
            or tags.get("dispensing") == "yes"
        )

    def _request_overpass(self, query: str, request_id: Optional[str] = None) -> dict[str, Any]:
        import time

        headers = {
            "User-Agent": "MediSenseAI-OverpassClient/1.0",
        }

        start_time = time.monotonic()
        last_error: Exception | None = None

        for endpoint in self.endpoints:
            elapsed = time.monotonic() - start_time
            if elapsed >= float(self.timeout_seconds):
                break
            remaining = max(0.5, float(self.timeout_seconds) - elapsed)
            timeout_cfg = httpx.Timeout(
                connect=min(0.8, remaining),
                read=remaining,
                write=min(0.8, remaining),
                pool=min(0.8, remaining),
            )

            try:
                with httpx.Client(
                    timeout=timeout_cfg,
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
                    raise ValueError(
                        "Overpass returned a non-object JSON response."
                    )

                return payload

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "OSM Overpass request failed: request_id=%s endpoint=%s error=%s",
                    request_id or "unknown",
                    endpoint,
                    exc,
                )

        if last_error is not None:
            raise last_error

        raise RuntimeError("No Overpass endpoint is configured.")

    def _normalize_elements(
        self,
        elements: list[Any],
        latitude: float,
        longitude: float,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()

        for element in elements:
            if not isinstance(element, dict):
                continue

            tags = element.get("tags") or {}
            coords = self._element_coordinates(element)
            if not isinstance(tags, dict) or not self._is_medical_shop(tags):
                logger.info(
                    "MEDICAL_SHOP_REJECT: reason=non_pharmacy_osm_element provider=osm name=%s coordinates=%s category=%s",
                    tags.get("name") or "(unnamed)" if isinstance(tags, dict) else "(unknown)",
                    coords,
                    tags.get("amenity") or tags.get("shop") or tags.get("healthcare") if isinstance(tags, dict) else None,
                )
                continue

            if coords is None:
                logger.info(
                    "MEDICAL_SHOP_REJECT: reason=missing_coordinates provider=osm name=%s coordinates=None category=%s",
                    tags.get("name") or "(unnamed)",
                    tags.get("amenity") or tags.get("shop") or tags.get("healthcare"),
                )
                continue
            item_lat, item_lon = coords

            name = str(
                tags.get("name")
                or tags.get("name:en")
                or tags.get("brand")
                or tags.get("operator")
                or tags.get("ref")
                or ""
            ).strip()
            if not name:
                name = "Medical Shop"

            element_type = str(element.get("type") or "").strip()
            element_id = element.get("id")
            if not element_type or element_id is None:
                continue

            osm_id = f"osm:{element_type}:{element_id}"
            if osm_id in seen:
                continue
            seen.add(osm_id)

            distance_km = calculate_distance_km(
                float(latitude), float(longitude), item_lat, item_lon
            )
            address = self._format_address(tags)
            city = self._city(tags)
            state = self._state(tags)
            district = self._district(tags)
            contact = self._contact(tags)
            website = self._website(tags)

            items.append({
                "shop_id": osm_id,
                "external_id": osm_id,
                "name": name,
                "location": address or city or None,
                "state": state,
                "district": district,
                "city": city,
                "address": address,
                "latitude": item_lat,
                "longitude": item_lon,
                "contact": contact,
                "source": "OpenStreetMap",
                "source_url": website or self._source_object_url(element_type, element_id),
                "last_verified": None,
                "distance_km": float(distance_km),
            })

        items.sort(key=lambda item: float(item.get("distance_km", float("inf"))))
        return items

    def _search_radius(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        request_id: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], str | None, int]:
        radius_meters = int(float(radius_km) * 1000)
        query = self._build_query(latitude, longitude, radius_meters)
        try:
            payload = self._request_overpass(query, request_id=request_id)
        except httpx.TimeoutException:
            return [], "TIMEOUT", 0
        except Exception as exc:
            return [], str(exc), 0

        elements = payload.get("elements") or []
        items = self._normalize_elements(elements, latitude, longitude)
        return items, None, len(elements)

    def search_pharmacies(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return real OSM pharmacy/medical-shop records.

        Queries bounded OSM Overpass interpreter for the specified radius.
        Returns valid records with real coordinates and distances.
        """
        try:
            lat = float(latitude)
            lon = float(longitude)
            requested_radius = float(radius_km)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": "Valid latitude, longitude and radius are required.",
                "data": [],
            }

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return {
                "status": "error",
                "message": "Invalid latitude or longitude.",
                "data": [],
            }
        if requested_radius <= 0:
            return {
                "status": "error",
                "message": "Search radius must be greater than zero.",
                "data": [],
            }

        items, error, raw_count = self._search_radius(lat, lon, requested_radius, request_id=request_id)
        accepted_count = len(items)
        rejected_count = max(0, raw_count - accepted_count)

        if error == "TIMEOUT":
            logger.warning(
                "MEDICAL_SHOP_OSM: request_id=%s radius=%s source=osm raw_count=0 accepted_count=0 rejected_count=0 nearest_distance=None status=timeout",
                request_id or "unknown",
                requested_radius,
            )
            return {
                "status": "timeout",
                "message": "OpenStreetMap search timed out.",
                "data": [],
                "search_radius_km": requested_radius,
                "raw_count": 0,
                "valid_count": 0,
                "error": "TIMEOUT",
            }
        elif error:
            logger.warning(
                "MEDICAL_SHOP_OSM: request_id=%s radius=%s source=osm raw_count=0 accepted_count=0 rejected_count=0 nearest_distance=None status=error error=%s",
                request_id or "unknown",
                requested_radius,
                error,
            )
            return {
                "status": "error",
                "message": f"OpenStreetMap search unavailable: {error}",
                "data": [],
                "search_radius_km": requested_radius,
                "raw_count": 0,
                "valid_count": 0,
                "error": error,
            }

        nearest_dist = items[0]["distance_km"] if items else None
        nearest_str = f"{nearest_dist:.3f}" if nearest_dist is not None else "None"
        osm_status = "success" if items else "no_results"

        logger.info(
            "MEDICAL_SHOP_OSM: request_id=%s radius=%s source=osm raw_count=%d accepted_count=%d rejected_count=%d nearest_distance=%s status=%s",
            request_id or "unknown",
            requested_radius,
            raw_count,
            accepted_count,
            rejected_count,
            nearest_str,
            osm_status,
        )
        return {
            "status": "success",
            "message": f"Found {len(items)} OSM medical shop(s).",
            "data": items,
            "search_radius_km": requested_radius,
            "raw_count": raw_count,
            "valid_count": len(items),
            "error": None,
        }

