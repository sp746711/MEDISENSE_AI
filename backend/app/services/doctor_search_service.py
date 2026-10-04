"""DoctorSearchService — unified search orchestration for healthcare providers.

Handles BOTH:
1. Assessment-based provider discovery (auto-filled specialty, assessment location context)
2. Dashboard direct provider search (manual specialty, state, district, city, pin)

Enforces zero-fabrication, legitimate provenance via Google Places API (New) or
verified registry database records.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.database import SessionLocal
from app.database.models import Assessment, Doctor
from app.providers.google_places_provider import (
    GooglePlacesClient,
    normalize_specialty_term,
)
from app.providers.location_utils import calculate_distance_km

logger = logging.getLogger(__name__)


class DoctorSearchService:
    """Unified service for doctor discovery across assessment and dashboard workflows."""

    def __init__(
        self,
        db: Optional[Session] = None,
        google_client: Optional[GooglePlacesClient] = None,
    ) -> None:
        self._owns_session = db is None
        self.db = db or SessionLocal()
        self.settings = get_settings()
        self.google_client = google_client or GooglePlacesClient()

    def close(self) -> None:
        if self._owns_session:
            self.db.close()

    def _resolve_assessment_location(
        self, assessment_id: str
    ) -> tuple[Optional[float], Optional[float], Optional[str], Optional[str], Optional[str]]:
        """Resolve location from assessment context according to strict priority rules.

        Priority:
        1. Assessment confirmed coordinates
        2. Explicit verified assessment location
        3. None (caller falls back to manual search parameters)

        Do NOT combine inconsistent profile location fields.
        Do NOT invent coordinates.
        """
        try:
            val_uuid = UUID(assessment_id.strip())
        except (ValueError, AttributeError):
            return None, None, None, None, None

        assessment = self.db.get(Assessment, val_uuid)
        if not assessment:
            return None, None, None, None, None

        payload = assessment.result_payload or {}

        # Priority 1: Assessment confirmed coordinates
        confirmed_coords = (
            payload.get("confirmed_coordinates")
            or payload.get("coordinates")
            or (payload.get("location") if isinstance(payload.get("location"), dict) and "latitude" in payload.get("location", {}) else None)
        )
        if isinstance(confirmed_coords, dict):
            lat = confirmed_coords.get("latitude")
            lon = confirmed_coords.get("longitude")
            if lat is not None and lon is not None:
                try:
                    return float(lat), float(lon), None, None, None
                except (ValueError, TypeError):
                    pass

        # Priority 2: Explicit verified assessment location
        loc_obj = payload.get("location")
        if isinstance(loc_obj, dict):
            state = loc_obj.get("state")
            district = loc_obj.get("district")
            city = loc_obj.get("city")
            if state or district or city:
                return None, None, state, district, city

        # Check assessment's user location if registered consistently
        if assessment.user:
            return None, None, assessment.user.state, assessment.user.district, assessment.user.city

        return None, None, None, None, None

    def _cache_google_places(
        self,
        places: list[dict[str, Any]],
        fallback_state: Optional[str] = None,
        fallback_district: Optional[str] = None,
    ) -> None:
        """Cache real Google Places records in local DB to support demo appointment bookings."""
        if not places:
            return

        try:
            for p in places:
                doc_id_str = p.get("doctor_id")
                if not doc_id_str:
                    continue
                try:
                    doc_uuid = UUID(doc_id_str)
                except ValueError:
                    continue

                existing = self.db.get(Doctor, doc_uuid)
                if not existing:
                    new_doc = Doctor(
                        doctor_id=doc_uuid,
                        external_id=p.get("external_id"),
                        name=p.get("name") or "Medical Provider",
                        specialization=p.get("specialization") or "General Physician",
                        qualification=None,  # Never invent qualifications
                        facility=p.get("facility"),
                        state=p.get("state") or fallback_state or "Unknown",
                        district=p.get("district") or fallback_district or "Unknown",
                        city=p.get("city"),
                        address=p.get("address"),
                        latitude=p.get("latitude"),
                        longitude=p.get("longitude"),
                        contact=p.get("contact"),
                        source="Google Places",
                        source_url=p.get("source_url"),
                        registration_number=None,
                        registration_council=None,
                        last_verified=None,
                    )
                    self.db.add(new_doc)
            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            logger.warning("Could not cache Google Places providers: %s", exc)

    def search_doctors(
        self,
        *,
        state: Optional[str] = None,
        district: Optional[str] = None,
        city: Optional[str] = None,
        pin: Optional[str] = None,
        specialization: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_km: Optional[float] = None,
        page: int = 1,
        page_size: int = 10,
        assessment_id: Optional[str] = None,
        page_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Unified doctor search handling both Assessment and Dashboard flows."""
        try:
            # 1. Resolve Assessment Location Context if assessment_id is provided
            eff_lat = latitude
            eff_lon = longitude
            eff_state = state
            eff_district = district
            eff_city = city

            if assessment_id:
                ass_lat, ass_lon, ass_state, ass_dist, ass_city = self._resolve_assessment_location(assessment_id)
                # Priority 1: Confirmed coordinates
                if ass_lat is not None and ass_lon is not None:
                    eff_lat = ass_lat
                    eff_lon = ass_lon
                # Priority 2: Verified assessment location if manual is absent
                elif not (eff_state or eff_district or eff_city):
                    eff_state = ass_state
                    eff_district = ass_dist
                    eff_city = ass_city

            # 2. Controlled specialty mapping
            normalized_specialty = normalize_specialty_term(specialization)
            search_term = normalized_specialty or "Doctor"

            # 3. Use Google Places API (New) if configured
            if self.google_client.is_configured():
                has_coords = eff_lat is not None and eff_lon is not None
                initial_radius = radius_km or self.settings.local_radius_km

                if has_coords:
                    # Coordinate-based search
                    res = self.google_client.search_text(
                        query=search_term,
                        latitude=eff_lat,
                        longitude=eff_lon,
                        radius_km=initial_radius,
                        page_size=page_size,
                        page_token=page_token,
                        specialization=normalized_specialty,
                        page=page,
                    )

                    # If 0 results within local radius (25km), expand to 50km (Requirement 7)
                    if (
                        res.get("status") == "success"
                        and not res.get("doctors")
                        and initial_radius < self.settings.expanded_radius_km
                    ):
                        expanded_radius = self.settings.expanded_radius_km
                        expanded_res = self.google_client.search_text(
                            query=search_term,
                            latitude=eff_lat,
                            longitude=eff_lon,
                            radius_km=expanded_radius,
                            page_size=page_size,
                            page_token=page_token,
                            specialization=normalized_specialty,
                            page=page,
                        )
                        if expanded_res.get("status") == "success" and expanded_res.get("doctors"):
                            expanded_res["expanded"] = True
                            spec_label = f" {normalized_specialty}" if normalized_specialty else ""
                            expanded_res["expansion_message"] = (
                                f"No matching{spec_label} provider was found within {int(initial_radius)} km. "
                                f"Search expanded to approximately {int(expanded_radius)} km."
                            )
                            self._cache_google_places(expanded_res.get("doctors", []), eff_state, eff_district)
                            return expanded_res

                    if res.get("status") == "success":
                        self._cache_google_places(res.get("doctors", []), eff_state, eff_district)

                    return res

                else:
                    # Manual location search (Dashboard flow)
                    loc_parts = [p.strip() for p in [eff_city, eff_district, eff_state, pin] if p and p.strip()]
                    if not loc_parts:
                        return {
                            "status": "unavailable",
                            "doctors": [],
                            "data": [],
                            "page": page,
                            "page_size": page_size,
                            "has_next": False,
                            "has_previous": page > 1,
                            "total_count": None,
                            "expanded": False,
                            "expansion_message": None,
                            "message": (
                                "Healthcare provider information is currently unavailable because "
                                "no location parameters (coordinates or State/District) were provided."
                            ),
                            "source_metadata": {"source": "Google Places", "configured": True},
                        }

                    query_str = f"{search_term} in {', '.join(loc_parts)}"
                    res = self.google_client.search_text(
                        query=query_str,
                        page_size=page_size,
                        page_token=page_token,
                        specialization=normalized_specialty,
                        page=page,
                    )
                    if res.get("status") == "success":
                        self._cache_google_places(res.get("doctors", []), eff_state, eff_district)
                    return res

            # 4. If Google Places is NOT configured, query ONLY legitimate local verified DB records
            base_query = self.db.query(Doctor).filter(Doctor.source.isnot(None))

            if specialization and specialization.strip():
                spec_clean = specialization.strip()
                if normalized_specialty and normalized_specialty.lower() != spec_clean.lower():
                    from sqlalchemy import or_
                    base_query = base_query.filter(
                        or_(
                            Doctor.specialization.ilike(f"%{spec_clean}%"),
                            Doctor.specialization.ilike(f"%{normalized_specialty}%"),
                        )
                    )
                else:
                    base_query = base_query.filter(Doctor.specialization.ilike(f"%{spec_clean}%"))
            elif normalized_specialty:
                base_query = base_query.filter(Doctor.specialization.ilike(f"%{normalized_specialty}%"))

            has_coords = eff_lat is not None and eff_lon is not None
            has_manual = bool(eff_state or eff_district or eff_city or pin)

            if not has_coords and not has_manual:
                return {
                    "status": "configuration_missing",
                    "doctors": [],
                    "data": [],
                    "page": page,
                    "page_size": page_size,
                    "has_next": False,
                    "has_previous": page > 1,
                    "expanded": False,
                    "expansion_message": None,
                    "message": (
                        "Google Places provider discovery is not configured, and no location parameters were provided."
                    ),
                    "source_metadata": {"source": None, "configured": False},
                }

            matched_records: list[tuple[Doctor, Optional[float]]] = []
            expanded = False
            expansion_message = None

            if has_coords:
                all_local = base_query.all()
                local_radius = radius_km or self.settings.local_radius_km
                expanded_radius = self.settings.expanded_radius_km

                for doc in all_local:
                    if doc.latitude is not None and doc.longitude is not None:
                        dist = calculate_distance_km(eff_lat, eff_lon, doc.latitude, doc.longitude)
                        if dist is not None and dist <= local_radius:
                            matched_records.append((doc, dist))

                if not matched_records:
                    for doc in all_local:
                        if doc.latitude is not None and doc.longitude is not None:
                            dist = calculate_distance_km(eff_lat, eff_lon, doc.latitude, doc.longitude)
                            if dist is not None and dist <= expanded_radius:
                                matched_records.append((doc, dist))
                    if matched_records:
                        expanded = True
                        spec_label = f" {normalized_specialty or specialization}" if (normalized_specialty or specialization) else ""
                        expansion_message = (
                            f"No matching{spec_label} provider was found within {int(local_radius)} km. "
                            f"Search expanded to approximately {int(expanded_radius)} km."
                        )

                matched_records.sort(key=lambda x: x[1] if x[1] is not None else float("inf"))

            else:
                current_query = base_query
                if eff_state:
                    current_query = current_query.filter(Doctor.state.ilike(eff_state.strip()))
                if eff_district:
                    current_query = current_query.filter(Doctor.district.ilike(eff_district.strip()))

                if eff_city:
                    city_query = current_query.filter(Doctor.city.ilike(eff_city.strip()))
                    city_rows = city_query.all()
                    if city_rows:
                        matched_records = [(d, None) for d in city_rows]
                    else:
                        district_rows = current_query.all()
                        if district_rows:
                            matched_records = [(d, None) for d in district_rows]
                            expanded = True
                            spec_label = f" {normalized_specialty or specialization}" if (normalized_specialty or specialization) else ""
                            expansion_message = (
                                f"No matching{spec_label} provider was found in {eff_city.strip()}. "
                                f"Search expanded to {eff_district.strip()} district."
                            )
                else:
                    district_rows = current_query.all()
                    if district_rows:
                        matched_records = [(d, None) for d in district_rows]
                    elif eff_state:
                        state_rows = base_query.filter(Doctor.state.ilike(eff_state.strip())).all()
                        if state_rows:
                            matched_records = [(d, None) for d in state_rows]
                            expanded = True
                            spec_label = f" {normalized_specialty or specialization}" if (normalized_specialty or specialization) else ""
                            expansion_message = (
                                f"No matching{spec_label} provider was found in {eff_district or 'the district'}. "
                                f"Search expanded to {eff_state.strip()} state."
                            )

            if not matched_records:
                return {
                    "status": "unavailable",
                    "doctors": [],
                    "data": [],
                    "page": page,
                    "page_size": page_size,
                    "has_next": False,
                    "has_previous": page > 1,
                    "expanded": False,
                    "expansion_message": None,
                    "message": (
                        "Healthcare provider information is currently unavailable. "
                        "No authorized provider data source is configured, and no "
                        "verified provider records match this search."
                    ),
                    "source_metadata": {"source": None, "configured": False},
                }

            total_count = len(matched_records)
            offset = (page - 1) * page_size
            paged = matched_records[offset : offset + page_size]
            has_next = (offset + page_size) < total_count
            has_previous = page > 1

            doctor_list = [self._doctor_dict(doc, dist) for doc, dist in paged]

            return {
                "status": "ok",
                "doctors": doctor_list,
                "data": doctor_list,
                "page": page,
                "page_size": page_size,
                "has_next": has_next,
                "has_previous": has_previous,
                "total_count": total_count,
                "expanded": expanded,
                "expansion_message": expansion_message,
                "message": f"Found {total_count} matching provider(s).",
                "source_metadata": {
                    "source": "local_verified",
                    "configured": False,
                },
            }

        finally:
            self.close()

    def get_doctor(self, doctor_id: str | UUID) -> Optional[dict[str, Any]]:
        """Retrieve single doctor by UUID or external Google Place ID."""
        try:
            doc_uuid: Optional[UUID] = None
            if isinstance(doctor_id, UUID):
                doc_uuid = doctor_id
            elif isinstance(doctor_id, str):
                try:
                    doc_uuid = UUID(doctor_id.strip())
                except ValueError:
                    doc_uuid = None

            # 1. Lookup by UUID in DB
            if doc_uuid:
                doctor = self.db.get(Doctor, doc_uuid)
                if doctor:
                    return self._doctor_dict(doctor)

            # 2. Lookup by external_id in DB
            str_id = str(doctor_id).strip()
            existing_by_ext = self.db.query(Doctor).filter(Doctor.external_id == str_id).first()
            if existing_by_ext:
                return self._doctor_dict(existing_by_ext)

            # 3. Lookup in Google Places API (New) if configured
            if self.google_client.is_configured():
                res = self.google_client.get_place_details(str_id)
                if res.get("status") == "success" and res.get("doctor"):
                    doc_dict = res["doctor"]
                    self._cache_google_places([doc_dict])
                    return doc_dict

            return None
        finally:
            self.close()

    @staticmethod
    def _doctor_dict(d: Doctor, distance_km: Optional[float] = None) -> dict[str, Any]:
        """Convert Doctor ORM model into canonical API response."""
        verification = "verified" if d.last_verified and d.registration_number else "not_verified"
        return {
            "doctor_id": str(d.doctor_id),
            "external_id": getattr(d, "external_id", None) or str(d.doctor_id),
            "name": d.name,
            "specialization": d.specialization,
            "entity_type": "doctor",
            "qualification": getattr(d, "qualification", None),
            "registration_number": getattr(d, "registration_number", None),
            "registration_council": getattr(d, "registration_council", None),
            "facility": getattr(d, "facility", None),
            "state": d.state,
            "district": d.district,
            "city": getattr(d, "city", None),
            "address": getattr(d, "address", None),
            "latitude": getattr(d, "latitude", None),
            "longitude": getattr(d, "longitude", None),
            "contact": getattr(d, "contact", None),
            "website": None,
            "source": d.source,
            "source_url": getattr(d, "source_url", None),
            "last_verified": d.last_verified.isoformat() if d.last_verified else None,
            "verification_status": verification,
            "distance_km": round(distance_km, 2) if distance_km is not None else None,
        }
