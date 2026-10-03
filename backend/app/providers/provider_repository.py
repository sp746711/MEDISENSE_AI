"""Provider data access and search orchestration — never fabricate doctors/facilities/shops."""

from typing import Any, Optional
from uuid import UUID
import math

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.database import SessionLocal
from app.database.models import Doctor, Facility, MedicalShop
from app.providers.authorized_provider_source import AuthorizedProviderSource
from app.providers.location_utils import calculate_distance_km


class ProviderRepository:
    def __init__(self, db: Session | None = None) -> None:
        self._owns_session = db is None
        self.db = db or SessionLocal()
        self.settings = get_settings()
        self.source = AuthorizedProviderSource()

    def close(self) -> None:
        if self._owns_session:
            self.db.close()

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
    ) -> dict[str, Any]:
        """Search doctors via authorized external source if configured, or legitimate verified local DB records."""
        try:
            # 1. Check if authorized external source is configured
            if self.source.is_configured():
                external_res = self.source.fetch_doctors(
                    specialization=specialization,
                    state=state,
                    district=district,
                    city=city,
                    pin=pin,
                    latitude=latitude,
                    longitude=longitude,
                    radius_km=radius_km or self.settings.local_radius_km,
                    page=page,
                    page_size=page_size,
                    assessment_id=assessment_id,
                )
                if external_res.get("status") == "success":
                    data = external_res.get("data", [])
                    # Compute distance if user coordinates provided and item coordinates exist
                    if latitude is not None and longitude is not None:
                        for doc in data:
                            if doc.get("latitude") is not None and doc.get("longitude") is not None:
                                doc["distance_km"] = calculate_distance_km(
                                    latitude, longitude, doc["latitude"], doc["longitude"]
                                )
                            else:
                                doc["distance_km"] = None
                    return {
                        "status": "ok",
                        "doctors": data,
                        "data": data,
                        "page": external_res.get("page", page),
                        "page_size": external_res.get("page_size", page_size),
                        "has_next": external_res.get("has_next", False),
                        "has_previous": external_res.get("has_previous", False),
                        "total_count": external_res.get("total_count"),
                        "expanded": False,
                        "expansion_message": None,
                        "message": f"Found {len(data)} matching provider(s).",
                        "source_metadata": external_res.get("source_metadata", {}),
                    }
                elif external_res.get("status") == "rate_limited":
                    return external_res
                # If external source error, pass through explicit status without falling back to fake records
                elif external_res.get("status") == "error":
                    return external_res

            # 2. Local verified records query (only records that originated from legitimate sources with source IS NOT NULL)
            base_query = self.db.query(Doctor).filter(Doctor.source.isnot(None))

            if specialization and specialization.strip():
                base_query = base_query.filter(Doctor.specialization.ilike(f"%{specialization.strip()}%"))

            has_coords = latitude is not None and longitude is not None
            has_manual = bool(state or district or city or pin)

            if not has_coords and not has_manual:
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
                    "message": "Healthcare provider information is currently unavailable because no location parameters were provided.",
                    "source_metadata": {"source": None, "configured": False},
                }

            matched_records: list[tuple[Doctor, Optional[float]]] = []
            expanded = False
            expansion_message = None

            if has_coords:
                # Geographic search with radius expansion
                all_local = base_query.all()
                local_radius = radius_km or self.settings.local_radius_km
                expanded_radius = self.settings.expanded_radius_km

                # Phase 1: Local radius (25km)
                for doc in all_local:
                    if doc.latitude is not None and doc.longitude is not None:
                        dist = calculate_distance_km(latitude, longitude, doc.latitude, doc.longitude)
                        if dist is not None and dist <= local_radius:
                            matched_records.append((doc, dist))

                # Phase 2: Search expansion (50km) if 0 found locally
                if not matched_records:
                    for doc in all_local:
                        if doc.latitude is not None and doc.longitude is not None:
                            dist = calculate_distance_km(latitude, longitude, doc.latitude, doc.longitude)
                            if dist is not None and dist <= expanded_radius:
                                matched_records.append((doc, dist))
                    if matched_records:
                        expanded = True
                        spec_label = f" {specialization}" if specialization else ""
                        expansion_message = (
                            f"No matching{spec_label} provider was found within {int(local_radius)} km. "
                            f"Search expanded to approximately {int(expanded_radius)} km."
                        )

                matched_records.sort(key=lambda x: x[1] if x[1] is not None else float("inf"))

            else:
                # Manual hierarchy search with expansion
                current_query = base_query
                if state:
                    current_query = current_query.filter(Doctor.state.ilike(state.strip()))
                if district:
                    current_query = current_query.filter(Doctor.district.ilike(district.strip()))
                
                if city:
                    city_query = current_query.filter(Doctor.city.ilike(city.strip()))
                    city_rows = city_query.all()
                    if city_rows:
                        matched_records = [(d, None) for d in city_rows]
                    else:
                        # Expand to district
                        district_rows = current_query.all()
                        if district_rows:
                            matched_records = [(d, None) for d in district_rows]
                            expanded = True
                            spec_label = f" {specialization}" if specialization else ""
                            expansion_message = (
                                f"No matching{spec_label} provider was found in {city.strip()}. "
                                f"Search expanded to {district.strip()} district."
                            )
                        elif state:
                            # Expand to state
                            state_rows = base_query.filter(Doctor.state.ilike(state.strip())).all()
                            if state_rows:
                                matched_records = [(d, None) for d in state_rows]
                                expanded = True
                                spec_label = f" {specialization}" if specialization else ""
                                expansion_message = (
                                    f"No matching{spec_label} provider was found in {district.strip()}. "
                                    f"Search expanded to {state.strip()} state."
                                )
                else:
                    district_rows = current_query.all()
                    if district_rows:
                        matched_records = [(d, None) for d in district_rows]
                    elif state:
                        state_rows = base_query.filter(Doctor.state.ilike(state.strip())).all()
                        if state_rows:
                            matched_records = [(d, None) for d in state_rows]
                            expanded = True
                            spec_label = f" {specialization}" if specialization else ""
                            expansion_message = (
                                f"No matching{spec_label} provider was found in {district or 'the district'}. "
                                f"Search expanded to {state.strip()} state."
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
                    "source_metadata": {
                        "source": None,
                        "configured": False,
                    },
                }

            # Paginate results
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

    def get_doctor(self, doctor_id: UUID) -> Optional[dict[str, Any]]:
        try:
            doctor = self.db.get(Doctor, doctor_id)
            if doctor is None:
                return None
            return self._doctor_dict(doctor)
        finally:
            self.close()

    def search_facilities(
        self,
        *,
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
        assessment_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Search facilities via authorized external source if configured, or legitimate verified local DB records."""
        try:
            is_geoapify = (
                self.source.facility_source_name
                and self.source.facility_source_name.strip().lower() == "geoapify"
            )

            if is_geoapify and latitude is not None and longitude is not None:
                if page > 2:
                    return {
                        "status": "success",
                        "facilities": [],
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "has_next": False,
                        "has_previous": True,
                        "expanded": False,
                        "expansion_message": None,
                        "message": "Maximum of 2 pages reached.",
                        "source_metadata": {
                            "source": "Geoapify / OpenStreetMap",
                            "configured": True,
                            "source_url": "https://api.geoapify.com/v2/places",
                        },
                    }

                if not self.source.is_facility_configured():
                    return {
                        "status": "configuration_missing",
                        "facilities": [],
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "has_next": False,
                        "has_previous": page > 1,
                        "expanded": False,
                        "expansion_message": None,
                        "message": "Nearby healthcare search is not configured. Please set GEOAPIFY_API_KEY in the backend environment.",
                        "source_metadata": {
                            "source": "Geoapify / OpenStreetMap",
                            "configured": False,
                            "source_url": "https://api.geoapify.com/v2/places",
                        },
                    }

                external_res = self.source.fetch_facilities(
                    state=state,
                    district=district,
                    city=city,
                    pin=pin,
                    latitude=latitude,
                    longitude=longitude,
                    radius_km=radius_km or self.settings.local_radius_km,
                    emergency_only=emergency_only,
                    page=page,
                    page_size=page_size,
                    assessment_id=assessment_id,
                )
                return external_res

            elif not is_geoapify and self.source.is_facility_configured():
                external_res = self.source.fetch_facilities(
                    state=state,
                    district=district,
                    city=city,
                    pin=pin,
                    latitude=latitude,
                    longitude=longitude,
                    radius_km=radius_km or self.settings.local_radius_km,
                    emergency_only=emergency_only,
                    page=page,
                    page_size=page_size,
                    assessment_id=assessment_id,
                )
                if external_res.get("status") in ("success", "ok"):
                    data = external_res.get("data") or external_res.get("facilities") or []
                    if latitude is not None and longitude is not None:
                        for fac in data:
                            if fac.get("distance_km") is None:
                                if fac.get("latitude") is not None and fac.get("longitude") is not None:
                                    fac["distance_km"] = calculate_distance_km(
                                        latitude, longitude, fac["latitude"], fac["longitude"]
                                    )
                    return {
                        "status": "ok",
                        "facilities": data,
                        "data": data,
                        "page": external_res.get("page", page),
                        "page_size": external_res.get("page_size", page_size),
                        "has_next": external_res.get("has_next", False),
                        "has_previous": external_res.get("has_previous", False),
                        "total_count": external_res.get("total_count"),
                        "expanded": external_res.get("expanded", False),
                        "expansion_message": external_res.get("expansion_message"),
                        "message": external_res.get("message", f"Found {len(data)} matching facility(s)."),
                        "source_metadata": external_res.get("source_metadata", {}),
                    }
                return external_res

            base_query = self.db.query(Facility).filter(Facility.source.isnot(None))
            if emergency_only:
                base_query = base_query.filter(Facility.emergency_available.ilike("yes"))

            has_coords = latitude is not None and longitude is not None
            has_manual = bool(state or district or city or pin)

            if not has_coords and not has_manual:
                return {
                    "status": "unavailable",
                    "facilities": [],
                    "data": [],
                    "page": page,
                    "page_size": page_size,
                    "has_next": False,
                    "has_previous": page > 1,
                    "expanded": False,
                    "expansion_message": None,
                    "message": "Healthcare facility information is currently unavailable because no location parameters were provided.",
                    "source_metadata": {"source": None, "configured": False},
                }

            matched_records: list[tuple[Facility, Optional[float]]] = []
            expanded = False
            expansion_message = None

            if has_coords:
                all_local = base_query.all()
                local_radius = radius_km or self.settings.local_radius_km
                expanded_radius = self.settings.expanded_radius_km

                for fac in all_local:
                    if fac.latitude is not None and fac.longitude is not None:
                        dist = calculate_distance_km(latitude, longitude, fac.latitude, fac.longitude)
                        if dist is not None and dist <= local_radius:
                            matched_records.append((fac, dist))

                if not matched_records:
                    for fac in all_local:
                        if fac.latitude is not None and fac.longitude is not None:
                            dist = calculate_distance_km(latitude, longitude, fac.latitude, fac.longitude)
                            if dist is not None and dist <= expanded_radius:
                                matched_records.append((fac, dist))
                    if matched_records:
                        expanded = True
                        expansion_message = (
                            f"No matching facilities found within {int(local_radius)} km. "
                            f"Search expanded to approximately {int(expanded_radius)} km."
                        )
                matched_records.sort(key=lambda x: x[1] if x[1] is not None else float("inf"))
            else:
                current_query = base_query
                if state:
                    current_query = current_query.filter(Facility.state.ilike(state.strip()))
                if district:
                    current_query = current_query.filter(Facility.district.ilike(district.strip()))
                if city:
                    city_query = current_query.filter(Facility.city.ilike(city.strip()))
                    city_rows = city_query.all()
                    if city_rows:
                        matched_records = [(f, None) for f in city_rows]
                    else:
                        district_rows = current_query.all()
                        if district_rows:
                            matched_records = [(f, None) for f in district_rows]
                            expanded = True
                            expansion_message = (
                                f"No matching facilities found in {city.strip()}. "
                                f"Search expanded to {district.strip()} district."
                            )
                else:
                    district_rows = current_query.all()
                    matched_records = [(f, None) for f in district_rows]

            if not matched_records:
                return {
                    "status": "unavailable",
                    "facilities": [],
                    "data": [],
                    "page": page,
                    "page_size": page_size,
                    "has_next": False,
                    "has_previous": page > 1,
                    "expanded": False,
                    "expansion_message": None,
                    "message": (
                        "Healthcare facility information is currently unavailable. "
                        "No authorized facility data source is configured, and no "
                        "verified facility records match this search."
                    ),
                    "source_metadata": {"source": None, "configured": False},
                }

            total_count = len(matched_records)
            offset = (page - 1) * page_size
            paged = matched_records[offset : offset + page_size]
            has_next = (offset + page_size) < total_count
            has_previous = page > 1

            facility_list = [self._facility_dict(fac, dist) for fac, dist in paged]

            return {
                "status": "ok",
                "facilities": facility_list,
                "data": facility_list,
                "page": page,
                "page_size": page_size,
                "has_next": has_next,
                "has_previous": has_previous,
                "total_count": total_count,
                "expanded": expanded,
                "expansion_message": expansion_message,
                "message": f"Found {total_count} matching facility(s).",
                "source_metadata": {"source": "local_verified", "configured": False},
            }
        finally:
            self.close()

    def search_medical_shops(
        self,
        *,
        state: Optional[str] = None,
        district: Optional[str] = None,
        city: Optional[str] = None,
        pin: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_km: Optional[float] = None,
        page: int = 1,
        page_size: int = 10,
        assessment_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Search medical shops via authorized external source if configured, or legitimate verified local DB records."""
        try:
            is_geoapify = (
                self.source.shop_source_name
                and self.source.shop_source_name.strip().lower() == "geoapify"
            )

            if is_geoapify and latitude is not None and longitude is not None:
                if page > 2:
                    return {
                        "status": "success",
                        "medical_shops": [],
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "has_next": False,
                        "has_previous": True,
                        "expanded": False,
                        "expansion_message": None,
                        "message": "Maximum of 2 pages reached.",
                        "disclaimer": "Medical shop listings do not include medication prescribing or dosage advice.",
                        "source_metadata": {
                            "source": "Geoapify / OpenStreetMap",
                            "configured": True,
                            "source_url": "https://api.geoapify.com/v2/places",
                        },
                    }

                if not self.source.is_shop_configured():
                    return {
                        "status": "configuration_missing",
                        "medical_shops": [],
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "has_next": False,
                        "has_previous": page > 1,
                        "expanded": False,
                        "expansion_message": None,
                        "message": "Nearby healthcare search is not configured. Please set GEOAPIFY_API_KEY in the backend environment.",
                        "disclaimer": "Medical shop listings do not include medication prescribing or dosage advice.",
                        "source_metadata": {
                            "source": "Geoapify / OpenStreetMap",
                            "configured": False,
                            "source_url": "https://api.geoapify.com/v2/places",
                        },
                    }

                external_res = self.source.fetch_medical_shops(
                    state=state,
                    district=district,
                    city=city,
                    pin=pin,
                    latitude=latitude,
                    longitude=longitude,
                    radius_km=radius_km or self.settings.local_radius_km,
                    page=page,
                    page_size=page_size,
                    assessment_id=assessment_id,
                )
                return external_res

            elif not is_geoapify and self.source.is_shop_configured():
                external_res = self.source.fetch_medical_shops(
                    state=state,
                    district=district,
                    city=city,
                    pin=pin,
                    latitude=latitude,
                    longitude=longitude,
                    radius_km=radius_km or self.settings.local_radius_km,
                    page=page,
                    page_size=page_size,
                    assessment_id=assessment_id,
                )
                if external_res.get("status") in ("success", "ok"):
                    data = external_res.get("data") or external_res.get("medical_shops") or []
                    if latitude is not None and longitude is not None:
                        for shop in data:
                            if shop.get("distance_km") is None:
                                if shop.get("latitude") is not None and shop.get("longitude") is not None:
                                    shop["distance_km"] = calculate_distance_km(
                                        latitude, longitude, shop["latitude"], shop["longitude"]
                                    )
                    return {
                        "status": "ok",
                        "medical_shops": data,
                        "data": data,
                        "page": external_res.get("page", page),
                        "page_size": external_res.get("page_size", page_size),
                        "has_next": external_res.get("has_next", False),
                        "has_previous": external_res.get("has_previous", False),
                        "total_count": external_res.get("total_count"),
                        "expanded": external_res.get("expanded", False),
                        "expansion_message": external_res.get("expansion_message"),
                        "message": external_res.get("message", f"Found {len(data)} matching medical shop(s)."),
                        "disclaimer": "Medical shop listings do not include medication prescribing or dosage advice.",
                        "source_metadata": external_res.get("source_metadata", {}),
                    }
                return external_res

            base_query = self.db.query(MedicalShop).filter(MedicalShop.source.isnot(None))

            has_coords = latitude is not None and longitude is not None
            has_manual = bool(state or district or city or pin)

            if not has_coords and not has_manual:
                return {
                    "status": "unavailable",
                    "medical_shops": [],
                    "data": [],
                    "page": page,
                    "page_size": page_size,
                    "has_next": False,
                    "has_previous": page > 1,
                    "expanded": False,
                    "expansion_message": None,
                    "message": "Medical shop information is currently unavailable because no location parameters were provided.",
                    "disclaimer": "Medical shop listings do not include medication prescribing or dosage advice.",
                    "source_metadata": {"source": None, "configured": False},
                }

            matched_records: list[tuple[MedicalShop, Optional[float]]] = []
            expanded = False
            expansion_message = None

            if has_coords:
                all_local = base_query.all()
                local_radius = radius_km or self.settings.local_radius_km
                expanded_radius = self.settings.expanded_radius_km

                for shop in all_local:
                    if shop.latitude is not None and shop.longitude is not None:
                        dist = calculate_distance_km(latitude, longitude, shop.latitude, shop.longitude)
                        if dist is not None and dist <= local_radius:
                            matched_records.append((shop, dist))

                if not matched_records:
                    for shop in all_local:
                        if shop.latitude is not None and shop.longitude is not None:
                            dist = calculate_distance_km(latitude, longitude, shop.latitude, shop.longitude)
                            if dist is not None and dist <= expanded_radius:
                                matched_records.append((shop, dist))
                    if matched_records:
                        expanded = True
                        expansion_message = (
                            f"No matching medical shops found within {int(local_radius)} km. "
                            f"Search expanded to approximately {int(expanded_radius)} km."
                        )
                matched_records.sort(key=lambda x: x[1] if x[1] is not None else float("inf"))
            else:
                current_query = base_query
                if state:
                    current_query = current_query.filter(MedicalShop.state.ilike(state.strip()))
                if district:
                    current_query = current_query.filter(MedicalShop.district.ilike(district.strip()))
                if city:
                    city_query = current_query.filter(MedicalShop.city.ilike(city.strip()))
                    city_rows = city_query.all()
                    if city_rows:
                        matched_records = [(s, None) for s in city_rows]
                    else:
                        district_rows = current_query.all()
                        if district_rows:
                            matched_records = [(s, None) for s in district_rows]
                            expanded = True
                            expansion_message = (
                                f"No matching medical shops found in {city.strip()}. "
                                f"Search expanded to {district.strip()} district."
                            )
                else:
                    district_rows = current_query.all()
                    matched_records = [(s, None) for s in district_rows]

            if not matched_records:
                return {
                    "status": "unavailable",
                    "medical_shops": [],
                    "data": [],
                    "page": page,
                    "page_size": page_size,
                    "has_next": False,
                    "has_previous": page > 1,
                    "expanded": False,
                    "expansion_message": None,
                    "message": (
                        "Medical shop information is currently unavailable for this search. "
                        "No authorized medical shop data source is configured, and no "
                        "verified medical shop records match this search."
                    ),
                    "disclaimer": "Medical shop listings do not include medication prescribing or dosage advice.",
                    "source_metadata": {"source": None, "configured": False},
                }

            total_count = len(matched_records)
            offset = (page - 1) * page_size
            paged = matched_records[offset : offset + page_size]
            has_next = (offset + page_size) < total_count
            has_previous = page > 1

            shop_list = [self._medical_shop_dict(s, dist) for s, dist in paged]

            return {
                "status": "ok",
                "medical_shops": shop_list,
                "data": shop_list,
                "page": page,
                "page_size": page_size,
                "has_next": has_next,
                "has_previous": has_previous,
                "total_count": total_count,
                "expanded": expanded,
                "expansion_message": expansion_message,
                "message": f"Found {total_count} matching medical shop(s).",
                "disclaimer": "Medical shop listings do not include medication prescribing or dosage advice.",
                "source_metadata": {"source": "local_verified", "configured": False},
            }
        finally:
            self.close()

    @staticmethod
    def _doctor_dict(d: Doctor, distance_km: Optional[float] = None) -> dict[str, Any]:
        return {
            "doctor_id": str(d.doctor_id),
            "external_id": getattr(d, "external_id", None),
            "name": d.name,
            "specialization": d.specialization,
            "qualification": d.qualification,
            "registration_number": getattr(d, "registration_number", None),
            "registration_council": getattr(d, "registration_council", None),
            "facility": d.facility,
            "state": d.state,
            "district": d.district,
            "city": d.city,
            "address": d.address,
            "latitude": getattr(d, "latitude", None),
            "longitude": getattr(d, "longitude", None),
            "contact": getattr(d, "contact", None),
            "source": d.source,
            "source_url": getattr(d, "source_url", None),
            "last_verified": d.last_verified.isoformat() if d.last_verified else None,
            "distance_km": round(distance_km, 2) if distance_km is not None else None,
        }

    @staticmethod
    def _facility_dict(f: Facility, distance_km: Optional[float] = None) -> dict[str, Any]:
        return {
            "facility_id": str(f.facility_id),
            "name": f.name,
            "type": f.type,
            "state": f.state,
            "district": f.district,
            "city": f.city,
            "address": f.address,
            "latitude": getattr(f, "latitude", None),
            "longitude": getattr(f, "longitude", None),
            "emergency_available": f.emergency_available,
            "contact": getattr(f, "contact", None),
            "source": f.source,
            "source_url": getattr(f, "source_url", None),
            "last_verified": f.last_verified.isoformat() if f.last_verified else None,
            "distance_km": round(distance_km, 2) if distance_km is not None else None,
        }

    @staticmethod
    def _medical_shop_dict(s: MedicalShop, distance_km: Optional[float] = None) -> dict[str, Any]:
        return {
            "shop_id": str(s.shop_id),
            "name": s.name,
            "location": s.location,
            "state": s.state,
            "district": s.district,
            "city": s.city,
            "address": s.address,
            "latitude": getattr(s, "latitude", None),
            "longitude": getattr(s, "longitude", None),
            "contact": getattr(s, "contact", None),
            "source": s.source,
            "source_url": getattr(s, "source_url", None),
            "last_verified": s.last_verified.isoformat() if s.last_verified else None,
            "distance_km": round(distance_km, 2) if distance_km is not None else None,
        }
