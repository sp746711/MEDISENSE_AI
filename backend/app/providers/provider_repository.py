"""Provider data access — never fabricate doctors/facilities/shops."""

from typing import Any, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.database import SessionLocal
from app.database.models import Doctor, Facility, MedicalShop


class ProviderRepository:
    def __init__(self, db: Session | None = None) -> None:
        self._owns_session = db is None
        self.db = db or SessionLocal()
        self.settings = get_settings()

    def close(self) -> None:
        if self._owns_session:
            self.db.close()

    def search_doctors(
        self,
        *,
        state: str,
        district: str,
        city: Optional[str] = None,
        pin: Optional[str] = None,
        specialization: Optional[str] = None,
    ) -> dict[str, Any]:
        try:
            if not self.settings.provider_data_source:
                # Still allow DB-backed records if any were imported from a legitimate source
                query = self.db.query(Doctor).filter(
                    Doctor.state.ilike(state.strip()),
                    Doctor.district.ilike(district.strip()),
                )
                if city:
                    query = query.filter(Doctor.city.ilike(city.strip()))
                if specialization:
                    query = query.filter(Doctor.specialization.ilike(f"%{specialization.strip()}%"))
                rows = query.limit(50).all()
                if not rows:
                    return {
                        "status": "unavailable",
                        "doctors": [],
                        "search_scope": {"state": state, "district": district, "city": city, "pin": pin},
                        "expanded": False,
                        "message": (
                            "Healthcare provider information is currently unavailable. "
                            "No legitimate provider data source is configured, and no "
                            "verified provider records match this search."
                        ),
                    }
                return {
                    "status": "ok",
                    "doctors": [self._doctor_dict(d) for d in rows],
                    "search_scope": {"state": state, "district": district, "city": city, "pin": pin},
                    "expanded": False,
                    "message": f"Found {len(rows)} provider(s) in the selected area.",
                }
            return {
                "status": "unavailable",
                "doctors": [],
                "message": "External provider data source adapter is not yet connected.",
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
        state: str,
        district: str,
        city: Optional[str] = None,
        emergency_only: bool = False,
    ) -> dict[str, Any]:
        try:
            query = self.db.query(Facility).filter(
                Facility.state.ilike(state.strip()),
                Facility.district.ilike(district.strip()),
            )
            if city:
                query = query.filter(Facility.city.ilike(city.strip()))
            if emergency_only:
                query = query.filter(Facility.emergency_available.ilike("yes"))
            rows = query.limit(50).all()
            if not rows:
                return {
                    "status": "unavailable",
                    "facilities": [],
                    "message": (
                        "Healthcare facility information is currently unavailable "
                        "for this search."
                    ),
                    "search_scope": {"state": state, "district": district, "city": city},
                    "expanded": False,
                }
            return {
                "status": "ok",
                "facilities": [
                    {
                        "facility_id": str(f.facility_id),
                        "name": f.name,
                        "type": f.type,
                        "state": f.state,
                        "district": f.district,
                        "city": f.city,
                        "address": f.address,
                        "emergency_available": f.emergency_available,
                        "source": f.source,
                        "last_verified": f.last_verified.isoformat() if f.last_verified else None,
                    }
                    for f in rows
                ],
                "search_scope": {"state": state, "district": district, "city": city},
                "expanded": False,
            }
        finally:
            self.close()

    def search_medical_shops(
        self,
        *,
        state: str,
        district: str,
        city: Optional[str] = None,
    ) -> dict[str, Any]:
        try:
            query = self.db.query(MedicalShop).filter(
                MedicalShop.state.ilike(state.strip()),
                MedicalShop.district.ilike(district.strip()),
            )
            if city:
                query = query.filter(MedicalShop.city.ilike(city.strip()))
            rows = query.limit(50).all()
            if not rows:
                return {
                    "status": "unavailable",
                    "medical_shops": [],
                    "message": (
                        "Medical shop information is currently unavailable for this search. "
                        "This system does not prescribe medicines."
                    ),
                    "search_scope": {"state": state, "district": district, "city": city},
                }
            return {
                "status": "ok",
                "medical_shops": [
                    {
                        "shop_id": str(s.shop_id),
                        "name": s.name,
                        "location": s.location,
                        "state": s.state,
                        "district": s.district,
                        "city": s.city,
                        "address": s.address,
                        "source": s.source,
                        "last_verified": s.last_verified.isoformat() if s.last_verified else None,
                    }
                    for s in rows
                ],
                "disclaimer": "Medical shop listings do not include medication prescribing or dosage advice.",
            }
        finally:
            self.close()

    @staticmethod
    def _doctor_dict(d: Doctor) -> dict[str, Any]:
        return {
            "doctor_id": str(d.doctor_id),
            "name": d.name,
            "specialization": d.specialization,
            "qualification": d.qualification,
            "facility": d.facility,
            "state": d.state,
            "district": d.district,
            "city": d.city,
            "address": d.address,
            "source": d.source,
            "last_verified": d.last_verified.isoformat() if d.last_verified else None,
        }
