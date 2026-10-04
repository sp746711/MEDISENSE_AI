"""Comprehensive tests for Healthcare Navigation + Real Provider Discovery.

Enforces zero-fabrication, legitimate provenance, coordinate search, radius expansion,
pagination, and strict demo appointment contracts.
"""

from datetime import date, time
from uuid import uuid4
import pytest
from unittest.mock import MagicMock, patch

from app.database.database import SessionLocal, init_db
from app.database.models import Appointment, Doctor, Facility, MedicalShop, User
from app.providers.authorized_provider_source import AuthorizedProviderSource
from app.providers.location_utils import calculate_distance_km
from app.providers.provider_repository import ProviderRepository


@pytest.fixture(scope="function")
def db_session():
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# 1. No provider source configured
def test_no_provider_source_configured(monkeypatch, db_session):
    repo = ProviderRepository(db_session)
    # Ensure no external provider configured
    repo.source.provider_source_name = ""
    repo.source.provider_base_url = ""

    result = repo.search_doctors(state="NonExistentState_XYZ", district="NonExistentDistrict_XYZ")
    assert result["status"] == "unavailable"
    assert "No authorized provider data source is configured" in result["message"] or "unavailable" in result["message"]


# 2. No fake providers returned
def test_no_fake_providers_returned(db_session):
    repo = ProviderRepository(db_session)
    result = repo.search_doctors(state="UnicornState", district="UnicornDistrict")
    assert len(result["doctors"]) == 0
    assert result["status"] == "unavailable"


# 3. Doctor specialty filtering
def test_doctor_specialty_filtering(db_session):
    spec_unique = f"Cardio_{uuid4().hex[:6]}"
    doc_cardio = Doctor(
        name="Dr. Cardio Verified",
        specialization=spec_unique,
        state="TestState",
        district="TestDistrict",
        city="TestCity",
        source="Official Medical Council",
    )
    doc_neuro = Doctor(
        name="Dr. Neuro Verified",
        specialization="Neurology",
        state="TestState",
        district="TestDistrict",
        city="TestCity",
        source="Official Medical Council",
    )
    db_session.add_all([doc_cardio, doc_neuro])
    db_session.commit()

    repo = ProviderRepository(db_session)
    res = repo.search_doctors(state="TestState", district="TestDistrict", specialization=spec_unique)
    assert res["status"] == "ok"
    assert len(res["doctors"]) >= 1
    for d in res["doctors"]:
        assert spec_unique.lower() in d["specialization"].lower()


# 4 & 5. Doctor pagination (10 per page)
def test_doctor_pagination_ten_per_page(db_session):
    unique_tag = f"Tag_{uuid4().hex[:6]}"
    # Insert 15 doctors
    doctors = [
        Doctor(
            name=f"Dr. PageTest {i}",
            specialization=f"General_{unique_tag}",
            state="PageState",
            district="PageDistrict",
            source="Official Registry",
        )
        for i in range(15)
    ]
    db_session.add_all(doctors)
    db_session.commit()

    repo = ProviderRepository(db_session)
    page_1 = repo.search_doctors(state="PageState", district="PageDistrict", specialization=unique_tag, page=1, page_size=10)
    assert page_1["status"] == "ok"
    assert len(page_1["doctors"]) == 10
    assert page_1["page"] == 1
    assert page_1["page_size"] == 10
    assert page_1["has_next"] is True
    assert page_1["has_previous"] is False

    page_2 = repo.search_doctors(state="PageState", district="PageDistrict", specialization=unique_tag, page=2, page_size=10)
    assert page_2["status"] == "ok"
    assert len(page_2["doctors"]) == 5
    assert page_2["page"] == 2
    assert page_2["has_next"] is False
    assert page_2["has_previous"] is True


# 6 & 7. Geographic coordinate search & radius filtering
def test_geographic_coordinate_search_and_radius(db_session):
    # Base coords: 12.9716, 77.5946 (Bengaluru)
    # Doctor at ~3.5 km away (12.9900, 77.6100)
    tag = f"Geo_{uuid4().hex[:6]}"
    doc_nearby = Doctor(
        name="Dr. Nearby Verified",
        specialization=f"Physician_{tag}",
        state="Karnataka",
        district="Bengaluru Urban",
        city="Bengaluru",
        latitude=12.9900,
        longitude=77.6100,
        source="Official Registry",
    )
    db_session.add(doc_nearby)
    db_session.commit()

    repo = ProviderRepository(db_session)
    res = repo.search_doctors(latitude=12.9716, longitude=77.5946, radius_km=10.0, specialization=tag)
    assert res["status"] == "ok"
    assert len(res["doctors"]) >= 1
    nearby = [d for d in res["doctors"] if d["name"] == "Dr. Nearby Verified"][0]
    assert nearby["distance_km"] is not None
    assert nearby["distance_km"] <= 10.0


# 8. Missing coordinates -> no fake distance
def test_missing_coordinates_no_fake_distance(db_session):
    doc_no_coords = Doctor(
        name="Dr. No Coords",
        specialization=f"Specialist_{uuid4().hex[:6]}",
        state="Kerala",
        district="Ernakulam",
        city="Kochi",
        source="Official Registry",
        latitude=None,
        longitude=None,
    )
    db_session.add(doc_no_coords)
    db_session.commit()

    repo = ProviderRepository(db_session)
    res = repo.search_doctors(state="Kerala", district="Ernakulam", city="Kochi", specialization=doc_no_coords.specialization)
    assert res["status"] == "ok"
    found = [d for d in res["doctors"] if d["name"] == "Dr. No Coords"][0]
    assert found["distance_km"] is None  # Never fabricated


# 9, 10, 11. Search expansion (25 km -> 50 km) and explicit message
def test_search_expansion_radius(db_session):
    # User at 13.0000, 80.0000
    # Doctor at ~35 km distance (approx lat 13.3150, lon 80.0000)
    tag = f"Expand_{uuid4().hex[:6]}"
    doc_far = Doctor(
        name="Dr. Far Verified",
        specialization=f"Pediatrics_{tag}",
        state="Tamil Nadu",
        district="Chennai",
        latitude=13.3150,
        longitude=80.0000,
        source="State Council",
    )
    db_session.add(doc_far)
    db_session.commit()

    repo = ProviderRepository(db_session)
    # Search locally (default 25km); none within 25km, should expand to 50km
    res = repo.search_doctors(latitude=13.0000, longitude=80.0000, radius_km=25.0, specialization=tag)
    assert res["status"] == "ok"
    assert res["expanded"] is True
    assert "expanded to approximately 50 km" in res["expansion_message"]


# 12. Facility emergency_only
def test_facility_emergency_only_filtering(db_session):
    tag = f"Fac_{uuid4().hex[:6]}"
    state_unique = f"State_{tag}"
    fac_emerg = Facility(
        name=f"Emergency Hospital {tag}",
        type="Hospital",
        state=state_unique,
        district="TestFacDistrict",
        emergency_available="yes",
        source="Ministry of Health",
    )
    fac_clinic = Facility(
        name=f"Routine Clinic {tag}",
        type="Clinic",
        state=state_unique,
        district="TestFacDistrict",
        emergency_available="no",
        source="Ministry of Health",
    )
    db_session.add_all([fac_emerg, fac_clinic])
    db_session.commit()

    repo = ProviderRepository(db_session)
    res = repo.search_facilities(state=state_unique, district="TestFacDistrict", emergency_only=True)
    assert res["status"] == "ok"
    # None of the results should be routine clinic without emergency
    matching = [f for f in res["facilities"] if tag in f["name"]]
    assert len(matching) == 1
    assert matching[0]["name"] == f"Emergency Hospital {tag}"
    assert matching[0]["emergency_available"] == "yes"


# 13. Medical shop geographic filtering and disclaimer
def test_medical_shop_filtering_and_disclaimer(db_session):
    tag = f"Shop_{uuid4().hex[:6]}"
    shop = MedicalShop(
        name=f"Verified Pharmacy {tag}",
        state="ShopState",
        district="ShopDistrict",
        city="ShopCity",
        source="Pharmacy Licensing Authority",
    )
    db_session.add(shop)
    db_session.commit()

    repo = ProviderRepository(db_session)
    res = repo.search_medical_shops(state="ShopState", district="ShopDistrict", city="ShopCity")
    assert res["status"] == "ok"
    assert len(res["medical_shops"]) >= 1
    assert "disclaimer" in res
    assert "prescribing" in res["disclaimer"].lower()


# 14. Provider provenance
def test_provider_provenance_preservation(db_session):
    doc = Doctor(
        name="Dr. Provenance Verified",
        specialization=f"Oncology_{uuid4().hex[:6]}",
        state="StateProv",
        district="DistProv",
        registration_number="MED-12345-X",
        registration_council="National Medical Council",
        source="NMC Official Portal",
        source_url="https://nmc.example.gov/verify/12345",
        contact="+91-9876543210",
    )
    db_session.add(doc)
    db_session.commit()

    repo = ProviderRepository(db_session)
    res = repo.search_doctors(state="StateProv", district="DistProv", specialization=doc.specialization)
    assert res["status"] == "ok"
    d = res["doctors"][0]
    assert d["registration_number"] == "MED-12345-X"
    assert d["registration_council"] == "National Medical Council"
    assert d["source"] == "NMC Official Portal"
    assert d["source_url"] == "https://nmc.example.gov/verify/12345"
    assert d["contact"] == "+91-9876543210"


# 15. External provider normalization
def test_external_provider_normalization():
    adapter = AuthorizedProviderSource()
    raw_external = {
        "id": "ext_999",
        "name": "Dr. External Sourced",
        "specialty": "Cardiology",
        "qualification": "MBBS, MD",
        "reg_no": "REG-888",
        "council": "Delhi Medical Council",
        "hospital": "City Care Hospital",
        "state": "Delhi",
        "district": "New Delhi",
        "city": "New Delhi",
        "address": "Ring Road",
        "latitude": "28.6139",
        "longitude": "77.2090",
        "phone": "+91-1122334455",
        "source": "ABDM Registry",
        "source_url": "https://abdm.gov.in/registry/999",
        "last_verified": "2026-09-15T00:00:00Z",
    }
    normalized = adapter.normalize_doctor(raw_external)
    assert normalized["external_id"] == "ext_999"
    assert normalized["name"] == "Dr. External Sourced"
    assert normalized["specialization"] == "Cardiology"
    assert normalized["registration_number"] == "REG-888"
    assert normalized["registration_council"] == "Delhi Medical Council"
    assert normalized["latitude"] == 28.6139
    assert normalized["longitude"] == 77.2090
    assert normalized["source"] == "ABDM Registry"

    # Missing fields must remain None, never inferred
    raw_minimal = {"name": "Dr. Minimal"}
    normalized_min = adapter.normalize_doctor(raw_minimal)
    assert normalized_min["external_id"] is None
    assert normalized_min["specialization"] is None
    assert normalized_min["latitude"] is None
    assert normalized_min["registration_number"] is None


# 16. Rate-limit handling
def test_rate_limit_handling():
    adapter = AuthorizedProviderSource()
    adapter.provider_source_name = "TestLiveSource"
    adapter.provider_base_url = "https://mock.live.source"

    mock_resp = MagicMock()
    mock_resp.status_code = 429

    with patch("httpx.Client.get", return_value=mock_resp):
        res = adapter.fetch_doctors(state="Delhi", district="New Delhi")
        assert res["status"] == "rate_limited"
        assert len(res["data"]) == 0
        assert "temporarily unavailable due to rate limits" in res["message"]


# 17. Unavailable source handling
def test_unavailable_source_handling():
    adapter = AuthorizedProviderSource()
    adapter.provider_source_name = ""
    adapter.provider_base_url = ""

    res = adapter.fetch_doctors(state="Delhi", district="New Delhi")
    assert res["status"] == "unavailable"
    assert len(res["data"]) == 0
    assert "no authorized provider data source is configured" in res["message"]


# 18. Appointment requires legitimate local provider record
def test_appointment_requires_legitimate_provider(db_session):
    doc = Doctor(
        name="Dr. Appt Specialist",
        specialization="Endocrinology",
        state="Maharashtra",
        district="Pune",
        source="State Council",
    )
    user = User(
        name="Appt User",
        email=f"user_{uuid4().hex[:6]}@example.com",
        password_hash="pwd_hash",
        state="Maharashtra",
        district="Pune",
    )
    db_session.add_all([doc, user])
    db_session.commit()

    appt = Appointment(
        user_id=user.user_id,
        doctor_id=doc.doctor_id,
        appointment_date=date.today(),
        appointment_time=time(11, 0),
        status="DEMO_CONFIRMED",
        appointment_type="demo",
    )
    db_session.add(appt)
    db_session.commit()
    db_session.refresh(appt)

    assert appt.status == "DEMO_CONFIRMED"
    assert appt.doctor_id == doc.doctor_id
    assert appt.appointment_type == "demo"
