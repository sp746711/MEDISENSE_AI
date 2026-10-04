"""Comprehensive test suite for Google Places Healthcare Provider Discovery.

Covers all 24 acceptance criteria:
1. Google Places configuration missing
2. Google Places configured
3. Successful doctor discovery
4. Specialty search
5. General Physician search
6. Cardiologist search
7. Location restriction
8. Multiple providers
9. Distance calculation
10. Missing optional fields
11. 401 authorization error
12. 403 access denied
13. 429 rate limited
14. Timeout
15. Malformed response
16. No results
17. assessment_id context
18. Specialization auto-fill
19. Assessment coordinates used
20. Dashboard manual search
21. No fake provider fallback
22. API key never appears in response/logging
23. Clinic is not mislabeled as individual doctor
24. Unverified provider is not shown as NMC verified
"""

import json
import logging
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from app.database.database import SessionLocal, init_db
from app.database.models import Assessment, Doctor, User
from app.providers.google_places_provider import (
    GooglePlacesClient,
    determine_entity_type,
    normalize_specialty_term,
)
from app.services.doctor_search_service import DoctorSearchService


@pytest.fixture(scope="function")
def db_session():
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# 1. Google Places configuration missing
def test_google_places_configuration_missing():
    client = GooglePlacesClient(api_key="")
    assert not client.is_configured()
    res = client.search_text(query="General Physician")
    assert res["status"] == "configuration_missing"
    assert "not configured" in res["message"]
    assert len(res["doctors"]) == 0
    assert res["source_metadata"]["configured"] is False


# 2. Google Places configured
def test_google_places_configured():
    client = GooglePlacesClient(api_key="test_mock_api_key_123")
    assert client.is_configured()
    assert client.api_key == "test_mock_api_key_123"


# 3. Successful doctor discovery
def test_successful_doctor_discovery():
    client = GooglePlacesClient(api_key="test_mock_api_key_123")
    mock_payload = {
        "places": [
            {
                "id": "ChIJ_mock_doc_1",
                "displayName": {"text": "Dr. Ramesh Gupta", "languageCode": "en"},
                "formattedAddress": "123 Park Street, Kolkata, West Bengal 700016, India",
                "location": {"latitude": 22.5510, "longitude": 88.3520},
                "nationalPhoneNumber": "033 2222 1111",
                "websiteUri": "https://drgupta.example.com",
                "types": ["doctor", "health", "point_of_interest"],
                "businessStatus": "OPERATIONAL",
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.search_text(
            query="General Physician",
            latitude=22.5726,
            longitude=88.3639,
            specialization="General Physician",
        )
        assert res["status"] == "success"
        assert len(res["doctors"]) == 1
        doc = res["doctors"][0]
        assert doc["name"] == "Dr. Ramesh Gupta"
        assert doc["external_id"] == "ChIJ_mock_doc_1"
        assert doc["source"] == "Google Places"
        assert doc["verification_status"] == "not_verified"
        assert doc["entity_type"] == "doctor"
        assert doc["contact"] == "033 2222 1111"
        assert doc["website"] == "https://drgupta.example.com"
        assert doc["distance_km"] is not None


# 4. Specialty search
def test_specialty_search():
    assert normalize_specialty_term("Dermatology") == "Dermatologist"
    assert normalize_specialty_term("Pulmonology") == "Pulmonologist"
    assert normalize_specialty_term("Neurology") == "Neurologist"
    assert normalize_specialty_term("Orthopedics") == "Orthopedic doctor"
    assert normalize_specialty_term("ENT") == "ENT doctor"
    assert normalize_specialty_term("Ophthalmology") == "Ophthalmologist"
    assert normalize_specialty_term("Gastroenterology") == "Gastroenterologist"


# 5. General Physician search
def test_general_physician_search():
    assert normalize_specialty_term("General Physician") == "General Physician"
    assert normalize_specialty_term("Internal Medicine") == "General Physician"
    assert normalize_specialty_term("general practice") == "General Physician"


# 6. Cardiologist search
def test_cardiologist_search():
    assert normalize_specialty_term("Cardiology") == "Cardiologist"
    assert normalize_specialty_term("Cardiologist") == "Cardiologist"


# 7. Location restriction
def test_location_restriction():
    client = GooglePlacesClient(api_key="test_mock_api_key_123")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"places": []}

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        client.search_text(
            query="Cardiologist",
            latitude=12.9716,
            longitude=77.5946,
            radius_km=15,
        )
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        assert "locationRestriction" in body
        circle = body["locationRestriction"]["circle"]
        assert circle["center"]["latitude"] == 12.9716
        assert circle["center"]["longitude"] == 77.5946
        assert circle["radius"] == 15000.0


# 8. Multiple providers
def test_multiple_providers():
    client = GooglePlacesClient(api_key="test_mock_api_key_123")
    mock_payload = {
        "places": [
            {
                "id": f"place_{i}",
                "displayName": {"text": f"Dr. Doctor {i}"},
                "formattedAddress": f"Address {i}",
                "types": ["doctor"],
            }
            for i in range(10)
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.search_text(query="General Physician")
        assert res["status"] == "success"
        assert len(res["doctors"]) == 10
        # Check that total_count is not fabricated
        assert res["total_count"] is None


# 9. Distance calculation
def test_distance_calculation():
    client = GooglePlacesClient(api_key="test_mock_api_key_123")
    # User at 22.5726, 88.3639
    # Place at 22.5800, 88.3700 (straight-line ~1.05 km)
    mock_payload = {
        "places": [
            {
                "id": "place_dist_1",
                "displayName": {"text": "Dr. Close"},
                "location": {"latitude": 22.5800, "longitude": 88.3700},
                "types": ["doctor"],
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.search_text(
            query="General Physician",
            latitude=22.5726,
            longitude=88.3639,
        )
        assert res["status"] == "success"
        doc = res["doctors"][0]
        assert doc["distance_km"] is not None
        assert 0.8 < doc["distance_km"] < 1.5


# 10. Missing optional fields
def test_missing_optional_fields():
    client = GooglePlacesClient(api_key="test_mock_api_key_123")
    mock_payload = {
        "places": [
            {
                "id": "place_sparse",
                "displayName": {"text": "Dr. Sparse Details"},
                # Missing phone, website, address, location
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.search_text(query="General Physician")
        assert res["status"] == "success"
        doc = res["doctors"][0]
        assert doc["name"] == "Dr. Sparse Details"
        assert doc["contact"] is None
        assert doc["website"] is None
        assert doc["address"] is None
        assert doc["distance_km"] is None
        assert doc["qualification"] is None
        assert doc["registration_number"] is None


# 11. 401 Authorization Error
def test_401_authorization_error():
    client = GooglePlacesClient(api_key="invalid_bad_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "API Key Invalid"

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.search_text(query="Doctor")
        assert res["status"] == "authorization_error"
        assert len(res["doctors"]) == 0
        assert "authorization" in res["message"].lower() or "key" in res["message"].lower()


# 12. 403 Access Denied
def test_403_access_denied():
    client = GooglePlacesClient(api_key="forbidden_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "Access Denied"

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.search_text(query="Doctor")
        assert res["status"] == "authorization_error"
        assert len(res["doctors"]) == 0
        assert "access denied" in res["message"].lower() or "enabled" in res["message"].lower()


# 13. 429 Rate Limited
def test_429_rate_limited():
    client = GooglePlacesClient(api_key="rate_limited_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 429

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.search_text(query="Doctor")
        assert res["status"] == "rate_limited"
        assert len(res["doctors"]) == 0
        assert "rate limits" in res["message"].lower()


# 14. Timeout handling
def test_timeout_handling():
    client = GooglePlacesClient(api_key="test_key")

    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Timed out")):
        res = client.search_text(query="Doctor")
        assert res["status"] == "unavailable"
        assert len(res["doctors"]) == 0
        assert "timed out" in res["message"].lower()


# 15. Malformed response handling
def test_malformed_response():
    client = GooglePlacesClient(api_key="test_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.search_text(query="Doctor")
        assert res["status"] == "source_error"
        assert len(res["doctors"]) == 0
        assert "malformed" in res["message"].lower()


# 16. No results
def test_no_results():
    client = GooglePlacesClient(api_key="test_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"places": []}

    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.search_text(query="Neurosurgeon in Sahara Desert")
        assert res["status"] == "success"
        assert len(res["doctors"]) == 0
        assert "no matching" in res["message"].lower()


# 17. Assessment ID context
def test_assessment_id_context(db_session):
    user = User(
        name="Assessment User",
        email=f"user_{uuid4().hex[:6]}@example.com",
        password_hash="pwd",
        state="West Bengal",
        district="Kolkata",
    )
    db_session.add(user)
    db_session.commit()

    ass = Assessment(
        user_id=user.user_id,
        input_types=["symptoms"],
        status="completed",
        specialty="General Physician",
        result_payload={
            "confirmed_coordinates": {"latitude": 22.5726, "longitude": 88.3639}
        },
    )
    db_session.add(ass)
    db_session.commit()

    mock_client = GooglePlacesClient(api_key="test_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "places": [
            {
                "id": "place_ass_1",
                "displayName": {"text": "Dr. Found In Assessment Area"},
                "location": {"latitude": 22.5750, "longitude": 88.3650},
                "types": ["doctor"],
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        service = DoctorSearchService(db=db_session, google_client=mock_client)
        res = service.search_doctors(
            specialization="General Physician",
            assessment_id=str(ass.assessment_id),
        )
        assert res["status"] == "success"
        assert len(res["doctors"]) == 1
        # Verify assessment coordinates were passed to Google Places
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        circle = body["locationRestriction"]["circle"]
        assert circle["center"]["latitude"] == 22.5726
        assert circle["center"]["longitude"] == 88.3639


# 18. Specialization auto-fill
def test_specialization_auto_fill(db_session):
    mock_client = GooglePlacesClient(api_key="test_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"places": []}

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        service = DoctorSearchService(db=db_session, google_client=mock_client)
        # Search with General Physician specialty
        service.search_doctors(specialization="General Physician", state="Delhi", district="New Delhi")
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        assert "General Physician" in body.get("textQuery", "")


# 19. Assessment coordinates used over manual/profile
def test_assessment_coordinates_used(db_session):
    user = User(
        name="Coord User",
        email=f"coord_{uuid4().hex[:6]}@example.com",
        password_hash="pwd",
        state="Maharashtra",
        district="Mumbai",
    )
    db_session.add(user)
    db_session.commit()

    ass = Assessment(
        user_id=user.user_id,
        input_types=["symptoms"],
        status="completed",
        result_payload={
            "confirmed_coordinates": {"latitude": 19.0760, "longitude": 72.8777}
        },
    )
    db_session.add(ass)
    db_session.commit()

    mock_client = GooglePlacesClient(api_key="test_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"places": []}

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        service = DoctorSearchService(db=db_session, google_client=mock_client)
        service.search_doctors(
            specialization="Cardiologist",
            assessment_id=str(ass.assessment_id),
        )
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        circle = body["locationRestriction"]["circle"]
        # Coordinates must match assessment confirmed coordinates
        assert circle["center"]["latitude"] == 19.0760
        assert circle["center"]["longitude"] == 72.8777


# 20. Dashboard manual search
def test_dashboard_manual_search(db_session):
    mock_client = GooglePlacesClient(api_key="test_key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "places": [
            {
                "id": "place_barasat_1",
                "displayName": {"text": "Dr. Barasat Specialist"},
                "formattedAddress": "Barasat Hospital Road, Barasat, North 24 Parganas, West Bengal 700124",
                "types": ["doctor"],
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        service = DoctorSearchService(db=db_session, google_client=mock_client)
        res = service.search_doctors(
            specialization="Cardiologist",
            state="West Bengal",
            district="North 24 Parganas",
            city="Barasat",
            pin="700124",
        )
        assert res["status"] == "success"
        assert len(res["doctors"]) == 1
        _, kwargs = mock_post.call_args
        body = kwargs.get("json", {})
        query = body.get("textQuery", "")
        assert "Cardiologist in Barasat, North 24 Parganas, West Bengal, 700124" in query


# 21. No fake provider fallback
def test_no_fake_provider_fallback(db_session):
    mock_client = GooglePlacesClient(api_key="test_key")

    # API failure (500)
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Server Error"

    with patch("httpx.Client.post", return_value=mock_resp):
        service = DoctorSearchService(db=db_session, google_client=mock_client)
        res = service.search_doctors(
            specialization="General Physician",
            state="NonExistentState",
            district="NonExistentDistrict",
        )
        # Must return error status without fabricating fallback doctor records
        assert res["status"] == "unavailable"
        assert len(res["doctors"]) == 0
        assert len(res["data"]) == 0


# 22. API key never appears in response or logging
def test_api_key_never_appears_in_response_or_logging(caplog):
    secret_key = "AIzaSy_SECRET_GOOGLE_KEY_NEVER_LEAK_999"
    client = GooglePlacesClient(api_key=secret_key)

    with caplog.at_level(logging.ERROR):
        # Trigger an error log
        client._safe_log_error(f"Error connecting to https://places.googleapis.com?key={secret_key}")

    # Check caplog: secret must NOT appear
    for record in caplog.records:
        assert secret_key not in record.message
        assert "[REDACTED_API_KEY]" in record.message

    # Also test that API response does not contain the key
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"places": []}
    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.search_text(query="Doctor")
        serialized = json.dumps(res)
        assert secret_key not in serialized


# 23. Clinic is not mislabeled as individual doctor
def test_clinic_is_not_mislabeled_as_individual_doctor():
    # Medical clinic
    assert determine_entity_type(["medical_clinic", "health"], "Metro Polyclinic") == "clinic"
    # Medical center
    assert determine_entity_type(["medical_center", "health"], "Central Medical Centre") == "medical_center"
    # Hospital
    assert determine_entity_type(["hospital", "health"], "St. Mary Hospital") == "hospital"
    # Individual doctor
    assert determine_entity_type(["doctor", "health"], "Dr. Anita Roy") == "doctor"


# 24. Unverified provider is not shown as NMC verified
def test_unverified_provider_is_not_shown_as_nmc_verified():
    client = GooglePlacesClient(api_key="test_key")
    raw_place = {
        "id": "place_google_unverified",
        "displayName": {"text": "Dr. Unverified Provider"},
        "types": ["doctor"],
        "formattedAddress": "Delhi Road",
    }
    normalized = client.normalize_place(raw_place)
    assert normalized["verification_status"] == "not_verified"
    assert normalized["qualification"] is None
    assert normalized["registration_number"] is None
    assert normalized["registration_council"] is None
    assert normalized["last_verified"] is None
    assert normalized["source"] == "Google Places"
