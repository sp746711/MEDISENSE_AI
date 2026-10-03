"""Comprehensive tests for Geoapify Places API integration.

Verifies the 19 required criteria:
1. Geoapify configured
2. Missing Geoapify API key
3. Hospital success
4. Pharmacy success
5. Correct coordinates
6. Correct distance
7. 25 km search
8. 50 km expansion
9. Pagination
10. 401 authorization error
11. 403 forbidden error
12. 429 rate limit
13. Timeout handling
14. Malformed response handling
15. No results handling
16. Missing coordinates skipped safely
17. emergency_available remains null when not explicitly provided
18. API key never appears in response
19. Source metadata is correct
"""

import json
from typing import Any
import pytest
from unittest.mock import MagicMock, patch
import httpx

from app.providers.geoapify_places import GeoapifyPlacesClient
from app.providers.provider_repository import ProviderRepository
from app.database.database import SessionLocal, init_db


@pytest.fixture(scope="function")
def db_session():
    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def make_mock_feature(
    place_id: str = "place_123",
    name: str = "Apollo Hospital",
    lat: float = 12.9716,
    lon: float = 77.5946,
    formatted: str = "Apollo Hospital, Bannerghatta Rd, Bengaluru",
    city: str = "Bengaluru",
    state: str = "Karnataka",
    phone: str = "+91-80-2630-4050",
    emergency: str = "",
    website: str = "https://www.apollohospitals.com",
) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if phone:
        raw["phone"] = phone
    if emergency:
        raw["emergency"] = emergency
    if website:
        raw["website"] = website

    return {
        "type": "Feature",
        "properties": {
            "place_id": place_id,
            "name": name,
            "formatted": formatted,
            "city": city,
            "state": state,
            "district": "Bengaluru Urban",
            "lat": lat,
            "lon": lon,
            "datasource": {
                "sourcename": "openstreetmap",
                "raw": raw,
            },
        },
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat],
        },
    }


# 1. Geoapify configured
def test_geoapify_configured():
    client = GeoapifyPlacesClient(api_key="mock_key_abc", base_url="https://api.geoapify.com/v2/places")
    assert client.is_configured() is True
    assert client.api_key == "mock_key_abc"
    assert client.base_url == "https://api.geoapify.com/v2/places"


# 2. Missing Geoapify API key
def test_missing_geoapify_api_key():
    client = GeoapifyPlacesClient(api_key="")
    assert not client.is_configured()

    res = client.search_hospitals(latitude=12.9716, longitude=77.5946)
    assert res["status"] == "configuration_missing"
    assert "not configured" in res["message"]
    assert len(res["facilities"]) == 0
    assert res["source_metadata"]["configured"] is False


# 3, 5, 17. Hospital success, correct coordinates & emergency_available=None
def test_hospital_success():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    mock_feat = make_mock_feature(
        place_id="hosp_001",
        name="St. Martha's Hospital",
        lat=12.9720,
        lon=77.5950,
        phone="+91-80-2227-5081",
        emergency="",  # No explicit emergency field
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"features": [mock_feat]}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    assert res["status"] == "success"
    assert len(res["facilities"]) == 1
    fac = res["facilities"][0]

    assert fac["external_id"] == "hosp_001"
    assert fac["name"] == "St. Martha's Hospital"
    assert fac["type"] == "Hospital"
    assert fac["latitude"] == 12.9720
    assert fac["longitude"] == 77.5950
    assert fac["contact"] == "+91-80-2227-5081"
    assert fac["source"] == "Geoapify / OpenStreetMap"
    assert fac["last_verified"] is None  # Never set to current time
    assert fac["emergency_available"] is None  # Never assumed to be "yes"
    assert fac["has_emergency"] is False
    assert fac["distance_km"] is not None


# 4. Pharmacy success
def test_pharmacy_success():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    mock_feat = make_mock_feature(
        place_id="pharm_001",
        name="MedPlus Pharmacy",
        lat=12.9730,
        lon=77.5960,
        phone="+91-80-2233-4455",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"features": [mock_feat]}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_pharmacies(latitude=12.9716, longitude=77.5946)

    assert res["status"] == "success"
    assert len(res["medical_shops"]) == 1
    shop = res["medical_shops"][0]

    assert shop["shop_id"] == "pharm_001"
    assert shop["external_id"] == "pharm_001"
    assert shop["name"] == "MedPlus Pharmacy"
    assert shop["latitude"] == 12.9730
    assert shop["longitude"] == 77.5960
    assert shop["contact"] == "+91-80-2233-4455"
    assert shop["source"] == "Geoapify / OpenStreetMap"
    assert shop["last_verified"] is None
    assert shop["distance_km"] is not None
    assert "disclaimer" in res


# 6. Correct distance & nearest-first sorting
def test_correct_distance_and_sorting():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    user_lat, user_lon = 12.9716, 77.5946
    # Place A: ~10 km away
    feat_a = make_mock_feature("p_far", "Far Hospital", lat=13.0600, lon=77.5946)
    # Place B: ~1 km away
    feat_b = make_mock_feature("p_near", "Near Hospital", lat=12.9800, lon=77.5946)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    # Return out of order
    mock_resp.json.return_value = {"features": [feat_a, feat_b]}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_hospitals(latitude=user_lat, longitude=user_lon)

    assert res["status"] == "success"
    assert len(res["facilities"]) == 2
    # Place B should be sorted first
    assert res["facilities"][0]["external_id"] == "p_near"
    assert res["facilities"][1]["external_id"] == "p_far"
    assert res["facilities"][0]["distance_km"] < res["facilities"][1]["distance_km"]


# 7 & 8. 25 km search and 50 km expansion
def test_25km_search_and_50km_expansion():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    feat_expanded = make_mock_feature("hosp_exp", "Expanded Hospital", lat=13.3000, lon=77.5946)

    resp_25km = MagicMock(status_code=200)
    resp_25km.json.return_value = {"features": []}

    resp_50km = MagicMock(status_code=200)
    resp_50km.json.return_value = {"features": [feat_expanded]}

    calls = []

    def mock_get(*args, **kwargs):
        calls.append(kwargs.get("params", {}))
        if len(calls) == 1:
            return resp_25km
        return resp_50km

    with patch("httpx.Client.get", side_effect=mock_get):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    assert len(calls) == 2
    # First call had 25km (25000m)
    assert "25000" in calls[0]["filter"]
    # Second call had 50km (50000m)
    assert "50000" in calls[1]["filter"]

    assert res["status"] == "success"
    assert res["expanded"] is True
    assert "expanded to approximately 50 km" in res["expansion_message"]
    assert len(res["facilities"]) == 1


# 8b. No unnecessary expansion when 25km has results
def test_no_unnecessary_expansion():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    feat_local = make_mock_feature("hosp_local", "Local Hospital", lat=12.9750, lon=77.5950)
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": [feat_local]}

    calls = []

    def mock_get(*args, **kwargs):
        calls.append(kwargs.get("params", {}))
        return mock_resp

    with patch("httpx.Client.get", side_effect=mock_get):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    assert len(calls) == 1
    assert res["status"] == "success"
    assert res["expanded"] is False
    assert res["expansion_message"] is None


# 9. Pagination: page 1 (offset 0), page 2 (offset 10), page 3 blocked
def test_pagination():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    calls = []
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": [make_mock_feature() for _ in range(10)]}

    with patch("httpx.Client.get", side_effect=lambda *a, **k: calls.append(k.get("params", {})) or mock_resp):
        res1 = client.search_hospitals(latitude=12.9716, longitude=77.5946, page=1, page_size=10)
        res2 = client.search_hospitals(latitude=12.9716, longitude=77.5946, page=2, page_size=10)
        res3 = client.search_hospitals(latitude=12.9716, longitude=77.5946, page=3, page_size=10)

    # Page 1 & 2 called Geoapify, Page 3 was BLOCKED without calling Geoapify
    assert len(calls) == 2
    assert calls[0]["offset"] == 0
    assert calls[0]["limit"] == 10
    assert res1["has_next"] is True
    assert res1["has_previous"] is False

    assert calls[1]["offset"] == 10
    assert calls[1]["limit"] == 10
    # Page 2 has_next is always False (maximum 2 pages)
    assert res2["has_next"] is False
    assert res2["has_previous"] is True

    # Page 3 blocked
    assert res3["status"] == "success"
    assert res3["facilities"] == []
    assert res3["has_next"] is False
    assert res3["has_previous"] is True
    assert "Maximum of 2 pages reached" in res3["message"]


# 10. 401 error
def test_401_authorization_error():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    mock_resp = MagicMock(status_code=401)
    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    assert res["status"] == "authorization_error"
    assert "authorization failed" in res["message"]


# 11. 403 error
def test_403_forbidden_error():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    mock_resp = MagicMock(status_code=403)
    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    assert res["status"] == "authorization_error"
    assert "authorization failed" in res["message"]


# 12. 429 rate limit
def test_429_rate_limit():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    mock_resp = MagicMock(status_code=429)
    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    assert res["status"] == "rate_limited"
    assert "rate limit reached" in res["message"]
    assert res["facilities"] == []


# 13. Timeout handling
def test_timeout_handling():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Read timed out")):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    assert res["status"] == "unavailable"
    assert "timed out" in res["message"]


# 14. Malformed response handling
def test_malformed_response():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.side_effect = ValueError("Invalid JSON response")

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    assert res["status"] == "source_error"
    assert "Malformed" in res["message"]


# 15. No results handling (Case D: 25km = 0, 50km = 0)
def test_no_results():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": []}

    with patch("httpx.Client.get", return_value=mock_resp):
        res_hosp = client.search_hospitals(latitude=12.9716, longitude=77.5946)
        res_pharm = client.search_pharmacies(latitude=12.9716, longitude=77.5946)

    assert res_hosp["status"] == "success"
    assert len(res_hosp["facilities"]) == 0
    assert "No matching hospitals were found within 50 km of your confirmed location." == res_hosp["message"]

    assert res_pharm["status"] == "success"
    assert len(res_pharm["medical_shops"]) == 0
    assert "No matching pharmacies/medical shops were found within 50 km of your confirmed location." == res_pharm["message"]


# 16. Missing coordinates skipped safely
def test_missing_coordinates_skipped():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    feat_valid = make_mock_feature("hosp_ok", "Valid Hospital", lat=12.9720, lon=77.5950)
    feat_no_coords = {
        "type": "Feature",
        "properties": {"name": "No Coords Hospital", "place_id": "hosp_bad_1"},
        "geometry": None,
    }
    feat_invalid_coords = {
        "type": "Feature",
        "properties": {"name": "Bad Lat Hospital", "place_id": "hosp_bad_2", "lat": 150.0, "lon": 77.0},
        "geometry": {"type": "Point", "coordinates": [77.0, 150.0]},
    }

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": [feat_valid, feat_no_coords, feat_invalid_coords]}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    assert res["status"] == "success"
    assert len(res["facilities"]) == 1
    assert res["facilities"][0]["external_id"] == "hosp_ok"


# 17. Emergency capability preserved when explicitly provided
def test_emergency_explicit():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    feat_emg = make_mock_feature("hosp_emg", "Trauma Center", lat=12.9720, lon=77.5950, emergency="yes")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": [feat_emg]}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946, emergency_only=False)

    assert res["facilities"][0]["emergency_available"] == "yes"
    assert res["facilities"][0]["has_emergency"] is True


# 18. API key never appears in response
def test_api_key_never_appears_in_response():
    secret_key = "very_secret_api_key_12345"
    client = GeoapifyPlacesClient(api_key=secret_key)

    feat = make_mock_feature("hosp_key_check", "General Hospital", lat=12.9720, lon=77.5950)
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": [feat]}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    serialized = json.dumps(res)
    assert secret_key not in serialized


# 19. Source metadata is correct
def test_source_metadata_is_correct():
    client = GeoapifyPlacesClient(api_key="mock_test_key")

    feat = make_mock_feature("hosp_meta", "City Clinic", lat=12.9720, lon=77.5950)
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": [feat]}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_hospitals(latitude=12.9716, longitude=77.5946)

    meta = res.get("source_metadata", {})
    assert meta["source"] == "Geoapify / OpenStreetMap"
    assert meta["configured"] is True
    assert meta["source_url"] == "https://api.geoapify.com/v2/places"


# ProviderRepository integration test
def test_provider_repository_geoapify_integration(db_session):
    repo = ProviderRepository(db_session)
    repo.source.facility_source_name = "geoapify"
    repo.source.geoapify_client = GeoapifyPlacesClient(api_key="mock_test_key")

    feat = make_mock_feature("hosp_repo_test", "City Hospital", lat=12.9720, lon=77.5950)
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": [feat]}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = repo.search_facilities(latitude=12.9716, longitude=77.5946)

    assert res["status"] == "success"
    assert len(res["facilities"]) == 1
    assert res["facilities"][0]["external_id"] == "hosp_repo_test"
    assert res["source_metadata"]["source"] == "Geoapify / OpenStreetMap"
    assert res["source_metadata"]["configured"] is True


def test_provider_repository_page_3_blocked(db_session):
    repo = ProviderRepository(db_session)
    repo.source.facility_source_name = "geoapify"
    repo.source.geoapify_client = GeoapifyPlacesClient(api_key="mock_test_key")

    res = repo.search_facilities(latitude=12.9716, longitude=77.5946, page=3)
    assert res["status"] == "success"
    assert res["facilities"] == []
    assert res["has_next"] is False
    assert res["has_previous"] is True
    assert "Maximum of 2 pages reached" in res["message"]


def test_api_key_redacted_in_logs():
    import logging
    from app.providers.geoapify_places import RedactApiKeyFilter

    filt = RedactApiKeyFilter()
    rec = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="GET https://api.geoapify.com/v2/places?categories=healthcare.hospital&apiKey=SECRET_KEY_12345",
        args=(),
        exc_info=None,
    )
    filt.filter(rec)
    assert "SECRET_KEY_12345" not in rec.msg
    assert "apiKey=[REDACTED]" in rec.msg
