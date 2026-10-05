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
from typing import Any, Optional
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
    categories: Optional[list[str]] = None,
) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if phone:
        raw["phone"] = phone
    if emergency:
        raw["emergency"] = emergency
    if website:
        raw["website"] = website

    props: dict[str, Any] = {
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
    }
    if categories is not None:
        props["categories"] = categories

    return {
        "type": "Feature",
        "properties": props,
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


# 20. Medical Shops: TEST 1 — Nearest ordering
def test_medical_shops_nearest_ordering():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    # Features at ~5.7, 1.2, 3.4, 0.6, 2.1 km
    # Calculated offsets approximately:
    # 0.6 km: lat offset ~0.0054
    # 1.2 km: lat offset ~0.0108
    # 2.1 km: lat offset ~0.0189
    # 3.4 km: lat offset ~0.0306
    # 5.7 km: lat offset ~0.0513
    features_unsorted = [
        make_mock_feature("s1", "Shop 5.7km", lat=user_lat + 0.0513, lon=user_lon),
        make_mock_feature("s2", "Shop 1.2km", lat=user_lat + 0.0108, lon=user_lon),
        make_mock_feature("s3", "Shop 3.4km", lat=user_lat + 0.0306, lon=user_lon),
        make_mock_feature("s4", "Shop 0.6km", lat=user_lat + 0.0054, lon=user_lon),
        make_mock_feature("s5", "Shop 2.1km", lat=user_lat + 0.0189, lon=user_lon),
    ]

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": features_unsorted}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_pharmacies(latitude=user_lat, longitude=user_lon)

    assert res["status"] == "success"
    shops = res["medical_shops"]
    assert len(shops) == 5

    distances = [s["distance_km"] for s in shops]
    # Check strictly ascending order
    assert distances == sorted(distances)
    assert shops[0]["name"] == "Shop 0.6km"
    assert shops[1]["name"] == "Shop 1.2km"
    assert shops[2]["name"] == "Shop 2.1km"
    assert shops[3]["name"] == "Shop 3.4km"
    assert shops[4]["name"] == "Shop 5.7km"


# 21. Medical Shops: TEST 3 — Missing coordinates sorted last, never fabricate
def test_medical_shops_missing_coordinates_sorted_last():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    feat_0_8 = make_mock_feature("s_08", "Shop 0.8km", lat=user_lat + 0.0072, lon=user_lon)
    feat_none = {
        "type": "Feature",
        "properties": {
            "place_id": "s_none",
            "name": "Shop Unknown Coords",
            "formatted": "Unknown Street",
            "lat": None,
            "lon": None,
        },
        "geometry": {"type": "Point", "coordinates": []},
    }
    feat_2_4 = make_mock_feature("s_24", "Shop 2.4km", lat=user_lat + 0.0216, lon=user_lon)

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": [feat_0_8, feat_none, feat_2_4]}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_pharmacies(latitude=user_lat, longitude=user_lon)

    assert res["status"] == "success"
    shops = res["medical_shops"]
    assert len(shops) == 3

    assert shops[0]["name"] == "Shop 0.8km"
    assert shops[0]["distance_km"] is not None
    assert shops[1]["name"] == "Shop 2.4km"
    assert shops[1]["distance_km"] is not None
    assert shops[2]["name"] == "Shop Unknown Coords"
    assert shops[2]["distance_km"] is None  # Never fabricated


# 22. Medical Shops: TEST 4 — 25 km returns results, NO 50 km expansion
def test_medical_shops_25km_has_results_no_expansion():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    features_5 = [
        make_mock_feature(f"s_{i}", f"Shop {i}", lat=user_lat + 0.005 * i, lon=user_lon)
        for i in range(1, 6)
    ]

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": features_5}

    with patch.object(client, "_query_places", wraps=client._query_places) as spy_query:
        mock_http = MagicMock(status_code=200)
        mock_http.json.return_value = {"features": features_5}
        with patch("httpx.Client.get", return_value=mock_http):
            res = client.search_pharmacies(latitude=user_lat, longitude=user_lon)

        assert res["status"] == "success"
        assert len(res["medical_shops"]) == 5
        assert res["expanded"] is False
        assert spy_query.call_count == 1  # Only 1 query at 25km


# 23. Medical Shops: TEST 5 — 25 km zero results expands to 50 km
def test_medical_shops_25km_zero_results_expands_to_50km():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    call_count = 0
    def mock_get(url, params=None, headers=None):
        nonlocal call_count
        call_count += 1
        resp = MagicMock(status_code=200)
        if "circle:77.5946,12.9716,25000" in params.get("filter", ""):
            resp.json.return_value = {"features": []}
        else:
            # 50km returns results
            feat = make_mock_feature("s_far", "Shop in 50km", lat=user_lat + 0.35, lon=user_lon)
            resp.json.return_value = {"features": [feat]}
        return resp

    with patch("httpx.Client.get", side_effect=mock_get):
        res = client.search_pharmacies(latitude=user_lat, longitude=user_lon)

    assert res["status"] == "success"
    assert res["expanded"] is True
    assert len(res["medical_shops"]) == 1
    assert res["medical_shops"][0]["name"] == "Shop in 50km"
    assert call_count == 2  # Exactly 2 calls: 25km then 50km


# 24. Medical Shops: TEST 6 — Pagination max 2 pages
def test_medical_shops_pagination_max_two_pages():
    client = GeoapifyPlacesClient(api_key="mock_key")
    res_page_3 = client.search_pharmacies(latitude=12.9716, longitude=77.5946, page=3)
    assert res_page_3["status"] == "success"
    assert res_page_3["medical_shops"] == []
    assert res_page_3["has_next"] is False
    assert res_page_3["has_previous"] is True
    assert "Maximum of 2 pages reached" in res_page_3["message"]


# 25. Medical Shops: Controlled distance list (Section 19)
def test_medical_shops_controlled_distance_list():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    # Input distances: 5.7, 1.2, 3.4, 0.6, 2.1, 0.8, 4.9, 1.5, 7.2, 3.0
    # Expected order: 0.6, 0.8, 1.2, 1.5, 2.1, 3.0, 3.4, 4.9, 5.7, 7.2
    distance_inputs = [5.7, 1.2, 3.4, 0.6, 2.1, 0.8, 4.9, 1.5, 7.2, 3.0]
    features = [
        make_mock_feature(f"c_{i}", f"Shop {d}km", lat=user_lat + (d / 111.0), lon=user_lon)
        for i, d in enumerate(distance_inputs)
    ]

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": features}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_pharmacies(latitude=user_lat, longitude=user_lon, page=1)

    assert res["status"] == "success"
    paged = res["medical_shops"]
    assert len(paged) == 10

    paged_distances = [p["distance_km"] for p in paged]
    assert paged_distances == sorted(paged_distances)
    assert paged[0]["name"] == "Shop 0.6km"
    assert paged[1]["name"] == "Shop 0.8km"
    assert paged[2]["name"] == "Shop 1.2km"
    assert paged[3]["name"] == "Shop 1.5km"
    assert paged[4]["name"] == "Shop 2.1km"
    assert paged[5]["name"] == "Shop 3.0km"
    assert paged[6]["name"] == "Shop 3.4km"
    assert paged[7]["name"] == "Shop 4.9km"
    assert paged[8]["name"] == "Shop 5.7km"
    assert paged[9]["name"] == "Shop 7.2km"


# 26. Medical Shops: Mixed candidate set (Section 20 - real bug scenario)
def test_medical_shops_mixed_candidate_pool_real_bug():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    # Candidate A=5.7, B=5.9, C=6.2, D=0.7, E=1.4, F=2.0
    candidates = [
        ("Candidate A", 5.7),
        ("Candidate B", 5.9),
        ("Candidate C", 6.2),
        ("Candidate D", 0.7),
        ("Candidate E", 1.4),
        ("Candidate F", 2.0),
    ]
    features = [
        make_mock_feature(f"cand_{i}", name, lat=user_lat + (d / 111.0), lon=user_lon)
        for i, (name, d) in enumerate(candidates)
    ]

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": features}

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_pharmacies(latitude=user_lat, longitude=user_lon, page=1)

    assert res["status"] == "success"
    paged = res["medical_shops"]
    assert len(paged) == 6

    expected_names = [
        "Candidate D",  # 0.7 km
        "Candidate E",  # 1.4 km
        "Candidate F",  # 2.0 km
        "Candidate A",  # 5.7 km
        "Candidate B",  # 5.9 km
        "Candidate C",  # 6.2 km
    ]
    actual_names = [p["name"] for p in paged]
    assert actual_names == expected_names


# 27. Medical Shops: 20 candidate pool slicing into Page 1 and Page 2 (Section 21/22)
def test_medical_shops_twenty_candidate_pool_pagination_continuity():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    # 20 candidates with distances 0.5 to 10.0
    features_20 = [
        make_mock_feature(f"pool_{i}", f"Shop {i}", lat=user_lat + ((i * 0.5 + 0.5) / 111.0), lon=user_lon)
        for i in range(20)
    ]

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": features_20}

    with patch("httpx.Client.get", return_value=mock_resp):
        page1 = client.search_pharmacies(latitude=user_lat, longitude=user_lon, page=1)
        page2 = client.search_pharmacies(latitude=user_lat, longitude=user_lon, page=2)

    assert len(page1["medical_shops"]) == 10
    assert len(page2["medical_shops"]) == 10
    assert page1["has_next"] is True
    assert page2["has_next"] is False  # Max 2 pages

    page1_max_dist = max(s["distance_km"] for s in page1["medical_shops"])
    page2_min_dist = min(s["distance_km"] for s in page2["medical_shops"])

    # Page 1 nearest 10 are all closer than or equal to Page 2
    assert page1_max_dist <= page2_min_dist


# 28. Root-cause test (Prompt Section 17 & 6):
# Demonstrates that combining healthcare.pharmacy and commercial.health_and_beauty.pharmacy
# in ONE request finds closest pharmacies (0.7km, 1.3km) instead of only distant ones (5.7km).
def test_medical_shops_combined_categories_root_cause():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    # Verify client uses the combined category
    expected_combined_category = "healthcare.pharmacy,commercial.health_and_beauty.pharmacy"
    assert client.medical_shop_category == expected_combined_category

    # Feature A: healthcare.pharmacy at 5.7 km
    feat_a = make_mock_feature(
        place_id="feat_a_5_7",
        name="Ms Bose Medical",
        lat=user_lat + (5.7 / 111.0),
        lon=user_lon,
        categories=["healthcare.pharmacy"],
    )
    # Feature B: commercial.health_and_beauty.pharmacy at 0.7 km
    feat_b = make_mock_feature(
        place_id="feat_b_0_7",
        name="Apollo Pharmacy Nearby",
        lat=user_lat + (0.7 / 111.0),
        lon=user_lon,
        categories=["commercial.health_and_beauty.pharmacy"],
    )
    # Feature C: commercial.health_and_beauty.pharmacy at 1.3 km
    feat_c = make_mock_feature(
        place_id="feat_c_1_3",
        name="Wellness Pharmacy Local",
        lat=user_lat + (1.3 / 111.0),
        lon=user_lon,
        categories=["commercial.health_and_beauty.pharmacy"],
    )

    calls = []
    mock_resp = MagicMock(status_code=200)
    # Return features in arbitrary order (A, B, C)
    mock_resp.json.return_value = {"features": [feat_a, feat_b, feat_c]}

    with patch("httpx.Client.get", side_effect=lambda *a, **k: calls.append(k.get("params", {})) or mock_resp):
        res = client.search_pharmacies(latitude=user_lat, longitude=user_lon)

    # Exactly ONE single Places request sent for both categories
    assert len(calls) == 1
    assert calls[0]["categories"] == expected_combined_category
    assert "25000" in calls[0]["filter"]

    # Results returned and sorted nearest first: 0.7 km, 1.3 km, 5.7 km
    assert res["status"] == "success"
    shops = res["medical_shops"]
    assert len(shops) == 3
    assert shops[0]["name"] == "Apollo Pharmacy Nearby"
    assert round(shops[0]["distance_km"], 1) == 0.7
    assert shops[1]["name"] == "Wellness Pharmacy Local"
    assert round(shops[1]["distance_km"], 1) == 1.3
    assert shops[2]["name"] == "Ms Bose Medical"
    assert round(shops[2]["distance_km"], 1) == 5.7


# 29. Duplicate removal test (Prompt Section 7 & 18):
# When the same pharmacy is returned in both categories, it must appear only ONCE.
def test_medical_shops_deduplication():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    # Duplicate 1: identical place_id "ABC123", one healthcare.pharmacy, one commercial.health_and_beauty.pharmacy
    feat_dup1 = make_mock_feature(
        place_id="ABC123",
        name="Apollo Pharmacy Central",
        lat=user_lat + (1.0 / 111.0),
        lon=user_lon,
        categories=["healthcare.pharmacy"],
    )
    feat_dup2 = make_mock_feature(
        place_id="ABC123",
        name="Apollo Pharmacy Central",
        lat=user_lat + (1.0 / 111.0),
        lon=user_lon,
        categories=["commercial.health_and_beauty.pharmacy"],
    )
    # Duplicate 2 (fallback): no place_id, but identical normalized name and coordinates
    feat_dup_fallback1 = {
        "type": "Feature",
        "properties": {
            "name": "MedPlus Local",
            "formatted": "123 Main St",
            "lat": user_lat + (2.0 / 111.0),
            "lon": user_lon,
            "categories": ["healthcare.pharmacy"],
        },
        "geometry": {"type": "Point", "coordinates": [user_lon, user_lat + (2.0 / 111.0)]},
    }
    feat_dup_fallback2 = {
        "type": "Feature",
        "properties": {
            "name": "medplus local",
            "formatted": "123 Main St",
            "lat": user_lat + (2.0 / 111.0),
            "lon": user_lon,
            "categories": ["commercial.health_and_beauty.pharmacy"],
        },
        "geometry": {"type": "Point", "coordinates": [user_lon, user_lat + (2.0 / 111.0)]},
    }
    # Unique place
    feat_unique = make_mock_feature(
        place_id="XYZ789",
        name="Unique Pharma",
        lat=user_lat + (3.0 / 111.0),
        lon=user_lon,
        categories=["commercial.health_and_beauty.pharmacy"],
    )

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "features": [feat_dup1, feat_dup2, feat_dup_fallback1, feat_dup_fallback2, feat_unique]
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.search_pharmacies(latitude=user_lat, longitude=user_lon)

    assert res["status"] == "success"
    shops = res["medical_shops"]
    # 5 features deduplicated down to 3 unique pharmacies
    assert len(shops) == 3

    # Primary place_id deduplication: Apollo Pharmacy appears only once
    apollo_shops = [s for s in shops if "Apollo Pharmacy Central" in s["name"]]
    assert len(apollo_shops) == 1

    # Fallback coordinate + name deduplication: MedPlus appears only once
    medplus_shops = [s for s in shops if "medplus" in s["name"].lower()]
    assert len(medplus_shops) == 1

    # Unique pharmacy appears once
    unique_shops = [s for s in shops if "Unique Pharma" in s["name"]]
    assert len(unique_shops) == 1


# 30. 25 km rule test (Prompt Section 5 & 19):
# When combined categories return results within 25 km, do NOT search 50 km.
def test_medical_shops_25km_combined_no_50km_expansion():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    distances = [0.7, 1.2, 2.1, 5.7]
    features = [
        make_mock_feature(
            place_id=f"shop_{d}",
            name=f"Shop {d}km",
            lat=user_lat + (d / 111.0),
            lon=user_lon,
            categories=["commercial.health_and_beauty.pharmacy" if d < 2.0 else "healthcare.pharmacy"],
        )
        for d in distances
    ]

    calls = []
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": features}

    with patch("httpx.Client.get", side_effect=lambda *a, **k: calls.append(k.get("params", {})) or mock_resp):
        res = client.search_pharmacies(latitude=user_lat, longitude=user_lon)

    # Exactly 1 query made at 25km (25000m)
    assert len(calls) == 1
    assert "25000" in calls[0]["filter"]
    assert res["expanded"] is False
    assert res["expansion_message"] is None

    # Results strictly in order: 0.7 km, 1.2 km, 2.1 km, 5.7 km
    shops = res["medical_shops"]
    assert len(shops) == 4
    result_distances = [round(s["distance_km"], 1) for s in shops]
    assert result_distances == [0.7, 1.2, 2.1, 5.7]


# 31. Zero-result fallback test (Prompt Section 5 & 20):
# When 25 km returns zero results, make exactly ONE 50 km fallback request with the SAME combined categories.
def test_medical_shops_zero_results_fallback_uses_same_combined_category():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946
    expected_categories = "healthcare.pharmacy,commercial.health_and_beauty.pharmacy"

    calls = []
    def mock_get(url, params=None, headers=None):
        calls.append(dict(params or {}))
        resp = MagicMock(status_code=200)
        if "25000" in params.get("filter", ""):
            resp.json.return_value = {"features": []}
        else:
            # 50km request returns features from both categories
            f1 = make_mock_feature("f_50_1", "Far Pharmacy 1", lat=user_lat + (35.0 / 111.0), lon=user_lon, categories=["healthcare.pharmacy"])
            f2 = make_mock_feature("f_50_2", "Far Pharmacy 2", lat=user_lat + (28.0 / 111.0), lon=user_lon, categories=["commercial.health_and_beauty.pharmacy"])
            resp.json.return_value = {"features": [f1, f2]}
        return resp

    with patch("httpx.Client.get", side_effect=mock_get):
        res = client.search_pharmacies(latitude=user_lat, longitude=user_lon)

    # Exactly TWO calls: 25 km followed by 50 km
    assert len(calls) == 2
    assert "25000" in calls[0]["filter"]
    assert calls[0]["categories"] == expected_categories
    assert "50000" in calls[1]["filter"]
    assert calls[1]["categories"] == expected_categories

    assert res["status"] == "success"
    assert res["expanded"] is True
    assert "expanded to approximately 50 km" in res["expansion_message"]

    # Results deduplicated and sorted nearest-first
    shops = res["medical_shops"]
    assert len(shops) == 2
    assert shops[0]["name"] == "Far Pharmacy 2"  # ~28 km
    assert shops[1]["name"] == "Far Pharmacy 1"  # ~35 km
    assert shops[0]["distance_km"] < shops[1]["distance_km"]


# 32. Pagination and candidate pool capping test (Prompt Section 10, 12, 21):
# 20 unique valid pharmacies -> Page 1 (nearest 10), Page 2 (next 10), Page 3 (blocked).
# Over 20 candidates in Geoapify -> capped to maximum 20 visible results across 2 pages.
def test_medical_shops_pagination_and_candidate_limit_capping():
    client = GeoapifyPlacesClient(api_key="mock_key")
    user_lat, user_lon = 12.9716, 77.5946

    # Generate 35 candidate features with distances from 0.5 km to 17.5 km
    features_35 = [
        make_mock_feature(
            place_id=f"shop_p_{i}",
            name=f"Shop P {i}",
            lat=user_lat + (((i + 1) * 0.5) / 111.0),
            lon=user_lon,
            categories=["commercial.health_and_beauty.pharmacy" if i % 2 == 0 else "healthcare.pharmacy"],
        )
        for i in range(35)
    ]

    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"features": features_35}

    with patch("httpx.Client.get", return_value=mock_resp):
        page1 = client.search_pharmacies(latitude=user_lat, longitude=user_lon, page=1)
        page2 = client.search_pharmacies(latitude=user_lat, longitude=user_lon, page=2)
        page3 = client.search_pharmacies(latitude=user_lat, longitude=user_lon, page=3)

    # Page 1: exactly nearest 10
    assert page1["status"] == "success"
    assert len(page1["medical_shops"]) == 10
    assert page1["has_next"] is True
    assert page1["has_previous"] is False
    assert page1["total_count"] == 20  # Capped at maximum 20 visible results

    # Page 2: next 10
    assert page2["status"] == "success"
    assert len(page2["medical_shops"]) == 10
    assert page2["has_next"] is False  # Max 2 pages reached
    assert page2["has_previous"] is True
    assert page2["total_count"] == 20

    # Nearest-first order continuity between Page 1 and Page 2
    assert page1["medical_shops"][-1]["distance_km"] <= page2["medical_shops"][0]["distance_km"]

    # Page 3: blocked
    assert page3["status"] == "success"
    assert page3["medical_shops"] == []
    assert page3["has_next"] is False
    assert page3["has_previous"] is True
    assert "Maximum of 2 pages reached" in page3["message"]



