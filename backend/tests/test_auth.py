"""Basic auth and health tests (require DATABASE_URL)."""

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Skip if database is not configured for tests
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url or "CHANGE_ME" in db_url:
        pytest.skip("DATABASE_URL not configured for tests")

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["app"] == "MediSense AI"


def test_register_login_me(client):
    email = "stage1_tester@example.com"
    password = "SecurePass123"

    # Clean-slate: registration may conflict if re-run
    reg = client.post(
        "/api/auth/register",
        json={
            "name": "Stage One Tester",
            "email": email,
            "password": password,
            "confirm_password": password,
            "state": "Karnataka",
            "district": "Bengaluru Urban",
            "city": "Bengaluru",
        },
    )
    if reg.status_code == 409:
        login = client.post(
            "/api/auth/login",
            json={"email": email, "password": password},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
    else:
        assert reg.status_code == 201
        token = reg.json()["access_token"]

    me = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me.status_code == 200
    assert me.json()["email"] == email
