
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import create_access_token
from app.database.database import SessionLocal, init_db
from app.database.models import User, Assessment, Symptom


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_db()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_records():
    """
    Create isolated test users and assessments.
    Remove their records after the test.
    """
    db = SessionLocal()
    user_ids = []
    assessment_ids = []

    try:
        suffix = uuid.uuid4().hex

        owner = User(
            name="Symptom API Test",
            email=f"symptom_owner_{suffix}@example.test",
            password_hash="test-only-hash",
            state="West Bengal",
            district="Test District",
            city="Test City",
        )
        other_user = User(
            name="Other Symptom API Test",
            email=f"symptom_other_{suffix}@example.test",
            password_hash="test-only-hash",
            state="West Bengal",
            district="Test District",
            city="Test City",
        )

        db.add_all([owner, other_user])
        db.commit()
        db.refresh(owner)
        db.refresh(other_user)

        user_ids.extend([owner.user_id, other_user.user_id])

        assessment = Assessment(
            user_id=owner.user_id,
            input_types=["symptoms"],
            status="draft",
        )
        other_assessment = Assessment(
            user_id=other_user.user_id,
            input_types=["symptoms"],
            status="draft",
        )

        db.add_all([assessment, other_assessment])
        db.commit()
        db.refresh(assessment)
        db.refresh(other_assessment)

        assessment_ids.extend([
            assessment.assessment_id,
            other_assessment.assessment_id,
        ])

        yield {
            "owner": owner,
            "other_user": other_user,
            "assessment": assessment,
            "other_assessment": other_assessment,
            "token": create_access_token(str(owner.user_id)),
            "other_token": create_access_token(str(other_user.user_id)),
        }

    finally:
        # Delete dependent rows first; the API uses its own DB sessions.
        if assessment_ids:
            db.query(Symptom).filter(
                Symptom.assessment_id.in_(assessment_ids)
            ).delete(synchronize_session=False)

            db.query(Assessment).filter(
                Assessment.assessment_id.in_(assessment_ids)
            ).delete(synchronize_session=False)

        if user_ids:
            db.query(User).filter(
                User.user_id.in_(user_ids)
            ).delete(synchronize_session=False)

        db.commit()
        db.close()


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_preview_returns_extracted_symptoms_and_followups(
    client, test_records
):
    assessment_id = test_records["assessment"].assessment_id

    response = client.post(
        f"/api/assessments/{assessment_id}/symptoms/analyze",
        headers=auth(test_records["token"]),
        json={"raw_text": "I have a headache for two days."},
    )

    assert response.status_code == 200, response.text

    data = response.json()
    assert data.get("status") == "analyzed"
    assert isinstance(data.get("extracted"), list)
    assert "followups" in data


def test_submit_persists_symptoms_and_returns_followup_result(
    client, test_records
):
    assessment_id = test_records["assessment"].assessment_id

    response = client.post(
        f"/api/assessments/{assessment_id}/symptoms",
        headers=auth(test_records["token"]),
        json={"raw_text": "I have a headache for two days."},
    )

    assert response.status_code == 200, response.text

    data = response.json()
    assert data.get("status") == "extracted"
    assert isinstance(data.get("extracted"), list)
    assert "followups" in data

    list_response = client.get(
        f"/api/assessments/{assessment_id}/symptoms",
        headers=auth(test_records["token"]),
    )

    assert list_response.status_code == 200, list_response.text
    saved_data = list_response.json()

    # Accommodate either a direct list or a wrapped response.
    saved = (
        saved_data
        if isinstance(saved_data, list)
        else saved_data.get("symptoms", saved_data.get("items", []))
    )
    assert isinstance(saved, list)
    assert len(saved) > 0


def test_missing_authentication_is_rejected(client, test_records):
    assessment_id = test_records["assessment"].assessment_id

    response = client.post(
        f"/api/assessments/{assessment_id}/symptoms/analyze",
        json={"raw_text": "I have a headache."},
    )

    assert response.status_code == 401


def test_user_cannot_access_another_users_assessment(
    client, test_records
):
    other_assessment_id = test_records["other_assessment"].assessment_id

    response = client.post(
        f"/api/assessments/{other_assessment_id}/symptoms/analyze",
        headers=auth(test_records["token"]),
        json={"raw_text": "I have a headache."},
    )

    assert response.status_code == 404
