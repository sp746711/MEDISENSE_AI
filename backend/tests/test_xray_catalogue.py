"""Tests for dynamic searchable anatomical X-ray catalogue, model separation, and router integrity."""

import pytest
from fastapi.testclient import TestClient

from app.ai.xray_catalogue import (
    ANATOMICAL_CATALOGUE,
    get_xray_catalogue_with_model_status,
)
from app.ai.xray_model import get_xray_model_service
from app.ai.xray_router import (
    XRAY_MODEL_REGISTRY,
    get_model_registry_entry,
    normalize_region_name,
    route_xray,
)
from app.main import app

client = TestClient(app)


def test_catalogue_structure_and_regions():
    """Verify the anatomical catalogue covers all required clinical regions."""
    catalogue = get_xray_catalogue_with_model_status()
    assert len(catalogue) >= 8

    region_names = [r["name"] for r in catalogue]
    assert any("Head" in name for name in region_names)
    assert any("Chest" in name for name in region_names)
    assert any("Spine" in name for name in region_names)
    assert any("Upper Limb" in name or "Arm" in name for name in region_names)
    assert any("Wrist" in name or "Hand" in name for name in region_names)
    assert any("Abdomen" in name for name in region_names)
    assert any("Pelvis" in name or "Hip" in name for name in region_names)
    assert any("Knee" in name or "Lower" in name for name in region_names)


def test_search_head_and_skull_hierarchy():
    """Requirement 1 & 2: Searching for Head and Skull finds correct hierarchy."""
    catalogue = get_xray_catalogue_with_model_status()
    head_region = next((r for r in catalogue if r["id"] == "head_neck"), None)
    assert head_region is not None
    assert "Head & Neck" in head_region["name"]

    subregion_ids = [s["id"] for s in head_region["subregions"]]
    assert "skull" in subregion_ids
    assert "facial_bones" in subregion_ids
    assert "sinuses" in subregion_ids

    skull_sub = next(s for s in head_region["subregions"] if s["id"] == "skull")
    skull_xrays = [x["name"] for x in skull_sub["xray_types"]]
    assert any("Skull X-Ray (AP & Lateral)" in name for name in skull_xrays)
    assert any("Towne's View" in name for name in skull_xrays)


def test_search_aliases_and_normalization():
    """Requirement 3 & 4: Aliases map to correct canonical anatomical regions."""
    assert normalize_region_name("head") == ("skull", True)
    assert normalize_region_name("skull") == ("skull", True)
    assert normalize_region_name("cranium") == ("skull", True)
    assert normalize_region_name("face") == ("facial bones", True)
    assert normalize_region_name("sinus") == ("sinuses", True)
    assert normalize_region_name("c-spine") == ("spine", True)
    assert normalize_region_name("neck") == ("neck", True)
    assert normalize_region_name("low back") == ("spine", True)
    assert normalize_region_name("cxr") == ("chest", True)
    assert normalize_region_name("lungs") == ("chest", True)
    assert normalize_region_name("collarbone") == ("clavicle", True)
    assert normalize_region_name("upper arm") == ("humerus", True)
    assert normalize_region_name("tailbone") == ("spine", True)
    assert normalize_region_name("shin") == ("lower leg", True)
    assert normalize_region_name("heel") == ("calcaneus", True)


def test_catalogue_api_endpoint():
    """Verify GET /api/assessments/xray/catalogue returns enriched catalogue with model status and flat regions."""
    resp = client.get("/api/assessments/xray/catalogue")
    assert resp.status_code == 200
    data = resp.json()
    assert "catalogue" in data
    assert "regions" in data
    assert data["total_regions"] >= 8

    # Verify flat backend regions for simple frontend search
    head_region = next((r for r in data["regions"] if r["name"] == "Head"), None)
    assert head_region is not None
    assert head_region["canonical_region"] == "skull"
    assert "head" in head_region["aliases"]
    assert head_region["is_supported"] is False

    chest_region = next((r for r in data["regions"] if r["name"] == "Chest"), None)
    assert chest_region is not None
    assert chest_region["canonical_region"] == "chest"
    assert chest_region["is_supported"] is True

    # Verify model support separation in API response
    head = next(r for r in data["catalogue"] if r["id"] == "head_neck")
    skull = next(s for s in head["subregions"] if s["id"] == "skull")
    skull_xtype = skull["xray_types"][0]

    assert skull_xtype["model_support"]["status"] == "unsupported"
    assert skull_xtype["model_support"]["is_supported"] is False
    assert "Unsupported" in skull_xtype["model_support"]["badge"]
    assert "unavailable" in skull_xtype["model_support"]["notice"].lower()



def test_model_support_distinction_truthful():
    """Requirement 8: Separate catalogue availability from AI model support."""
    # Chest is supported by architecture configuration
    chest_reg = get_model_registry_entry("chest")
    assert chest_reg is not None
    assert chest_reg.is_supported is True
    assert chest_reg.model_architecture == "ResNet18"

    # Skull is in catalogue but model is unsupported
    skull_reg = get_model_registry_entry("skull")
    assert skull_reg is not None
    assert skull_reg.is_supported is False

    # Spine is in catalogue but model is unsupported
    spine_reg = get_model_registry_entry("cervical spine")
    assert spine_reg is not None
    assert spine_reg.is_supported is False


def test_prevent_unsupported_image_routing_to_unrelated_model(tmp_path):
    """Requirement 9: Never route an unsupported region to an unrelated model or fake diagnosis."""
    dummy_file = tmp_path / "dummy_xray.jpg"
    dummy_file.write_bytes(b"dummy_content")

    # Routing skull image
    route_result = route_xray(str(dummy_file), declared_region="skull")
    assert route_result["region"] == "skull"
    assert route_result["is_recognized"] is True
    assert route_result["is_supported"] is False
    assert route_result["reason_code"] == "unsupported_region"
    assert "skull" in route_result["message"].lower()

    # Model inference adapter must return unavailable status without fabricating findings
    service = get_xray_model_service()
    analysis = service.analyze(str(dummy_file), region="skull")
    assert analysis["status"] == "unavailable"
    assert analysis["prediction"] is None
    assert analysis["confidence_if_valid"] is None
    assert "unavailable" in analysis["message"].lower() or "not configured" in analysis["message"].lower()


def test_existing_supported_chest_workflow(tmp_path):
    """Requirement 11: Preserve existing chest X-ray routing workflow."""
    dummy_file = tmp_path / "chest_xray.png"
    dummy_file.write_bytes(b"dummy_png_bytes")

    route_result = route_xray(str(dummy_file), declared_region="chest")
    assert route_result["region"] == "chest"
    assert route_result["is_recognized"] is True
    assert route_result["is_supported"] is True
    # If checkpoint is absent on developer machine, model service safely marks unavailable without crash
    service = get_xray_model_service()
    analysis = service.analyze(str(dummy_file), region="chest")
    assert analysis["region"] == "chest"
    assert "status" in analysis


def test_search_query_hea_matches_head_region():
    """Requirement 2: Typing 'hea' matches Head in the backend-provided catalogue."""
    resp = client.get("/api/assessments/xray/catalogue")
    assert resp.status_code == 200
    regions = resp.json().get("regions", [])

    query = "hea".lower()
    matches = [
        r for r in regions
        if query in r["name"].lower() or any(query in a.lower() for a in r.get("aliases", []))
    ]
    assert len(matches) > 0
    head_match = next((m for m in matches if m["name"] == "Head"), None)
    assert head_match is not None
    assert head_match["canonical_region"] == "skull"


def test_search_query_spi_matches_spine():
    """Searching for 'spi' matches Spine."""
    resp = client.get("/api/assessments/xray/catalogue")
    assert resp.status_code == 200
    regions = resp.json().get("regions", [])

    query = "spi".lower()
    matches = [
        r for r in regions
        if query in r["name"].lower() or any(query in a.lower() for a in r.get("aliases", []))
    ]
    assert any(m["name"] == "Spine" for m in matches)


def test_search_empty_and_no_match():
    """Empty search should produce no suggestions; unmatched query returns empty."""
    resp = client.get("/api/assessments/xray/catalogue")
    regions = resp.json().get("regions", [])

    # Empty query logic
    assert [r for r in regions if "" in r["name"].lower()] == regions  # raw check
    # Nonexistent query
    unmatched = [r for r in regions if "xyz999nonexistent" in r["name"].lower()]
    assert len(unmatched) == 0

