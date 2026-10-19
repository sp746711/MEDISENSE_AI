"""Comprehensive test suite for Phase 4 multi-input assessment combinations and model registry.

Verifies the 9 required input combinations:
A. X-ray image only
B. Radiology/X-ray report PDF only
C. Symptoms only
D. General medical report or laboratory report only
E. X-ray image + radiology report
F. X-ray image + symptoms
G. Radiology report + symptoms
H. X-ray image + medical/lab reports + symptoms
I. Multiple supported images and/or reports attached to one assessment

Also tests:
- Region-aware model registry & canonical region normalization
- Cross-modal concordance tracking
- Fault isolation: one modality failure never erases or corrupts another modality
- No fake predictions / negative assumption when model is unavailable
"""

import io
from pathlib import Path
import pytest
from PIL import Image

from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.ai.xray_router import (
    CHEST_CLASSES,
    XRAY_MODEL_REGISTRY,
    get_all_registry_entries,
    get_model_registry_entry,
    normalize_region_name,
    route_xray,
)
from app.ai.xray_model import (
    XRayModelService,
    DEFAULT_UNAVAILABLE_MESSAGE,
    reset_xray_model_service,
)
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def isolate_test_environment():
    """Ensure shared settings and X-ray model singleton are reset before and after every test."""
    settings = get_settings()
    orig_checkpoint = settings.xray_chest_checkpoint
    reset_xray_model_service()
    try:
        yield
    finally:
        settings.xray_chest_checkpoint = orig_checkpoint
        reset_xray_model_service()


def _make_dummy_image_bytes(width=224, height=224) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Region Registry Tests
# ---------------------------------------------------------------------------

def test_model_registry_entries():
    """Registry must contain definitions for all supported anatomical regions."""
    entries = get_all_registry_entries()
    assert len(entries) >= 16
    regions = {e["canonical_region"] for e in entries}
    assert "chest" in regions
    assert "wrist" in regions
    assert "hand" in regions
    assert "knee" in regions
    assert "spine" in regions

    # Verify chest entry
    chest = get_model_registry_entry("chest")
    assert chest is not None
    assert chest.canonical_region == "chest"
    assert chest.is_supported is True
    assert chest.model_architecture == "ResNet18"
    assert chest.class_label_mapping == CHEST_CLASSES

    # Verify unsupported anatomical region entry
    wrist = get_model_registry_entry("wrist")
    assert wrist is not None
    assert wrist.canonical_region == "wrist"
    assert wrist.is_supported is False


def test_region_normalization():
    """Region names and aliases normalize consistently to canonical regions."""
    assert normalize_region_name("chest") == ("chest", True)
    assert normalize_region_name("thorax") == ("chest", True)
    assert normalize_region_name("CHEST PA") == ("chest", True)
    assert normalize_region_name("lungs") == ("chest", True)
    assert normalize_region_name("wrist") == ("wrist", True)
    assert normalize_region_name("carpal") == ("wrist", True)
    assert normalize_region_name("knee") == ("knee", True)
    assert normalize_region_name("patella") == ("knee", True)
    assert normalize_region_name("cervical spine") == ("spine", True)
    assert normalize_region_name("lumbar spine") == ("spine", True)
    assert normalize_region_name("foot") == ("foot", True)
    assert normalize_region_name("ankle") == ("ankle", True)
    assert normalize_region_name("unknown_region") == ("unknown region", False)


def test_route_xray_honest_availability(tmp_path):
    """Router accurately distinguishes recognized unsupported vs supported missing checkpoint."""
    dummy_file = tmp_path / "xray.png"
    dummy_file.write_bytes(_make_dummy_image_bytes())

    # Unsupported region
    route_wrist = route_xray(str(dummy_file), declared_region="wrist")
    assert route_wrist["supported"] is False
    assert route_wrist["reason_code"] == "unsupported_region"
    assert "specialized for chest" in route_wrist["message"].lower()

    # Chest without checkpoint configured on disk
    route_chest = route_xray(str(dummy_file), declared_region="chest")
    assert route_chest["canonical_region"] == "chest"
    assert route_chest["supported"] is True
    assert route_chest["is_available"] is False
    assert route_chest["reason_code"] == "missing_checkpoint"


# ---------------------------------------------------------------------------
# Required Input Combination Tests (A through I)
# ---------------------------------------------------------------------------

def test_combination_a_xray_image_only():
    """Combination A: X-ray image only."""
    evidence = combine_evidence(
        symptoms=[],
        report_findings=[],
        xray_result={
            "file_name": "chest_sample.png",
            "region": "chest",
            "status": "UNAVAILABLE",
            "prediction": None,
            "message": DEFAULT_UNAVAILABLE_MESSAGE,
        },
    )
    assert evidence["xray"]["received"] is True
    assert evidence["xray"]["status"] == "UNAVAILABLE"
    assert evidence["xray"]["prediction"] is None
    assert len(evidence["symptoms"]["present"]) == 0
    assert evidence["report"]["provided"] is False

    triage = apply_triage(evidence)
    assert triage["pathway"] is None
    assert triage["status"] == "INSUFFICIENT_EVIDENCE"


def test_combination_b_radiology_report_only():
    """Combination B: Radiology/X-ray report PDF only."""
    evidence = combine_evidence(
        symptoms=[],
        report_findings=[],
        xray_reports=[
            {
                "file_name": "wrist_radiology_report.pdf",
                "extracted_text": "Findings: Transverse fracture of distal radius. Impression: Distal radial fracture.",
                "structured_findings": {
                    "qualitative_findings": [
                        {"finding": "distal radius fracture", "state": "PRESENT"}
                    ],
                    "narrative_findings": [
                        {"section": "IMPRESSION", "content": "Distal radial fracture noted."}
                    ],
                },
            }
        ],
    )
    assert evidence["xray"]["received"] is False  # No image uploaded
    assert len(evidence["xray"]["reports"]) == 1
    assert evidence["xray"]["reports"][0]["file_name"] == "wrist_radiology_report.pdf"
    assert len(evidence["supporting"]) > 0
    assert any("distal radius fracture" in s.lower() for s in evidence["supporting"])


def test_combination_c_symptoms_only():
    """Combination C: Symptoms only."""
    evidence = combine_evidence(
        symptoms=[
            {"symptom": "fever", "severity": "MILD", "state": "PRESENT"},
            {"symptom": "cough", "severity": "MILD", "state": "PRESENT"},
            {"symptom": "chest pain", "state": "ABSENT"},
        ],
        report_findings=[],
        xray_result=None,
    )
    assert len(evidence["symptoms"]["present"]) == 2
    assert len(evidence["symptoms"]["absent"]) == 1
    assert evidence["xray"]["received"] is False
    assert evidence["report"]["provided"] is False


def test_combination_d_lab_report_only():
    """Combination D: General medical report or laboratory report only."""
    evidence = combine_evidence(
        symptoms=[],
        report_findings=[
            {
                "test_name": "hemoglobin",
                "value": "13.5",
                "unit": "g/dL",
                "status": "NORMAL",
                "interpretation": "NORMAL",
            },
            {
                "test_name": "wbc",
                "value": "14.2",
                "unit": "10^3/uL",
                "status": "HIGH",
                "interpretation": "HIGH",
            },
        ],
        medical_reports=[{"file_name": "cbc_panel.pdf", "report_type": "lab"}],
    )
    assert len(evidence["symptoms"]["present"]) == 0
    assert evidence["report"]["provided"] is True
    assert len(evidence["report"]["abnormal_findings"]) == 1
    assert len(evidence["report"]["normal_findings"]) == 1
    assert evidence["xray"]["received"] is False


def test_combination_e_xray_image_and_radiology_report():
    """Combination E: X-ray image + radiology report."""
    evidence = combine_evidence(
        symptoms=[],
        xray_results=[
            {
                "file_name": "chest_xray.png",
                "region": "chest",
                "status": "UNAVAILABLE",
                "prediction": None,
                "message": DEFAULT_UNAVAILABLE_MESSAGE,
            }
        ],
        xray_reports=[
            {
                "file_name": "chest_radiology_report.pdf",
                "extracted_text": "Lungs: Clear. No pleural effusion or pneumothorax.",
                "structured_findings": {
                    "qualitative_findings": [
                        {"finding": "pleural effusion", "state": "ABSENT"},
                        {"finding": "pneumothorax", "state": "ABSENT"},
                    ]
                },
            }
        ],
    )
    assert evidence["xray"]["received"] is True
    assert len(evidence["xray"]["images"]) == 1
    assert len(evidence["xray"]["reports"]) == 1
    # Check that reassuring contains the negative report findings
    assert any("pleural effusion" in r.lower() or "pneumothorax" in r.lower() for r in evidence["reassuring"])
    # Cross-modal analysis records honest NOT_APPLICABLE status since image model is unavailable
    conc = evidence["xray"]["cross_modal_concordance"]
    assert len(conc) == 1
    assert conc[0]["concordance_status"] == "NOT_APPLICABLE"


def test_combination_f_xray_image_and_symptoms():
    """Combination F: X-ray image + symptoms."""
    evidence = combine_evidence(
        symptoms=[
            {"symptom": "cough", "severity": "MODERATE", "state": "PRESENT", "duration": "5 days"},
            {"symptom": "fever", "state": "PRESENT"},
        ],
        xray_results=[
            {
                "file_name": "chest_scan.png",
                "region": "chest",
                "status": "UNAVAILABLE",
                "prediction": None,
                "message": DEFAULT_UNAVAILABLE_MESSAGE,
            }
        ],
    )
    assert len(evidence["symptoms"]["present"]) == 2
    assert evidence["xray"]["received"] is True
    assert evidence["xray"]["status"] == "UNAVAILABLE"
    # Cough and fever are preserved; unavailable model does not fabricate disease or clear lungs
    assert not any("clear" in r.lower() for r in evidence["reassuring"])


def test_combination_g_radiology_report_and_symptoms():
    """Combination G: Radiology report + symptoms."""
    evidence = combine_evidence(
        symptoms=[
            {"symptom": "wrist pain", "state": "PRESENT", "body_area": "wrist"},
        ],
        xray_reports=[
            {
                "file_name": "wrist_xray_report.pdf",
                "extracted_text": "Nondisplaced fracture of the scaphoid.",
                "structured_findings": {
                    "qualitative_findings": [
                        {"finding": "scaphoid fracture", "state": "PRESENT"}
                    ]
                },
            }
        ],
    )
    assert len(evidence["symptoms"]["present"]) == 1
    assert len(evidence["xray"]["reports"]) == 1
    assert any("scaphoid fracture" in s.lower() for s in evidence["supporting"])


def test_combination_h_all_modalities():
    """Combination H: X-ray image + medical/lab reports + symptoms."""
    evidence = combine_evidence(
        symptoms=[
            {"symptom": "shortness of breath", "severity": "MODERATE", "state": "PRESENT"},
            {"symptom": "cough", "state": "PRESENT"},
        ],
        report_findings=[
            {
                "test_name": "crp",
                "value": "45.0",
                "unit": "mg/L",
                "status": "HIGH",
                "interpretation": "HIGH",
            }
        ],
        medical_reports=[{"file_name": "bloodwork.pdf"}],
        xray_results=[
            {
                "file_name": "chest.jpg",
                "region": "chest",
                "status": "UNAVAILABLE",
                "prediction": None,
            }
        ],
    )
    assert len(evidence["symptoms"]["present"]) == 2
    assert evidence["report"]["provided"] is True
    assert len(evidence["report"]["abnormal_findings"]) == 1
    assert evidence["xray"]["received"] is True
    assert evidence["xray"]["images"][0]["region"] == "chest"


def test_combination_i_multiple_images_and_multiple_reports():
    """Combination I: Multiple supported images and/or reports attached to one assessment."""
    evidence = combine_evidence(
        symptoms=[{"symptom": "polyarthralgia", "state": "PRESENT"}],
        xray_results=[
            {
                "id": "img-1",
                "file_name": "chest_xray_pa.png",
                "region": "chest",
                "status": "UNAVAILABLE",
                "prediction": None,
            },
            {
                "id": "img-2",
                "file_name": "wrist_xray_ap.png",
                "region": "wrist",
                "status": "UNAVAILABLE",
                "prediction": None,
            },
        ],
        xray_reports=[
            {
                "file_name": "chest_report.pdf",
                "extracted_text": "Chest is clear.",
                "structured_findings": {
                    "qualitative_findings": [{"finding": "clear lungs", "state": "PRESENT"}]
                },
            },
            {
                "file_name": "wrist_report.pdf",
                "extracted_text": "No acute fracture.",
                "structured_findings": {
                    "qualitative_findings": [{"finding": "no fracture", "state": "PRESENT"}]
                },
            },
        ],
        medical_reports=[{"file_name": "esr_crp.pdf"}],
        report_findings=[
            {"test_name": "esr", "value": "12", "unit": "mm/hr", "status": "NORMAL", "interpretation": "NORMAL"}
        ],
    )
    # Check that both images are preserved
    assert len(evidence["xray"]["images"]) == 2
    assert {img["file_name"] for img in evidence["xray"]["images"]} == {"chest_xray_pa.png", "wrist_xray_ap.png"}

    # Check that both radiology reports are preserved
    assert len(evidence["xray"]["reports"]) == 2
    assert {rep["file_name"] for rep in evidence["xray"]["reports"]} == {"chest_report.pdf", "wrist_report.pdf"}

    # Check that lab report is preserved
    assert len(evidence["report"]["medical_reports"]) == 1


# ---------------------------------------------------------------------------
# Fault Isolation Tests
# ---------------------------------------------------------------------------

def test_fault_isolation_unreadable_pdf_does_not_corrupt_symptoms():
    """Unreadable or empty PDF report does not erase or invalidate entered symptoms."""
    evidence = combine_evidence(
        symptoms=[{"symptom": "fever", "state": "PRESENT", "severity": "HIGH"}],
        xray_reports=[
            {
                "file_name": "corrupt_report.pdf",
                "extracted_text": "",
                "structured_findings": {},
            }
        ],
    )
    assert len(evidence["symptoms"]["present"]) == 1
    assert evidence["symptoms"]["present"][0]["symptom"] == "fever"
    assert len(evidence["xray"]["reports"]) == 1
    assert evidence["xray"]["reports"][0]["status"] == "UNREADABLE"


def test_fault_isolation_unavailable_xray_never_creates_normal_reassurance():
    """An unavailable X-ray model must never be converted into normal / negative findings."""
    service = XRayModelService()
    res = service.analyze("nonexistent_path.png", region="chest")
    assert res["status"] == "unavailable"
    assert res["prediction"] is None
    assert res["confidence_if_valid"] is None

    # Check evidence representation
    evidence = combine_evidence(
        symptoms=[{"symptom": "cough", "state": "PRESENT"}],
        xray_result=res,
    )
    for reassur in evidence.get("reassuring", []):
        assert "normal" not in reassur.lower() or "chest" not in reassur.lower()
    assert any("unavailable" in u.lower() for u in evidence.get("unassessed", []))
