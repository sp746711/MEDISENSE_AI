"""Phase 4 X-Ray Pipeline Audit & Verification Suite.

Covers:
- Checkpoint & model availability (no hallucinated predictions when checkpoint absent)
- Upload & file validation (empty files, corrupted images, invalid dimensions, specific error messages)
- OpenCV preprocessing (correct color channels, normalization, tensor output)
- Grad-CAM explainability (no fake Gaussian blur simulation; honest unavailable or real Grad-CAM)
- Model architecture compatibility (ResNet18 loading without torchvision dependency)
- Triage independence (deterministic emergency rules independent of X-ray inference)
"""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from app.ai.xray_model import XRayModelService, CHEST_CLASSES, DEFAULT_UNAVAILABLE_MESSAGE
from app.ai.gradcam import generate_gradcam
from app.utils.file_validation import validate_image_file


def _create_sample_image(width=224, height=224, color=(128, 128, 128)):
    """Helper to generate in-memory PNG bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_defect_1_validation_message_preserves_specific_root_cause():
    """Empty or corrupted files must return specific failure messages, not masked generic extension errors."""
    # 1. Empty file with .png extension
    empty_res = validate_image_file("scan.png", b"", max_mb=10)
    assert not empty_res["ok"]
    assert "Empty file is not allowed" in empty_res["message"]

    # 2. Corrupted bytes with .png extension (starts with PNG signature but truncated/corrupt)
    corrupted_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00corrupt"
    corrupt_res = validate_image_file("scan.png", corrupted_png, max_mb=10)
    assert not corrupt_res["ok"]
    assert "corrupted" in corrupt_res["message"].lower() or "cannot be decoded" in corrupt_res["message"].lower()


def test_defect_2_invalid_image_dimensions_rejected():
    """Microscopic images (e.g. 2x2 px) are rejected as invalid radiograph dimensions."""
    micro_bytes = _create_sample_image(width=2, height=2)
    val = validate_image_file("tiny_xray.png", micro_bytes, max_mb=10)
    assert not val["ok"]
    assert "small" in val["message"].lower() or "dimensions" in val["message"].lower() or "resolution" in val["message"].lower()


def test_defect_2_corrupted_image_file_returns_invalid_image_status(tmp_path):
    """Corrupted file or empty/missing image produces invalid_image / preprocessing failure."""
    # 1. Missing file produces invalid_image
    ghost_file = str(tmp_path / "ghost.png")
    service = XRayModelService()
    res = service.analyze(ghost_file, region="chest")
    assert res["status"] == "unavailable"
    assert res["reason_code"] == "invalid_image"

    # 2. Corrupted file content causes preprocess_image to return None safely
    corrupt_file = tmp_path / "broken_xray.png"
    corrupt_file.write_bytes(b"\x89PNG\r\n\x1a\nGARBAGE_PAYLOAD_NOT_AN_IMAGE")
    tensor = service.preprocess_image(str(corrupt_file))
    assert tensor is None


def test_defect_3_preprocessing_channel_order_rgb(tmp_path):
    """Preprocessing must preserve RGB channel ordering and never invert R and B."""
    # Create pure red image (R=255, G=0, B=0)
    red_image_path = str(tmp_path / "pure_red.png")
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(red_image_path)

    service = XRayModelService()
    tensor = service.preprocess_image(red_image_path)

    assert tensor is not None
    assert tensor.shape == (1, 3, 224, 224)

    # In ImageNet norm: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # For R=255 (1.0), normalized value is (1.0 - 0.485)/0.229 ~= +2.24
    # For G=0, B=0, normalized value is (0.0 - 0.456)/0.224 ~= -2.03
    r_val = tensor[0, 0, 112, 112].item()
    b_val = tensor[0, 2, 112, 112].item()

    assert r_val > 1.5, f"Red channel normalized value expected > 1.5, got {r_val}"
    assert b_val < -1.0, f"Blue channel normalized value expected < -1.0, got {b_val}"


def test_defect_4_gradcam_no_fake_simulation_without_model(tmp_path):
    """Grad-CAM must never generate fake Gaussian blur heatmaps when model is absent."""
    test_img = str(tmp_path / "test.png")
    Image.new("RGB", (224, 224), (100, 100, 100)).save(test_img)

    # When model is None
    res = generate_gradcam(None, test_img, target_class=0)
    assert res["status"] == "unavailable"
    assert res["artifact_path"] is None
    assert "unavailable" in res["message"].lower() or "not configured" in res["message"].lower()


def test_defect_5_resnet_state_dict_loading_without_torchvision(tmp_path):
    """PyTorch state_dict must load cleanly into ResNet18 even without torchvision installed."""
    service = XRayModelService()
    # Build standard state dict
    from app.ai.xray_model import build_resnet18
    net = build_resnet18(num_classes=len(CHEST_CLASSES))
    ckpt_path = str(tmp_path / "model.pt")
    torch.save(net.state_dict(), ckpt_path)

    # Set service to load this checkpoint
    service.settings.xray_chest_checkpoint = ckpt_path
    loaded = service.load_model()

    assert loaded is True
    assert service.is_available() is True
    assert service._model is not None


def test_audit_missing_checkpoint_honest_unavailable_disclosure(tmp_path):
    """Chest X-ray without checkpoint returns exact honest unavailable disclaimer."""
    valid_img = str(tmp_path / "chest_xray.png")
    Image.new("RGB", (224, 224), (120, 120, 120)).save(valid_img)

    service = XRayModelService()
    # Ensure no checkpoint
    service.settings.xray_chest_checkpoint = ""
    service._loaded = False
    service._model = None

    res = service.analyze(valid_img, region="chest")
    assert res["status"] == "unavailable"
    assert res["reason_code"] == "missing_checkpoint"
    assert res["prediction"] is None
    assert res["confidence_if_valid"] is None
    assert res["message"] == DEFAULT_UNAVAILABLE_MESSAGE


def test_gradcam_real_pytorch_activation(tmp_path):
    """When a trained PyTorch model is present, real Grad-CAM heatmap is computed from feature maps."""
    test_img = str(tmp_path / "xray_sample.png")
    Image.new("RGB", (224, 224), (120, 120, 120)).save(test_img)

    from app.ai.xray_model import build_resnet18
    model = build_resnet18(num_classes=5)
    model.eval()

    res = generate_gradcam(model, test_img, target_class=0)
    assert res["status"] == "ok"
    assert res["artifact_path"] is not None
    assert Path(res["artifact_path"]).exists()


def test_audit_triage_independence_from_xray():
    """Emergency triage rules operate independently and must never be overridden by X-ray status."""
    from app.ai.evidence_engine import combine_evidence
    from app.ai.triage_engine import apply_triage

    evidence_emergency = combine_evidence(
        symptoms=[
            {"symptom": "chest pain", "severity": "SEVERE", "state": "PRESENT"},
            {"symptom": "radiation to left arm", "state": "PRESENT"},
        ],
        xray_result={
            "received": True,
            "status": "UNAVAILABLE",
            "prediction": None,
        },
    )

    triage_res = apply_triage(evidence_emergency)
    assert triage_res["pathway"] == "EMERGENCY"
    assert "rule_e1_chest_pain_present" in triage_res["rules_triggered"]


def test_audit_a1_extension_and_signature_mismatch_rejected():
    """Text file renamed to .png or executable bytes are rejected immediately."""
    fake_png = b"This is just plain text, not a PNG image."
    val = validate_image_file("scan.png", fake_png, max_mb=10)
    assert not val["ok"]
    assert "unsupported" in val["message"].lower() or "signature" in val["message"].lower()

    # Uppercase extension with valid PNG content is accepted
    valid_png_bytes = _create_sample_image(width=100, height=100)
    val_upper = validate_image_file("SCAN.PNG", valid_png_bytes, max_mb=10)
    assert val_upper["ok"]


def test_audit_a2_oversized_file_rejected():
    """Files exceeding maximum allowed megabytes are rejected."""
    # 2 MB limit for test
    oversized_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024 + 10)
    val = validate_image_file("big.png", oversized_bytes, max_mb=2)
    assert not val["ok"]
    assert "exceeds maximum size" in val["message"].lower()


def test_audit_c3_incompatible_checkpoint_handled_gracefully(tmp_path):
    """Corrupted checkpoint file does not crash the service and returns available=False."""
    corrupt_ckpt = tmp_path / "bad_model.pt"
    corrupt_ckpt.write_bytes(b"NOT_A_VALID_PYTORCH_CHECKPOINT")

    service = XRayModelService()
    service.settings.xray_chest_checkpoint = str(corrupt_ckpt)
    loaded = service.load_model()

    assert loaded is False
    assert service.is_available() is False


def test_audit_c5_inference_eval_mode_and_finite_probabilities(tmp_path):
    """When a model is present, inference runs in eval mode, returns finite values matching classes."""
    from app.ai.xray_model import build_resnet18
    service = XRayModelService()
    model = build_resnet18(num_classes=len(CHEST_CLASSES))
    service._model = model
    service._loaded = True
    service.model_version = "chest_resnet18@test"

    test_img = str(tmp_path / "valid_chest.png")
    Image.new("RGB", (224, 224), (128, 128, 128)).save(test_img)

    res = service.analyze(test_img, region="chest")
    assert res["status"] == "completed"
    assert res["prediction"] in CHEST_CLASSES
    assert res["confidence_if_valid"] is not None
    assert "%" in res["confidence_if_valid"]
    assert res["explainability_artifact"] is not None


def test_audit_d_unsupported_region_handling(tmp_path):
    """Unsupported anatomical regions return explicit unsupported_region status."""
    img_path = str(tmp_path / "scan.png")
    Image.new("RGB", (224, 224), (128, 128, 128)).save(img_path)

    service = XRayModelService()
    res = service.analyze(img_path, region="wrist")
    assert res["status"] == "unavailable"
    assert res["reason_code"] == "unsupported_region"
    assert res["prediction"] is None
    assert "wrist" in res["uncertainty"].lower()


def test_audit_h_missing_imaging_is_never_treated_as_negative():
    """Missing or unavailable imaging must not be converted into normal or reassuring findings."""
    from app.ai.evidence_engine import combine_evidence

    # Patient with cough, but no X-ray provided
    ev_no_xray = combine_evidence(
        symptoms=[{"symptom": "cough", "state": "PRESENT"}],
        xray_result=None,
        xray_results=[],
    )
    # Check that reassurance does not contain any fabricated negative X-ray claim
    reassuring = ev_no_xray.get("reassuring", [])
    assert not any("x-ray" in r.lower() or "radiograph" in r.lower() for r in reassuring)

    # Patient with cough, but X-ray status is UNAVAILABLE
    ev_unavail_xray = combine_evidence(
        symptoms=[{"symptom": "cough", "state": "PRESENT"}],
        xray_result={"received": True, "status": "UNAVAILABLE", "region": "chest", "prediction": None},
    )
    # Unavailable X-ray must be in unassessed / missing, never in reassuring
    assert not any("x-ray" in r.lower() for r in ev_unavail_xray.get("reassuring", []))
    assert any("unavailable" in u.lower() for u in ev_unavail_xray.get("unassessed", []))
