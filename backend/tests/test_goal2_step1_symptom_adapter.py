"""Targeted regression tests for Goal 2 - Step 1: Multimodal Symptom Adapter Fix.

Validates that:
1. Evidence engine symptom objects with 'symptom', 'status', 'state', 'duration', 'severity',
   'body_area', 'laterality', 'qualifiers', and 'context' are extracted with high fidelity.
2. Legacy 'name' dicts and plain strings are seamlessly supported.
3. All clinical states (PRESENT, ABSENT, UNKNOWN, HISTORICAL, NOT_ASKED) are preserved.
4. Duration strings (e.g. '3 weeks', '14 days', '2 months') are parsed accurately to numeric days.
5. Severity, body area, laterality, and follow-up answers are preserved.
6. Missing symptoms are not turned into negative findings.
7. Empty symptoms yield an all-zero symptom tensor and a 0.0 modality mask (never false positive).
8. Symptom tokens and feature vectors correctly reach SymptomModalityEncoder producing a 64-dim output.
9. Protected X-ray pipeline integrity is fully maintained.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import pytest
import torch

from app.ai.multimodal_condition_model import (
    MultimodalPatientRecord,
    ClinicalSymptomObservation,
    MultimodalFeatureExtractor,
    SymptomModalityEncoder,
    get_multimodal_inference_service,
    parse_duration_to_days,
    parse_symptom_observation,
)
from app.ai.evidence_engine import combine_evidence


def test_duration_parsing_units():
    """Verify parse_duration_to_days handles various clinical duration strings and numeric inputs."""
    assert parse_duration_to_days("3 weeks") == 21.0
    assert parse_duration_to_days("14 days") == 14.0
    assert parse_duration_to_days("2 months") == 60.0
    assert parse_duration_to_days("1 year") == 365.0
    assert parse_duration_to_days("48 hours") == 2.0
    assert parse_duration_to_days(21) == 21.0
    assert parse_duration_to_days(7.5) == 7.5
    assert parse_duration_to_days("5") == 5.0
    assert parse_duration_to_days(None) is None
    assert parse_duration_to_days("") is None
    assert parse_duration_to_days("unknown") is None
    assert parse_duration_to_days("unspecified") is None


def test_evidence_engine_exact_symptom_objects_extraction():
    """Verify build_record_from_inputs extracts exact evidence engine symptom dictionaries."""
    multimodal_svc = get_multimodal_inference_service()

    # Exact dictionary produced by evidence engine for present symptoms
    present_sym = {
        "finding": "knee pain",
        "symptom": "knee pain",
        "status": "PRESENT",
        "state": "PRESENT",
        "duration": "3 weeks",
        "severity": "MILD",
        "body_area": "knee",
        "laterality": "RIGHT",
        "quality": "dull",
        "domain": "musculoskeletal",
        "context": "[qualifiers: laterality=RIGHT, type=CURRENT] swollen",
        "trigger": "walking",
        "type": "CURRENT",
        "onset": "GRADUAL",
        "source": "user_input",
        "provenance": "symptom_nlp",
    }

    # Exact dictionary produced by evidence engine for absent symptoms
    absent_sym = {
        "finding": "chest pain",
        "symptom": "chest pain",
        "status": "ABSENT",
        "state": "ABSENT",
        "type": "NEGATIVE",
        "body_area": "chest",
        "context": "denies chest pain",
        "domain": "cardiovascular",
        "source": "user_input",
        "provenance": "symptom_nlp",
    }

    # Exact dictionary produced by evidence engine for unknown symptoms
    unknown_sym = {
        "finding": "shortness of breath",
        "symptom": "shortness of breath",
        "status": "UNKNOWN",
        "state": "UNKNOWN",
        "type": "UNCERTAIN",
        "body_area": "chest",
        "domain": "respiratory",
        "source": "user_input",
        "provenance": "symptom_nlp",
    }

    # Exact dictionary produced by evidence engine for historical symptoms
    historical_sym = {
        "finding": "headache",
        "symptom": "headache",
        "status": "PRESENT",
        "state": "PRESENT",
        "duration": None,
        "severity": "MODERATE",
        "body_area": "head",
        "laterality": "BILATERAL",
        "type": "HISTORICAL",
        "history": "historical occurrence",
        "source": "user_input",
        "provenance": "symptom_nlp",
    }

    not_asked_sym = {
        "symptom": "fever",
        "state": "NOT_ASKED",
    }

    record, tensor_bundle = multimodal_svc.build_record_from_inputs(
        subject_id="patient_goal2_step1",
        symptoms_present=[present_sym],
        symptoms_absent=[absent_sym],
        symptoms_unknown=[unknown_sym],
        symptoms_historical=[historical_sym],
        symptoms_not_asked=[not_asked_sym],
    )

    assert len(record.symptoms) == 5

    # 1. Knee pain (PRESENT, 21 days, MILD, laterality=RIGHT, body_area=knee)
    obs_knee = record.symptoms[0]
    assert obs_knee.symptom == "knee pain"
    assert obs_knee.canonical == "knee pain"
    assert obs_knee.state == "PRESENT"
    assert obs_knee.duration_days == 21.0
    assert obs_knee.severity == "MILD"
    assert obs_knee.body_area == "knee"
    assert obs_knee.laterality == "RIGHT"
    assert any("laterality=RIGHT" in q for q in obs_knee.qualifiers)
    assert any("swollen" in q for q in obs_knee.qualifiers)

    # 2. Chest pain (ABSENT, body_area=chest)
    obs_chest = record.symptoms[1]
    assert obs_chest.symptom == "chest pain"
    assert obs_chest.state == "ABSENT"
    assert obs_chest.body_area == "chest"

    # 3. Shortness of breath (UNKNOWN)
    obs_sob = record.symptoms[2]
    assert obs_sob.symptom == "shortness of breath"
    assert obs_sob.state == "UNKNOWN"

    # 4. Headache (HISTORICAL - recognized despite status='PRESENT' due to type='HISTORICAL')
    obs_head = record.symptoms[3]
    assert obs_head.symptom == "headache"
    assert obs_head.state == "HISTORICAL"
    assert obs_head.severity == "MODERATE"

    # 5. Fever (NOT_ASKED)
    obs_fever = record.symptoms[4]
    assert obs_fever.symptom == "fever"
    assert obs_fever.state == "NOT_ASKED"


def test_symptom_features_reach_multimodal_encoder():
    """Verify that extracted features produce correct 6-dim vectors and reach SymptomModalityEncoder."""
    multimodal_svc = get_multimodal_inference_service()

    present_sym = {
        "symptom": "knee pain",
        "duration": "3 weeks",
        "severity": "MILD",
        "body_area": "knee",
        "laterality": "RIGHT",
        "follow_up_answer": "worse after stairs",
    }
    absent_sym = {
        "symptom": "chest pain",
        "state": "ABSENT",
    }
    historical_sym = {
        "symptom": "headache",
        "type": "HISTORICAL",
    }
    not_asked_sym = {
        "symptom": "fever",
        "state": "NOT_ASKED",
    }
    unknown_sym = {
        "symptom": "fatigue",
        "state": "UNKNOWN",
    }

    record, tensor_bundle = multimodal_svc.build_record_from_inputs(
        subject_id="test_tensors",
        symptoms_present=[present_sym],
        symptoms_absent=[absent_sym],
        symptoms_unknown=[unknown_sym],
        symptoms_historical=[historical_sym],
        symptoms_not_asked=[not_asked_sym],
    )

    tokens = tensor_bundle["symptom_tokens"]
    features = tensor_bundle["symptom_features"]
    mask = tensor_bundle["modality_mask"]

    # Check tensor shapes
    assert tokens.shape == (1, 128)
    assert features.shape == (1, 128, 6)
    assert mask.shape == (1, 5)

    # Symptom modality must be active (1.0)
    assert mask[0, 0].item() == 1.0

    # Token IDs must be non-zero for occupied slots
    for i in range(5):
        assert tokens[0, i].item() > 0

    # Knee pain feature vector checks:
    # [state_code, dur_clamped, sev_code, has_qual, has_followup, is_active]
    knee_feat = features[0, 0].tolist()
    assert knee_feat[0] == 1.0  # PRESENT
    expected_dur = math.log1p(21.0) / 5.0
    assert pytest.approx(knee_feat[1], abs=1e-3) == expected_dur
    assert pytest.approx(knee_feat[2], abs=1e-3) == 0.33  # MILD
    assert knee_feat[3] == 1.0  # qualifiers/laterality present
    assert knee_feat[4] == 1.0  # follow_up_answer present
    assert knee_feat[5] == 1.0  # is_active

    # Chest pain: ABSENT -> state_code == -1.0
    chest_feat = features[0, 1].tolist()
    assert chest_feat[0] == -1.0
    assert chest_feat[5] == 1.0

    # Fatigue: UNKNOWN -> state_code == 0.0
    fatigue_feat = features[0, 2].tolist()
    assert fatigue_feat[0] == 0.0
    assert fatigue_feat[5] == 1.0

    # Headache: HISTORICAL -> state_code == 0.5
    head_feat = features[0, 3].tolist()
    assert head_feat[0] == 0.5
    assert head_feat[5] == 1.0

    # Fever: NOT_ASKED -> state_code == -0.5
    fever_feat = features[0, 4].tolist()
    assert fever_feat[0] == -0.5
    assert fever_feat[5] == 1.0

    # Forward pass through SymptomModalityEncoder
    encoder = SymptomModalityEncoder(vocab_size=512, embed_dim=64, out_dim=64)
    h_sym = encoder(tokens, features)
    assert h_sym.shape == (1, 64)
    assert not torch.isnan(h_sym).any()


def test_legacy_name_representation_and_plain_strings():
    """Verify backward compatibility with legacy name dicts and raw strings."""
    multimodal_svc = get_multimodal_inference_service()

    legacy_present = {"name": "cough", "duration_days": 5, "severity": "MODERATE"}
    legacy_absent = {"name": "fever"}
    string_sym = "joint pain"

    record, tensor_bundle = multimodal_svc.build_record_from_inputs(
        subject_id="test_legacy",
        symptoms_present=[legacy_present, string_sym],
        symptoms_absent=[legacy_absent],
    )

    assert len(record.symptoms) == 3
    assert record.symptoms[0].symptom == "cough"
    assert record.symptoms[0].duration_days == 5.0
    assert record.symptoms[0].severity == "MODERATE"
    assert record.symptoms[1].symptom == "joint pain"
    assert record.symptoms[1].state == "PRESENT"
    assert record.symptoms[2].symptom == "fever"
    assert record.symptoms[2].state == "ABSENT"


def test_missing_symptoms_do_not_create_negative_findings_or_activate_modality():
    """Verify that when no symptoms are provided, symptom features are all 0 and mask is 0.0."""
    multimodal_svc = get_multimodal_inference_service()

    record, tensor_bundle = multimodal_svc.build_record_from_inputs(
        subject_id="test_empty",
        symptoms_present=[],
        symptoms_absent=[],
        symptoms_unknown=[],
    )

    assert len(record.symptoms) == 0

    tokens = tensor_bundle["symptom_tokens"]
    features = tensor_bundle["symptom_features"]
    mask = tensor_bundle["modality_mask"]

    # All tokens are 0
    assert (tokens == 0).all()
    # All features are 0.0
    assert (features == 0.0).all()
    # Modality mask for symptoms is 0.0 (NOT active)
    assert mask[0, 0].item() == 0.0


def test_live_evidence_engine_output_to_multimodal_record():
    """Verify that actual output from evidence_engine.combine_evidence routes cleanly into build_record_from_inputs."""
    evidence = combine_evidence(
        symptoms=[
            {
                "symptom": "knee pain",
                "state": "PRESENT",
                "duration": "3 weeks",
                "severity": "MILD",
                "body_area": "knee",
                "laterality": "RIGHT",
            },
            {
                "symptom": "chest pain",
                "state": "ABSENT",
                "body_area": "chest",
            },
        ]
    )

    multimodal_svc = get_multimodal_inference_service()
    record, tensor_bundle = multimodal_svc.build_record_from_inputs(
        subject_id="live_evidence_test",
        symptoms_present=evidence["symptoms"]["present"],
        symptoms_absent=evidence["symptoms"]["absent"],
        symptoms_unknown=evidence["symptoms"]["unknown"],
    )

    assert len(record.symptoms) == 2
    knee_obs = next(s for s in record.symptoms if s.symptom == "knee pain")
    assert knee_obs.state == "PRESENT"
    assert knee_obs.duration_days == 21.0
    assert knee_obs.body_area == "knee"
    assert knee_obs.laterality == "RIGHT"

    chest_obs = next(s for s in record.symptoms if s.symptom == "chest pain")
    assert chest_obs.state == "ABSENT"

    mask = tensor_bundle["modality_mask"]
    assert mask[0, 0].item() == 1.0


def test_protected_xray_assets_sha256_unmodified():
    """Verify every protected X-ray asset against the flat repository manifest.

    The manifest stores paths as top-level keys (not under an ``assets`` key).
    Resolve paths from the repository root so this test works regardless of
    the current working directory used to invoke pytest.
    """
    backend_root = Path(__file__).resolve().parents[1]
    repository_root = backend_root.parent
    manifest_path = backend_root / "models" / "xray_protected_manifest.json"

    assert manifest_path.is_file(), (
        f"Protected X-ray manifest missing: {manifest_path}"
    )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    assert isinstance(manifest, dict), "Manifest must be a JSON object."
    assert len(manifest) == 15, (
        f"Expected exactly 15 protected assets; found {len(manifest)}."
    )

    for relative_path, expected_hash in manifest.items():
        assert isinstance(relative_path, str) and relative_path.strip(), (
            "Manifest contains an invalid asset path."
        )
        assert isinstance(expected_hash, str), (
            f"Invalid hash value for {relative_path}."
        )
        assert len(expected_hash) == 64 and all(
            char in "0123456789abcdefABCDEF" for char in expected_hash
        ), f"Invalid SHA-256 format for {relative_path}."

        relative = Path(relative_path)
        assert not relative.is_absolute(), (
            f"Absolute paths are not permitted: {relative_path}"
        )
        assert ".." not in relative.parts, (
            f"Path traversal detected: {relative_path}"
        )

        asset_path = repository_root / relative
        assert asset_path.is_file(), (
            f"Protected asset missing: {relative_path}"
        )

        digest = hashlib.sha256()
        with asset_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        actual_hash = digest.hexdigest()
        assert actual_hash.lower() == expected_hash.lower(), (
            f"Protected asset SHA-256 mismatch: {relative_path}"
        )
