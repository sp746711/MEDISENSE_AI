"""MediSense AI — Goal 2 Step 6.6 Security Audit Test Suite.
Runtime Authorization, Trusted Signing Boundary & Source-Record Verification Tests.

Sections:
1. Authorization Tests (1-6)
2. Secret Management Tests (7-11)
3. Source-Record Reality Tests (12-20)
4. Verification & Fail-Closed Tests (21-25)
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from app.ai.multimodal_dataset_ingestion import (
    AuthenticMultimodalDatasetIngester,
    CrosswalkSigningAuthority,
    SourceRecordValidator,
    _get_hmac_signing_key,
    compute_crosswalk_integrity_hash,
    create_authenticated_imaging_crosswalk,
    validate_study_provenance,
)


# ==============================================================================
# Group 1: Authorization Tests (1-6)
# ==============================================================================

def test_01_untrusted_caller_requests_trusted_generator():
    """1. Untrusted caller supplying caller_role='trusted_generator' without signing authority is rejected."""
    with pytest.raises(PermissionError, match="Unauthorized caller role"):
        create_authenticated_imaging_crosswalk(
            subject_id="10001",
            hadm_id="20001",
            study_id="50001",
            caller_role="trusted_generator",
            signing_authority=None,
        )


def test_02_untrusted_caller_requests_authorized_pipeline():
    """2. Untrusted caller supplying caller_role='authorized_pipeline' without signing authority is rejected."""
    with pytest.raises(PermissionError, match="Unauthorized caller role"):
        create_authenticated_imaging_crosswalk(
            subject_id="10001",
            hadm_id="20001",
            study_id="50001",
            caller_role="authorized_pipeline",
            signing_authority=None,
        )


def test_03_untrusted_caller_requests_test_authority():
    """3. Untrusted caller supplying caller_role='test_authority' without signing authority is rejected."""
    with pytest.raises(PermissionError, match="Unauthorized caller role"):
        create_authenticated_imaging_crosswalk(
            subject_id="10001",
            hadm_id="20001",
            study_id="50001",
            caller_role="test_authority",
            signing_authority=None,
        )


def test_04_caller_modifies_role_arguments_after_obtaining_legitimate_object():
    """4. Caller modifies generator/role metadata after legitimate signing; ingestion verification flags REJECTED."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    cw = authority.create_signed_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
    )
    # Caller attempts to alter generator identity post-signing
    cw["generator"] = "INSTITUTIONAL_PACS_BRIDGE_V1"

    rec = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50001"},
        "study_hadm_crosswalk": cw,
    }
    valid, report = AuthenticMultimodalDatasetIngester().validate_and_ingest_records([rec])
    assert len(valid) == 1
    assert valid[0]["study_linkage_status"] == "REJECTED"
    assert valid[0]["xray_pathology_eligible"] == 0.0


def test_05_caller_invokes_internal_signing_functions_directly():
    """5. Direct invocation of internal key resolution or hash functions without environment credentials fails closed."""
    old_env = os.environ.pop("MEDISENSE_PROVENANCE_HMAC_SECRET", None)
    old_key = os.environ.pop("CROSSWALK_SIGNING_KEY", None)
    try:
        # In strict mode without environment key, internal key retriever returns None
        key = _get_hmac_signing_key(strict=True)
        assert key is None

        # Hash computation with strict_key=True returns None
        h = compute_crosswalk_integrity_hash(
            subject_id="10001",
            hadm_id="20001",
            study_id="50001",
            bridge_key="ED_STAY_DIRECT_LINK",
            source="MIMIC_ED_CXR_CROSSWALK",
            source_record_reference="MIMIC_IV_ED.ED_RADIOLOGY",
            generator="MIMIC_CXR_ED_BRIDGE_V1",
            strict_key=True,
        )
        assert h is None
    finally:
        if old_env is not None: os.environ["MEDISENSE_PROVENANCE_HMAC_SECRET"] = old_env
        if old_key is not None: os.environ["CROSSWALK_SIGNING_KEY"] = old_key


def test_06_caller_attempts_to_use_test_only_signing_route_in_production_mode():
    """6. Caller attempting to use test-only signing authority in production mode is strictly rejected."""
    old_env = os.environ.get("APP_ENV")
    os.environ["APP_ENV"] = "production"
    try:
        with pytest.raises(PermissionError, match="Test signing authority disabled in production environment"):
            CrosswalkSigningAuthority.get_test_authority()

        with pytest.raises(PermissionError, match="Test signing authority disabled in production environment"):
            CrosswalkSigningAuthority(is_test_authority=True)
    finally:
        if old_env is not None:
            os.environ["APP_ENV"] = old_env
        else:
            os.environ.pop("APP_ENV", None)


# ==============================================================================
# Group 2: Secret Management Tests (7-11)
# ==============================================================================

def test_07_missing_production_secret():
    """7. Missing production secret causes authority creation to fail closed."""
    old_env = os.environ.pop("MEDISENSE_PROVENANCE_HMAC_SECRET", None)
    old_key = os.environ.pop("CROSSWALK_SIGNING_KEY", None)
    old_app = os.environ.get("APP_ENV")
    os.environ["APP_ENV"] = "production"
    try:
        with pytest.raises(PermissionError, match="production HMAC secret not configured"):
            CrosswalkSigningAuthority(is_test_authority=False)
    finally:
        if old_env is not None: os.environ["MEDISENSE_PROVENANCE_HMAC_SECRET"] = old_env
        if old_key is not None: os.environ["CROSSWALK_SIGNING_KEY"] = old_key
        if old_app is not None: os.environ["APP_ENV"] = old_app
        else: os.environ.pop("APP_ENV", None)


def test_08_empty_secret():
    """8. Explicitly empty or whitespace secret fails closed."""
    os.environ["MEDISENSE_PROVENANCE_HMAC_SECRET"] = "   "
    try:
        key = _get_hmac_signing_key(strict=True)
        assert key is None
    finally:
        os.environ.pop("MEDISENSE_PROVENANCE_HMAC_SECRET", None)


def test_09_invalid_secret_configuration():
    """9. Invalid/weak secret configuration (<16 bytes) in strict mode raises ValueError."""
    os.environ["MEDISENSE_PROVENANCE_HMAC_SECRET"] = "short_key_8b"
    try:
        with pytest.raises(ValueError, match="at least 16 bytes"):
            _get_hmac_signing_key(strict=True, min_bytes=16)
    finally:
        os.environ.pop("MEDISENSE_PROVENANCE_HMAC_SECRET", None)


def test_10_attempted_secret_extraction_through_public_module_attributes():
    """10. Ingestion module does not export or expose secret keys as module attributes."""
    import app.ai.multimodal_dataset_ingestion as mod
    forbidden = ["HMAC_KEY", "HMAC_SECRET", "CROSSWALK_SALT", "_SECRET", "SECRET_KEY", "SIGNING_KEY"]
    for attr in forbidden:
        assert not hasattr(mod, attr), f"Forbidden secret attribute '{attr}' exposed on module!"


def test_11_secret_leakage_through_logs_or_exceptions():
    """11. Secret values are never formatted into exception messages, string representations, or logs."""
    raw_secret = "super_sensitive_secret_value_12345"
    authority = CrosswalkSigningAuthority(signing_key=raw_secret.encode(), is_test_authority=False)
    # Check string representation
    assert raw_secret not in str(authority)
    assert raw_secret not in repr(authority)
    assert "[PROTECTED]" in str(authority)

    # Check exception messages
    try:
        authority.create_signed_crosswalk(subject_id="INVALID_ID", hadm_id="20001", study_id="50001")
    except ValueError as e:
        assert raw_secret not in str(e)


# ==============================================================================
# Group 3: Source-Record Reality Tests (12-20)
# ==============================================================================

def test_12_fabricated_patient_identifiers():
    """12. Fabricated, negative, or empty patient identifiers cannot be signed."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    with pytest.raises(ValueError, match="fabricated"):
        authority.create_signed_crosswalk(subject_id="FABRICATED_PATIENT_999", hadm_id="20001", study_id="50001")

    with pytest.raises(ValueError, match="invalid or negative"):
        authority.create_signed_crosswalk(subject_id="10001", hadm_id="-20001", study_id="50001")

    with pytest.raises(ValueError, match="empty"):
        authority.create_signed_crosswalk(subject_id="", hadm_id="20001", study_id="50001")


def test_13_approved_source_string_with_nonexistent_record():
    """13. Approved source string referencing a nonexistent physical source file reports NOT VERIFIED."""
    validator = SourceRecordValidator(data_root=Path("nonexistent_directory_for_test"))
    manifest = validator.validate_source_record(
        dataset="MIMIC-IV-ED",
        table_ref="MIMIC_IV_ED.ED_RADIOLOGY",
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
        bridge_key="ED_STAY_DIRECT_LINK",
    )
    assert manifest["verified"] is False
    assert manifest["status"] == "NOT VERIFIED"
    assert manifest["reason"] == "source_file_missing_from_disk"


def test_14_approved_generator_name_with_fabricated_source_data():
    """14. Approved generator name with fabricated source reference is rejected during signing."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    with pytest.raises(ValueError, match="Invalid or fabricated source record reference"):
        authority.create_signed_crosswalk(
            subject_id="10001",
            hadm_id="20001",
            study_id="50001",
            source_record_reference="NONEXISTENT_FABRICATED_TABLE.CSV",
        )


def test_15_valid_hmac_over_fabricated_but_structurally_valid_payload():
    """15. A valid HMAC over an unverified source record fails closed to UNLINKED (HMAC alone does not authenticate clinical reality)."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    # Explicitly signed with source_records_verified=False
    cw = authority.create_signed_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
        source_records_verified=False,  # Unverified source data!
    )
    rec = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50001"},
        "study_hadm_crosswalk": cw,
    }
    valid, report = AuthenticMultimodalDatasetIngester().validate_and_ingest_records([rec])
    assert len(valid) == 1
    # Must fail closed: valid HMAC alone cannot grant verified encounter linkage
    assert valid[0]["study_linkage_status"] == "UNLINKED"
    assert valid[0]["xray_pathology_eligible"] == 0.0
    assert valid[0]["modality_mask"][3] == 0.0


def test_16_mismatched_subject_admission_study_or_bridge_identifiers():
    """16. Crosswalk with mismatched study_id or hadm_id is flagged REJECTED."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    cw = authority.create_signed_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
        source_records_verified=True,
    )
    # Record claims study_id 50002 while crosswalk was signed for 50001
    rec = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50002",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50002"},
        "study_hadm_crosswalk": cw,
    }
    valid, report = AuthenticMultimodalDatasetIngester().validate_and_ingest_records([rec])
    assert valid[0]["study_linkage_status"] == "REJECTED"


def test_17_missing_source_record_reference():
    """17. Crosswalk missing source_record_reference fails closed as UNLINKED."""
    cw = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "stay_id": "30001",
        "bridge_key": "30001",
        "source": "MIMIC_ED_CXR_CROSSWALK",
        "source_record_reference": "",  # Missing
        "generator": "MIMIC_CXR_ED_BRIDGE_V1",
        "link_mechanism": "ED_STAY_DIRECT_LINK",
        "crosswalk_integrity_hash": "a" * 64,
    }
    rec = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50001"},
        "study_hadm_crosswalk": cw,
    }
    valid, report = AuthenticMultimodalDatasetIngester().validate_and_ingest_records([rec])
    assert valid[0]["study_linkage_status"] == "UNLINKED"


def test_18_temporal_only_association():
    """18. Temporal-only linkage mechanism disguised as crosswalk falls back to TEMPORAL_CANDIDATE_LINK."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    cw = authority.create_signed_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
        link_mechanism="TEMPORAL_PROXIMITY_ONLY",
        source_records_verified=True,
    )
    rec = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50001"},
        "study_hadm_crosswalk": cw,
    }
    valid, report = AuthenticMultimodalDatasetIngester().validate_and_ingest_records([rec])
    assert valid[0]["study_linkage_status"] == "TEMPORAL_CANDIDATE_LINK"
    assert valid[0]["xray_pathology_eligible"] == 0.0


def test_19_missing_source_dataset():
    """19. Crosswalk missing source dataset string fails closed as UNLINKED."""
    status, src, elig, reason = validate_study_provenance(
        study_id="50001",
        subject_id="10001",
        hadm_id="20001",
        study_data={"study_hadm_crosswalk": {"subject_id": "10001", "hadm_id": "20001", "study_id": "50001", "source": ""}},
        is_temporal_match=False,
    )
    assert status == "UNLINKED"
    assert reason == "missing_provenance_source"


def test_20_conflicting_duplicate_imaging_records():
    """20. Conflicting duplicate study records fail closed."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    cw = authority.create_signed_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
        source_records_verified=True,
    )
    rec = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50001"},
        "xray_studies": [
            {"study_id": "50001", "study_hadm_crosswalk": cw, "features": [0.1, 0.2]},
            {"study_id": "50001", "study_hadm_crosswalk": cw, "features": [0.9, 0.8]},  # Conflicting features!
        ],
    }
    valid, report = AuthenticMultimodalDatasetIngester().validate_and_ingest_records([rec])
    assert valid[0]["study_linkage_status"] == "REJECTED"
    assert valid[0]["xray_pathology_eligible"] == 0.0


# ==============================================================================
# Group 4: Verification & Fail-Closed Tests (21-25)
# ==============================================================================

def test_21_validly_signed_and_genuinely_source_validated_crosswalk():
    """21. Validly signed and source-verified crosswalk achieves VERIFIED_ENCOUNTER_LINK."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    cw = authority.create_signed_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
        stay_id="30001",
        source_records_verified=True,
        source_record_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    rec = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50001"},
        "study_hadm_crosswalk": cw,
    }
    valid, report = AuthenticMultimodalDatasetIngester().validate_and_ingest_records([rec])
    assert len(valid) == 1
    assert valid[0]["study_linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert valid[0]["imaging_study_linked"] is True
    assert valid[0]["xray_pathology_eligible"] == 1.0
    assert valid[0]["modality_mask"][3] == 1.0


def test_22_tampered_crosswalk():
    """22. Tampered crosswalk (e.g. modified hadm_id) fails HMAC check and is flagged REJECTED."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    cw = authority.create_signed_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
        source_records_verified=True,
    )
    cw["hadm_id"] = "29999"  # Tampered!

    rec = {
        "subject_id": "10001",
        "hadm_id": "29999",
        "study_id": "50001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50001"},
        "study_hadm_crosswalk": cw,
    }
    valid, report = AuthenticMultimodalDatasetIngester().validate_and_ingest_records([rec])
    assert valid[0]["study_linkage_status"] == "REJECTED"


def test_23_replayed_crosswalk_for_different_study():
    """23. Replayed crosswalk for a different study fails identifier binding check."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    cw = authority.create_signed_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
        source_records_verified=True,
    )
    # Attached to study 50002
    rec = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50002",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50002"},
        "study_hadm_crosswalk": cw,
    }
    valid, report = AuthenticMultimodalDatasetIngester().validate_and_ingest_records([rec])
    assert valid[0]["study_linkage_status"] == "REJECTED"


def test_24_missing_or_malformed_signature():
    """24. Missing signature fails closed as UNLINKED; malformed signature flags REJECTED."""
    authority = CrosswalkSigningAuthority.get_test_authority()
    cw = authority.create_signed_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
        source_records_verified=True,
    )
    # Test missing signature
    cw_missing = dict(cw)
    cw_missing["crosswalk_integrity_hash"] = ""
    status, _, _, reason = validate_study_provenance("50001", "10001", "20001", {"study_hadm_crosswalk": cw_missing}, False)
    assert status == "UNLINKED"

    # Test malformed signature
    cw_malformed = dict(cw)
    cw_malformed["crosswalk_integrity_hash"] = "invalid_garbage_hash"
    status2, _, _, reason2 = validate_study_provenance("50001", "10001", "20001", {"study_hadm_crosswalk": cw_malformed}, False)
    assert status2 == "REJECTED"


def test_25_signing_and_verification_key_mismatch():
    """25. Crosswalk signed with Key A but verified with Key B fails HMAC verification and flags REJECTED."""
    key_a = b"signing_key_authority_alpha_16b"
    key_b = b"verification_key_beta_diff_16b"

    authority_a = CrosswalkSigningAuthority(signing_key=key_a, is_test_authority=False)
    cw = authority_a.create_signed_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
        source_records_verified=True,
    )

    # Validate using Key B
    status, _, _, reason = validate_study_provenance(
        study_id="50001",
        subject_id="10001",
        hadm_id="20001",
        study_data={"study_hadm_crosswalk": cw},
        is_temporal_match=False,
        verification_key=key_b,
    )
    assert status == "REJECTED"
    assert reason == "crosswalk_integrity_hash_mismatch_tampered_record"
