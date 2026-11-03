"""Focused regression and adversarial tests for Goal 2 Step 6.1 and 6.2:
Verified Imaging Provenance, Fail-Closed Linkage, and Multi-Study Isolation.

Test cases:
Part A — Core Fail-Closed Linkage Scenarios:
1. Explicitly verified encounter linkage.
2. Missing study_encounter_linked.
3. study_encounter_linked=None.
4. study_encounter_linked=False.
5. Temporal match without verified linkage.
6. Temporal match with explicit verified linkage.
7. X-ray image present but encounter linkage unavailable.
8. Radiology report linked but image not linked.
9. Image linked but radiology report not linked.
10. Multiple X-ray studies with different linkage states.

Part B — Adversarial Provenance Verification & Isolation (Section 3):
11. Empty provenance dictionary.
12. Arbitrary non-empty provenance dictionary.
13. Provenance containing only a temporal match.
14. Malformed crosswalk (non-dict).
15. Crosswalk with mismatched subject ID.
16. Crosswalk with mismatched study ID.
17. Crosswalk with mismatched encounter ID.
18. Explicit VERIFIED status without supporting provenance.
19. Two studies where only one is verified.
20. Multiple studies where none are verified.
21. One verified study plus one rejected study.
22. Duplicate study IDs with conflicting linkage states.
"""

from __future__ import annotations

import pytest
from app.ai.multimodal_dataset_ingestion import (
    AuthenticMultimodalDatasetIngester,
    CrosswalkSigningAuthority,
    create_authenticated_imaging_crosswalk as _base_create_authenticated_imaging_crosswalk,
)

_test_signing_auth = CrosswalkSigningAuthority.get_test_authority()


def create_authenticated_imaging_crosswalk(*args, **kwargs):
    """Test helper that attaches authorized test signing authority unless caller specifies authority/role."""
    if "signing_authority" not in kwargs and "caller_role" not in kwargs:
        kwargs["signing_authority"] = _test_signing_auth
    return _base_create_authenticated_imaging_crosswalk(*args, **kwargs)


# ==============================================================================
# Part A: Core Fail-Closed Linkage Scenarios
# ==============================================================================

def test_01_explicitly_verified_encounter_linkage():
    """1. Explicitly verified encounter linkage sets study_linked=True, xray_eligible=1.0, and mask[3]=1.0."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10003001",
        "hadm_id": "20004001",
        "study_id": "50005001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough", "fever"],
        "xray": {"study_id": "50005001"},
        "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
            subject_id="10003001",
            hadm_id="20004001",
            study_id="50005001",
            source="MIMIC_IV_ED_TRANSFER",
        ),
        "xray_pathology_eligible": 1.0,
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["imaging_study_linked"] is True
    assert r["study_linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert r["xray_pathology_eligible"] == 1.0
    assert r["modality_mask"] == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert r["temporal_linkage_evidence"]["study_encounter_linked"] is True


def test_02_missing_study_encounter_linkage():
    """2. Missing study_encounter_linked fails closed (mask[3]=0.0, xray_eligible=0.0, status='UNLINKED')."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10003002",
        "hadm_id": "20004002",
        "study_id": "50005002",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50005002"},
        # study_encounter_linked is missing!
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["imaging_study_linked"] is False
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"] == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_03_study_encounter_linked_is_none():
    """3. study_encounter_linked=None fails closed (mask[3]=0.0, xray_eligible=0.0)."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10003003",
        "hadm_id": "20004003",
        "study_id": "50005003",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50005003"},
        "study_encounter_linked": None,
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["imaging_study_linked"] is False
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"] == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_04_study_encounter_linked_is_false():
    """4. study_encounter_linked=False explicitly rejected (mask[3]=0.0, status='REJECTED')."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10003004",
        "hadm_id": "20004004",
        "study_id": "50005004",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50005004"},
        "study_encounter_linked": False,
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["imaging_study_linked"] is False
    assert r["study_linkage_status"] == "REJECTED"
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"] == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_05_temporal_match_without_verified_linkage():
    """5. Temporal match without verified linkage is TEMPORAL_CANDIDATE_LINK, fails closed from supervision."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10003005",
        "hadm_id": "20004005",
        "study_id": "50005005",
        "target_condition": "Pneumonia",
        "admittime": "2026-03-01T12:00:00Z",
        "dischtime": "2026-03-05T12:00:00Z",
        "study_time": "2026-03-02T12:00:00Z",  # In window!
        "symptoms": ["cough"],
        "xray": {"study_id": "50005005"},
        # No verified linkage field!
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["temporal_linkage_evidence"]["imaging_in_window"] is True
    assert r["study_linkage_status"] == "TEMPORAL_CANDIDATE_LINK"
    # Crucial: temporal match ALONE must NOT enable supervised training
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"] == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_06_temporal_match_with_explicit_verified_linkage():
    """6. Temporal match with explicit verified linkage establishes VERIFIED_ENCOUNTER_LINK."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10003006",
        "hadm_id": "20004006",
        "study_id": "50005006",
        "target_condition": "Pneumonia",
        "admittime": "2026-03-01T12:00:00Z",
        "dischtime": "2026-03-05T12:00:00Z",
        "study_time": "2026-03-02T12:00:00Z",  # In window
        "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
            subject_id="10003006",
            hadm_id="20004006",
            study_id="50005006",
            source="MIMIC_CXR_HADM_MATCH",
        ),
        "symptoms": ["cough"],
        "xray": {"study_id": "50005006"},
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["temporal_linkage_evidence"]["imaging_in_window"] is True
    assert r["study_linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert r["imaging_study_linked"] is True
    assert r["xray_pathology_eligible"] == 1.0
    assert r["modality_mask"] == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_07_xray_image_present_but_encounter_linkage_unavailable():
    """7. X-ray image present but encounter linkage unavailable preserves encounter while masking X-ray."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10003007",
        "hadm_id": "20004007",
        "study_id": "50005007",
        "target_condition": "Pneumonia",
        "symptoms": ["cough", "fever"],
        "laboratory": {"wbc": 12.0},
        "xray": {"study_id": "50005007"},
        "study_linkage_status": "UNKNOWN",
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["imaging_study_linked"] is False
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"] == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_08_radiology_report_linked_but_image_not_linked():
    """8. Radiology report linked but image not linked keeps radiology mask=1.0 and image mask=0.0."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10003008",
        "hadm_id": "20004008",
        "study_id": "50005008",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "radiology": {"consolidation": 1.0},   # Report present and linked
        "xray": {"study_id": "50005008"},     # Image present
        "study_encounter_linked": False,       # But image linkage failed!
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["temporal_linkage_evidence"]["radiology_linked"] is True
    assert r["temporal_linkage_evidence"]["xray_study_linked"] is False
    assert r["study_linkage_status"] == "REJECTED"
    assert r["xray_pathology_eligible"] == 0.0
    # Modality mask: symptoms=1, lab=0, rad=1, xray=0, exam=0
    assert r["modality_mask"] == [1.0, 0.0, 1.0, 0.0, 0.0]


def test_09_image_linked_but_radiology_report_not_linked():
    """9. Image linked but radiology report not linked keeps image mask=1.0 and radiology mask=0.0."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10003009",
        "hadm_id": "20004009",
        "study_id": "50005009",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50005009"},
        "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
            subject_id="10003009",
            hadm_id="20004009",
            study_id="50005009",
            source="MIMIC_DIRECT_CROSSWALK",
        ),
        # No radiology report
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["temporal_linkage_evidence"]["radiology_linked"] is False
    assert r["temporal_linkage_evidence"]["xray_study_linked"] is True
    assert r["study_linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert r["xray_pathology_eligible"] == 1.0
    # Modality mask: symptoms=1, lab=0, rad=0, xray=1, exam=0
    assert r["modality_mask"] == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_10_multiple_xray_studies_with_different_linkage_states():
    """10. Multiple X-ray studies with different linkage states evaluate individually."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10003010",
        "hadm_id": "20004010",
        "study_id": "50005010_1",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50005010_1"},
        "xray_studies": [
            {
                "study_id": "50005010_1",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10003010",
                    hadm_id="20004010",
                    study_id="50005010_1",
                    source="MIMIC_ED_ANCHOR",
                ),
            },
            {"study_id": "50005010_2", "study_encounter_linked": False},
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert r["imaging_study_linked"] is True
    assert r["xray_pathology_eligible"] == 1.0
    assert r["modality_mask"] == [1.0, 0.0, 0.0, 1.0, 0.0]

    # Check that individual study provenance entries are preserved
    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 2
    assert prov[0]["study_id"] == "50005010_1"
    assert prov[0]["linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert prov[0]["pathology_eligible"] == 1.0
    assert prov[1]["study_id"] == "50005010_2"
    assert prov[1]["linkage_status"] == "REJECTED"
    assert prov[1]["pathology_eligible"] == 0.0


# ==============================================================================
# Part B: Adversarial Provenance Verification & Isolation (Section 3)
# ==============================================================================

def test_11_adversarial_empty_provenance_dictionary():
    """11. Empty provenance dictionary {} must fail closed and be classified as UNLINKED."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004001",
        "hadm_id": "20005001",
        "study_id": "50006001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006001"},
        "study_hadm_crosswalk": {},  # Empty dict!
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_12_adversarial_arbitrary_non_empty_provenance_dictionary():
    """12. Arbitrary non-empty provenance dictionary without required keys must fail closed."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004002",
        "hadm_id": "20005002",
        "study_id": "50006002",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006002"},
        "encounter_link_provenance": {"random_key": "some_arbitrary_value"},  # Arbitrary dict!
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_13_adversarial_provenance_containing_only_temporal_match():
    """13. Provenance containing only a temporal match marker must NOT establish verified linkage."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004003",
        "hadm_id": "20005003",
        "study_id": "50006003",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006003"},
        "study_hadm_crosswalk": {
            "subject_id": "10004003",
            "hadm_id": "20005003",
            "study_id": "50006003",
            "link_type": "TEMPORAL_MATCH_ONLY",  # Temporal only!
        },
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "TEMPORAL_CANDIDATE_LINK"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_14_adversarial_malformed_crosswalk_non_dict():
    """14. Malformed crosswalk (string, int, list instead of dict) must fail closed as UNLINKED."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004004",
        "hadm_id": "20005004",
        "study_id": "50006004",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006004"},
        "study_hadm_crosswalk": "MALFORMED_STRING_PROVENANCE",  # Non-dict!
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_15_adversarial_crosswalk_with_mismatched_subject_id():
    """15. Crosswalk with mismatched subject ID must be strictly REJECTED."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004005",
        "hadm_id": "20005005",
        "study_id": "50006005",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006005"},
        "study_hadm_crosswalk": {
            "subject_id": "99999999",  # WRONG SUBJECT!
            "hadm_id": "20005005",
            "study_id": "50006005",
            "source": "MIMIC_CROSSWALK",
        },
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_16_adversarial_crosswalk_with_mismatched_study_id():
    """16. Crosswalk with mismatched study ID must be strictly REJECTED."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004006",
        "hadm_id": "20005006",
        "study_id": "50006006",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006006"},
        "study_hadm_crosswalk": {
            "subject_id": "10004006",
            "hadm_id": "20005006",
            "study_id": "88888888",  # WRONG STUDY ID!
            "source": "MIMIC_CROSSWALK",
        },
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_17_adversarial_crosswalk_with_mismatched_encounter_id():
    """17. Crosswalk with mismatched encounter ID must be strictly REJECTED."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004007",
        "hadm_id": "20005007",
        "study_id": "50006007",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006007"},
        "study_hadm_crosswalk": {
            "subject_id": "10004007",
            "hadm_id": "77777777",  # WRONG ENCOUNTER ID!
            "study_id": "50006007",
            "source": "MIMIC_CROSSWALK",
        },
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_18_adversarial_explicit_verified_status_without_supporting_provenance():
    """18. Explicit VERIFIED claim without supporting provenance mapping must fail closed."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004008",
        "hadm_id": "20005008",
        "study_id": "50006008",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006008"},
        "study_encounter_linked": True,  # Claimed True
        "study_linkage_status": "VERIFIED_ENCOUNTER_LINK",  # Claimed VERIFIED
        # But NO crosswalk or encounter_link_provenance dict provided!
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    # Self-assertion without provenance must NOT yield verified encounter link
    assert r["study_linkage_status"] in ("UNLINKED", "TEMPORAL_CANDIDATE_LINK")
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_19_adversarial_two_studies_where_only_one_is_verified():
    """19. Two studies where only one is verified: only the verified study is eligible."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004009",
        "hadm_id": "20005009",
        "study_id": "50006009_A",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006009_A"},
        "xray_studies": [
            {
                "study_id": "50006009_A",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10004009",
                    hadm_id="20005009",
                    study_id="50006009_A",
                    source="MIMIC_ED_ANCHOR",
                ),
            },
            {
                "study_id": "50006009_B",
                # Study B has no provenance
            },
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert r["imaging_study_linked"] is True
    assert r["xray_pathology_eligible"] == 1.0
    assert r["modality_mask"][3] == 1.0

    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 2
    # Study A is eligible
    assert prov[0]["study_id"] == "50006009_A"
    assert prov[0]["linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert prov[0]["pathology_eligible"] == 1.0
    # Study B is NOT eligible
    assert prov[1]["study_id"] == "50006009_B"
    assert prov[1]["linkage_status"] == "UNLINKED"
    assert prov[1]["pathology_eligible"] == 0.0


def test_20_adversarial_multiple_studies_where_none_are_verified():
    """20. Multiple studies where none are verified: whole encounter masks out X-ray."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004010",
        "hadm_id": "20005010",
        "study_id": "50006010_1",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006010_1"},
        "xray_studies": [
            {"study_id": "50006010_1"},
            {"study_id": "50006010_2", "study_encounter_linked": None},
            {"study_id": "50006010_3", "study_hadm_crosswalk": {}},
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0

    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 3
    assert all(p["pathology_eligible"] == 0.0 for p in prov)


def test_21_adversarial_one_verified_study_plus_one_rejected_study():
    """21. One verified study plus one rejected study: verified study remains eligible, rejected excluded."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004011",
        "hadm_id": "20005011",
        "study_id": "50006011_V",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006011_V"},
        "xray_studies": [
            {
                "study_id": "50006011_V",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10004011",
                    hadm_id="20005011",
                    study_id="50006011_V",
                    source="MIMIC_VALID_TRANSFER",
                ),
            },
            {
                "study_id": "50006011_R",
                "study_encounter_linked": False,  # Explicitly rejected
            },
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    # Verified study establishes encounter linkage
    assert r["study_linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert r["imaging_study_linked"] is True
    assert r["xray_pathology_eligible"] == 1.0
    assert r["modality_mask"][3] == 1.0

    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 2
    assert prov[0]["study_id"] == "50006011_V"
    assert prov[0]["linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert prov[0]["pathology_eligible"] == 1.0
    assert prov[1]["study_id"] == "50006011_R"
    assert prov[1]["linkage_status"] == "REJECTED"
    assert prov[1]["pathology_eligible"] == 0.0


def test_22_adversarial_duplicate_study_ids_with_conflicting_linkage_states():
    """22. Duplicate study IDs with conflicting linkage states must be flagged REJECTED and fail closed."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10004012",
        "hadm_id": "20005012",
        "study_id": "50006012_DUP",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50006012_DUP"},
        "xray_studies": [
            {
                # Duplicate entry 1 claims verified
                "study_id": "50006012_DUP",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10004012",
                    hadm_id="20005012",
                    study_id="50006012_DUP",
                    source="MIMIC_DIRECT_CROSSWALK",
                ),
            },
            {
                # Duplicate entry 2 claims rejected
                "study_id": "50006012_DUP",
                "study_encounter_linked": False,
            },
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    # Conflicting duplicate claims must be rejected and fail closed!
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0

    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 2
    assert prov[0]["linkage_status"] == "REJECTED"
    assert prov[0]["exclusion_reason"] == "duplicate_study_ids_with_conflicting_linkage_states"
    assert prov[1]["linkage_status"] == "REJECTED"


# ==============================================================================
# Part C: Goal 2 Step 6.3 Duplicate Study Protection & Source Authenticity
# ==============================================================================

def test_23_duplicate_study_id_repeated_twice_with_identical_verified_provenance():
    """23. Same study ID repeated twice with identical verified provenance: deduplicate cleanly."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10005001",
        "hadm_id": "20006001",
        "study_id": "50007001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50007001"},
        "xray_studies": [
            {
                "study_id": "50007001",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005001",
                    hadm_id="20006001",
                    study_id="50007001",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
            {
                "study_id": "50007001",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005001",
                    hadm_id="20006001",
                    study_id="50007001",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert r["imaging_study_linked"] is True
    assert r["xray_pathology_eligible"] == 1.0
    assert r["modality_mask"][3] == 1.0

    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 2
    # First instance is verified
    assert prov[0]["study_id"] == "50007001"
    assert prov[0]["linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert prov[0]["pathology_eligible"] == 1.0
    assert prov[0]["exclusion_reason"] is None
    # Duplicate instance is excluded
    assert prov[1]["study_id"] == "50007001"
    assert prov[1]["linkage_status"] == "DUPLICATE_EXCLUDED"
    assert prov[1]["pathology_eligible"] == 0.0
    assert prov[1]["exclusion_reason"] == "identical_duplicate_study_deduplicated"


def test_24_duplicate_study_id_repeated_with_identical_features():
    """24. Same study ID repeated with identical features: contributes exactly once to aggregation."""
    ingester = AuthenticMultimodalDatasetIngester()

    feat = [0.25] * 14
    rec = {
        "subject_id": "10005002",
        "hadm_id": "20006002",
        "study_id": "50007002",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50007002"},
        "xray_studies": [
            {
                "study_id": "50007002",
                "features": list(feat),
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005002",
                    hadm_id="20006002",
                    study_id="50007002",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
            {
                "study_id": "50007002",
                "features": list(feat),
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005002",
                    hadm_id="20006002",
                    study_id="50007002",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["imaging_study_linked"] is True
    # Aggregated features must equal single contribution
    assert len(r["xray_pathology_features"]) == 14
    assert all(abs(val - 0.25) < 1e-5 for val in r["xray_pathology_features"])


def test_25_duplicate_study_id_repeated_with_different_features():
    """25. Same study ID repeated with different features: reject conflicting duplicate."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10005003",
        "hadm_id": "20006003",
        "study_id": "50007003",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50007003"},
        "xray_studies": [
            {
                "study_id": "50007003",
                "features": [0.1] * 14,
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005003",
                    hadm_id="20006003",
                    study_id="50007003",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
            {
                "study_id": "50007003",
                "features": [0.9] * 14,
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005003",
                    hadm_id="20006003",
                    study_id="50007003",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0

    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 2
    assert all(p["linkage_status"] == "REJECTED" for p in prov)
    assert all(p["exclusion_reason"] == "duplicate_study_ids_with_conflicting_features" for p in prov)


def test_26_duplicate_study_id_with_different_provenance_sources():
    """26. Same study ID with different provenance sources: reject conflicting duplicate."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10005004",
        "hadm_id": "20006004",
        "study_id": "50007004",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50007004"},
        "xray_studies": [
            {
                "study_id": "50007004",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005004",
                    hadm_id="20006004",
                    study_id="50007004",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
            {
                "study_id": "50007004",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005004",
                    hadm_id="20006004",
                    study_id="50007004",
                    source="INSTITUTIONAL_PACS_CROSSWALK",
                    source_record_reference="INSTITUTIONAL_PACS_ORDER_CROSSWALK",
                    generator="INSTITUTIONAL_PACS_BRIDGE_V1",
                ),
            },
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0

    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 2
    assert all(p["linkage_status"] == "REJECTED" for p in prov)
    assert all(p["exclusion_reason"] == "duplicate_study_ids_with_conflicting_provenance_sources" for p in prov)


def test_27_duplicate_study_id_with_conflicting_encounter_ids():
    """27. Same study ID with conflicting encounter IDs: reject conflicting duplicate."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10005005",
        "hadm_id": "20006005",
        "study_id": "50007005",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50007005"},
        "xray_studies": [
            {
                "study_id": "50007005",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005005",
                    hadm_id="20006005",
                    study_id="50007005",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
            {
                "study_id": "50007005",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005005",
                    hadm_id="29999999",  # Conflicting encounter ID!
                    study_id="50007005",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0

    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 2
    assert all(p["linkage_status"] == "REJECTED" for p in prov)
    assert all(p["exclusion_reason"] == "duplicate_study_ids_with_conflicting_encounter_ids" for p in prov)


def test_28_duplicate_study_id_with_verified_and_unverified_entries():
    """28. Same study ID with verified and unverified entries: reject conflicting duplicate."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10005006",
        "hadm_id": "20006006",
        "study_id": "50007006",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50007006"},
        "xray_studies": [
            {
                "study_id": "50007006",
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005006",
                    hadm_id="20006006",
                    study_id="50007006",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
            {
                "study_id": "50007006",
                # Second entry has no crosswalk (unverified)
            },
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0

    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 2
    assert all(p["linkage_status"] == "REJECTED" for p in prov)
    assert all(p["exclusion_reason"] == "duplicate_study_ids_with_conflicting_linkage_states" for p in prov)


def test_29_different_study_ids_with_separate_verified_provenance():
    """29. Different study IDs with separate verified provenance: both contribute, neither merged nor excluded."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10005007",
        "hadm_id": "20006007",
        "study_id": "50007007_A",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50007007_A"},
        "xray_studies": [
            {
                "study_id": "50007007_A",
                "features": [0.2] * 14,
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005007",
                    hadm_id="20006007",
                    study_id="50007007_A",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
            {
                "study_id": "50007007_B",
                "features": [0.6] * 14,
                "study_hadm_crosswalk": create_authenticated_imaging_crosswalk(
                    subject_id="10005007",
                    hadm_id="20006007",
                    study_id="50007007_B",
                    source="MIMIC_ED_CXR_CROSSWALK",
                ),
            },
        ],
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert r["imaging_study_linked"] is True
    assert r["xray_pathology_eligible"] == 1.0
    assert r["modality_mask"][3] == 1.0

    prov = r["temporal_linkage_evidence"]["xray_studies_provenance"]
    assert len(prov) == 2
    assert prov[0]["study_id"] == "50007007_A"
    assert prov[0]["linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert prov[0]["pathology_eligible"] == 1.0
    assert prov[1]["study_id"] == "50007007_B"
    assert prov[1]["linkage_status"] == "VERIFIED_ENCOUNTER_LINK"
    assert prov[1]["pathology_eligible"] == 1.0

    # Aggregated features must be the mean of study A and study B: (0.2 + 0.6) / 2 = 0.4
    assert len(r["xray_pathology_features"]) == 14
    assert all(abs(val - 0.4) < 1e-5 for val in r["xray_pathology_features"])


def test_30_unapproved_or_missing_provenance_source_fails_closed():
    """30. Unapproved or missing provenance source fails closed and does not assign a trusted default."""
    ingester = AuthenticMultimodalDatasetIngester()

    # Case A: Missing source in crosswalk
    rec_missing = {
        "subject_id": "10005008",
        "hadm_id": "20006008",
        "study_id": "50007008",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50007008"},
        "study_hadm_crosswalk": {
            "subject_id": "10005008",
            "hadm_id": "20006008",
            "study_id": "50007008",
            # No "source" field!
        },
    }
    valid_m, _ = ingester.validate_and_ingest_records([rec_missing])
    assert len(valid_m) == 1
    assert valid_m[0]["study_linkage_status"] == "UNLINKED"
    assert valid_m[0]["imaging_study_linked"] is False
    assert valid_m[0]["xray_pathology_eligible"] == 0.0
    assert valid_m[0]["modality_mask"][3] == 0.0

    # Case B: Arbitrary/unapproved source string
    rec_arbitrary = {
        "subject_id": "10005009",
        "hadm_id": "20006009",
        "study_id": "50007009",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50007009"},
        "study_hadm_crosswalk": {
            "subject_id": "10005009",
            "hadm_id": "20006009",
            "study_id": "50007009",
            "source": "my_arbitrary_untrusted_source",
        },
    }
    valid_a, _ = ingester.validate_and_ingest_records([rec_arbitrary])
    assert len(valid_a) == 1
    assert valid_a[0]["study_linkage_status"] == "UNLINKED"
    assert valid_a[0]["imaging_study_linked"] is False
    assert valid_a[0]["xray_pathology_eligible"] == 0.0
    assert valid_a[0]["modality_mask"][3] == 0.0


# ==============================================================================
# Part D: Goal 2 Step 6.4 Authenticated Provenance & Crosswalk Trust-Boundary
# ==============================================================================

def test_31_adversarial_forged_dictionary_with_approved_source_name():
    """31. Forged dictionary with an approved source name fails closed without authentic generator signature."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10006001",
        "hadm_id": "20007001",
        "study_id": "50008001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50008001"},
        "study_hadm_crosswalk": {
            "subject_id": "10006001",
            "hadm_id": "20007001",
            "study_id": "50008001",
            "source": "MIMIC_ED_CXR_CROSSWALK",
            # Forged: No generator, no signature/hash, no source-record reference!
        },
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    # Whitelist spoofing attempt MUST fail closed!
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_32_adversarial_manually_supplied_matching_identifiers():
    """32. Manually supplied matching identifiers without verifiable pipeline contract fail closed."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10006002",
        "hadm_id": "20007002",
        "study_id": "50008002",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50008002"},
        "encounter_link_provenance": {
            "subject_id": "10006002",
            "hadm_id": "20007002",
            "study_id": "50008002",
            "source": "MIMIC_IV_ED_TRANSFER",
            "caller_claim": "manual_entry_by_physician",
        },
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_33_adversarial_missing_source_record_reference():
    """33. Crosswalk missing source-record reference fails closed."""
    ingester = AuthenticMultimodalDatasetIngester()

    cw = create_authenticated_imaging_crosswalk(
        subject_id="10006003",
        hadm_id="20007003",
        study_id="50008003",
    )
    cw["source_record_reference"] = ""  # Strip reference!

    rec = {
        "subject_id": "10006003",
        "hadm_id": "20007003",
        "study_id": "50008003",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50008003"},
        "study_hadm_crosswalk": cw,
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_34_adversarial_invalid_or_nonexistent_source_record_reference():
    """34. Invalid or nonexistent source-record reference fails closed."""
    ingester = AuthenticMultimodalDatasetIngester()

    cw = {
        "subject_id": "10006004",
        "hadm_id": "20007004",
        "study_id": "50008004",
        "stay_id": "ED_STAY_DIRECT_LINK",
        "bridge_key": "ED_STAY_DIRECT_LINK",
        "source": "MIMIC_ED_CXR_CROSSWALK",
        "source_record_reference": "NONEXISTENT_TABLE_FABRICATED.CSV",
        "generator": "MIMIC_CXR_ED_BRIDGE_V1",
        "link_mechanism": "ED_STAY_DIRECT_LINK",
        "crosswalk_integrity_hash": "a" * 64,
    }

    rec = {
        "subject_id": "10006004",
        "hadm_id": "20007004",
        "study_id": "50008004",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50008004"},
        "study_hadm_crosswalk": cw,
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_35_adversarial_modified_crosswalk_after_generation():
    """35. Crosswalk modified after generation: integrity hash mismatch flags REJECTED."""
    ingester = AuthenticMultimodalDatasetIngester()

    # Validly generated for hadm_id 20007005
    cw = create_authenticated_imaging_crosswalk(
        subject_id="10006005",
        hadm_id="20007005",
        study_id="50008005",
    )
    # Attacker tampers with hadm_id after signature was computed
    cw["hadm_id"] = "20007999"

    rec = {
        "subject_id": "10006005",
        "hadm_id": "20007999",
        "study_id": "50008005",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50008005"},
        "study_hadm_crosswalk": cw,
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    # Tamper detection catches the forged modification
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_36_adversarial_unknown_crosswalk_generator():
    """36. Unknown crosswalk generator fails closed."""
    ingester = AuthenticMultimodalDatasetIngester()

    cw = {
        "subject_id": "10006006",
        "hadm_id": "20007006",
        "study_id": "50008006",
        "stay_id": "ED_STAY_DIRECT_LINK",
        "bridge_key": "ED_STAY_DIRECT_LINK",
        "source": "MIMIC_ED_CXR_CROSSWALK",
        "source_record_reference": "MIMIC_IV_ED.ED_RADIOLOGY",
        "generator": "UNAPPROVED_COMMUNITY_SCRAPER_BOT",
        "link_mechanism": "ED_STAY_DIRECT_LINK",
        "crosswalk_integrity_hash": "b" * 64,
    }

    rec = {
        "subject_id": "10006006",
        "hadm_id": "20007006",
        "study_id": "50008006",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50008006"},
        "study_hadm_crosswalk": cw,
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "UNLINKED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_37_adversarial_crosswalk_created_outside_trusted_pipeline():
    """37. Crosswalk created outside trusted pipeline with forged integrity hash fails closed."""
    ingester = AuthenticMultimodalDatasetIngester()

    rec = {
        "subject_id": "10006007",
        "hadm_id": "20007007",
        "study_id": "50008007",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50008007"},
        "study_hadm_crosswalk": {
            "subject_id": "10006007",
            "hadm_id": "20007007",
            "study_id": "50008007",
            "stay_id": "999999",
            "bridge_key": "999999",
            "source": "MIMIC_ED_CXR_CROSSWALK",
            "source_record_reference": "MIMIC_IV_ED.ED_RADIOLOGY",
            "generator": "MIMIC_CXR_ED_BRIDGE_V1",
            "crosswalk_integrity_hash": "deadbeefcafebabe0123456789abcdef0123456789abcdef0123456789abcdef",
        },
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_38_adversarial_temporal_only_matching_disguised_as_approved_crosswalk():
    """38. Temporal-only matching disguised as approved crosswalk falls back to TEMPORAL_CANDIDATE_LINK."""
    ingester = AuthenticMultimodalDatasetIngester()

    cw = create_authenticated_imaging_crosswalk(
        subject_id="10006008",
        hadm_id="20007008",
        study_id="50008008",
        link_mechanism="TEMPORAL_PROXIMITY_ONLY",  # Disallowed mechanism!
    )

    rec = {
        "subject_id": "10006008",
        "hadm_id": "20007008",
        "study_id": "50008008",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50008008"},
        "study_hadm_crosswalk": cw,
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "TEMPORAL_CANDIDATE_LINK"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


# ==============================================================================
# Part E: Task 2 Step 6.5 HMAC Secret Security & Signing-Boundary Attacks
# ==============================================================================

def test_39_adversarial_caller_supplying_fabricated_identifiers():
    """39. A caller supplying fabricated, synthetic, negative, or empty identifiers to the signing function is rejected."""
    # Fabricated marker in subject_id
    with pytest.raises(ValueError, match="fabricated"):
        create_authenticated_imaging_crosswalk(
            subject_id="FABRICATED_PATIENT_999",
            hadm_id="20001",
            study_id="50001",
        )

    # Synthetic marker in study_id
    with pytest.raises(ValueError, match="fabricated"):
        create_authenticated_imaging_crosswalk(
            subject_id="10001",
            hadm_id="20001",
            study_id="MOCK_STUDY_999",
        )

    # Negative integer in hadm_id
    with pytest.raises(ValueError, match="invalid or negative"):
        create_authenticated_imaging_crosswalk(
            subject_id="10001",
            hadm_id="-20001",
            study_id="50001",
        )

    # Empty identifier
    with pytest.raises(ValueError, match="empty"):
        create_authenticated_imaging_crosswalk(
            subject_id="",
            hadm_id="20001",
            study_id="50001",
        )


def test_40_adversarial_caller_invoking_signing_function_directly_unauthorized():
    """40. A caller invoking the signing function with an unauthorized caller role is rejected."""
    with pytest.raises(PermissionError, match="Unauthorized caller role"):
        create_authenticated_imaging_crosswalk(
            subject_id="10001",
            hadm_id="20001",
            study_id="50001",
            caller_role="untrusted_caller",
        )

    with pytest.raises(PermissionError, match="Unauthorized caller role"):
        create_authenticated_imaging_crosswalk(
            subject_id="10001",
            hadm_id="20001",
            study_id="50001",
            caller_role="dataset_consumer",
        )


def test_41_adversarial_caller_supplying_fabricated_source_record_reference():
    """41. A caller supplying a fabricated or unapproved source-record reference is rejected."""
    with pytest.raises(ValueError, match="Invalid or fabricated source record reference"):
        create_authenticated_imaging_crosswalk(
            subject_id="10001",
            hadm_id="20001",
            study_id="50001",
            source_record_reference="CUSTOM_FABRICATED_LEDGER_TABLE.CSV",
        )


def test_42_adversarial_caller_using_approved_generator_without_running_generator():
    """42. An adversarial caller crafts an envelope with an approved generator ID but forged signature; ingestion fails closed."""
    ingester = AuthenticMultimodalDatasetIngester()

    # Attacker manually constructs dictionary with approved generator name and bogus hash
    cw = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "stay_id": "ED_STAY_DIRECT_LINK",
        "bridge_key": "ED_STAY_DIRECT_LINK",
        "source": "MIMIC_ED_CXR_CROSSWALK",
        "source_record_reference": "MIMIC_IV_ED.ED_RADIOLOGY",
        "generator": "MIMIC_CXR_ED_BRIDGE_V1",  # Whitelisted generator
        "link_mechanism": "ED_STAY_DIRECT_LINK",
        "crosswalk_integrity_hash": "deadbeef" * 8,  # Forged/fake signature
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

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0
    assert r["modality_mask"][3] == 0.0


def test_43_adversarial_caller_accessing_hmac_key_from_ingestion_module():
    """43. Ingestion module does not export or expose HMAC key or secret attributes to callers."""
    from app.ai import multimodal_dataset_ingestion as ingestion_mod

    # Verify no secret or key attributes exist as module-level variables
    forbidden_attrs = ["HMAC_KEY", "HMAC_SECRET", "CROSSWALK_SALT", "SALT", "_SECRET", "SECRET_KEY"]
    for attr in forbidden_attrs:
        assert not hasattr(ingestion_mod, attr), f"Ingestion module must not expose '{attr}'"
        assert getattr(ingestion_mod, attr, None) is None


def test_44_adversarial_modified_payload_after_signing():
    """44. A legitimate crosswalk modified after signing fails integrity verification and is flagged REJECTED."""
    ingester = AuthenticMultimodalDatasetIngester()

    cw = create_authenticated_imaging_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
    )
    # Post-signing alteration of study_id
    cw["study_id"] = "50002"

    rec = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50002",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50002"},
        "study_hadm_crosswalk": cw,
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0


def test_45_adversarial_crosswalk_copied_from_different_study_or_encounter():
    """45. A crosswalk copied from a different study/encounter is rejected on identifier mismatch."""
    ingester = AuthenticMultimodalDatasetIngester()

    # Crosswalk legitimately signed for study 50001 and hadm 20001
    cw = create_authenticated_imaging_crosswalk(
        subject_id="10001",
        hadm_id="20001",
        study_id="50001",
    )

    # Attacker binds this crosswalk to a different study 50002
    rec = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50002",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50002"},
        "study_hadm_crosswalk": cw,
    }

    valid, report = ingester.validate_and_ingest_records([rec])
    assert len(valid) == 1
    r = valid[0]
    assert r["study_linkage_status"] == "REJECTED"
    assert r["imaging_study_linked"] is False
    assert r["xray_pathology_eligible"] == 0.0


def test_46_adversarial_unsigned_or_incorrectly_signed_crosswalk():
    """46. Unsigned or malformed signatures fail closed."""
    ingester = AuthenticMultimodalDatasetIngester()

    # Missing hash entirely
    cw_unsigned = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "stay_id": "ED_STAY_DIRECT_LINK",
        "bridge_key": "ED_STAY_DIRECT_LINK",
        "source": "MIMIC_ED_CXR_CROSSWALK",
        "source_record_reference": "MIMIC_IV_ED.ED_RADIOLOGY",
        "generator": "MIMIC_CXR_ED_BRIDGE_V1",
        "link_mechanism": "ED_STAY_DIRECT_LINK",
        "crosswalk_integrity_hash": "",
    }

    rec1 = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "target_condition": "Pneumonia",
        "symptoms": ["cough"],
        "xray": {"study_id": "50001"},
        "study_hadm_crosswalk": cw_unsigned,
    }
    valid1, _ = ingester.validate_and_ingest_records([rec1])
    assert valid1[0]["study_linkage_status"] == "UNLINKED"

    # Garbage signature
    cw_corrupt = dict(cw_unsigned)
    cw_corrupt["crosswalk_integrity_hash"] = "NOT_A_VALID_SIGNATURE"
    rec2 = dict(rec1)
    rec2["study_hadm_crosswalk"] = cw_corrupt
    valid2, _ = ingester.validate_and_ingest_records([rec2])
    assert valid2[0]["study_linkage_status"] == "REJECTED"


def test_47_adversarial_crosswalk_generated_outside_trusted_boundary():
    """47. A crosswalk produced by an unapproved generator fails closed as UNLINKED."""
    ingester = AuthenticMultimodalDatasetIngester()

    cw = {
        "subject_id": "10001",
        "hadm_id": "20001",
        "study_id": "50001",
        "stay_id": "ED_STAY_DIRECT_LINK",
        "bridge_key": "ED_STAY_DIRECT_LINK",
        "source": "MIMIC_ED_CXR_CROSSWALK",
        "source_record_reference": "MIMIC_IV_ED.ED_RADIOLOGY",
        "generator": "UNTRUSTED_EXTERNAL_PIPELINE_V9",
        "link_mechanism": "ED_STAY_DIRECT_LINK",
        "crosswalk_integrity_hash": "0123456789abcdef" * 4,
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

    valid, _ = ingester.validate_and_ingest_records([rec])
    assert valid[0]["study_linkage_status"] == "UNLINKED"
    assert valid[0]["xray_pathology_eligible"] == 0.0


def test_48_adversarial_missing_or_improperly_configured_secret():
    """48. In strict mode or when secret is unconfigured/empty, signing raises and validation fails closed."""
    from app.ai.multimodal_dataset_ingestion import compute_crosswalk_integrity_hash
    import os

    old_env = os.environ.pop("MEDISENSE_PROVENANCE_HMAC_SECRET", None)
    old_key = os.environ.pop("CROSSWALK_SIGNING_KEY", None)

    try:
        # Strict key requirement must fail closed (return None)
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

        # Signing with strict_key must raise PermissionError
        with pytest.raises(PermissionError, match="HMAC secret is unconfigured"):
            create_authenticated_imaging_crosswalk(
                subject_id="10001",
                hadm_id="20001",
                study_id="50001",
                strict_key=True,
            )
    finally:
        if old_env is not None:
            os.environ["MEDISENSE_PROVENANCE_HMAC_SECRET"] = old_env
        if old_key is not None:
            os.environ["CROSSWALK_SIGNING_KEY"] = old_key


