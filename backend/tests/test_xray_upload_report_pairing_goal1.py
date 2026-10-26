"""Regression tests for Goal 1 image/report region checks.

Run from backend root:
    .venv\\Scripts\\python.exe -m pytest tests\\test_xray_upload_report_pairing_goal1.py -v
"""
from app.routes.xray import _build_report_region_verification


def test_unique_same_region_pair_is_region_compatible_but_not_explicitly_confirmed():
    result = _build_report_region_verification(
        resolved_region="chest",
        report_regions=["chest"],
        unrecognized_report_region_count=0,
        report_count=1,
        existing_image_count=0,
    )
    assert result["status"] == "MATCH"
    assert result["region_compatible"] is True
    # Body-region agreement is not proof that report and image have been linked.
    assert result["association_explicitly_confirmed"] is False
    assert result["can_combine"] is False


def test_unique_different_regions_are_marked_mismatch():
    result = _build_report_region_verification(
        resolved_region="knee",
        report_regions=["chest"],
        unrecognized_report_region_count=0,
        report_count=1,
        existing_image_count=0,
    )
    assert result["status"] == "MISMATCH"
    assert result["region_compatible"] is False
    assert result["can_combine"] is False


def test_multiple_reports_do_not_create_false_mismatch_for_each_image():
    result = _build_report_region_verification(
        resolved_region="knee",
        report_regions=["chest", "knee"],
        unrecognized_report_region_count=0,
        report_count=2,
        existing_image_count=0,
    )
    assert result["status"] == "UNVERIFIED"
    assert result["association_scope"] == "AMBIGUOUS_OR_INCOMPLETE"
    assert result["can_combine"] is False


def test_existing_image_prevents_global_report_assignment_to_new_image():
    result = _build_report_region_verification(
        resolved_region="knee",
        report_regions=["chest"],
        unrecognized_report_region_count=0,
        report_count=1,
        existing_image_count=1,
    )
    assert result["status"] == "UNVERIFIED"
    assert result["can_combine"] is False


def test_unknown_anatomy_cannot_be_declared_a_match_or_mismatch():
    result = _build_report_region_verification(
        resolved_region="unknown",
        report_regions=["chest"],
        unrecognized_report_region_count=0,
        report_count=1,
        existing_image_count=0,
    )
    assert result["status"] == "UNVERIFIED"
    assert result["detected_image_region"] is None
    assert result["can_combine"] is False


def test_unrecognized_report_region_keeps_pair_unverified():
    result = _build_report_region_verification(
        resolved_region="chest",
        report_regions=[],
        unrecognized_report_region_count=1,
        report_count=1,
        existing_image_count=0,
    )
    assert result["status"] == "UNVERIFIED"
    assert result["unrecognized_report_region_count"] == 1
    assert result["can_combine"] is False


def test_comparison_works_with_canonicalized_region_casing():
    result = _build_report_region_verification(
        resolved_region="CHEST",
        report_regions=["Chest"],
        unrecognized_report_region_count=0,
        report_count=1,
        existing_image_count=0,
    )
    assert result["status"] == "MATCH"
    assert result["detected_image_region"] == "chest"
