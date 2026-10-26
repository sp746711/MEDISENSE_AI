from app.ai.evidence_engine import combine_evidence


def test_region_mismatch_is_not_combined():
    evidence = combine_evidence(
        xray_results=[{
            "file_name": "knee_xray.png",
            "region": "knee",
            "status": "COMPLETED",
            "prediction": "fracture",
        }],
        xray_reports=[{
            "file_name": "chest_report.pdf",
            "extracted_text": "CHEST RADIOGRAPH: clear lungs.",
            "structured_findings": {
                "qualitative_findings": [
                    {"finding": "clear lungs", "state": "PRESENT"}
                ]
            },
        }],
    )
    assert evidence["xray"]["has_region_mismatch"] is True
    assert evidence["xray"]["status"] == "REGION_MISMATCH"
    assert evidence["xray"]["prediction"] is None


def test_xray_report_only_does_not_require_image():
    evidence = combine_evidence(
        xray_reports=[{
            "file_name": "chest_report.pdf",
            "extracted_text": "CHEST RADIOGRAPH: clear lungs.",
            "structured_findings": {
                "qualitative_findings": [
                    {"finding": "clear lungs", "state": "PRESENT"}
                ]
            },
        }],
        selected_input_types=["xray_report"],
    )
    assert evidence["xray"]["reports"]
    assert evidence["xray"]["reports"][0]["region_verification"]["status"] == "REPORT_ONLY"
