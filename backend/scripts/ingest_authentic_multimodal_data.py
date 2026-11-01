"""CLI Driver for Authentic Multimodal Dataset Ingestion (MediSense AI Goal 1).

Usage:
    python scripts/ingest_authentic_multimodal_data.py --input <path_to_raw_cohort.json> [--output backend/data/multimodal_linked/linked_patient_cohort.json] [--source MIMIC-IV-CXR] [--version v2.2]

Constraints:
- Strictly rejects synthetic or placeholder patient data.
- Enforces subject_id, hadm_id, and study_id authentic linkage.
- Maps condition labels via ICD-9/10 to canonical 24 clinical classes.
- Produces ingestion_audit_report.json alongside the linked cohort.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure backend directory is in sys.path
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.ai.multimodal_dataset_ingestion import AuthenticMultimodalDatasetIngester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_cli")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest authentic multimodal EHR cohort.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to raw linked records (JSON or NDJSON).")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(backend_root / "data" / "multimodal_linked" / "linked_patient_cohort.json"),
        help="Path where validated linked cohort JSON will be saved.",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=str,
        default=str(backend_root / "data" / "multimodal_linked" / "ingestion_audit_report.json"),
        help="Path where ingestion audit report will be saved.",
    )
    parser.add_argument("--source", type=str, default="MIMIC-IV-CXR", help="Name of authentic source dataset.")
    parser.add_argument("--version", type=str, default="v2.2", help="Version identifier of source dataset.")

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info(f"Loading raw records from: {input_path}")
    raw_records = []
    try:
        if input_path.suffix.lower() == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    raw_records = content
                elif isinstance(content, dict) and "records" in content:
                    raw_records = content["records"]
                else:
                    raw_records = [content]
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_records.append(json.loads(line))
    except Exception as exc:
        logger.error(f"Failed to parse input file: {exc}")
        return 1

    logger.info(f"Read {len(raw_records)} raw records. Initializing {args.source} ({args.version}) ingester...")
    ingester = AuthenticMultimodalDatasetIngester(data_source_name=args.source, source_version=args.version)
    valid_records, audit_report = ingester.validate_and_ingest_records(raw_records, source_file_path=input_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report.to_dict(), f, indent=2)
    logger.info(f"Saved ingestion audit report to: {report_path}")

    if not audit_report.is_valid:
        logger.error(
            f"Ingestion rejected! 0 valid records retained from {len(raw_records)} raw records. "
            f"Reasons: {audit_report.rejection_reasons}"
        )
        return 2

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_records, f, indent=2)

    logger.info(
        f"Successfully ingested and saved {len(valid_records)} valid records "
        f"across {audit_report.unique_patients_count} unique patients to {output_path}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
