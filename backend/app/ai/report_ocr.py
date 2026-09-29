"""Medical report text extraction (OCR/PDF) service for MediSense AI.

Extracts text from PDF documents using PyMuPDF and validates text readability.
Never invents content if document extraction fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

OCR_VERSION = "report-ocr-v1.0"


def extract_text_from_file(file_path: str) -> dict[str, Any]:
    """Extract text from an uploaded PDF or image report file."""
    path = Path(file_path)
    if not path.exists():
        return {
            "status": "unavailable",
            "extracted_text": None,
            "message": f"File does not exist: {file_path}",
            "ocr_version": OCR_VERSION,
            "file_path": file_path,
        }

    ext = path.suffix.lower()

    if ext == ".pdf":
        try:
            doc = fitz.open(file_path)
            extracted_pages: list[str] = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text") or ""
                if text.strip():
                    extracted_pages.append(text.strip())
            doc.close()

            full_text = "\n\n".join(extracted_pages).strip()
            if not full_text:
                return {
                    "status": "unreadable",
                    "extracted_text": None,
                    "pages": len(doc),
                    "message": (
                        "PDF contains scanned or rasterized pages without selectable text. "
                        "Please provide a digital PDF or report with readable text."
                    ),
                    "ocr_version": OCR_VERSION,
                    "file_path": file_path,
                }

            return {
                "status": "extracted",
                "extracted_text": full_text,
                "pages": len(extracted_pages),
                "message": f"Successfully extracted text across {len(extracted_pages)} page(s).",
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }
        except Exception as exc:
            return {
                "status": "error",
                "extracted_text": None,
                "message": f"Failed to parse PDF document: {exc}",
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }

    # For image formats (.jpg, .png, .jpeg)
    if ext in {".png", ".jpg", ".jpeg"}:
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page in doc:
                txt = page.get_text("text") or ""
                if txt.strip():
                    full_text += txt.strip() + "\n"
            doc.close()

            if full_text.strip():
                return {
                    "status": "extracted",
                    "extracted_text": full_text.strip(),
                    "message": "Extracted text layer from image.",
                    "ocr_version": OCR_VERSION,
                    "file_path": file_path,
                }

            return {
                "status": "unreadable",
                "extracted_text": None,
                "message": (
                    "Image file received. OCR text layer is unavailable for this image. "
                    "For accurate automated analysis, please upload a digital PDF report."
                ),
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }
        except Exception as exc:
            return {
                "status": "error",
                "extracted_text": None,
                "message": f"Failed to read image document: {exc}",
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }

    return {
        "status": "unsupported",
        "extracted_text": None,
        "message": f"Unsupported report file format: {ext}",
        "ocr_version": OCR_VERSION,
        "file_path": file_path,
    }
