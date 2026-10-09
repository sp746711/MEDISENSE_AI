"""Medical report text extraction (OCR/PDF) service for MediSense AI.

Extracts text and tabular structures from PDF documents using PyMuPDF and validates text readability.
Never invents content if document extraction fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

OCR_VERSION = "report-ocr-v2.0"


def extract_text_from_file(file_path: str) -> dict[str, Any]:
    """Extract text and tables from an uploaded PDF or image report file."""
    path = Path(file_path)
    if not path.exists():
        return {
            "status": "unavailable",
            "extracted_text": None,
            "tables": [],
            "message": f"File does not exist: {file_path}",
            "ocr_version": OCR_VERSION,
            "file_path": file_path,
        }

    ext = path.suffix.lower()

    if ext == ".pdf":
        if fitz is None and PdfReader is None:
            return {
                "status": "unavailable",
                "extracted_text": None,
                "tables": [],
                "pages": 0,
                "message": "No PDF parser installed. Digital PDF extraction unavailable.",
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }
        try:
            extracted_pages: list[str] = []
            tables_data: list[list[list[str]]] = []
            if fitz is not None:
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text") or ""
                    if text.strip():
                        extracted_pages.append(text.strip())
                    if hasattr(page, "find_tables"):
                        try:
                            for tab in page.find_tables():
                                extracted = tab.extract()
                                if extracted:
                                    tables_data.append(extracted)
                        except Exception:
                            pass
                doc.close()
            elif PdfReader is not None:
                reader = PdfReader(file_path)
                for p in reader.pages:
                    txt = p.extract_text() or ""
                    if txt.strip():
                        extracted_pages.append(txt.strip())

            full_text = "\n\n".join(extracted_pages).strip()
            if not full_text:
                return {
                    "status": "unreadable",
                    "extracted_text": None,
                    "tables": [],
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
                "tables": tables_data,
                "pages": len(extracted_pages),
                "message": f"Successfully extracted text across {len(extracted_pages)} page(s).",
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }
        except Exception as exc:
            return {
                "status": "error",
                "extracted_text": None,
                "tables": [],
                "message": f"Failed to parse PDF document: {exc}",
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }

    # For image formats (.jpg, .png, .jpeg)
    if ext in {".png", ".jpg", ".jpeg"}:
        if fitz is None:
            return {
                "status": "unavailable",
                "extracted_text": None,
                "tables": [],
                "message": "PyMuPDF (fitz) is not installed. Image text extraction unavailable.",
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }
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
                    "tables": [],
                    "message": "Extracted text layer from image.",
                    "ocr_version": OCR_VERSION,
                    "file_path": file_path,
                }

            return {
                "status": "unreadable",
                "extracted_text": None,
                "tables": [],
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
                "tables": [],
                "message": f"Failed to read image document: {exc}",
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }

    return {
        "status": "unsupported",
        "extracted_text": None,
        "tables": [],
        "message": f"Unsupported report file format: {ext}",
        "ocr_version": OCR_VERSION,
        "file_path": file_path,
    }
