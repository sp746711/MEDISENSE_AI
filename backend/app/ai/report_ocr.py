"""Medical report text extraction (OCR/PDF) service for MediSense AI.

Extracts text and tabular structures from digital PDFs, scanned PDFs, and image-based reports (PNG, JPG, JPEG).
Integrates Tesseract OCR for pixel-level optical character recognition on images and scanned PDF pages.
Never invents clinical content if document extraction fails.
"""

from __future__ import annotations

import io
import os
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

OCR_VERSION = "report-ocr-v3.0"


def find_tesseract_binary() -> str | None:
    """Locate the Tesseract executable in environment variables, PATH, or standard Windows paths."""
    # 1. Custom environment variable override
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).is_file():
        if pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = env_cmd
        return env_cmd

    # 2. System PATH
    which_cmd = shutil.which("tesseract") or shutil.which("tesseract.exe")
    if which_cmd and Path(which_cmd).is_file():
        if pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = which_cmd
        return which_cmd

    # 3. Standard Windows installation paths
    candidate_dirs = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Tesseract-OCR",
        Path(os.environ.get("LOCALAPPDATA", r"C:\Users\Hp\AppData\Local")) / "Programs" / "Tesseract-OCR",
        Path(r"C:\Tesseract-OCR"),
    ]
    for d in candidate_dirs:
        exe_path = d / "tesseract.exe"
        if exe_path.is_file():
            str_path = str(exe_path)
            if pytesseract is not None:
                pytesseract.pytesseract.tesseract_cmd = str_path
            return str_path

    return None


def is_ocr_engine_available() -> bool:
    """Return True if pytesseract is installed and a working tesseract executable is detected."""
    if pytesseract is None:
        return False
    bin_path = find_tesseract_binary()
    return bin_path is not None


def normalize_ocr_text(text: str | None) -> str:
    """Clean and normalize OCR text without destroying clinical numbers, signs, units, or ranges."""
    if not text:
        return ""

    # Replace non-breaking spaces and irregular whitespace
    t = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

    # Normalize various unicode dashes to standard hyphen-minus without mangling math signs
    t = re.sub(r"[\u2013\u2014\u2212\x96]", "-", t)

    # Normalize unicode quotes
    t = re.sub(r"[\u2018\u2019]", "'", t)
    t = re.sub(r"[\u201c\u201d]", '"', t)

    # Clean multiple spaces on each line while preserving indentation/linebreaks
    cleaned_lines: list[str] = []
    for line in t.splitlines():
        line_clean = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line_clean)

    # Remove excessive blank lines (>2)
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


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
            "ocr_engine": "none",
            "file_path": file_path,
        }

    ext = path.suffix.lower()

    # -------------------------------------------------------------
    # 1. PDF Reports (Digital text first, OCR fallback for scans)
    # -------------------------------------------------------------
    if ext == ".pdf":
        if fitz is None and PdfReader is None:
            return {
                "status": "unavailable",
                "extracted_text": None,
                "tables": [],
                "pages": 0,
                "message": "No PDF parser installed. Digital PDF extraction unavailable.",
                "ocr_version": OCR_VERSION,
                "ocr_engine": "none",
                "file_path": file_path,
            }

        try:
            extracted_pages: list[str] = []
            tables_data: list[list[list[str]]] = []
            total_doc_pages = 0

            if fitz is not None:
                doc = fitz.open(file_path)
                total_doc_pages = len(doc)
                for page_num in range(total_doc_pages):
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

                # If digital text was extracted successfully from the PDF
                full_text = "\n\n".join(extracted_pages).strip()
                if full_text:
                    doc.close()
                    clean_full = normalize_ocr_text(full_text)
                    return {
                        "status": "extracted",
                        "extracted_text": clean_full,
                        "tables": tables_data,
                        "pages": len(extracted_pages),
                        "ocr_engine": "pdf_text_layer",
                        "message": f"Successfully extracted text across {len(extracted_pages)} page(s).",
                        "ocr_version": OCR_VERSION,
                        "file_path": file_path,
                    }

                # If no selectable digital text found: attempt OCR fallback on rasterized PDF pages
                ocr_available = is_ocr_engine_available()
                if ocr_available and pytesseract is not None:
                    ocr_pages: list[str] = []
                    for page_num in range(total_doc_pages):
                        page = doc[page_num]
                        pix = page.get_pixmap(dpi=200)
                        img_bytes = pix.tobytes("png")
                        with Image.open(io.BytesIO(img_bytes)) as pil_img:
                            page_ocr = pytesseract.image_to_string(pil_img, config="--oem 3 --psm 6")
                            if not page_ocr or len(page_ocr.strip()) < 5:
                                page_ocr = pytesseract.image_to_string(pil_img)
                            if page_ocr.strip():
                                ocr_pages.append(page_ocr.strip())

                    doc.close()
                    full_ocr_text = "\n\n".join(ocr_pages).strip()
                    if full_ocr_text:
                        clean_ocr = normalize_ocr_text(full_ocr_text)
                        return {
                            "status": "extracted",
                            "extracted_text": clean_ocr,
                            "tables": [],
                            "pages": len(ocr_pages),
                            "ocr_engine": "pytesseract",
                            "message": f"Successfully extracted text via OCR across {len(ocr_pages)} scanned page(s).",
                            "ocr_version": OCR_VERSION,
                            "file_path": file_path,
                        }
                    else:
                        return {
                            "status": "unreadable",
                            "extracted_text": None,
                            "tables": [],
                            "pages": total_doc_pages,
                            "ocr_engine": "pytesseract",
                            "message": "Scanned PDF processed via OCR, but no legible medical report text was detected.",
                            "ocr_version": OCR_VERSION,
                            "file_path": file_path,
                        }
                else:
                    doc.close()
                    return {
                        "status": "unreadable",
                        "extracted_text": None,
                        "tables": [],
                        "pages": total_doc_pages,
                        "ocr_engine": "none",
                        "message": (
                            "PDF contains scanned or rasterized pages without selectable text. "
                            "Tesseract OCR engine is not installed or configured on the server. "
                            "Please provide a digital PDF or install Tesseract OCR on Windows."
                        ),
                        "ocr_version": OCR_VERSION,
                        "file_path": file_path,
                    }

            elif PdfReader is not None:
                reader = PdfReader(file_path)
                total_doc_pages = len(reader.pages)
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
                        "pages": total_doc_pages,
                        "ocr_engine": "none",
                        "message": "PDF contains scanned pages without selectable text.",
                        "ocr_version": OCR_VERSION,
                        "file_path": file_path,
                    }

                clean_full = normalize_ocr_text(full_text)
                return {
                    "status": "extracted",
                    "extracted_text": clean_full,
                    "tables": [],
                    "pages": len(extracted_pages),
                    "ocr_engine": "pdf_text_layer",
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
                "ocr_engine": "none",
                "file_path": file_path,
            }

    # -------------------------------------------------------------
    # 2. Image Reports (.png, .jpg, .jpeg)
    # -------------------------------------------------------------
    if ext in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(file_path) as img:
                # Basic sanity check
                width, height = img.size
                if width < 10 or height < 10:
                    return {
                        "status": "unreadable",
                        "extracted_text": None,
                        "tables": [],
                        "pages": 1,
                        "ocr_engine": "none",
                        "message": "Image dimensions are too small to contain readable report data.",
                        "ocr_version": OCR_VERSION,
                        "file_path": file_path,
                    }

                # Check if OCR engine is available
                ocr_available = is_ocr_engine_available()
                if not ocr_available or pytesseract is None:
                    return {
                        "status": "unreadable",
                        "extracted_text": None,
                        "tables": [],
                        "pages": 1,
                        "ocr_engine": "none",
                        "message": (
                            "Image report received. Tesseract OCR engine is not installed or configured on the server. "
                            "To enable automated OCR analysis for PNG/JPG medical reports, install Tesseract OCR on Windows "
                            "(e.g., from https://github.com/UB-Mannheim/tesseract/wiki) and configure TESSERACT_CMD or system PATH. "
                            "Alternatively, upload a digital PDF report with selectable text."
                        ),
                        "ocr_version": OCR_VERSION,
                        "file_path": file_path,
                    }

                # Convert image for robust OCR processing
                proc_img = img
                if proc_img.mode in ("RGBA", "P", "LA"):
                    proc_img = proc_img.convert("RGB")

                # Perform OCR
                raw_ocr = pytesseract.image_to_string(proc_img, config="--oem 3 --psm 6")
                if not raw_ocr or len(raw_ocr.strip()) < 5:
                    raw_ocr = pytesseract.image_to_string(proc_img)

                clean_text = normalize_ocr_text(raw_ocr)
                if not clean_text or len(clean_text.strip()) < 4:
                    return {
                        "status": "unreadable",
                        "extracted_text": None,
                        "tables": [],
                        "pages": 1,
                        "ocr_engine": "pytesseract",
                        "message": "Image report processed via OCR, but no legible medical text was recognized.",
                        "ocr_version": OCR_VERSION,
                        "file_path": file_path,
                    }

                return {
                    "status": "extracted",
                    "extracted_text": clean_text,
                    "tables": [],
                    "pages": 1,
                    "ocr_engine": "pytesseract",
                    "message": "Successfully extracted text from image report via OCR.",
                    "ocr_version": OCR_VERSION,
                    "file_path": file_path,
                }

        except Exception as exc:
            return {
                "status": "error",
                "extracted_text": None,
                "tables": [],
                "pages": 1,
                "ocr_engine": "none",
                "message": f"Failed to process image document: {exc}",
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }

    return {
        "status": "unsupported",
        "extracted_text": None,
        "tables": [],
        "message": f"Unsupported report file format: {ext}",
        "ocr_version": OCR_VERSION,
        "ocr_engine": "none",
        "file_path": file_path,
    }
