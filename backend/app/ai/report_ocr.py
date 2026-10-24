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

OCR_VERSION = "report-ocr-v3.2-task1"


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
        Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Programs" / "Tesseract-OCR",
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
    """Return True only when pytesseract and a runnable Tesseract binary are available."""
    if pytesseract is None:
        return False
    bin_path = find_tesseract_binary()
    if not bin_path:
        return False
    try:
        # File existence alone is insufficient (bad permissions or broken installations).
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


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


def _result(
    *,
    status: str,
    file_path: str,
    message: str,
    extracted_text: str | None = None,
    tables: list[Any] | None = None,
    pages: int | None = None,
    ocr_engine: str = "none",
) -> dict[str, Any]:
    """Build a consistent, non-clinical extraction result without leaking exceptions."""
    result: dict[str, Any] = {
        "status": status,
        "extracted_text": extracted_text,
        "tables": tables if tables is not None else [],
        "message": message,
        "ocr_version": OCR_VERSION,
        "ocr_engine": ocr_engine,
        "file_path": file_path,
    }
    if pages is not None:
        result["pages"] = pages
    return result


def _needs_ocr(page_text: str) -> bool:
    """Identify pages with absent or sparse selectable text (common in hybrid PDFs)."""
    text = (page_text or "").strip()
    alnum_count = sum(ch.isalnum() for ch in text)
    # A selectable header alone is not evidence that the report body/table was extracted.
    return len(text) < 80 or alnum_count < 35


def _merge_page_text(digital_text: str, ocr_text: str) -> str:
    """Merge a page's digital and OCR text while removing obvious repeated lines.

    Keep distinct OCR lines (often the scanned table body) and never silently replace
    the selectable text layer. This is a conservative duplicate-line filter, not a
    guarantee that OCR output is clinically accurate.
    """
    digital = (digital_text or "").strip()
    ocr = (ocr_text or "").strip()
    if not digital:
        return ocr
    if not ocr:
        return digital

    def key(line: str) -> str:
        return re.sub(r"\s+", " ", line).strip().casefold()

    digital_keys = {key(line) for line in digital.splitlines() if key(line)}
    extra_lines = [line for line in ocr.splitlines() if key(line) and key(line) not in digital_keys]
    if not extra_lines:
        return digital
    return digital + "\n" + "\n".join(extra_lines)


def extract_text_from_file(file_path: str) -> dict[str, Any]:
    """Extract text and tables from an uploaded PDF or image report file."""
    path = Path(file_path)
    if not path.exists():
        return _result(
            status="unavailable", file_path=file_path,
            message="The uploaded report file could not be found on the server.",
        )

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
            ocr_used = False
            has_digital_text = False
            ocr_unavailable_pages: list[int] = []

            if fitz is not None:
                # Process each page independently to support mixed digital/scanned PDFs.
                with fitz.open(file_path) as doc:
                    total_doc_pages = len(doc)
                    ocr_available = (
                        pytesseract is not None and is_ocr_engine_available()
                    )

                    for page_num in range(total_doc_pages):
                        page = doc[page_num]
                        page_text = (page.get_text("text") or "").strip()

                        # Preserve table extraction from digital pages.
                        if hasattr(page, "find_tables"):
                            try:
                                for tab in page.find_tables():
                                    extracted_table = tab.extract()
                                    if extracted_table:
                                        tables_data.append(extracted_table)
                            except Exception:
                                # Table extraction is optional; continue text extraction.
                                pass

                        if page_text:
                            has_digital_text = True

                        # Hybrid PDFs may have a selectable header but scanned tables/body.
                        # Keep the digital layer and OCR sparse pages; never replace original text.
                        if page_text and not _needs_ocr(page_text):
                            extracted_pages.append(page_text)
                            continue

                        if not ocr_available:
                            # A sparse digital layer may omit scanned tables or body content.
                            # Report that OCR could not complete this page even if a header exists.
                            ocr_unavailable_pages.append(page_num + 1)
                            extracted_pages.append(page_text)
                            continue

                        try:
                            pix = page.get_pixmap(dpi=250, alpha=False)
                            img_bytes = pix.tobytes("png")
                            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                                if pil_img.mode in ("RGBA", "P", "LA"):
                                    pil_img = pil_img.convert("RGB")
                                page_ocr = pytesseract.image_to_string(
                                    pil_img, config="--oem 3 --psm 6"
                                )
                                if not page_ocr or len(page_ocr.strip()) < 5:
                                    page_ocr = pytesseract.image_to_string(pil_img)

                            page_ocr = (page_ocr or "").strip()
                            if page_ocr:
                                ocr_used = True
                                # Preserve the digital text layer and add OCR text only when
                                # useful; duplicates are possible, so downstream NLP must not
                                # treat repeated text as independent evidence.
                                extracted_pages.append(_merge_page_text(page_text, page_ocr))
                            elif page_text:
                                extracted_pages.append(page_text)
                                ocr_unavailable_pages.append(page_num + 1)
                            else:
                                extracted_pages.append("")
                                ocr_unavailable_pages.append(page_num + 1)
                        except Exception:
                            extracted_pages.append(page_text)
                            ocr_unavailable_pages.append(page_num + 1)

                # Join non-empty page text in original document order.
                full_text = "\n\n".join(
                    page_text for page_text in extracted_pages if page_text
                ).strip()

                if full_text:
                    clean_text = normalize_ocr_text(full_text)
                    if ocr_unavailable_pages:
                        message = (
                            "Text extracted from readable pages, but scanned page(s) "
                            + ", ".join(map(str, ocr_unavailable_pages))
                            + " could not be read. Review the original report to verify completeness."
                        )
                    elif ocr_used:
                        message = (
                            f"Successfully extracted text from readable pages "
                            f"({total_doc_pages} total page(s)); OCR was used for scanned pages."
                        )
                    else:
                        message = (
                            f"Successfully extracted selectable text from the PDF "
                            f"({total_doc_pages} total page(s))."
                        )

                    if ocr_used and has_digital_text:
                        ocr_engine = "pdf_text_layer+pytesseract"
                    elif ocr_used:
                        ocr_engine = "pytesseract"
                    elif ocr_unavailable_pages:
                        ocr_engine = "pdf_text_layer_partial" if has_digital_text else "none"
                    else:
                        ocr_engine = "pdf_text_layer"

                    return {
                        "status": "extracted",
                        "extracted_text": clean_text,
                        "tables": tables_data,
                        "pages": total_doc_pages,
                        "ocr_engine": ocr_engine,
                        "message": message,
                        "ocr_version": OCR_VERSION,
                        "file_path": file_path,
                    }

                return {
                    "status": "unreadable",
                    "extracted_text": None,
                    "tables": tables_data,
                    "pages": total_doc_pages,
                    "ocr_engine": "pytesseract" if ocr_used else "none",
                    "message": (
                        "No legible medical report text was detected. "
                        + ("Tesseract OCR was unavailable or failed for scanned pages; verify its installation and configuration." 
                           if not (pytesseract is not None and is_ocr_engine_available())
                           else "OCR ran but did not recover readable text; inspect the original PDF or try a clearer scan.")
                    ),
                    "ocr_version": OCR_VERSION,
                    "file_path": file_path,
                }

            # pypdf fallback: digital text only; it cannot rasterize pages for OCR.
            reader = PdfReader(file_path)
            total_doc_pages = len(reader.pages)
            for page in reader.pages:
                page_text = (page.extract_text() or "").strip()
                extracted_pages.append(page_text)

            full_text = "\n\n".join(
                page_text for page_text in extracted_pages if page_text
            ).strip()
            if not full_text:
                return {
                    "status": "unreadable",
                    "extracted_text": None,
                    "tables": [],
                    "pages": total_doc_pages,
                    "ocr_engine": "none",
                    "message": (
                        "No selectable PDF text was found. Scanned-page OCR requires "
                        "PyMuPDF and a configured Tesseract OCR engine."
                    ),
                    "ocr_version": OCR_VERSION,
                    "file_path": file_path,
                }

            return {
                "status": "extracted",
                "extracted_text": normalize_ocr_text(full_text),
                "tables": [],
                "pages": total_doc_pages,
                "ocr_engine": "pdf_text_layer",
                "message": (
                    f"Extracted selectable text from the PDF ({total_doc_pages} total page(s)). "
                    "Scanned pages cannot be OCR-processed by the pypdf-only fallback."
                ),
                "ocr_version": OCR_VERSION,
                "file_path": file_path,
            }

        except Exception as exc:
            return {
                "status": "error",
                "extracted_text": None,
                "tables": [],
                "message": "The PDF could not be parsed. Verify that it is a valid, readable PDF and try again.",
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
                # Apply camera orientation metadata before OCR and normalize palette/alpha/gray modes.
                try:
                    from PIL import ImageOps
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                img.load()
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

                # Prepare a readable OCR copy; keep the uploaded original untouched.
                from PIL import ImageOps, ImageEnhance
                proc_img = img.convert("L")
                proc_img = ImageOps.autocontrast(proc_img)
                if max(proc_img.size) < 1800:
                    scale = min(2.0, 1800 / max(proc_img.size))
                    proc_img = proc_img.resize(
                        (max(1, int(proc_img.width * scale)), max(1, int(proc_img.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                proc_img = ImageEnhance.Sharpness(proc_img).enhance(1.2)

                # Perform OCR. OCR output remains unverified text, not a validated lab result.
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
                    "message": "Successfully extracted text from image report via OCR. Verify recognized values, units, and reference ranges against the original image.",
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
                "message": "The image report could not be processed. Verify that the file is a valid PNG/JPG image and try again.",
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
