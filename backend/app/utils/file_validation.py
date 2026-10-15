import io
import re
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

ALLOWED_REPORT_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MAGIC_PREFIXES = {
    ".pdf": [b"%PDF"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
}

EXECUTABLE_MARKERS = (b"MZ", b"\x7fELF", b"#!", b"<!DOCTYPE html", b"<html", b"<?php")


def _extension(filename: str) -> str:
    # Strip any directory path components to avoid traversal
    clean_name = Path(filename).name
    # Remove any dangerous null bytes or path separators
    clean_name = re.sub(r"[\x00/\\:]", "", clean_name)
    return Path(clean_name).suffix.lower()


def sanitize_filename(filename: str) -> str:
    """Return a safe base filename free of directory traversal sequences."""
    clean_name = Path(filename or "upload").name
    clean_name = re.sub(r"[\x00/\\:]", "", clean_name).strip()
    return clean_name or "medical_report"


def detect_file_type_from_content(content: bytes) -> str | None:
    """Identify the actual file extension from binary signatures."""
    if not content:
        return None
    if content.startswith(b"%PDF"):
        return ".pdf"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return None


def validate_report_file(filename: str, content: bytes, max_mb: int) -> dict[str, Any]:
    return _validate(filename, content, max_mb, ALLOWED_REPORT_EXTENSIONS)


def validate_image_file(filename: str, content: bytes, max_mb: int) -> dict[str, Any]:
    return _validate(filename, content, max_mb, ALLOWED_IMAGE_EXTENSIONS)


def _validate(
    filename: str,
    content: bytes,
    max_mb: int,
    allowed: set[str],
) -> dict[str, Any]:
    if not filename:
        return {"ok": False, "message": "Filename is required."}

    clean_filename = sanitize_filename(filename)
    declared_ext = _extension(clean_filename)

    max_bytes = max_mb * 1024 * 1024
    if len(content) == 0:
        return {"ok": False, "message": "Empty file is not allowed."}
    if len(content) > max_bytes:
        return {
            "ok": False,
            "message": f"File exceeds maximum size of {max_mb} MB.",
        }

    for marker in EXECUTABLE_MARKERS:
        if content.startswith(marker):
            return {"ok": False, "message": "Executable or script uploads are not allowed."}

    # Sniff actual type by magic bytes
    detected_ext = detect_file_type_from_content(content)
    if detected_ext is None:
        return {
            "ok": False,
            "message": "File format is unsupported or binary signature is invalid.",
        }

    # Verify that the detected format is in the allowed set
    if detected_ext not in allowed:
        return {
            "ok": False,
            "message": f"Unsupported file type '{detected_ext}'. Allowed: {', '.join(sorted(allowed))}",
        }

    # Verify declared extension matches or is acceptable
    if declared_ext and declared_ext in allowed:
        # Normalize .jpeg -> .jpg for comparison
        norm_declared = ".jpg" if declared_ext == ".jpeg" else declared_ext
        norm_detected = ".jpg" if detected_ext == ".jpeg" else detected_ext
        if norm_declared != norm_detected:
            return {
                "ok": False,
                "message": f"Declared file extension '{declared_ext}' does not match actual file content format '{detected_ext}'.",
            }

    # Deep integrity verification
    if detected_ext in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
            with Image.open(io.BytesIO(content)) as img:
                w, h = img.size
                if w < 10 or h < 10:
                    return {
                        "ok": False,
                        "extension": detected_ext,
                        "message": f"Image dimensions ({w}x{h}) are too small to be a valid radiograph. Minimum resolution is 10x10.",
                    }
        except Exception:
            return {
                "ok": False,
                "extension": detected_ext,
                "message": "Image file is corrupted, malformed, or cannot be decoded.",
            }
    elif detected_ext == ".pdf":
        if fitz is not None and len(content) > 100:
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                if len(doc) == 0:
                    doc.close()
                    return {"ok": False, "extension": detected_ext, "message": "PDF document contains 0 pages."}
                doc.close()
            except Exception:
                return {
                    "ok": False,
                    "extension": detected_ext,
                    "message": "PDF file is corrupted or cannot be read.",
                }

    file_type = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(detected_ext, "application/octet-stream")

    return {
        "ok": True,
        "filename": clean_filename,
        "extension": detected_ext if detected_ext != ".jpeg" else ".jpg",
        "file_type": file_type,
        "message": "ok",
    }

