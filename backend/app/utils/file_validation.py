"""File validation utilities for medical uploads."""

from pathlib import Path
from typing import Any

ALLOWED_REPORT_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Simple magic-byte checks (not exhaustive; blocks obvious executables)
MAGIC_PREFIXES = {
    ".pdf": [b"%PDF"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
}

EXECUTABLE_MARKERS = (b"MZ", b"\x7fELF", b"#!")


def _extension(filename: str) -> str:
    return Path(filename).suffix.lower()


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

    ext = _extension(filename)
    if ext not in allowed:
        return {
            "ok": False,
            "message": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed))}",
        }

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
            return {"ok": False, "message": "Executable uploads are not allowed."}

    expected = MAGIC_PREFIXES.get(ext, [])
    if expected and not any(content.startswith(p) for p in expected):
        return {
            "ok": False,
            "message": "File content does not match the declared file type.",
        }

    file_type = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(ext, "application/octet-stream")

    return {
        "ok": True,
        "extension": ext if ext != ".jpeg" else ".jpg",
        "file_type": file_type,
        "message": "ok",
    }
