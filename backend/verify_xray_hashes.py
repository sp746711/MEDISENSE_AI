"""Cryptographic SHA-256 hash verification for protected X-ray assets (Part 2 & Part 13)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parent
manifest_path = backend_dir / "models" / "xray_protected_manifest.json"

if not manifest_path.exists():
    print(f"Error: Protected manifest not found at {manifest_path}")
    sys.exit(1)

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

repo_root = backend_dir.parent
all_passed = True
results = []

print("=" * 80)
print("PROTECTED X-RAY ASSET SHA-256 HASH VERIFICATION")
print("=" * 80)

for rel_path, expected_hash in manifest.items():
    # rel_path is relative to repo root (e.g., backend/app/ai/xray_router.py)
    file_path = repo_root / rel_path
    if not file_path.exists():
        print(f"MISSING: {rel_path}")
        all_passed = False
        results.append((rel_path, expected_hash, "FILE_NOT_FOUND", False))
        continue

    # Read binary bytes
    content = file_path.read_bytes()
    # For python text files, test both raw bytes and normalized LF bytes
    raw_hash = hashlib.sha256(content).hexdigest()
    lf_hash = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()

    matches = (raw_hash.lower() == expected_hash.lower()) or (lf_hash.lower() == expected_hash.lower())
    active_hash = raw_hash if (raw_hash.lower() == expected_hash.lower()) else lf_hash

    status_str = "MATCH (BIT-EXACT)" if matches else "MISMATCH"
    if not matches:
        all_passed = False

    print(f"File: {rel_path}")
    print(f"  Expected: {expected_hash}")
    print(f"  Observed: {active_hash} [{'PASS' if matches else 'FAIL'}]")
    results.append((rel_path, expected_hash, active_hash, matches))

print("=" * 80)
total = len(manifest)
passed = sum(1 for r in results if r[3])
print(f"SUMMARY: {passed}/{total} protected files matched manifest.")
if all_passed:
    print("ALL PROTECTED X-RAY ASSETS VERIFIED INTACT AND UNMODIFIED.")
    sys.exit(0)
else:
    print("WARNING: ONE OR MORE PROTECTED ASSETS DEVIATE FROM MANIFEST!")
    sys.exit(1)
