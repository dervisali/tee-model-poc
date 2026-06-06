"""
Verify the DELF expert-validation packet ZIP against its embedded manifest.

This is a cheap provenance check before sending or archiving the expert packet:
every file listed in the manifest must exist inside the ZIP, and its byte size,
SHA-256, and CSV row count must match the recorded values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


DEFAULT_PACKET = Path("evaluation/delf_expert_validation_packet.zip")
MANIFEST_ARCNAME = "evaluation/delf_expert_validation_packet_manifest.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv_row_count_bytes(path: str, data: bytes) -> int | None:
    if not path.lower().endswith(".csv"):
        return None
    text = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline="")
    return sum(1 for _ in csv.DictReader(text))


def verify_packet(packet_path: Path) -> dict:
    failures: list[str] = []
    packet_sha = _sha256_bytes(packet_path.read_bytes()) if packet_path.exists() else None
    if not packet_path.exists():
        return {
            "ok": False,
            "packet": str(packet_path),
            "packet_sha256": packet_sha,
            "failures": [f"packet missing: {packet_path}"],
        }

    try:
        with zipfile.ZipFile(packet_path) as zf:
            names = set(zf.namelist())
            if MANIFEST_ARCNAME not in names:
                return {
                    "ok": False,
                    "packet": str(packet_path),
                    "packet_sha256": packet_sha,
                    "failures": [f"manifest missing: {MANIFEST_ARCNAME}"],
                }
            manifest = json.loads(zf.read(MANIFEST_ARCNAME).decode("utf-8"))
            entries = manifest.get("files", [])
            if not isinstance(entries, list) or not entries:
                failures.append("manifest files must be a non-empty list")
                entries = []
            for entry in entries:
                if not isinstance(entry, dict):
                    failures.append("manifest file entries must be objects")
                    continue
                arcname = str(entry.get("path", ""))
                if not arcname:
                    failures.append("manifest file entry missing path")
                    continue
                if arcname not in names:
                    failures.append(f"packet file missing: {arcname}")
                    continue
                data = zf.read(arcname)
                if entry.get("bytes") != len(data):
                    failures.append(f"{arcname} byte count mismatch")
                if entry.get("sha256") != _sha256_bytes(data):
                    failures.append(f"{arcname} sha256 mismatch")
                if entry.get("csv_rows") != _csv_row_count_bytes(arcname, data):
                    failures.append(f"{arcname} csv_rows mismatch")
    except Exception as exc:  # noqa: BLE001 - malformed ZIP/JSON is a no-go
        failures.append(f"packet unreadable: {type(exc).__name__}: {exc}")

    return {
        "ok": not failures,
        "packet": str(packet_path),
        "packet_sha256": packet_sha,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DELF expert validation packet ZIP.")
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    args = parser.parse_args()

    result = verify_packet(args.packet.expanduser())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
