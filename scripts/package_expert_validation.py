"""
Create a DELF expert-validation packet ZIP with a manifest.

The packet is intended to be sent to the DELF/DALF expert. The manifest records
row counts and SHA-256 hashes so we can later confirm which review packet was
sent and avoid mixing stale CSVs with newer eval/source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO / "evaluation" / "delf_expert_validation_packet.zip"
DEFAULT_FILES = [
    REPO / "evaluation" / "delf_questions.json",
    REPO / "evaluation" / "corpus_files.json",
    REPO / "evaluation" / "delf_questions_expert_review.csv",
    REPO / "evaluation" / "delf_source_catalog.csv",
    REPO / "evaluation" / "EXPERT_VALIDATION_GUIDE.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_row_count(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def build_manifest(files: list[Path]) -> dict:
    manifest_files = []
    for path in files:
        manifest_files.append({
            "path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "csv_rows": csv_row_count(path),
        })
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "DELF/DALF retrieval evaluation expert validation",
        "files": manifest_files,
    }


def create_packet(files: list[Path], output: Path) -> dict:
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing packet files: " + ", ".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(files)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=str(path.relative_to(REPO)))
        zf.writestr(
            "evaluation/delf_expert_validation_packet_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "manifest": manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Package DELF expert validation files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--file", type=Path, action="append", dest="files")
    args = parser.parse_args()

    files = [path.expanduser() for path in (args.files or DEFAULT_FILES)]
    payload = create_packet(files, args.output.expanduser())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
