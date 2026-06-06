"""
Export evaluation/delf_questions.json to a DELF-expert review CSV.

The JSON file remains the source of truth for the harness. This script creates a
review-friendly table with explicit columns for the expert's decision, corrected
sources, and notes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUT = Path("evaluation/delf_questions.json")
DEFAULT_OUTPUT = Path("evaluation/delf_questions_expert_review.csv")
DEFAULT_CORPUS_FILES = Path("evaluation/corpus_files.json")
DEFAULT_SOURCE_CATALOG = Path("evaluation/delf_source_catalog.csv")
EXPERT_INPUT_COLUMNS = [
    "validation_decision",
    "expert_corrected_sources",
    "expert_corrected_ground_truth",
    "expert_notes",
]

REVIEW_COLUMNS = [
    "id",
    "validation_decision",
    "expert_corrected_sources",
    "expert_corrected_ground_truth",
    "expert_notes",
    "category",
    "level",
    "skill",
    "language",
    "type",
    "difficulty",
    "question",
    "ground_truth",
    "expected_sources",
    "split",
    "current_validation_status",
]

SOURCE_COLUMNS = [
    "filename",
    "child_count",
    "doc_type",
    "level",
    "skill",
]


def _load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", data) if isinstance(data, dict) else data


def _as_cell(value) -> str:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def _compact_counts(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return "; ".join(f"{name}:{count}" for name, count in value)
    return str(value)


def _has_existing_expert_work(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(str(row.get(column, "")).strip() for column in EXPERT_INPUT_COLUMNS):
                return True
    return False


def export_review_csv(input_path: Path, output_path: Path, *, force_overwrite: bool = False) -> int:
    if not force_overwrite and _has_existing_expert_work(output_path):
        raise FileExistsError(
            f"Refusing to overwrite existing expert review work in {output_path}. "
            "Pass --force-overwrite only after saving/archiving the reviewed CSV."
        )

    questions = _load_questions(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for item in questions:
            writer.writerow({
                "id": item.get("id", ""),
                "validation_decision": "",
                "expert_corrected_sources": "",
                "expert_corrected_ground_truth": "",
                "expert_notes": "",
                "category": item.get("category", ""),
                "level": item.get("level", ""),
                "skill": item.get("skill", ""),
                "language": item.get("language", ""),
                "type": item.get("type", ""),
                "difficulty": item.get("difficulty", ""),
                "question": item.get("question", ""),
                "ground_truth": item.get("ground_truth", ""),
                "expected_sources": _as_cell(item.get("expected_sources", [])),
                "split": item.get("split", ""),
                "current_validation_status": item.get("validation_status", ""),
            })
    return len(questions)


def export_source_catalog(corpus_files_path: Path, output_path: Path) -> int:
    data = json.loads(corpus_files_path.read_text(encoding="utf-8"))
    files = data.get("files", data) if isinstance(data, dict) else data
    if not isinstance(files, dict):
        raise ValueError("corpus_files.json must contain a 'files' object keyed by filename.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        for filename, meta in sorted(files.items()):
            writer.writerow({
                "filename": filename,
                "child_count": meta.get("child_count", ""),
                "doc_type": _compact_counts(meta.get("doc_type")),
                "level": _compact_counts(meta.get("level")),
                "skill": _compact_counts(meta.get("skill")),
            })
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare DELF expert validation CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus-files", type=Path, default=DEFAULT_CORPUS_FILES)
    parser.add_argument("--source-catalog", type=Path, default=DEFAULT_SOURCE_CATALOG)
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite an existing review CSV even if expert decision/note fields are already filled.",
    )
    args = parser.parse_args()

    count = export_review_csv(args.input, args.output, force_overwrite=args.force_overwrite)
    source_count = export_source_catalog(args.corpus_files, args.source_catalog)
    print(f"wrote {count} review rows: {args.output}")
    print(f"wrote {source_count} source rows: {args.source_catalog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
