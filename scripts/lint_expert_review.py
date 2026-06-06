"""
Lint the DELF expert-review CSV before applying it.

This catches mechanical review issues early:
  - missing/unknown IDs,
  - missing validation decisions,
  - invalid decision values,
  - empty correction fields for FIX_* decisions,
  - retained/corrected source filenames that do not exist in evaluation/corpus_files.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from scripts.apply_expert_validation import (
    VALID_DECISIONS,
    _item_sources,
    _non_expert_review_reason,
)


DEFAULT_JSON = Path("evaluation/delf_questions.json")
DEFAULT_REVIEW = Path("evaluation/delf_questions_expert_review.csv")
DEFAULT_CORPUS_FILES = Path("evaluation/corpus_files.json")


@dataclass
class LintResult:
    rows: int
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", data) if isinstance(data, dict) else data


def _load_corpus_basenames(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files", data) if isinstance(data, dict) else data
    if isinstance(files, dict):
        names = files.keys()
    else:
        names = [
            item.get("filename") or item.get("source_filename") or item.get("path") or item
            for item in files
        ]
    return {unicodedata.normalize("NFC", Path(str(name)).name) for name in names}


def _split_sources(cell: str) -> list[str]:
    return [part.strip() for part in cell.split(";") if part.strip()]


def lint_review(
    *,
    eval_path: Path,
    review_path: Path,
    corpus_files_path: Path,
    require_complete: bool,
) -> LintResult:
    questions = _load_questions(eval_path)
    question_ids = [str(item.get("id", "")).strip() for item in questions]
    by_id = {str(item.get("id", "")).strip(): item for item in questions}
    ids = set(question_ids)
    corpus_names = _load_corpus_basenames(corpus_files_path)
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()

    duplicate_eval_ids = sorted(qid for qid, count in Counter(question_ids).items() if qid and count > 1)
    if any(not qid for qid in question_ids):
        errors.append("Eval JSON contains blank or missing IDs")
    if duplicate_eval_ids:
        errors.append(
            "Eval JSON contains duplicate IDs: "
            + ", ".join(duplicate_eval_ids[:10])
            + (" ..." if len(duplicate_eval_ids) > 10 else "")
        )

    with review_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required_columns = {
        "id",
        "validation_decision",
        "expert_corrected_sources",
        "expert_corrected_ground_truth",
    }
    missing_columns = sorted(required_columns - set(fieldnames))
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return LintResult(rows=len(rows), errors=errors, warnings=warnings)

    for i, row in enumerate(rows, start=2):  # header is line 1
        qid = str(row.get("id", "")).strip()
        decision = str(row.get("validation_decision", "")).strip().upper()
        prefix = f"line {i} id={qid or '<missing>'}"

        if not qid:
            errors.append(f"{prefix}: missing id")
            continue
        if qid not in ids:
            errors.append(f"{prefix}: unknown id")
            continue
        if qid in seen:
            errors.append(f"{prefix}: duplicate id")
        seen.add(qid)

        if not decision:
            message = f"{prefix}: missing validation_decision"
            if require_complete:
                errors.append(message)
            else:
                warnings.append(message)
            continue
        if decision not in VALID_DECISIONS:
            errors.append(f"{prefix}: invalid validation_decision={decision}")
            continue

        corrected_sources = _split_sources(str(row.get("expert_corrected_sources", "")))
        corrected_answer = str(row.get("expert_corrected_ground_truth", "")).strip()
        expert_notes = str(row.get("expert_notes", "")).strip()
        non_expert_reason = _non_expert_review_reason(expert_notes)
        if non_expert_reason:
            errors.append(
                f"{prefix}: expert_notes indicate non-expert/AI review evidence: {non_expert_reason}"
            )

        if decision in {"FIX_SOURCE", "FIX_BOTH"} and not corrected_sources:
            errors.append(f"{prefix}: {decision} requires expert_corrected_sources")
        if decision in {"FIX_ANSWER", "FIX_BOTH"} and not corrected_answer:
            errors.append(f"{prefix}: {decision} requires expert_corrected_ground_truth")

        sources_to_check = corrected_sources
        if decision in {"VALIDATED", "FIX_ANSWER"}:
            sources_to_check = _item_sources(by_id[qid])

        for source in sources_to_check:
            basename = unicodedata.normalize("NFC", Path(source).name)
            if basename not in corpus_names:
                errors.append(f"{prefix}: source not in corpus_files.json: {source}")

    missing_ids = sorted(ids - seen)
    if missing_ids:
        errors.append(f"Review CSV missing eval IDs: {', '.join(missing_ids[:10])}"
                      + (" ..." if len(missing_ids) > 10 else ""))

    return LintResult(rows=len(rows), errors=errors, warnings=warnings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint DELF expert-review CSV.")
    parser.add_argument("--json", "--eval-path", dest="json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--corpus-files", type=Path, default=DEFAULT_CORPUS_FILES)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    result = lint_review(
        eval_path=args.json,
        review_path=args.review,
        corpus_files_path=args.corpus_files,
        require_complete=args.require_complete,
    )
    payload = {
        "ok": result.ok,
        "rows": result.rows,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
