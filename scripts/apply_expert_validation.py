"""
Apply a completed DELF expert-review CSV back to delf_questions.json.

Accepted validation_decision values:
  - VALIDATED: keep current question, mark as expert validated
  - FIX_SOURCE: replace expected_sources from expert_corrected_sources, mark validated
  - FIX_ANSWER: replace ground_truth from expert_corrected_ground_truth, mark validated
  - FIX_BOTH: replace both sources and answer, mark validated
  - DROP: set answerable=false and mark dropped

Rows with an empty validation_decision are left unchanged unless --strict is set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_JSON = Path("evaluation/delf_questions.json")
DEFAULT_REVIEW = Path("evaluation/delf_questions_expert_review.csv")
DEFAULT_CORPUS_FILES = Path("evaluation/corpus_files.json")

VALID_DECISIONS = {"VALIDATED", "FIX_SOURCE", "FIX_ANSWER", "FIX_BOTH", "DROP"}
NON_EXPERT_REVIEW_MARKERS = (
    "ai-grounded",
    "ai grounded",
    "llm-grounded",
    "llm generated",
    "llm-generated",
    "machine-generated",
    "machine generated",
    "automated review",
    "chatgpt",
    "claude",
    "codex",
    "gemini",
)


def _load_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data, data.get("questions", [])
    return data, data


def _split_sources(cell: str) -> list[str]:
    return [part.strip() for part in cell.split(";") if part.strip()]


def _item_sources(item: dict) -> list[str]:
    values = item.get("expected_sources", [])
    if isinstance(values, str):
        return _split_sources(values)
    if isinstance(values, list):
        return [str(value).strip() for value in values if str(value).strip()]
    return []


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


def _missing_corpus_sources(sources: list[str], corpus_names: set[str] | None) -> list[str]:
    if corpus_names is None:
        return []
    missing: list[str] = []
    for source in sources:
        basename = unicodedata.normalize("NFC", Path(source).name)
        if basename not in corpus_names:
            missing.append(source)
    return missing


def _non_expert_review_reason(notes: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", notes).casefold()
    for marker in NON_EXPERT_REVIEW_MARKERS:
        if marker in normalized:
            return marker
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".validation_manifest.json")


def _same_path(a: Path, b: Path) -> bool:
    return a.expanduser().resolve(strict=False) == b.expanduser().resolve(strict=False)


def _validate_output_path(
    *,
    json_path: Path,
    review_path: Path,
    output_path: Path,
    corpus_files_path: Path | None,
) -> None:
    collisions = [
        ("input eval JSON", json_path),
        ("expert review CSV", review_path),
    ]
    if corpus_files_path is not None:
        collisions.append(("corpus catalog", corpus_files_path))
    for label, path in collisions:
        if _same_path(output_path, path):
            raise SystemExit(f"--output must not overwrite the {label}: {path}")


def _write_manifest(
    *,
    json_path: Path,
    review_path: Path,
    output_path: Path,
    manifest_path: Path,
    strict: bool,
    counts: dict,
    corpus_files_path: Path | None,
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strict": strict,
        "input_eval_path": str(json_path),
        "review_path": str(review_path),
        "output_eval_path": str(output_path),
        "input_eval_sha256": _sha256(json_path),
        "review_sha256": _sha256(review_path),
        "output_eval_sha256": _sha256(output_path),
        "counts": counts,
    }
    if corpus_files_path is not None:
        payload["corpus_files_path"] = str(corpus_files_path)
        payload["corpus_files_sha256"] = _sha256(corpus_files_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_review(
    json_path: Path,
    review_path: Path,
    output_path: Path,
    *,
    strict: bool,
    corpus_files_path: Path | None = None,
) -> dict:
    _validate_output_path(
        json_path=json_path,
        review_path=review_path,
        output_path=output_path,
        corpus_files_path=corpus_files_path,
    )
    data, questions = _load_json(json_path)
    question_ids = [str(item.get("id", "")).strip() for item in questions]
    by_id = {str(item.get("id", "")).strip(): item for item in questions}
    updated = deepcopy(data)
    if isinstance(updated, dict):
        updated_questions = updated.get("questions", [])
    else:
        updated_questions = updated
    updated_by_id = {str(item.get("id", "")).strip(): item for item in updated_questions}

    counts = {"validated": 0, "fixed": 0, "dropped": 0, "unchanged": 0}
    errors: list[str] = []
    seen: set[str] = set()
    corpus_names = _load_corpus_basenames(corpus_files_path) if corpus_files_path else None
    if any(not qid for qid in question_ids):
        errors.append("Eval JSON contains blank or missing IDs")
    duplicate_eval_ids = sorted(qid for qid, count in Counter(question_ids).items() if qid and count > 1)
    if duplicate_eval_ids:
        errors.append(
            "Eval JSON contains duplicate IDs: "
            + ", ".join(duplicate_eval_ids[:10])
            + (" ..." if len(duplicate_eval_ids) > 10 else "")
        )

    with review_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {
            "id",
            "validation_decision",
            "expert_corrected_sources",
            "expert_corrected_ground_truth",
        }
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            raise SystemExit("\n".join(errors))
        for row in reader:
            qid = str(row.get("id", "")).strip()
            decision = str(row.get("validation_decision", "")).strip().upper()
            if not qid or qid not in by_id:
                errors.append(f"Unknown id in review CSV: {qid}")
                continue
            if qid in seen:
                errors.append(f"Duplicate id in review CSV: {qid}")
                continue
            seen.add(qid)
            if not decision:
                if strict:
                    errors.append(f"Missing validation_decision for {qid}")
                else:
                    counts["unchanged"] += 1
                continue
            if decision not in VALID_DECISIONS:
                errors.append(f"Invalid validation_decision for {qid}: {decision}")
                continue

            item = updated_by_id[qid]
            notes = str(row.get("expert_notes", "")).strip()
            non_expert_reason = _non_expert_review_reason(notes)
            if non_expert_reason:
                errors.append(
                    f"{qid} expert_notes indicate non-expert/AI review evidence: {non_expert_reason}"
                )
                continue
            if notes:
                item["expert_notes"] = notes

            if decision == "DROP":
                item["answerable"] = False
                item["validation_status"] = "DROPPED_BY_DELF_EXPERT"
                counts["dropped"] += 1
                continue

            row_fixed = False
            retained_sources = _item_sources(item)
            if decision in {"FIX_SOURCE", "FIX_BOTH"}:
                sources = _split_sources(str(row.get("expert_corrected_sources", "")))
                if not sources:
                    errors.append(f"{qid} marked {decision} but expert_corrected_sources is empty")
                    continue
                missing = _missing_corpus_sources(sources, corpus_names)
                if missing:
                    errors.append(
                        f"{qid} marked {decision} but corrected source is not in corpus_files.json: "
                        + ", ".join(missing[:5])
                    )
                    continue
                item["expected_sources"] = sources
                retained_sources = sources
                row_fixed = True

            if decision in {"FIX_ANSWER", "FIX_BOTH"}:
                answer = str(row.get("expert_corrected_ground_truth", "")).strip()
                if not answer:
                    errors.append(f"{qid} marked {decision} but expert_corrected_ground_truth is empty")
                    continue
                item["ground_truth"] = answer
                row_fixed = True

            if decision in {"VALIDATED", "FIX_ANSWER"}:
                missing = _missing_corpus_sources(retained_sources, corpus_names)
                if missing:
                    errors.append(
                        f"{qid} marked {decision} but retained source is not in corpus_files.json: "
                        + ", ".join(missing[:5])
                    )
                    continue

            item["answerable"] = True
            item["validation_status"] = "VALIDATED_BY_DELF_EXPERT"
            counts["validated"] += 1
            if row_fixed:
                counts["fixed"] += 1

    if strict:
        missing_ids = sorted(set(by_id) - seen)
        if missing_ids:
            preview = ", ".join(missing_ids[:10])
            suffix = " ..." if len(missing_ids) > 10 else ""
            errors.append(f"Review CSV missing eval IDs: {preview}{suffix}")

    if errors:
        raise SystemExit("\n".join(errors))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_manifest(
        json_path=json_path,
        review_path=review_path,
        output_path=output_path,
        manifest_path=default_manifest_path(output_path),
        strict=strict,
        counts=counts,
        corpus_files_path=corpus_files_path,
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply DELF expert validation review.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=Path("evaluation/delf_questions.validated.json"))
    parser.add_argument("--corpus-files", type=Path, default=DEFAULT_CORPUS_FILES)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    counts = apply_review(
        args.json,
        args.review,
        args.output,
        strict=args.strict,
        corpus_files_path=args.corpus_files,
    )
    print(json.dumps(
        {"output": str(args.output), "manifest": str(default_manifest_path(args.output)), **counts},
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
