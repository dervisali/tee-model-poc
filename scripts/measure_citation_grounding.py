"""
Measure chatbot citation grounding over the DELF evaluation set.

This is the M4-style gate companion to recall@k: for each evaluated question it
runs the non-streaming chatbot path, reads the built-in citation_report, and
writes an auditable JSON artifact. It is intentionally explicit because it makes
live generation calls unless MOCK_MODE=true.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_PATH = REPO / "evaluation" / "delf_questions.json"
DEFAULT_RESULTS_DIR = REPO / "evaluation" / "results"
BASELINE_CHROMA_DIR = REPO / "chroma_db"


def _set_env(chroma_dir: Path | None, eval_path: Path) -> None:
    if chroma_dir is not None:
        os.environ["CHROMA_DIR"] = str(chroma_dir)
    os.environ["RAGAS_TEST_SET_PATH"] = str(eval_path)


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", data) if isinstance(data, dict) else data


def is_validated(item: dict) -> bool:
    return str(item.get("validation_status", "")).startswith("VALIDATED")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def select_questions(
    questions: list[dict],
    *,
    require_validated: bool,
    max_questions: int | None,
) -> list[dict]:
    selected = [
        q for q in questions
        if q.get("answerable", True) and (is_validated(q) or not require_validated)
    ]
    if max_questions is not None:
        selected = selected[:max_questions]
    return selected


def validate_certification_inputs(
    *,
    certification_run_id: str | None,
    require_validated: bool,
    max_questions: int | None,
    chroma_dir: Path | None,
) -> None:
    """Fail fast for artifacts intended to count as M4 production evidence."""
    if not certification_run_id:
        return
    if not require_validated:
        raise SystemExit("--certification-run-id requires --require-validated")
    if max_questions is not None:
        raise SystemExit("--certification-run-id must cover the full validated eval set; omit --max-questions")
    if chroma_dir is None:
        raise SystemExit("--certification-run-id requires --chroma-dir")
    resolved = chroma_dir.expanduser().resolve(strict=False)
    baseline = BASELINE_CHROMA_DIR.resolve(strict=False)
    repo = REPO.resolve(strict=False)
    if resolved.name == "chroma_db" or resolved == baseline:
        raise SystemExit("--certification-run-id must not measure citation grounding against baseline chroma_db")
    if _is_relative_to(resolved, baseline):
        raise SystemExit(
            "--certification-run-id must not measure citation grounding inside baseline chroma_db"
        )
    if _is_relative_to(repo, resolved) or _is_relative_to(baseline, resolved):
        raise SystemExit(
            "--certification-run-id must not measure citation grounding against a parent of the repo or baseline chroma_db"
        )
    from src.readiness_check import check_readiness

    readiness = check_readiness(run_smoke=False, chroma_dir=resolved)
    if not readiness.ok:
        raise SystemExit(
            "Refusing M4 certification run; CHROMA_DIR is not ready:\n"
            + "\n".join(readiness.failures)
        )


def validate_certification_eval_questions(
    questions: list[dict],
    *,
    certification_run_id: str | None,
) -> None:
    if not certification_run_id:
        return
    unvalidated = [
        str(item.get("id", idx + 1))
        for idx, item in enumerate(questions)
        if item.get("answerable", True) and not is_validated(item)
    ]
    if unvalidated:
        raise SystemExit(
            "Refusing M4 certification run; eval file contains unvalidated answerable items: "
            + ", ".join(unvalidated[:10])
        )


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    passed = sum(1 for row in rows if row["citation_passed"])
    no_error = sum(1 for row in rows if not row.get("error"))
    grounding_allowed = sum(
        1 for row in rows
        if row.get("safety", {}).get("grounding_allowed") is not False
    )
    return {
        "n": n,
        "citation_passed": passed,
        "citation_pass_rate": (passed / n if n else None),
        "no_error": no_error,
        "no_error_rate": (no_error / n if n else None),
        "grounding_allowed": grounding_allowed,
        "grounding_allowed_rate": (grounding_allowed / n if n else None),
    }


def _run_chat_rows(questions: list[dict], *, language_override: str | None) -> list[dict]:
    from src.chatbot import chat

    rows: list[dict] = []
    for idx, item in enumerate(questions, start=1):
        language = language_override or item.get("language") or None
        result = chat(
            item["question"],
            history=[],
            language=language,
            session_id=f"citation-eval-{item.get('id', idx)}",
        )
        report = result.get("citation_report") or {}
        rows.append({
            "id": item.get("id", f"q{idx:02d}"),
            "language": item.get("language"),
            "category": item.get("category"),
            "level": item.get("level"),
            "question": item.get("question"),
            "validation_status": item.get("validation_status"),
            "citation_passed": bool(report.get("passed")),
            "cited_count": report.get("cited_count", 0),
            "retrieved_count": report.get("retrieved_count", 0),
            "invalid_parent_ids": report.get("invalid_parent_ids", []),
            "cited_parent_ids": report.get("cited_parent_ids", []),
            "error": result.get("error"),
            "safety": result.get("safety", {}),
            "answer_preview": str(result.get("answer", ""))[:500],
        })
    return rows


def write_report(payload: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"citation_grounding_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure chatbot citation grounding.")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--chroma-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--require-validated", action="store_true")
    parser.add_argument("--language", choices=["tr", "fr"], default=None)
    parser.add_argument("--certification-run-id")
    parser.add_argument("--preflight", action="store_true", help="Report selection only; no model calls.")
    args = parser.parse_args()

    eval_path = args.eval_path.expanduser()
    if not eval_path.exists():
        raise SystemExit(f"Eval set does not exist: {eval_path}")

    _set_env(args.chroma_dir.expanduser() if args.chroma_dir else None, eval_path)
    validate_certification_inputs(
        certification_run_id=args.certification_run_id,
        require_validated=args.require_validated,
        max_questions=args.max_questions,
        chroma_dir=args.chroma_dir,
    )

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    questions = load_questions(eval_path)
    validate_certification_eval_questions(
        questions,
        certification_run_id=args.certification_run_id,
    )
    selected = select_questions(
        questions,
        require_validated=args.require_validated,
        max_questions=args.max_questions,
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_path": str(eval_path),
        "chroma_dir": str(args.chroma_dir) if args.chroma_dir else None,
        "questions_total": len(questions),
        "questions_selected": len(selected),
        "require_validated": args.require_validated,
        "max_questions": args.max_questions,
        "certification_run_id": args.certification_run_id,
    }
    if args.preflight:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0 if selected else 2
    if not selected:
        raise SystemExit("No questions selected for citation grounding measurement.")

    rows = _run_chat_rows(selected, language_override=args.language)
    payload = {
        "metadata": metadata,
        "summary": summarize(rows),
        "per_question": rows,
    }
    out = write_report(payload, args.results_dir.expanduser())
    s = payload["summary"]
    print(
        f"citation pass rate = {s['citation_pass_rate']:.1%} "
        f"({s['citation_passed']}/{s['n']}); artifact: {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
