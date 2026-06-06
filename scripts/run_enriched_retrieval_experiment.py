"""
Run the controlled enriched-ingestion retrieval experiment.

This script intentionally writes to a separate ChromaDB directory by default
(`chroma_db_enriched`) so the current baseline DB remains a control. It sets
environment variables before importing application modules because `settings` is
loaded at import time.

Typical sequence:
    python -m scripts.run_enriched_retrieval_experiment --preflight
    python -m scripts.run_enriched_retrieval_experiment --run-ingest --force
    python -m scripts.run_enriched_retrieval_experiment --skip-ingest --evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_DIR = REPO / "chroma_db_enriched"
DEFAULT_DATA_DIR = REPO / "new_docs"
BASELINE_CHROMA_DIR = REPO / "chroma_db"


def _set_env(chroma_dir: Path, data_dir: Path) -> None:
    os.environ["CHROMA_DIR"] = str(chroma_dir)
    os.environ["DATA_DIR"] = str(data_dir)
    # Keep settings-based fallback consumers aligned with the explicit CLI path.
    # The recall call below also passes the loaded questions directly.
    os.environ["ENABLE_CONTEXTUAL_ENRICHMENT"] = "true"
    os.environ["CHUNKING_STRATEGY"] = "paragraph"
    os.environ["EMBEDDING_DIMENSION"] = "3072"
    os.environ["ENABLE_CROSS_LINGUAL_BM25"] = "false"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _refuse_unsafe_target(chroma_dir: Path, *, force: bool) -> None:
    resolved = chroma_dir.resolve()
    baseline = BASELINE_CHROMA_DIR.resolve()
    repo = REPO.resolve()
    if resolved == baseline or resolved.name == "chroma_db":
        raise SystemExit(
            f"Refusing to use baseline CHROMA_DIR as experiment target: {resolved}"
        )
    if _is_relative_to(resolved, baseline):
        raise SystemExit(
            f"Refusing to use a path inside baseline CHROMA_DIR as experiment target: {resolved}"
        )
    if resolved == repo:
        raise SystemExit(f"Refusing to use repository root as experiment target: {resolved}")
    if _is_relative_to(repo, resolved) or _is_relative_to(baseline, resolved):
        raise SystemExit(
            f"Refusing to use a parent of the repository or baseline DB as experiment target: {resolved}"
        )
    if chroma_dir.is_symlink():
        raise SystemExit(f"Refusing to use symlinked experiment target: {chroma_dir}")
    if resolved.exists() and not force:
        raise SystemExit(
            f"Experiment CHROMA_DIR already exists: {resolved}\n"
            "Use --force to clear/rebuild it, or --skip-ingest --evaluate to reuse it."
        )


def _prepare_reingest_target(chroma_dir: Path, *, force: bool) -> dict:
    """Prepare the experiment DB directory for a zero-state rebuild."""
    _refuse_unsafe_target(chroma_dir, force=force)
    resolved = chroma_dir.resolve()
    cleared = False
    if resolved.exists():
        if not resolved.is_dir():
            raise SystemExit(f"Experiment CHROMA_DIR is not a directory: {resolved}")
        shutil.rmtree(resolved)
        cleared = True

    return {
        "chroma_dir": str(resolved),
        "cleared_existing_directory": cleared,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_fingerprint(root: Path) -> dict:
    resolved = root.resolve()
    if not resolved.exists():
        return {
            "path": str(resolved),
            "present": False,
            "file_count": 0,
            "total_bytes": 0,
            "sha256": None,
        }

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(p for p in resolved.rglob("*") if p.is_file()):
        rel = path.relative_to(resolved).as_posix()
        size = path.stat().st_size
        file_hash = _file_sha256(path)
        file_count += 1
        total_bytes += size
        digest.update(rel.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        digest.update(file_hash.encode("ascii"))

    return {
        "path": str(resolved),
        "present": True,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _baseline_integrity(before: dict, after: dict) -> dict:
    unchanged = before == after
    return {
        "ok": unchanged,
        "baseline_chroma_dir": str(BASELINE_CHROMA_DIR.resolve()),
        "before": before,
        "after": after,
    }


def _experiment_db_readiness(chroma_dir: Path) -> dict:
    failures: list[str] = []
    parents_path = chroma_dir / "parents.json"
    bm25_path = chroma_dir / "bm25_index.pkl"
    parent_count: int | None = None
    child_count: int | None = None

    if not chroma_dir.exists():
        failures.append(f"CHROMA_DIR does not exist: {chroma_dir}")
    if not parents_path.exists():
        failures.append(f"parents.json missing: {parents_path}")
    else:
        try:
            parent_count = len(json.loads(parents_path.read_text(encoding="utf-8")))
            if parent_count == 0:
                failures.append("parents.json is empty")
        except Exception as exc:  # noqa: BLE001 - diagnostic guard only
            failures.append(f"parents.json unreadable: {type(exc).__name__}: {exc}")
    if not bm25_path.exists():
        failures.append(f"bm25_index.pkl missing: {bm25_path}")

    if chroma_dir.exists():
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(chroma_dir))
            child_count = client.get_collection("tee_children").count()
            if child_count == 0:
                failures.append("tee_children collection is empty")
        except Exception as exc:  # noqa: BLE001 - absent/corrupt collection
            failures.append(f"tee_children collection unreadable: {type(exc).__name__}: {exc}")

    return {
        "ok": not failures,
        "failures": failures,
        "parent_count": parent_count,
        "child_count": child_count,
    }


def _load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", data) if isinstance(data, dict) else data


def _preflight(chroma_dir: Path, data_dir: Path, eval_path: Path) -> dict:
    from src.document_loaders import iter_documents

    docs = list(iter_documents(data_dir))
    failures = [str(d.path) + ": " + str(d.error) for d in docs if d.error]
    nonempty = [
        d for d in docs
        if not d.error and any(page.text.strip() for page in d.pages)
    ]
    questions = _load_questions(eval_path)
    unvalidated = [
        q["id"] for q in questions
        if not str(q.get("validation_status", "")).startswith("VALIDATED")
    ]
    return {
        "experiment_chroma_dir": str(chroma_dir),
        "data_dir": str(data_dir),
        "eval_path": str(eval_path),
        "documents_total": len(docs),
        "documents_nonempty": len(nonempty),
        "document_failures": failures,
        "experiment_db_readiness": _experiment_db_readiness(chroma_dir),
        "eval_questions": len(questions),
        "eval_unvalidated": len(unvalidated),
        "eval_unvalidated_ids": unvalidated[:10],
    }


def _write_experiment_summary(payload: dict, results_dir: Path | None = None) -> Path:
    out_dir = results_dir or REPO / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"enriched_experiment_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _run_ingest() -> dict:
    from src.ingestion import clear_and_reingest

    return clear_and_reingest()


def _run_recall(eval_path: Path, results_dir: Path, *, rerank: bool) -> dict:
    from src.config import settings
    from src.evaluator import load_test_questions, run_recall_report

    settings.ENABLE_RERANKING = rerank
    settings.RAGAS_TEST_SET_PATH = eval_path
    settings.RAGAS_RESULTS_DIR = results_dir
    questions = load_test_questions(eval_path)
    return run_recall_report(test_questions=questions, save=True)


def _recall_gate_summary(report: dict) -> dict:
    metadata = report["metadata"]
    aggregate = report["aggregate"]["overall"]
    m3_k = metadata["m3_k"]
    return {
        "artifact_json": metadata.get("artifact_json"),
        "artifact_md": metadata.get("artifact_md"),
        "n_questions_evaluated": metadata["n_questions_evaluated"],
        "m3_k": m3_k,
        "m3_recall_at_k": metadata["m3_recall_at_k"],
        "m3_hit_at_k": metadata["m3_hit_at_k"],
        "m3_passed": bool(metadata["m3_recall_at_k"] is not None and metadata["m3_recall_at_k"] >= 0.90),
        "recall_at_1": aggregate.get("recall@1"),
        "recall_at_3": aggregate.get("recall@3"),
        "recall_at_5": aggregate.get("recall@5"),
        "recall_at_10": aggregate.get("recall@10"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled enriched retrieval experiment.")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--eval-path", type=Path, default=REPO / "evaluation" / "delf_questions.json")
    parser.add_argument("--results-dir", type=Path, default=REPO / "evaluation" / "results")
    parser.add_argument("--certification-run-id")
    parser.add_argument("--preflight", action="store_true", help="Check inputs only; no ingest/eval.")
    parser.add_argument("--run-ingest", action="store_true", help="Clear/rebuild the experiment DB.")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse an existing experiment DB.")
    parser.add_argument("--evaluate", action="store_true", help="Run recall reports after ingest/reuse.")
    parser.add_argument(
        "--require-validated",
        action="store_true",
        help="Refuse evaluation unless every answerable eval item is expert-validated.",
    )
    parser.add_argument("--force", action="store_true", help="Allow clearing an existing experiment DB.")
    args = parser.parse_args()

    chroma_dir = args.chroma_dir.expanduser()
    data_dir = args.data_dir.expanduser()
    eval_path = args.eval_path.expanduser()
    results_dir = args.results_dir.expanduser()

    if not data_dir.exists():
        raise SystemExit(f"DATA_DIR does not exist: {data_dir}")
    if not eval_path.exists():
        raise SystemExit(f"Eval set does not exist: {eval_path}")

    _set_env(chroma_dir, data_dir)
    os.environ["RAGAS_TEST_SET_PATH"] = str(eval_path)
    os.environ["RAGAS_RESULTS_DIR"] = str(results_dir)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    preflight = _preflight(chroma_dir, data_dir, eval_path)
    if args.preflight:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if not preflight["document_failures"] else 2

    if not args.run_ingest and not args.skip_ingest:
        raise SystemExit("Choose --run-ingest or --skip-ingest. Use --preflight for checks only.")
    if args.run_ingest and args.skip_ingest:
        raise SystemExit("Choose only one of --run-ingest or --skip-ingest.")
    if args.evaluate and args.require_validated and preflight["eval_unvalidated"]:
        raise SystemExit(
            f"Refusing certification evaluation: {preflight['eval_unvalidated']} eval items "
            "are not expert-validated."
        )

    if args.run_ingest:
        target_preparation = _prepare_reingest_target(chroma_dir, force=args.force)
    else:
        _refuse_unsafe_target(chroma_dir, force=True)
        target_preparation = {
            "chroma_dir": str(chroma_dir.resolve()),
            "cleared_existing_directory": False,
        }
    baseline_before = _directory_fingerprint(BASELINE_CHROMA_DIR)

    summary: dict = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_chroma_dir": str(chroma_dir),
            "data_dir": str(data_dir),
            "eval_path": str(eval_path),
            "contextual_enrichment": True,
            "chunking_strategy": "paragraph",
            "embedding_dimension": 3072,
            "cross_lingual_bm25": False,
            "certification_run_id": args.certification_run_id,
        },
        "preflight": preflight,
        "baseline_integrity": {
            "baseline_chroma_dir": str(BASELINE_CHROMA_DIR.resolve()),
            "before": baseline_before,
            "after": None,
            "ok": None,
        },
        "target_preparation": target_preparation,
    }

    if args.run_ingest:
        summary["ingestion"] = _run_ingest()

    if args.evaluate:
        readiness = _experiment_db_readiness(chroma_dir)
        summary["experiment_db_readiness"] = readiness
        if not readiness["ok"]:
            baseline_after = _directory_fingerprint(BASELINE_CHROMA_DIR)
            summary["baseline_integrity"] = _baseline_integrity(baseline_before, baseline_after)
            out = _write_experiment_summary(summary, results_dir)
            raise SystemExit(
                f"Refusing evaluation; experiment DB is incomplete. Summary: {out}\n"
                + json.dumps(readiness, ensure_ascii=False, indent=2)
            )
        summary["recall_no_rerank"] = _recall_gate_summary(
            _run_recall(eval_path, results_dir, rerank=False)
        )
        summary["recall_with_rerank"] = _recall_gate_summary(
            _run_recall(eval_path, results_dir, rerank=True)
        )

    baseline_after = _directory_fingerprint(BASELINE_CHROMA_DIR)
    summary["baseline_integrity"] = _baseline_integrity(baseline_before, baseline_after)
    out = _write_experiment_summary(summary, results_dir)
    print(f"experiment summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
