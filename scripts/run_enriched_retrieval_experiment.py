"""
Run the controlled enriched-ingestion retrieval experiment.

This script intentionally writes to a separate ChromaDB directory by default
(`chroma_db_enriched`) so the current baseline DB remains a control. It sets
environment variables before importing application modules because `settings` is
loaded at import time.

Typical sequence:
    python -m scripts.run_enriched_retrieval_experiment --preflight
    python -m scripts.run_enriched_retrieval_experiment --warm-cache-only --max-cache-misses 100
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
    """Prepare the experiment DB directory for a zero-state rebuild.

    `enrichment_cache.json` is preserved across forced rebuilds because it is
    keyed by source document/chunk text, not by ChromaDB state. Keeping it avoids
    repeating thousands of LLM enrichment calls while still clearing every DB
    artifact that can affect retrieval results.
    """
    _refuse_unsafe_target(chroma_dir, force=force)
    resolved = chroma_dir.resolve()
    cleared = False
    cache_name = "enrichment_cache.json"
    preserved_cache: bytes | None = None
    if resolved.exists():
        if not resolved.is_dir():
            raise SystemExit(f"Experiment CHROMA_DIR is not a directory: {resolved}")
        cache_path = resolved / cache_name
        if cache_path.exists() and cache_path.is_file():
            preserved_cache = cache_path.read_bytes()
        shutil.rmtree(resolved)
        cleared = True
        if preserved_cache is not None:
            resolved.mkdir(parents=True, exist_ok=True)
            (resolved / cache_name).write_bytes(preserved_cache)

    return {
        "chroma_dir": str(resolved),
        "cleared_existing_directory": cleared,
        "preserved_enrichment_cache": preserved_cache is not None,
        "preserved_enrichment_cache_bytes": len(preserved_cache or b""),
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


def _preflight(
    chroma_dir: Path,
    data_dir: Path,
    eval_path: Path,
    *,
    check_experiment_db: bool = True,
) -> dict:
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
        "experiment_db_readiness": (
            _experiment_db_readiness(chroma_dir)
            if check_experiment_db
            else {
                "ok": None,
                "failures": ["skipped before destructive rebuild"],
                "parent_count": None,
                "child_count": None,
            }
        ),
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


def _limit_missing_chunks(
    document_id: str,
    chunks: list[str],
    cache: dict[str, str],
    *,
    max_misses: int | None,
) -> list[str]:
    """Return chunks not already in the enrichment cache, capped if requested."""
    from src.contextual_enrichment import _cache_key

    missing: list[str] = []
    for chunk in chunks:
        if _cache_key(document_id, chunk) in cache:
            continue
        missing.append(chunk)
        if max_misses is not None and len(missing) >= max_misses:
            break
    return missing


def _warm_enrichment_cache(max_cache_misses: int | None = None) -> dict:
    """
    Populate contextual enrichment cache without writing retrieval DB artifacts.

    This is a resumable helper for the long enriched-ingestion path. The final
    controlled experiment still needs --run-ingest --evaluate to produce
    parents.json, BM25, Chroma collections, baseline fingerprints, and recall
    measurements.
    """
    from src.chunkers import get_parent_chunker
    from src.config import settings
    from src.contextual_enrichment import _load_cache, enrich_chunks
    from src.ingestion import _load_documents, _sanitize_id, _split_into_child_chunks

    strategy = settings.CHUNKING_STRATEGY
    parent_chunker = get_parent_chunker(strategy)
    documents = _load_documents()
    cache = _load_cache()
    cache_entries_before = len(cache)

    total_child_chunks = 0
    total_missing_before = 0
    warmed_misses = 0
    docs_seen = 0
    docs_warmed = 0
    stopped_after_limit = False

    for doc_load in documents:
        docs_seen += 1
        safe_stem = _sanitize_id(doc_load.metadata.source_filename)
        child_chunks: list[str] = []
        for pc in doc_load.pages:
            for parent_text in parent_chunker(pc.text):
                child_chunks.extend(_split_into_child_chunks(parent_text))

        total_child_chunks += len(child_chunks)
        missing = _limit_missing_chunks(
            safe_stem,
            child_chunks,
            cache,
            max_misses=None,
        )
        total_missing_before += len(missing)
        if not missing:
            continue

        remaining = None if max_cache_misses is None else max_cache_misses - warmed_misses
        if remaining is not None and remaining <= 0:
            stopped_after_limit = True
            break
        to_warm = missing[:remaining]
        if len(to_warm) < len(missing):
            stopped_after_limit = True

        _, stats = enrich_chunks(
            doc_load.full_text,
            to_warm,
            document_id=safe_stem,
        )
        docs_warmed += 1
        warmed_misses += stats["misses"]

        cache = _load_cache()
        if max_cache_misses is not None and warmed_misses >= max_cache_misses:
            stopped_after_limit = True
            break

    cache_entries_after = len(_load_cache())
    return {
        "chunking_strategy": strategy,
        "documents_seen": docs_seen,
        "documents_warmed": docs_warmed,
        "total_child_chunks_seen": total_child_chunks,
        "cache_entries_before": cache_entries_before,
        "cache_entries_after": cache_entries_after,
        "missing_chunks_before": total_missing_before,
        "warmed_misses": warmed_misses,
        "max_cache_misses": max_cache_misses,
        "stopped_after_limit": stopped_after_limit,
    }


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
    parser.add_argument(
        "--warm-cache-only",
        action="store_true",
        help="Populate enrichment_cache.json only; no Chroma/parents/BM25 write.",
    )
    parser.add_argument(
        "--max-cache-misses",
        type=int,
        default=None,
        help="Maximum cache misses to warm in this run; only valid with --warm-cache-only.",
    )
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

    if args.max_cache_misses is not None and args.max_cache_misses < 1:
        raise SystemExit("--max-cache-misses must be >= 1")

    preflight = _preflight(
        chroma_dir,
        data_dir,
        eval_path,
        check_experiment_db=not args.run_ingest and not args.warm_cache_only,
    )
    if args.preflight:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if not preflight["document_failures"] else 2

    if args.warm_cache_only:
        if args.run_ingest or args.skip_ingest or args.evaluate:
            raise SystemExit("--warm-cache-only cannot be combined with ingest/evaluate flags.")
        _refuse_unsafe_target(chroma_dir, force=True)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "experiment_chroma_dir": str(chroma_dir),
                "data_dir": str(data_dir),
                "contextual_enrichment": True,
                "chunking_strategy": "paragraph",
                "mode": "warm-cache-only",
            },
            "preflight": preflight,
            "warm_cache": _warm_enrichment_cache(args.max_cache_misses),
        }
        out = _write_experiment_summary(summary, results_dir)
        print(f"cache warm summary: {out}")
        print(json.dumps(summary["warm_cache"], ensure_ascii=False, indent=2))
        return 0

    if args.max_cache_misses is not None:
        raise SystemExit("--max-cache-misses is only valid with --warm-cache-only.")

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
