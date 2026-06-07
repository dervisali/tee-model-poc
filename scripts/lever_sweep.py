"""
Phase 2.3 — non-destructive retrieval lever sweep (REAL API).

Runs the deterministic recall@k report (src.evaluator.run_recall_report) under
retrieval-TIME configs that require NO re-ingestion:
  - baseline                (committed defaults)
  - rerank_on               (ENABLE_RERANKING=true; over-fetch RERANK_FETCH_K → top_k)
  - rerank_auto_metadata    (rerank + guarded PO-only auto metadata filtering)
  - rerank_auto_metadata_source_hints
                            (rerank + guarded metadata + explicit source hints)
  - full_recall_mode_no_rerank
                            (guarded metadata + source hints + source diversification)
  - full_recall_mode        (rerank + guarded metadata + source hints + source diversification)
  - cross_lingual_bm25_on   (TR→FR translation + hybrid BM25)
  - alpha_0.5               (more BM25 weight in fusion)

Writes ONE artifact incrementally (evaluation/results/lever_sweep_<UTC>.json) so a
mid-run failure keeps partial results. The summary headlines M3 at the configured
gate k (currently recall@3) while keeping recall@5 for comparison with older
Phase 2 notes. Reproduce: `MOCK_MODE=false python -m scripts.lever_sweep`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE_CHROMA_DIR = REPO / "chroma_db"
DEFAULT_SWEEP_CHROMA_DIR = REPO / "chroma_db_sweep"

sys.path.insert(0, str(REPO))

K_VALUES = [1, 3, 5, 10]
RESULTS_DIR = REPO / "evaluation" / "results"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _refuse_baseline_target(chroma_dir: Path) -> None:
    resolved = chroma_dir.resolve()
    baseline = BASELINE_CHROMA_DIR.resolve()
    if resolved == baseline or resolved.name == "chroma_db":
        raise SystemExit(f"Refusing to run lever sweep against baseline CHROMA_DIR: {resolved}")
    if _is_relative_to(resolved, baseline):
        raise SystemExit(f"Refusing to run lever sweep inside baseline CHROMA_DIR: {resolved}")
    if chroma_dir.is_symlink():
        raise SystemExit(f"Refusing symlinked lever-sweep CHROMA_DIR: {chroma_dir}")


def _prepare_sweep_copy(chroma_dir: Path, *, reuse_copy: bool) -> dict:
    """
    Materialize a disposable copy of the baseline DB for retrieval-time sweeps.

    Chroma may update its SQLite file when opened, even for read-heavy workflows.
    Running against a copy keeps the production baseline byte fingerprint stable.
    """
    _refuse_baseline_target(chroma_dir)
    if not BASELINE_CHROMA_DIR.exists():
        raise SystemExit(f"Baseline CHROMA_DIR missing: {BASELINE_CHROMA_DIR}")

    resolved = chroma_dir.resolve()
    if resolved.exists() and not reuse_copy:
        if not resolved.is_dir():
            raise SystemExit(f"Lever-sweep CHROMA_DIR is not a directory: {resolved}")
        shutil.rmtree(resolved)
    if not resolved.exists():
        shutil.copytree(BASELINE_CHROMA_DIR, resolved)

    return {
        "baseline_chroma_dir": str(BASELINE_CHROMA_DIR.resolve()),
        "sweep_chroma_dir": str(resolved),
        "reuse_copy": reuse_copy,
    }


def _summarize(report: dict) -> dict:
    agg = report["aggregate"]
    o = agg["overall"]
    md = report["metadata"]
    m3_k = md["m3_k"]
    byl = agg["buckets"].get("language", {})
    bydt = agg["buckets"].get("doc_type", {})
    bylv = agg["buckets"].get("level", {})

    def g(d, key, k):
        return round(d.get(key, {}).get(f"recall@{k}", 0.0), 4) if key in d else None

    return {
        "m3_k": m3_k,
        "m3_recall": round(md["m3_recall_at_k"], 4),
        "m3_hit": round(md["m3_hit_at_k"], 4),
        "m3_passed": bool(md["m3_recall_at_k"] is not None and md["m3_recall_at_k"] >= 0.90),
        "recall@3": round(o["recall@3"], 4),
        "recall@5": round(o["recall@5"], 4),
        "recall@10": round(o["recall@10"], 4),
        "hit@3": round(o["hit@3"], 4),
        "hit@5": round(o["hit@5"], 4),
        "tr@3": g(byl, "tr", 3),
        "fr@3": g(byl, "fr", 3),
        "tr@5": g(byl, "tr", 5),
        "fr@5": g(byl, "fr", 5),
        "grille@3": g(bydt, "grille", 3),
        "grille@5": g(bydt, "grille", 5),
        "descripteur@3": g(bydt, "descripteur", 3),
        "B2@3": g(bylv, "B2", 3),
        "B2@5": g(bylv, "B2", 5),
    }


def _config_specs():
    from src.config import settings
    from src.retrieval import retrieve_context

    def _set(attr, val):
        old = getattr(settings, attr)
        setattr(settings, attr, val)
        return lambda: setattr(settings, attr, old)

    def _noop():
        return lambda: None

    def _chain(*setups):
        restores = [setup() for setup in setups]
        return lambda: [restore() for restore in reversed(restores)]

    return [
        ("baseline", _noop, None),
        ("rerank_on", lambda: _set("ENABLE_RERANKING", True), None),
        (
            "rerank_auto_metadata",
            lambda: _chain(
                lambda: _set("ENABLE_RERANKING", True),
                lambda: _set("ENABLE_AUTO_METADATA_FILTER", True),
            ),
            None,
        ),
        (
            "rerank_auto_metadata_source_hints",
            lambda: _chain(
                lambda: _set("ENABLE_RERANKING", True),
                lambda: _set("ENABLE_AUTO_METADATA_FILTER", True),
                lambda: _set("ENABLE_SOURCE_HINTS", True),
            ),
            None,
        ),
        (
            "full_recall_mode_no_rerank",
            lambda: _chain(
                lambda: _set("ENABLE_RERANKING", False),
                lambda: _set("ENABLE_AUTO_METADATA_FILTER", True),
                lambda: _set("ENABLE_SOURCE_HINTS", True),
                lambda: _set("ENABLE_SOURCE_DIVERSIFICATION", True),
            ),
            None,
        ),
        (
            "full_recall_mode",
            lambda: _chain(
                lambda: _set("ENABLE_RERANKING", True),
                lambda: _set("ENABLE_AUTO_METADATA_FILTER", True),
                lambda: _set("ENABLE_SOURCE_HINTS", True),
                lambda: _set("ENABLE_SOURCE_DIVERSIFICATION", True),
            ),
            None,
        ),
        ("cross_lingual_bm25_on", lambda: _set("ENABLE_CROSS_LINGUAL_BM25", True), None),
        ("alpha_0.5", _noop, lambda q, top_k: retrieve_context(q, top_k=top_k, alpha=0.5)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_SWEEP_CHROMA_DIR,
        help="Disposable ChromaDB copy used for the sweep; baseline chroma_db is refused.",
    )
    ap.add_argument(
        "--reuse-copy",
        action="store_true",
        help="Reuse --chroma-dir instead of refreshing it from baseline before the sweep.",
    )
    args = ap.parse_args()

    from scripts.run_enriched_retrieval_experiment import _baseline_integrity, _directory_fingerprint

    baseline_before = _directory_fingerprint(BASELINE_CHROMA_DIR)
    sweep_target = _prepare_sweep_copy(args.chroma_dir, reuse_copy=args.reuse_copy)
    os.environ["CHROMA_DIR"] = sweep_target["sweep_chroma_dir"]
    os.environ.setdefault("DATA_DIR", str(REPO / "new_docs"))

    from src.config import settings
    from src.evaluator import load_test_set, run_recall_report
    from src.retrieval import retrieve_context

    if settings.MOCK_MODE:
        print("REFUSING: MOCK_MODE=true gives fixture retrieval, not a real sweep.")
        return 2

    try:
        if not retrieve_context("B2 PE", top_k=3):
            print("AUTH/RETRIEVAL probe returned no hits — aborting.")
            return 3
    except Exception as exc:  # noqa: BLE001
        print(f"AUTH/RETRIEVAL probe failed: {exc}")
        return 3

    test_set = load_test_set()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = RESULTS_DIR / f"lever_sweep_{ts}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = _config_specs()
    if args.only:
        configs = [c for c in configs if c[0] == args.only]
        if not configs:
            print(f"Unknown config: {args.only}")
            return 4

    results = {
        "metadata": {
            "generated_utc": ts, "n_questions": len(test_set), "ks": K_VALUES,
            "baseline_alpha": settings.HYBRID_ALPHA, "rrf_k": settings.HYBRID_RRF_K,
            "top_k": settings.RETRIEVAL_TOP_K, "rerank_fetch_k": settings.RERANK_FETCH_K,
            "mock_mode": settings.MOCK_MODE,
            "sweep_target": sweep_target,
            "baseline_integrity": {
                "before": baseline_before,
                "after": None,
                "ok": None,
            },
        },
        "configs": {},
    }

    for name, setup, fn in configs:
        restore = setup()
        try:
            print(f"running {name} ...", flush=True)
            rep = run_recall_report(test_questions=test_set, ks=K_VALUES, retrieve_fn=fn, save=False)
            results["configs"][name] = _summarize(rep)
        except Exception as exc:  # noqa: BLE001
            results["configs"][name] = {"error": str(exc)}
            print(f"  {name} FAILED: {exc}")
        finally:
            restore()
        baseline_after = _directory_fingerprint(BASELINE_CHROMA_DIR)
        results["metadata"]["baseline_integrity"] = _baseline_integrity(
            baseline_before,
            baseline_after,
        )
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_after = _directory_fingerprint(BASELINE_CHROMA_DIR)
    results["metadata"]["baseline_integrity"] = _baseline_integrity(baseline_before, baseline_after)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nartifact: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
