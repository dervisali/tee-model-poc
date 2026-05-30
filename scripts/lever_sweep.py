"""
Phase 2.3 — non-destructive retrieval lever sweep (REAL API).

Runs the deterministic recall@k report (src.evaluator.run_recall_report) under
retrieval-TIME configs that require NO re-ingestion:
  - baseline                (committed defaults)
  - rerank_on               (ENABLE_RERANKING=true; over-fetch RERANK_FETCH_K → top_k)
  - cross_lingual_bm25_on   (TR→FR translation + hybrid BM25)
  - alpha_0.5               (more BM25 weight in fusion)

Writes ONE artifact incrementally (evaluation/results/lever_sweep_<UTC>.json) so a
mid-run failure keeps partial results. Read it back with the Read tool — do not rely
on stdout. Reproduce: `MOCK_MODE=false python -m scripts.lever_sweep`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.evaluator import load_test_set, run_recall_report  # noqa: E402
from src.retrieval import retrieve_context  # noqa: E402

K_VALUES = [1, 3, 5, 10]
RESULTS_DIR = settings.BASE_DIR / "evaluation" / "results"


def _summarize(report: dict) -> dict:
    agg = report["aggregate"]
    o = agg["overall"]
    byl = agg["buckets"].get("language", {})
    bydt = agg["buckets"].get("doc_type", {})
    bylv = agg["buckets"].get("level", {})

    def g(d, key, k):
        return round(d.get(key, {}).get(f"recall@{k}", 0.0), 4) if key in d else None

    return {
        "recall@5": round(o["recall@5"], 4),
        "recall@10": round(o["recall@10"], 4),
        "hit@5": round(o["hit@5"], 4),
        "tr@5": g(byl, "tr", 5),
        "fr@5": g(byl, "fr", 5),
        "grille@5": g(bydt, "grille", 5),
        "B2@5": g(bylv, "B2", 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

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

    def _set(attr, val):
        old = getattr(settings, attr)
        setattr(settings, attr, val)
        return lambda: setattr(settings, attr, old)

    def _noop():
        return lambda: None

    configs = [
        ("baseline", _noop, None),
        ("rerank_on", lambda: _set("ENABLE_RERANKING", True), None),
        ("cross_lingual_bm25_on", lambda: _set("ENABLE_CROSS_LINGUAL_BM25", True), None),
        ("alpha_0.5", _noop, lambda q, top_k: retrieve_context(q, top_k=top_k, alpha=0.5)),
    ]
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
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nartifact: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
