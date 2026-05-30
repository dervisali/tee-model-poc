"""Regenerate error_analysis_<UTC>.md from the latest recall_baseline artifact.

Driven entirely by the artifact so the numbers always match the last live run
(including the post-NFC-normalization-fix baseline). Run:
    python evaluation/_make_error_analysis.py
"""
import glob
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
RES = BASE / "results"

artifact = sorted(RES.glob("recall_baseline_*.json"))[-1]
r = json.loads(artifact.read_text(encoding="utf-8"))
qs = {q["id"]: q for q in json.loads((BASE / "delf_questions.json").read_text(encoding="utf-8"))["questions"]}
md = r["metadata"]
agg = r["aggregate"]
pq = r["per_question"]
ks = md["ks"]
stamp = artifact.stem.replace("recall_baseline_", "")


def pct(v):
    return f"{v * 100:.1f}%" if v is not None else "—"


def bucket_table(dim):
    rows = []
    for val, a in agg["buckets"][dim].items():
        if a["n"] == 0:
            continue
        rows.append((val, a))
    rows.sort(key=lambda x: (x[1].get("recall@5") if x[1].get("recall@5") is not None else 9))
    out = ["| value | n | recall@5 | hit@5 | recall@10 |", "|---|---|---|---|---|"]
    for val, a in rows:
        out.append(f"| {val} | {a['n']} | {pct(a.get('recall@5'))} | {pct(a.get('hit@5'))} | {pct(a.get('recall@10'))} |")
    return "\n".join(out)


zero5 = [q for q in pq if q["hit@5"] == 0.0]
lines = []
lines.append("# Phase 2 — Retrieval recall error analysis (PROVISIONAL)")
lines.append("")
lines.append("> **PROVISIONAL & UNVALIDATED.** `expected_sources` were auto-remapped to real corpus")
lines.append("> files from sampled chunk content; NOT validated by a DELF expert. Filenames are")
lines.append("> compared NFC-normalized (corpus stores some names NFD on macOS).")
lines.append("")
lines.append(f"- Source artifact: `evaluation/results/{artifact.name}`")
lines.append("- Reproduce: `MOCK_MODE=false python -m src.evaluator --recall-report`")
lines.append(f"- Eval set: `evaluation/delf_questions.json` v2.0 — {md['n_questions_evaluated']} answerable bilingual Q&A")
lines.append(f"- Config: search_mode={md['search_mode']}, RETRIEVAL_TOP_K={md['retrieval_top_k']}, "
             f"embedding={md['embedding_model']}, MOCK_MODE={md['mock_mode']}")
lines.append("")
lines.append("## Headline")
lines.append("")
o = agg["overall"]
lines.append("| Metric | " + " | ".join(f"@{k}" for k in ks) + " |")
lines.append("|---|" + "|".join("---" for _ in ks) + "|")
lines.append("| mean source-recall | " + " | ".join(pct(o.get(f"recall@{k}")) for k in ks) + " |")
lines.append("| hit@k | " + " | ".join(pct(o.get(f"hit@{k}")) for k in ks) + " |")
lines.append("")
lines.append(f"**M3 = mean recall@{md['m3_k']} = {pct(md['m3_recall_at_k'])}** "
             f"(hit@{md['m3_k']} = {pct(md['m3_hit_at_k'])}); target ≥ 90% → "
             f"{'MET' if (md['m3_recall_at_k'] or 0) >= 0.9 else 'NOT MET'}.")
lines.append("")
lines.append("## Failure buckets (worst recall@5 first)")
for dim in ("language", "doc_type", "level", "type", "split"):
    lines.append("")
    lines.append(f"### By {dim}")
    lines.append("")
    lines.append(bucket_table(dim))
lines.append("")
lines.append("## Zero-hit@5 misses (gold not in top-5)")
lines.append("")
if zero5:
    lines.append("| id | lang | doc_type | level | gold | top-3 retrieved | recall@10 |")
    lines.append("|---|---|---|---|---|---|---|")
    for q in zero5:
        gold = ", ".join(q["gold"])
        top3 = ", ".join(q["retrieved"][:3]) if q["retrieved"] else "(none)"
        lines.append(f"| {q['id']} | {q['language']} | {q['doc_type']} | {q['level']} | {gold} | {top3} | {pct(q['recall@10'])} |")
else:
    lines.append("(none — every question has ≥1 gold source in top-5)")
lines.append("")
lines.append("## Hypotheses for the 2c lever sweep (NOT yet applied)")
lines.append("")
lines.append("1. **Two generalist docs dominate.** `Inventaire_ONLINE_full.pdf` (2,109 children) and")
lines.append("   `manuel-exacor.pdf` (367) are strong attractors and crowd out small format-specific")
lines.append("   files (`.pptx`/`.docx`, 13–48 chunks) even when those are the precise source. Levers:")
lines.append("   metadata pre-filter by doc_type (already in `retrieve_context`), reranking, per-source caps.")
lines.append("2. **Cross-lingual (TR) recall lags FR.** See the by-language table; revisit")
lines.append("   `ENABLE_CROSS_LINGUAL_BM25` / query translation on this corrected set.")
lines.append("3. **Ranking vs coverage.** recall@10 > recall@5 → several golds sit at rank 6–10; an LLM")
lines.append("   reranker over a wider fetch (`ENABLE_RERANKING=true`, `RERANK_FETCH_K=20`) targets the @5 gate")
lines.append("   without re-ingestion.")
lines.append("4. **Transient embedding failures.** A query can occasionally return 0 hits on one run and the")
lines.append("   correct doc at rank 1 on a re-run; the harness scores empties as 0 (no crash), so the")
lines.append("   reported floor slightly understates true recall. Retry-on-empty would stabilize the metric.")
lines.append("")

out_path = RES / f"error_analysis_{stamp}.md"
out_path.write_text("\n".join(lines), encoding="utf-8")

# tiny numbers file for doc updates
nums = {
    "artifact": artifact.name,
    "stamp": stamp,
    "n": md["n_questions_evaluated"],
    "m3_k": md["m3_k"],
    "recall": {f"@{k}": round(o.get(f"recall@{k}"), 4) for k in ks},
    "hit": {f"@{k}": round(o.get(f"hit@{k}"), 4) for k in ks},
    "by_language": {v: {"n": a["n"], "recall@5": round(a.get("recall@5") or 0, 4), "hit@5": round(a.get("hit@5") or 0, 4)} for v, a in agg["buckets"]["language"].items()},
    "zero_hit5_ids": [q["id"] for q in zero5],
}
(RES / "_nums.json").write_text(json.dumps(nums, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out_path.name, "and _nums.json; M3 recall@%s = %.3f" % (md["m3_k"], md["m3_recall_at_k"]))
