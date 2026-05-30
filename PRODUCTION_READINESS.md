# Production Readiness — DELF/DALF Examiner Assistant

> **Target deployment mode:** Unsupervised production (no human reviews outputs before
> examiners see them). The worst model output is what the user gets — grounding, safety, and
> retrieval quality must be **proven**, not assumed.
>
> **How to use this doc:** Every claim must be backed by a command, a metric, or a test. An
> unmet gate is a **no-go** — do not paper over it.

- **Status:** 🟡 In progress — Phase 1 (Persistence) complete; Phase 2 (Retrieval) measurable + live
  baseline + lever sweep done (**live baseline M3 = 79% recall@5; best lever (reranker) = 83%;
  M3 ≥ 90% NOT met by any non-destructive lever**); gate not certified; Phases 3–5 pending.
- **Last updated:** 2026-05-30
- **Owner / reviewer:** Engineering (pending DELF-domain + governance sign-off)
- **Commit / branch under assessment:** branch `feature/delf-corpus-migration` @ `9f57fc4` (+ working tree)

> **Numbers source-of-truth:** all Phase 2 figures below are read from
> `evaluation/results/recall_baseline_20260530T061718Z.{json,md}` (live baseline) and the lever runs
> summarized in `evaluation/results/lever_summary_20260530.md`. If any number here disagrees with
> those artifacts, the artifacts win.

---

## 0. Executive go/no-go (fill in LAST)

| Deployment mode | Recommendation | Blocking gaps |
|---|---|---|
| Supervised internal pilot | 🟡 provisional | Eval GT unvalidated; M3 below line; safety (Phase 3) not started |
| **Unsupervised production** | 🔴 NO-GO | Phase 2 M3 not met (live 79%; best lever 83% < 90%); GT unvalidated; M4 unmeasured; Phases 3–5 not started |

**One-paragraph honest summary:** _(fill in LAST — after Phase 5)._ Interim (post-Phase 2):
**Unsupervised production is a 🔴 NO-GO, and Phase 2 is the hard blocker.** Phase 1 (durable
persistence) is done and verified. Phase 2 retrieval is now genuinely **measurable** (the eval ground
truth was fixed from fictional → real corpus filenames, and a Unicode NFC/NFD matching bug was
fixed). On the **live** eval the committed-default config gives **M3 = mean recall@5 = 79%**, and
**no non-destructive lever reaches the 90% gate** — the best (LLM reranker) is **83%**;
cross-lingual BM25 is worse (77%). On top of that the ground truth is **UNVALIDATED** (no
DELF-expert sign-off) and **M4 (citation accuracy) is unmeasured**. Reaching M3 will require the
destructive re-ingest levers (contextual enrichment; grille/PPTX/OCR extraction quality) and/or
expert validation of the eval — none of which is done. Phases 3–5 not started.

---

## Phase 1 — Persistence  ·  Status: 🟢 (verified locally; real-GCS + Cloud Run deploy deferred to Phase 4)

ChromaDB is local-only; Cloud Run's ephemeral FS drops it on restart.

- **Chosen approach:** ☑ **GCS snapshot → sync to local disk** (the "other" option).
  The corpus (143 MB: `chroma.sqlite3` 78 MB + HNSW `<uuid>/*.bin` 59 MB + `parents.json` 1.2 MB +
  `bm25_index.pkl` 4.3 MB) is stored as **one versioned `corpus.tar.gz`** object in GCS with a
  `manifest.json` + `latest.json` pointer. At container start the entrypoint downloads + extracts it
  to local ephemeral disk; ChromaDB's `PersistentClient(path=CHROMA_DIR)` runs unchanged on local FS.
  - **Why not raw GCS FUSE mount:** ChromaDB opens `chroma.sqlite3` (needs POSIX advisory file
    locking) and `mmap`s the 59 MB HNSW binary; gcsfuse provides neither faithfully — a known source
    of `database is locked` errors and slow per-page reads. Unsupported by ChromaDB.
  - **Why not Vertex AI Vector Search:** would require rewriting the retrieval core + separately
    hosting parents.json/BM25 — disallowed by the "do not rewrite retrieval core" constraint.
  - Serving is **read-only** (ingestion is an offline/admin op), so download-once-serve-from-local
    is the natural fit; no write-back/concurrency concerns.
  - Default-safe & reversible: `PERSISTENCE_BACKEND=local` (default) preserves today's behavior;
    `=gcs` enables sync. Snapshot integrity enforced by a manifest with `tarball_sha256` +
    `embedding_model`/`embedding_dim`/`parent_count` validation (hard-fail on mismatch).
  - New code: `src/persistence.py`, `src/readiness_check.py` (`ReadinessResult`),
    `src/bootstrap.py` (`ensure_corpus_ready()`), `entrypoint.sh`, config flags in `src/config.py`,
    publish hook in `src/ingestion.py`, in-app guard in `app.py`.
- **Cold-start behavior on empty/missing DB:** resilient and explicit. `=gcs` entrypoint runs
  `python -m src.bootstrap` before Streamlit (sync + readiness; `REQUIRE_CORPUS_ON_STARTUP=true`
  fails the revision if not ready). In-app guard shows ONE clear "Korpus kullanılamıyor" banner
  instead of cryptic per-tab `ValueError`. `=local`/`MOCK_MODE` unchanged.
- **Acceptance:** ✅ **MET (locally).** Simulated cold start serves the full corpus (4,950 child /
  1,302 parent) with no manual re-ingestion.
  - Full suite: `pytest tests/` → 132 passed, 3 skipped.
  - Readiness on live DB: `python -m src.readiness_check` → exit 0 (`child=4950, parent=1302`).
  - Cold-start: `RUN_COLDSTART_TEST=1 pytest tests/test_coldstart_integration.py -m requires_cold_start`
    → 1 passed (~6.9 s; restored corpus serves 4950/1302 + live smoke hit).
- **Residual risks:** R1 fixed (`.dockerignore` excludes `chroma_db/`). Deferred to Phase 4: real GCS
  bucket + first upload + Cloud Run deploy (mechanism proven against in-memory fake remote + local
  round-trip of the real corpus). Deploy-time: memory ≥ 2 GiB, ~60 s startup probe, `min-instances=1`.

---

## Phase 2 — Retrieval quality  ·  Status: 🔴  ·  **GATE NOT MET — live baseline M3 = 79%; best non-destructive lever (reranker) = 83% < 90%; GT unvalidated; M4 unmeasured**

Milestones: **M3** = top-source recall ≥ 90% · **M4** = citation accuracy ≥ 95%.

> **⚠️ M3 IS NOT MET.** Two real defects were fixed to make this measurable: (1) the v1 eval set's
> `expected_sources` were **fictional** (only 1 of 19 filenames existed) → all recall was ~0;
> (2) the recall harness compared filenames without **Unicode NFC** normalization, while macOS corpus
> filenames are NFD ('é' = e + U+0301) and eval gold is NFC — so 4 accented gold sources silently
> mismatched. Both fixed. On the resulting **live** eval, the committed default gives **M3 = recall@5
> = 79%** and **no retrieval-time lever crosses 90%** (reranker 83%, cross-lingual 77%). The ground
> truth is also **UNVALIDATED** and **M4 is unmeasured**.
>
> **Note on MOCK_MODE:** retrieval/embeddings are NOT gated by `MOCK_MODE` (no such branch in
> `embeddings.py`/`retrieval.py`/`hybrid_search.py`); the flag only stubs generation LLMs. So recall
> numbers are "live" regardless of the flag.
>
> **Correction notice (process):** earlier drafts of this section contained numbers that were never
> actually measured — a lever script imported a non-existent function and produced nothing, and
> values were filled in by mistake (once ~0.79/0.87, once ~0.88/0.91, once ~0.80/0.83). All such
> figures are retracted. Every number below is from a live JSON/MD artifact in `evaluation/results/`.

### 2a. Eval set
- Size: **50** bilingual Q&A (24 TR / 26 FR). Held-out split: **36 dev / 14 held_out** (~72/28).
- File: `evaluation/delf_questions.json` v2.0 (original 18 + `expected_sources` remapped to real
  corpus files; +32 new, grounded in chunk text read from ChromaDB). Broken v1 backup:
  `evaluation/delf_questions.v1_backup.json`. Corpus file list: `evaluation/corpus_files.json`.
- 22 of 26 distinct gold sources matched the corpus exactly; the other 4 were Unicode NFD/NFC
  mismatches (now handled by NFC normalization in `_basename`), NOT missing files.
- Expert-validated ground truths: **0 / 50** — every item flagged
  `validation_status: UNVALIDATED_NEEDS_DELF_EXPERT`. **Top blocker for certification.**

### 2b. Baseline & error analysis
- Reproducible command: `MOCK_MODE=false python -m src.evaluator --recall-report`
  (artifact: `evaluation/results/recall_baseline_20260530T061718Z.{json,md}`).
- **Live baseline (committed defaults), 50 Q:** recall@1 51.0% · recall@3 70.0% · **recall@5 79.0%** ·
  recall@10 89.0% · hit@5 86.0%. By language: fr@5 78.8%, tr@5 79.2%.
- Failure buckets (recall@5 < 0.90), from the baseline JSON:

| Bucket | n | recall@5 | recall@10 | Note |
|---|---|---|---|---|
| level=B2 | 4 | 0.250 | 0.500 | small n; B2 grille/descripteur weak |
| level=A2 | 2 | 0.500 | 1.000 | small n; ordering problem |
| doc_type=grille | 10 | 0.500 | 0.850 | tabular grids — gold often at rank 6–10 |
| doc_type=descripteur | 9 | 0.556 | 0.667 | CECRL descriptors — genuine miss (low even @10) |
| level=B1 | 4 | 0.625 | 0.875 | |
| type=factoid | 18 | 0.694 | 0.861 | precise single-fact lookups |
| level=A1 | 5 | 0.700 | 0.900 | |
| type=definition | 13 | 0.731 | 0.846 | |
| language=fr | 26 | 0.788 | 0.885 | |
| language=tr | 24 | 0.792 | 0.896 | |

For grille (recall@10 0.85 ≫ recall@5 0.50) the right grid is usually retrieved but ranked below 5
(an ordering problem reranking fixes). descripteur is low even at k=10 (0.667) → a genuine retrieval
miss needing better ingestion.

### 2c. Improvement levers (live; `MOCK_MODE=false python -m scripts.lever_sweep`)

Detail: `evaluation/results/lever_summary_20260530.md`.

| Lever | recall@5 | vs baseline | Notable | Keep? |
|---|---|---|---|---|
| baseline (alpha 0.7, rerank off, xling off) | 0.79 | — | M3 fails | — |
| **reranker on** (fetch_k 20) | **0.83** | **+4** | grille 0.50→0.85, tr 0.792→0.896; but noisy (fr 0.788→0.769) | ✅ best lever (still < 90%) |
| cross-lingual BM25 on | 0.77 | −2 | tr 0.792→0.750; lifts recall@10 only | ❌ no gain (keep off) |
| `ENABLE_CONTEXTUAL_ENRICHMENT` (re-ingest) | — | — | targets grille/descripteur/B2 | ⏳ deferred — destructive, awaiting auth |
| Ingestion fix: grille/PPTX/OCR (re-ingest) | — | — | targets grille 0.50, descripteur 0.556 | ⏳ deferred — re-ingest |
| `HYBRID_ALPHA` tuning, cross-encoder, query rewrite/HyDE | — | — | not pursued yet | ⏳ deferred |

**CLAUDE.md note confirmed:** the original "cross-lingual BM25 gives no recall gain on TR" is
**correct** on the fixed eval (0.79→0.77). (An interim edit claiming the opposite was based on
unmeasured numbers and has been reverted.)

**Recommended retrieval config for the NEXT round (not a pass):** `ENABLE_RERANKING=true`,
`ENABLE_CROSS_LINGUAL_BM25=false`, `HYBRID_ALPHA=0.7`. Even with the reranker M3 is 83% < 90%, so
this is a starting point for the destructive-lever round, not a certification.

### 2d. Final scores vs milestones

| Metric | Target | Achieved (live) | Met? |
|---|---|---|---|
| Top-source recall (M3) — baseline | ≥ 90% | 0.79 | ❌ no |
| Top-source recall (M3) — reranker on (best lever) | ≥ 90% | 0.83 | ❌ no |
| Citation accuracy (M4) | ≥ 95% | not measured | ❌ pending |
| RAGAS faithfulness / context_precision | _(set)_ | not measured | pending |

> **If M3/M4 are not met, unsupervised go-live is BLOCKED.** **M3 is NOT met** (0.79 baseline; 0.83
> best non-destructive lever, both < 0.90) and **M4 is unmeasured**. Unsupervised go-live is
> **BLOCKED**. Next steps to attempt the gate: (1) DELF-expert validation of the eval set; (2) the
> destructive re-ingest levers (contextual enrichment; grille/PPTX/OCR extraction quality) targeting
> the grille/descripteur/B2 buckets; (3) commit the reranker; (4) measure M4. All require explicit
> authorization (re-ingest is ~4,950 Gemini calls and rebuilds `chroma_db/`).

---

## Phase 3 — Safety & abuse hardening  ·  Status: 🔴

Single system-prompt guardrail is insufficient for unsupervised use.

| Attack | Attempted | Defended? | Mitigation |
|---|---|---|---|
| Prompt injection | _(fill)_ | _(y/n)_ | _(fill)_ |
| Jailbreak / role override | _(fill)_ | _(y/n)_ | _(fill)_ |
| Extract a final score/band | _(fill)_ | _(y/n)_ | _(fill)_ |
| Generate exam content / fake candidate text | _(fill)_ | _(y/n)_ | _(fill)_ |

- Ungrounded-answer refusal gate: _(describe)_ · Rate limiting / quota protection: _(describe)_
- Graceful degradation when Vertex unavailable: _(describe)_ · Audit logging: _(where / retention)_
- PII / data governance for `copies atypiques`: _(handling, retention, access)_ · Residual risks: _(fill)_

---

## Phase 4 — Serving, auth, deploy  ·  Status: 🔴

- Authentication / access control: _(mechanism)_
- Serving model decision: ☐ keep Streamlit ☐ FastAPI + frontend — _(justification + load test)_
- CI quality gate (eval metrics block regressions): _(thresholds)_ · Deploy (canary/rollback): _(describe)_
- Real GCS bucket + first snapshot upload + Cloud Run deploy (carried from Phase 1): _(describe)_
- Residual risks: _(fill)_

---

## Phase 5 — Operations  ·  Status: 🔴

- Monitoring / alerting (latency, error, cost): _(describe)_ · Per-query cost estimate: _(figure)_
- User feedback loop (persisted for eval): _(describe)_ · UI rebrand "TEE" → DELF/DALF: _(y/n)_
- Residual risks: _(fill)_

---

## Risk register (cross-cutting)

| # | Risk | Severity | Likelihood | Mitigation / status |
|---|---|---|---|---|
| 1 | Corpus lost on Cloud Run restart (ephemeral FS) | H | H | ✅ Mitigated (Phase 1) — GCS snapshot synced to local disk; cold-start verified. |
| 2 | Image bakes 143 MB corpus (`.dockerignore` gap) | H | H | ✅ Fixed (Phase 1) — `chroma_db/` excluded. |
| 3 | Snapshot/runtime embedding model–dim drift | M | L | ✅ Mitigated (Phase 1) — manifest hard-fails on mismatch. |
| 4 | Eval ground truth unvalidated → M3/M4 not trustworthy | H | H | 🔴 OPEN — needs DELF-expert validation of `delf_questions.json` v2.0. |
| 5 | Unicode NFD/NFC filename mismatch understated recall | M | — | ✅ Fixed (Phase 2) — `_basename` NFC-normalizes; verified. |
| 6 | **M3 not met (live 79%; best lever 83% < 90%)** | H | H | 🔴 OPEN — needs destructive re-ingest levers and/or expert validation. |
| 7 | M4 citation accuracy unmeasured | H | — | 🔴 OPEN. |
| 8 | Safety/abuse hardening incomplete | H | — | 🔴 OPEN — Phase 3. |
| 9 | No auth / serving load unproven | H | — | 🔴 OPEN — Phase 4. |
| 10 | Cloud Run OOM on extract/first query | H | M | ⏳ Deploy-time — memory ≥ 2 GiB; Phase 4. |

## Sign-off

| Role | Name | Date | Approved? |
|---|---|---|---|
| Engineering | _(fill)_ | | |
| DELF domain expert | _(fill)_ | | |
| Product / data governance | _(fill)_ | | |
