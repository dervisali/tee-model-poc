# Session Handoff — DELF/DALF Corpus Migration

## Goal

Migrate the TEE-Model RAG system from its Turkish payroll corpus to a French
DELF/DALF examiner-corrector knowledge base (~56 files in `new_docs/`), so a
Turkish examiner can query in Turkish or French and generate training
materials (process map / error cards / glossary / simulation) in either
language. Work follows an 8-phase goal-driven plan with measurable gates.

## Current Status

Branch: `feature/delf-corpus-migration` (forked from `feature/cloud-vertex-rag-v2`).
All committed work is **pushed**; working tree clean except untracked data/artifacts.
PR #2 open: `feature/delf-corpus-migration` → `feature/cloud-vertex-rag-v2`.

Plan file (source of truth): `/Users/dervis/.claude/plans/refine-the-plan-and-silly-puzzle.md`

Phases **done** (committed + pushed):
- Phase 0 — metadata schema locked
- Phase 1 — multi-format loaders (gate M1 = 100% file coverage, 56/56)
- Phase 2 — path classifier + ingestion wiring (gate M2 = 100%, 4950/4950 chunks tagged in dry-run audit)
- Phase 3 — French BM25 tokenizer
- Phase 3.5 — cross-lingual query expansion
- Phase 8 — bilingual generators (TR/FR output toggle)

Phases **deferred** (not started):
- Phase 4 — bilingual examiner eval set (`evaluation/examiner_questions.json`)
- Phase 5 — retrieval eval harness (`scripts/eval_retrieval.py`)
- Phase 6 — metadata-filter query router (`src/metadata_filter.py`)
- Phase 7 — Cloud Storage FUSE persistence for Cloud Run cold-start

**Live-verified** (real Vertex AI, 2026-05-16): full 56-file corpus (4950 chunks)
ingested with zero 429 errors after Vertex AI quota was raised to 100k RPM.
Both bilingual output modes confirmed in the running app:
- TR output: process map generated in Turkish with DELF domain content
- FR output: process map generated in French ("Habilitation de l'examinateur-correcteur", etc.)

Gates M3 (top-3 source recall ≥90%) and M4 (citation accuracy ≥95%) are
**not formally measured** — they need the Phase 4 eval set.

**Previously blocked** (now resolved): full 56-file corpus ingestion was blocked
by Vertex AI quota on project `woven-operative-491610-u6`. Quota raised to
100k RPM via Cloud Console → IAM & Admin → Quotas.

## Key Decisions

- **New branch `feature/delf-corpus-migration`** off `feature/cloud-vertex-rag-v2`,
  not committed onto cloud-vertex directly. Reason: the DELF migration is a
  distinct concern; keeps the cloud-vertex PR reviewable on its own.
- **Goal-driven phased execution**: every phase has a measurable gate; do not
  advance past a failed gate. Rejected: building everything then testing.
- **Metadata schema locked once** in `src/chunk_metadata.py` before any loader
  code. Reason: every loader/classifier/ingestion writer must emit the same
  shape; avoids ad-hoc drift.
- **PDF OCR fallback** via `pdfplumber` page → `pypdfium2` render → `pytesseract`.
  Added after the smoke gate showed `niveaux_mots_cles.pdf` was image-only.
  Rejected: leaving image-only PDFs as 0-text (would fail M1).
- **Regex-first path classifier** (`src/document_classifier.py`), not LLM.
  ~95% of the corpus is classifiable from filename/folder tokens deterministically;
  LLM reserved as fallback (not yet needed). Trade-off accepted: a few
  edge-case files land in a low-confidence manual-review bucket.
- **Per-page chunking** done caller-side in `src/ingestion.py`; the chunker
  contract (`src/chunkers.py`, all take `str`) was left unchanged. Reason:
  page-precise citations without touching the chunker API.
- **French BM25**: élision split + ~85-word stop-word filter + Snowball French
  stemmer (`nltk`). Turkish tokenizer path preserved for backward compat.
  Rejected: spaCy lemmatization (too heavy, +500MB).
- **Cross-lingual retrieval (Phase 3.5)**: dense path uses the original query
  (multilingual `gemini-embedding-001` handles it natively, free); only the
  BM25 path gets a Gemini-Flash translation of the query. Rejected:
  translating the whole corpus to Turkish at ingest (~3000 extra LLM calls).
- **Bilingual generators (Phase 8)**: each generator takes `language:
  Literal["tr","fr"]`; prompts/fixtures exist in both languages; Pydantic
  schemas unchanged (field names stay Turkish — they are identifiers, not
  user-facing). User explicitly chose "both via toggle" over single-language.
- **`INFERENCE_BACKEND` setting** (`vertex` | `public_genai`): added so the
  same `google-genai` SDK code targets either Vertex AI (ADC) or the public
  Gemini API (api_key).
- **Switched to Vertex AI** for live tests (user chose it over public Gemini
  API). Quota raised to 100k RPM; full corpus now ingests cleanly.
- **`_sanitize_id` uses full filename** (not `.stem`) — fixed in `src/ingestion.py`
  to avoid `DuplicateIDError` when two files share the same base name but
  different extensions (e.g. `WhatsApp Image…docx` vs `…jpeg`).
- **Embedding sleep 0.5s** (vertex) after quota raise. Was 1.5s (hit ~40 RPM
  limit). Now 0.5s → ~120 calls/min, safely under 100k RPM.

## Files Touched

### Created
- `src/chunk_metadata.py` — Pydantic `ChunkMetadata` model + `Level`/`Skill`/`DocType`/`Language` type aliases. The locked metadata contract.
- `src/document_loaders.py` — `load_pdf` (+OCR fallback), `load_docx`, `load_pptx`, `load_image_ocr`, `iter_documents`; `PageContent`/`LoadResult` NamedTuples.
- `src/document_classifier.py` — `classify_path(path, root)` regex classifier; `classify_corpus` returns (high-confidence, manual-review).
- `src/query_translator.py` — `detect_query_language()` heuristic + `translate_query()` (Gemini Flash, `lru_cache`, MOCK_MODE short-circuit).
- `scripts/smoke_load_all.py` — Phase 1 M1 coverage gate.
- `scripts/verify_metadata_completeness.py` — Phase 2 M2 audit (loader→classifier→chunker dry run, no Vertex calls).
- `tests/test_query_translator.py` — 10 tests: language detection, translation cache, cross-lingual contract.
- `tests/test_generators_bilingual.py` — 7 tests: TR/FR fixture dispatch, prompt registry completeness.
- `evaluation/phase1_smoke.json` — captured M1 smoke output (56/56).

### Modified
- `src/config.py` — added `CORPUS_PRIMARY_LANGUAGE` (renamed from `CORPUS_LANGUAGE`), `QUERY_LANGUAGE_AUTO_DETECT`, `OUTPUT_LANGUAGE`, `INFERENCE_BACKEND`, `GOOGLE_API_KEY`. `EMBEDDING_DIMENSION` is `Literal[768,1536,3072]`.
- `src/ingestion.py` — replaced `_load_source_files` with `_load_documents`/`DocumentLoad`/`_sanitize_id`; per-page chunking; metadata dict carries `ChunkMetadata.model_dump()`; summary has `by_doc_type`/`by_level` + `explicit_chunks`/`tacit_chunks`=0 placeholders. **Fixed**: `_sanitize_id` now uses full `source_filename` (not `.stem`) to avoid duplicate IDs across files with same base name.
- `src/hybrid_search.py` — language-aware `_tokenize` (French + Turkish); `BM25Index.search` and `rebuild_bm25_index_from_collection` take `language`; `hybrid_search_children` takes `bm25_query` override.
- `src/retrieval.py` — `retrieve_context` detects query language; when ≠ `CORPUS_PRIMARY_LANGUAGE`, translates and passes `bm25_query`.
- `src/generators.py` — `_GROUNDING_TEMPLATES` (TR/FR), `_GENERATOR_PROMPTS` registry, `_MOCK_*_FR` fixtures, `_MOCK_FIXTURES_BY_LANG`; all 4 generators take `language` param; `_load_optimized_prompt` tries `{key}_{lang}` first.
- `src/llm.py` — `get_genai_client()` dispatches on `INFERENCE_BACKEND`; `get_vertex_client` kept as alias.
- `src/embeddings.py` — `embed_passages` sleeps 0.5s/batch (vertex, after quota raise to 100k RPM) / 4.5s (public_genai). Tenacity retry with `_is_retryable_exception` catches `ResourceExhausted`/429.
- `app.py` — sidebar radio "Çıktı Dili / Langue de sortie" → `st.session_state["output_language"]`; passed to all 4 generator calls.
- `tests/test_retrieval.py` — Turkish tokenizer tests pass `language="tr"`; 2 new French tokenizer tests.
- `requirements.txt` — added `pdfplumber`, `python-docx`, `python-pptx`, `pytesseract`, `Pillow`, `nltk`.

### Read-only for context
- `src/chunkers.py` — `paragraph_parent_chunks` etc., all take `str`. Not modified.
- `src/confidence.py` — Phase 2.2 confidence scorer. Used unchanged by generators.
- `/Users/dervis/.claude/plans/refine-the-plan-and-silly-puzzle.md` — the 8-phase plan.

## Important Snippets

Current `.env` (gitignored — do not commit; contains a live API key):
```
MOCK_MODE=false
INFERENCE_BACKEND=vertex
GOOGLE_CLOUD_PROJECT=woven-operative-491610-u6
GOOGLE_CLOUD_LOCATION=us-central1
DATA_DIR=new_docs
ENABLE_CONTEXTUAL_ENRICHMENT=false
GOOGLE_API_KEY=AIza...   # public-API fallback, ignored in vertex mode
```

`ChunkMetadata` fields (see `src/chunk_metadata.py`): `source_file`,
`source_filename`, `page` (int|None), `level`, `skill`, `doc_type`,
`language`, `is_authoritative`, `classifier_confidence`.

Commands:
```
MOCK_MODE=true python3 -m pytest tests/ -v          # 68 passed, 2 skipped
python3 -m scripts.smoke_load_all new_docs/         # M1 gate
python3 -m scripts.verify_metadata_completeness new_docs/   # M2 gate
python3 -m src.ingestion                            # real ingest (reads DATA_DIR)
python3 -m streamlit run app.py                     # start app (bare streamlit not on PATH)
```

Commit hashes on `origin/feature/delf-corpus-migration`:
`1c53bc4` fix duplicate IDs + restore 0.5s sleep · `8711391` quota-speed commit (has rpm_cap — superseded) · `27d4ed0` backend dispatch · `7d8444c` phases 3.5+8 · `26ed010` phases 0-3.

## Open Questions

- **M3 / M4 unmeasured.** Need the Phase 4 eval set (30-50 bilingual examiner
  questions, ~60% TR / 40% FR, with `expected_source_files` + `expected_pages`).
- **TR mock fixtures still payroll-themed.** `_MOCK_PROCESS_MAP` etc. in
  `src/generators.py` are the old payroll content; only the `_MOCK_*_FR`
  fixtures are DELF. Tests are shape-based so this passes, but TR MOCK_MODE
  output is off-domain. Decide whether to rewrite TR fixtures to DELF.
- **Should `new_docs/` be committed?** Currently untracked (treated as data).
  ~56 files, includes non-authoritative drafts/WhatsApp images.
- **`GOOGLE_API_KEY` in `.env` was printed to the terminal** during a prior
  session — rotate if it is not a throwaway key.

## Next Step

**Build the Phase 4 bilingual eval set** → `evaluation/examiner_questions.json`
(30-50 questions, ~60% TR / ~40% FR, each with `expected_source_files` +
`expected_pages`), then run `scripts/eval_retrieval.py` (Phase 5) to formally
measure M3 (top-3 source recall ≥90%) and M4 (citation accuracy ≥95%).

Follow-ups:
1. Fix the `filename`/`source` metadata-key bug in `src/retrieval.py` (see Gotchas).
2. Decide on TR mock fixture rewrite and the `app.py` payroll-string cleanup.
3. Phase 6 — metadata-filter query router (`src/metadata_filter.py`).

## Gotchas / Watch-outs

- **`DATA_DIR=new_docs` in `.env`** — full corpus active. Do not switch back
  to `new_docs_subset` unless re-testing quota behaviour.
- **`chroma_db/` holds the full 4950-chunk corpus** after the 2026-05-16
  ingest. `rm -rf chroma_db` before any re-ingest — `run_ingestion` upserts,
  it does not clear, so stale chunks would mix in.
- **Bug: `src/retrieval.py:153-154` and `:243-244`** read `meta.get("filename")`
  and `meta.get("source")`, but the DELF `ChunkMetadata` writes
  `source_filename` / `source_file` — there is no `filename`/`source` key.
  Result: citation label at `src/retrieval.py:292`
  (`[Kaynak N: {filename} — {parent_id}]`) displays `"bilinmiyor"`. Fix: map
  `source_filename`→`filename` or update the reads. Affects M4 citation display.
- **Vertex AI quota raised to 100k RPM** on project `woven-operative-491610-u6`
  via Cloud Console → IAM & Admin → Quotas. The old default was ~40 RPM which
  caused 429s on batch 41 of the full ingest.
- **`ENABLE_CONTEXTUAL_ENRICHMENT=false`** in `.env` — set deliberately to
  skip the per-chunk LLM enrichment call during cost-sensitive testing. Flip
  to `true` for a production-quality ingest (adds one Gemini call per chunk).
- **`app.py` still says "Maaş Mutemedi Onboarding"** in the caption and the DB
  Explorer tab (~line 879) still groups chunks by the legacy `source` field
  (`tacit`/`explicit`) — DELF chunks won't appear there. Cosmetic, deferred.
- **`get_vertex_client` is an alias** for `get_genai_client` in `src/llm.py` —
  the name lies (it may return a public-API client). Kept for import compat.
- **`src/evaluator.py` (RAGAS) still hard-wired to `langchain-google-vertexai`**
  — it will not work under `INFERENCE_BACKEND=public_genai`. Not exercised
  this session.
- **Streamlit launches via `python3 -m streamlit run app.py`** — the bare
  `streamlit` binary is not on PATH on this machine.
- **`initial_sidebar_state="collapsed"`** in `app.py` — the language toggle
  lives in a collapsed sidebar; expand it (">>" chevron, top-left) to reach
  the radio.
- **Generator output is non-deterministic** — temperature variance, not a bug.
- **System deps installed**: `tesseract` 5.5.2 + `tesseract-lang` (French) via
  Homebrew; Python pkgs via `python3 -m pip` (no venv).
