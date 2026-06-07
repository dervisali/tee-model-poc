# TEE-Model — DELF/DALF Examiner Training Assistant

RAG-based bilingual assistant for **DELF/DALF examiner-correctors in Turkey**.
Turkish-speaking examiners query a French-language official corpus (grilles, descripteurs, methodologie docs) and receive grounded answers in Turkish or French.

> **Domain note:** README.md and ARCHITECTURE.md describe the old payroll (maaş mutemedi) use case. They are stale. The current application is DELF/DALF examiner training. `system-overview.html` is the authoritative reference.

## Quick Start

```bash
# Auth (Vertex AI ADC)
gcloud auth application-default login
gcloud config set project woven-operative-491610-u6

# Install
pip install -r requirements.txt

# Run
python3 -m streamlit run app.py

# Test without API calls
MOCK_MODE=true python3 -m streamlit run app.py
```

## Re-ingestion

Required when: corpus files change, `EMBEDDING_DIMENSION` changes, or `ENABLE_CONTEXTUAL_ENRICHMENT` is toggled.

```bash
rm -rf chroma_db/
python -m src.ingestion
```

`DATA_DIR` defaults to `data/` in config but is set to `new_docs/` in `.env` — the live DELF corpus is in `new_docs/`.

**Never change `EMBEDDING_DIMENSION` without full re-ingestion.** Mixing dimension sizes corrupts ChromaDB.

## Key Files

| File | Role |
|---|---|
| `app.py` | Streamlit UI — 6 tabs, 1039 lines |
| `src/config.py` | All settings via Pydantic-Settings; read from `.env` |
| `src/retrieval.py` | `retrieve_context()` — hybrid BM25+dense, parent-doc fetch, RRF fusion |
| `src/generators.py` | `generate_process_map/error_cards/glossary/simulation()` |
| `src/ingestion.py` | Document loading, chunking, embedding, ChromaDB write |
| `src/chunk_metadata.py` | `ChunkMetadata` Pydantic schema for all chunk metadata fields |
| `src/query_translator.py` | Language detection + TR→FR translation for BM25 path |
| `src/hybrid_search.py` | BM25 index (French: Snowball+élision+stop-words; Turkish: basic norm) |
| `src/prompt_optimizer.py` | A/B prompt variants; LLM judge scores grounding/completeness/usefulness |
| `src/llm.py` | `llm_generate()` — wraps Vertex AI or public Gemini API |
| `src/chatbot.py` | `chat()` — Tab 7 corpus-grounded conversational assistant |

## Architecture Summary

```
User query
  → detect language → translate query (TR→FR, BM25 path only)
  → dense retrieval (gemini-embedding-001, ChromaDB tee_children)
  → BM25 retrieval (bm25_index.pkl)
  → RRF fusion (HYBRID_ALPHA=0.7 dense, 0.3 BM25, k=60)
  → parent fetch (child hit → tee_parents by parent_id, deduplicate)
  → build grounded system prompt
  → llm_generate(response_format=PydanticSchema)
```

**Process map uses multi-phase retrieval:** 3 targeted queries (preparation / individual correction / double correction+finalisation) × top_k=8, merged to top-20 unique parents. A single generic query only retrieves 1-2 phases.

## Corpus Stats (as of 2026-05-16)

- 56 source files (PDF, DOCX, PPTX, JPEG) in `new_docs/`
- 4 950 child chunks in `tee_children` ChromaDB collection
- 1 302 parent chunks in `tee_parents` ChromaDB collection
- Primary language: French (`CORPUS_PRIMARY_LANGUAGE=fr`)
- Embedding: `gemini-embedding-001`, 3072 dimensions, L2-normalised

## App Tabs (app.py)

| # | Tab name | Status |
|---|---|---|
| 0 | Veri Yükleme ve İşleme | Live |
| 1 | Süreç Haritası ve Hata Kartları | Live |
| 2 | İnteraktif Simülasyon | Live |
| 3 | Uzman Onay Paneli | Live |
| 4 | Prompt Optimizer | Live |
| 5 | Veritabanı Gezgini | Live (patched for DELF metadata at lines 909-911) |
| 6 | Asistan | Live — corpus-grounded chatbot (`src/chatbot.py`) |

## Feature Flags (runtime state in `.env`)

| Flag | config.py default | `.env` override | Reason |
|---|---|---|---|
| `ENABLE_CONTEXTUAL_ENRICHMENT` | `true` | **`false`** | Cost control — requires ~4 950 Gemini calls at ingest |
| `ENABLE_HYBRID_SEARCH` | `true` | — | Active |
| `ENABLE_CONFIDENCE_SCORING` | `true` | — | Active; second LLM pass per generation |
| `ENABLE_QUERY_REWRITING` | `false` | — | Off — slow, 3× LLM calls at retrieval time |
| `MOCK_MODE` | `false` | — | Off; set to `true` for UI testing without API |

## Known Issues

- **Milestones M3/M4 — MEASURED 2026-06-07, both clear their numeric gates (engineering-grade; mind the caveats).** Eval = `evaluation/delf_questions.json` (50 bilingual Q&A).
  - **M3** (top-3 recall ≥90%): **93%** (recall@5 = 0.95). ⚠️ ONLY with the source-recall levers now in `.env` (`ENABLE_AUTO_METADATA_FILTER`, `ENABLE_SOURCE_HINTS`, `ENABLE_SOURCE_DIVERSIFICATION`); committed defaults (levers off) ≈ 70%. Artifacts: `evaluation/results/recall_baseline_*.json`, `lever_sweep_*.json`.
  - **M4** (citation accuracy ≥95%): **96% (48/50)** after the d39/d03 safety fixes (commit `e217f42`); deterministic floor ~94%, ~1 pt is generation variance; d04/d40 still fail (model emits 0 citations despite correct retrieval). Artifact: `evaluation/results/citation_grounding_*.json`.
  - **⚠️ Eval is AI-validated, NOT expert-validated** (project decision). Engineering metrics, **not** a formal certification (`scripts.apply_expert_validation` rejects the AI-marked review; no `*.validated.json`/manifest). **Unsupervised-production = still 🔴 NO-GO** (`PRODUCTION_READINESS.md`).
  - **DB-read caution:** direct ChromaDB reads can mutate `chroma.sqlite3`; measure against a disposable copy and verify the live DB hash (see `scripts/lever_sweep.py`).
- **App header says "TEE"** — The main title still shows the old acronym; DELF rebrand of the header is pending.
- **ChromaDB is local only** — Cloud Run ephemeral filesystem drops the DB on restart. GCS FUSE mount for persistence not yet implemented.
- **`src/metadata_filter.py` not implemented** — Pre-filtering by level/skill/doc_type before vector search (Phase 6) is planned but missing.
- **`src/evaluator.py` hardcoded to langchain-google-vertexai** — breaks under `INFERENCE_BACKEND=public_genai`.

## Deferred / Not Yet Implemented

- Metadata pre-filtering (`src/metadata_filter.py`) — queries like "B2 PO grilles only"
- Cross-encoder reranking after hybrid search
- GCS FUSE mount for ChromaDB persistence on Cloud Run
- Bilingual evaluation set (50 Q&A pairs)
- User feedback loop (thumbs up/down on generated content)
- **Vertex context caching for chatbot/generators** — investigated, skipped: static system-prompt portions (~700-900 tokens for chatbot, ~150-250 for generators) are below Gemini 2.5 Flash's 1,024-token implicit cache threshold. Revisit if (a) static rules grow past 1,024 tokens by embedding examples/FAQ, or (b) migrating to Gemini Pro where per-token cost makes explicit caching worthwhile.

## Running Tests

```bash
pytest tests/
# Specific suites
pytest tests/test_generators_bilingual.py
pytest tests/test_retrieval.py
pytest tests/test_integration.py   # requires live Vertex AI auth
```

## Environment Variables

All in `src/config.py`. Key ones for `.env`:

| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | — | Required for Vertex AI |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | Vertex AI region. `global` is also valid and gives automatic capacity routing; kept regional for predictable quota/data locality. |
| `INFERENCE_BACKEND` | `vertex` | `vertex` (ADC) or `public_genai` (API key) |
| `DATA_DIR` | `data/` | Override to `new_docs` for DELF corpus |
| `GENERATION_MODEL` | `gemini-2.5-flash` | Generation model |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Changing this requires full re-ingestion |
| `EMBEDDING_DIMENSION` | `3072` | Changing this requires full re-ingestion |
| `CORPUS_PRIMARY_LANGUAGE` | `fr` | Controls BM25 tokenizer + stop-words |
| `OUTPUT_LANGUAGE` | `tr` | Default generation output language |
| `MOCK_MODE` | `false` | `true` = all LLM calls return fixtures |
