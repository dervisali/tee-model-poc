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
| `src/retrieval.py` | `retrieve_context()` — hybrid BM25+dense, parent-doc fetch, RRF fusion, optional metadata pre-filter + LLM rerank |
| `src/generators.py` | `generate_process_map/error_cards/glossary/simulation()` |
| `src/ingestion.py` | Document loading, chunking, embedding, ChromaDB write |
| `src/chunk_metadata.py` | `ChunkMetadata` Pydantic schema for all chunk metadata fields |
| `src/metadata_filter.py` | `build_where_clause()` / `detect_filters()` — level/skill/doc_type pre-filter before vector search |
| `src/reranker.py` | `rerank()` — LLM-as-reranker second-stage precision (gated by `ENABLE_RERANKING`) |
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
| `ENABLE_RERANKING` | `false` | — | Off — LLM-as-reranker adds 1 LLM call (latency/cost). On = over-fetch `RERANK_FETCH_K` (20) candidates → rerank to top_k. Quality win, not speed. |
| `MOCK_MODE` | `false` | — | Off; set to `true` for UI testing without API |

## Known Issues

- **Milestones M3/M4 not yet measured** — top-3 source recall ≥90% and citation accuracy ≥95% still need a measurement run. DELF eval set now exists (`evaluation/delf_questions.json`, 18 bilingual Q&A); run `python -m src.evaluator` to baseline.
- **App header says "TEE"** — The main title still shows the old acronym; DELF rebrand of the header is pending.
- **ChromaDB is local only** — Cloud Run ephemeral filesystem drops the DB on restart. GCS FUSE mount for persistence not yet implemented.

## Deferred / Not Yet Implemented

- **Metadata pre-filter not wired into UI** — `src/metadata_filter.py` exists and `retrieve_context(metadata_filter=...)` works; no UI control surfaces it yet (callers pass filters programmatically). `detect_filters()` auto-detect is available but not called by default.
- Local cross-encoder / hosted rerank API — rejected in favor of LLM-as-reranker (`src/reranker.py`); revisit only if rerank latency becomes a bottleneck.
- GCS FUSE mount for ChromaDB persistence on Cloud Run
- Expand DELF eval set toward 50 Q&A pairs
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
| `ENABLE_RERANKING` | `false` | `true` = LLM reranks top candidates after fusion |
| `RERANK_FETCH_K` | `20` | Candidates over-fetched for the reranker before slicing to top_k |
| `RERANK_MODEL` | `None` | Reranker model; `None` reuses `GENERATION_MODEL` |
| `MOCK_MODE` | `false` | `true` = all LLM calls return fixtures |
