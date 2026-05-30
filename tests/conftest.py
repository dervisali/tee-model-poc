"""pytest yardımcıları — tüm test dosyaları için ortak fixture'lar."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

# Tüm testlerden ÖNCE MOCK_MODE'u açıyoruz — fonksiyonlar fixture'a düşsün.
os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("ENABLE_CONFIDENCE_SCORING", "true")
os.environ.setdefault("ENABLE_HYBRID_SEARCH", "true")
os.environ.setdefault("ENABLE_CONTEXTUAL_ENRICHMENT", "false")  # ingestion'ı LLM'e bağımlı kılmasın
os.environ.setdefault("ENABLE_DEDUPLICATION", "false")


def _have(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


HAS_CHROMADB = _have("chromadb")
HAS_GOOGLE_GENAI = _have("google.genai")
HAS_RANK_BM25 = _have("rank_bm25")
RUN_VERTEX_TESTS = os.getenv("RUN_VERTEX_TESTS", "false").lower() == "true"
RUN_COLDSTART_TEST = os.getenv("RUN_COLDSTART_TEST", "false").lower() in ("1", "true", "yes")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_vertex_embeddings: canlı Vertex AI embedding çağrısı gerektirir.")
    config.addinivalue_line("markers", "requires_chromadb: chromadb gerektirir.")
    config.addinivalue_line("markers", "requires_bm25: rank_bm25 gerektirir.")
    config.addinivalue_line("markers", "integration: Tam yığını gerektiren entegrasyon testi.")
    config.addinivalue_line("markers", "requires_chromadb_seeded: gerçek bir ChromaDB tohumlar (chromadb gerekir).")
    config.addinivalue_line("markers", "requires_cold_start: RUN_COLDSTART_TEST=1 ve gerçek korpus gerektiren soğuk-başlangıç testi.")


def pytest_collection_modifyitems(config, items):
    skip_vertex = pytest.mark.skip(reason="canlı Vertex AI testleri kapalı veya google-genai yüklü değil")
    skip_chroma = pytest.mark.skip(reason="chromadb yüklü değil")
    skip_bm25 = pytest.mark.skip(reason="rank_bm25 yüklü değil")
    skip_seeded = pytest.mark.skip(reason="chromadb yüklü değil (tohumlanmış DB testi)")
    skip_coldstart = pytest.mark.skip(reason="RUN_COLDSTART_TEST ayarlı değil veya chromadb yok")

    for item in items:
        if "requires_vertex_embeddings" in item.keywords and not (RUN_VERTEX_TESTS and HAS_GOOGLE_GENAI):
            item.add_marker(skip_vertex)
        if "requires_chromadb" in item.keywords and not HAS_CHROMADB:
            item.add_marker(skip_chroma)
        if "requires_bm25" in item.keywords and not HAS_RANK_BM25:
            item.add_marker(skip_bm25)
        if "requires_chromadb_seeded" in item.keywords and not HAS_CHROMADB:
            item.add_marker(skip_seeded)
        if "requires_cold_start" in item.keywords and not (RUN_COLDSTART_TEST and HAS_CHROMADB):
            item.add_marker(skip_coldstart)
