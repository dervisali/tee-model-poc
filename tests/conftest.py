"""
pytest yardımcıları — tüm test dosyaları için ortak fixture'lar.

Ağır bağımlılıklar (torch, sentence-transformers, ollama, chromadb) yüklü
değilse, ilgili testler `requires_heavy_deps` marker'ı üzerinden atlanır.
MOCK_MODE=true ile çalıştırıldığında üretici testleri Ollama'ya hiç
dokunmaz; bu, mission Phase 4 başarı kriterine uygundur.
"""

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


HAS_TORCH = _have("torch")
HAS_SENTENCE_TRANSFORMERS = _have("sentence_transformers")
HAS_CHROMADB = _have("chromadb")
HAS_OLLAMA_PY = _have("ollama")
HAS_RANK_BM25 = _have("rank_bm25")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_torch: torch + sentence-transformers gerektirir.")
    config.addinivalue_line("markers", "requires_chromadb: chromadb gerektirir.")
    config.addinivalue_line("markers", "requires_ollama_lib: ollama python paketi gerektirir.")
    config.addinivalue_line("markers", "requires_bm25: rank_bm25 gerektirir.")
    config.addinivalue_line("markers", "integration: Tam yığını gerektiren entegrasyon testi.")


def pytest_collection_modifyitems(config, items):
    skip_torch = pytest.mark.skip(reason="torch / sentence-transformers yüklü değil")
    skip_chroma = pytest.mark.skip(reason="chromadb yüklü değil")
    skip_ollama = pytest.mark.skip(reason="ollama python paketi yüklü değil")
    skip_bm25 = pytest.mark.skip(reason="rank_bm25 yüklü değil")

    for item in items:
        if "requires_torch" in item.keywords and not (HAS_TORCH and HAS_SENTENCE_TRANSFORMERS):
            item.add_marker(skip_torch)
        if "requires_chromadb" in item.keywords and not HAS_CHROMADB:
            item.add_marker(skip_chroma)
        if "requires_ollama_lib" in item.keywords and not HAS_OLLAMA_PY:
            item.add_marker(skip_ollama)
        if "requires_bm25" in item.keywords and not HAS_RANK_BM25:
            item.add_marker(skip_bm25)
