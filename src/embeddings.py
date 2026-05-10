"""
Vertex AI Gemini embedding modeli sarmalayıcısı.

gemini-embedding-001 modeli, Google Gen AI SDK üzerinden Vertex AI'da
çalıştırılır ve bütün ingestion + retrieval bileşenleri tarafından kullanılır.

Önemli notlar:
- Belge gömerken task_type="RETRIEVAL_DOCUMENT" kullanılır.
- Sorgu gömerken task_type="RETRIEVAL_QUERY" kullanılır.

Karar gerekçesi:
- Cloud Run üzerinde torch / sentence-transformers yükü taşınmaz.
- gemini-embedding-001, Türkçe dahil çok dilli retrieval için Google'ın
  en yüksek kaliteli cloud embedding modelidir.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config import settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client singleton — Google Gen AI SDK istemcisi yeniden kullanılır.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_client():
    """Vertex AI Gemini embedding istemcisini döndürür."""
    from src.llm import get_vertex_client

    return get_vertex_client()


def _embed(texts: list[str], *, task_type: str) -> list[list[float]]:
    """Google Gen AI SDK ile metinleri gömer."""
    if not texts:
        return []
    from google.genai.types import EmbedContentConfig

    client = get_embedding_client()
    response = client.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=texts,
        config=EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.EMBEDDING_DIMENSION,
        ),
    )
    vectors = [embedding.values for embedding in response.embeddings]
    logger.info(
        "Embedding tamamlandı",
        extra={
            "event": "embedding_complete",
            "model": settings.EMBEDDING_MODEL,
            "task_type": task_type,
            "count": len(vectors),
            "dimension": settings.EMBEDDING_DIMENSION,
        },
    )
    return vectors


# ---------------------------------------------------------------------------
# Genel API
# ---------------------------------------------------------------------------

def embed_passage(text: str) -> list[float]:
    """Belge/parça gömesi — depolama yönü."""
    return _embed([text], task_type="RETRIEVAL_DOCUMENT")[0]


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Toplu belge gömeleri — ingestion sırasında verim için kullanılır."""
    output: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        output.extend(
            _embed(texts[start:start + batch_size], task_type="RETRIEVAL_DOCUMENT")
        )
    return output


def embed_query(text: str) -> list[float]:
    """Sorgu gömesi — arama yönü."""
    return _embed([text], task_type="RETRIEVAL_QUERY")[0]


def embedding_dimension() -> int:
    """Aktif modelin vektör boyutunu döndürür (collection metadata için)."""
    return settings.EMBEDDING_DIMENSION


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """İki normalize edilmiş vektör için kosinüs benzerliği (= iç çarpım)."""
    return float(sum(x * y for x, y in zip(a, b)))
