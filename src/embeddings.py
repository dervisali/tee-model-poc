"""
Vertex AI Gemini embedding modeli sarmalayıcısı.

gemini-embedding-001 modeli, Google Gen AI SDK üzerinden Vertex AI'da
çalıştırılır ve bütün ingestion + retrieval bileşenleri tarafından kullanılır.

Önemli notlar:
- Belge gömerken task_type="RETRIEVAL_DOCUMENT" kullanılır.
- Sorgu gömerken task_type="RETRIEVAL_QUERY" kullanılır.

Karar gerekçesi:
- Cloud Run üzerinde ağır yerel embedding runtime yükü taşınmaz.
- gemini-embedding-001, Türkçe dahil çok dilli retrieval için Google'ın
  en yüksek kaliteli cloud embedding modelidir.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import settings
from src.llm import _is_retryable_exception


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client singleton — Google Gen AI SDK istemcisi yeniden kullanılır.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_client():
    """Vertex AI Gemini embedding istemcisini döndürür."""
    from src.llm import get_vertex_client

    return get_vertex_client()


def _l2_normalize(vector: list[float]) -> list[float]:
    """Cosine retrieval için vektörü birim norma taşır."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm <= 1e-12:
        return vector
    return [x / norm for x in vector]


@retry(
    stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
    wait=wait_exponential(
        multiplier=1,
        min=settings.LLM_RETRY_MIN_WAIT,
        max=settings.LLM_RETRY_MAX_WAIT,
    ),
    retry=retry_if_exception(_is_retryable_exception),
    reraise=True,
)
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
    vectors = [_l2_normalize(list(embedding.values)) for embedding in response.embeddings]
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


def embed_passages(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Toplu belge gömeleri — ingestion sırasında verim için kullanılır.

    Hem Vertex AI hem de public Gemini API RPM kotalarına tabidir.
    Yeni Vertex projeleri ~40 RPM gibi düşük varsayılan kotayla gelir;
    bu nedenle sabit sleep yerine gerçek bir RPM limiti uygulanır:
    her batch çağrısı arasında en az (60 / rpm_cap) saniye beklenir.
    """
    import time

    # Conservative cap: well below observed ~40 RPM default quota.
    # public_genai free tier is tighter; vertex new-project default ~40 RPM.
    rpm_cap = 12 if settings.INFERENCE_BACKEND == "public_genai" else 25
    min_interval = 60.0 / rpm_cap  # seconds that must elapse between batch calls

    output: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    last_call_at: float = 0.0

    for i, start in enumerate(range(0, len(texts), batch_size)):
        elapsed = time.monotonic() - last_call_at
        wait = min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        last_call_at = time.monotonic()
        output.extend(
            _embed(texts[start:start + batch_size], task_type="RETRIEVAL_DOCUMENT")
        )
        if total_batches > 10 and (i + 1) % 10 == 0:
            logger.info("Embedding progress: %d/%d batch", i + 1, total_batches)
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
