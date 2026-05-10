"""
Yerel gömme (embedding) modeli sarmalayıcısı.

multilingual-e5-large modeli, sentence-transformers üzerinden bir kez yüklenir
ve bütün ingestion + retrieval bileşenleri tarafından bu tek örnek kullanılır.

Önemli notlar (e5 modelleri için zorunlu önek konvansiyonu):
- Belge gömerken metin başına "passage: " eklenir.
- Sorgu gömerken metin başına "query: " eklenir.
Bu önekler atlanırsa retrieval kalitesi belirgin biçimde düşer
(model kartında belirtilmiş).

Karar gerekçesi:
- Gemini gömme modelinden ayrılma: tamamen yerel, ücretsiz, çevrimdışı çalışır.
- e5-large 1024 boyut üretir (Gemini 3072'di); chroma_db dim değiştiği için
  tüm koleksiyon yeniden ingest edilmelidir.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config import settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model singleton — torch ve sentence-transformers ağır olduğundan, model
# yalnızca gerçekten gömme yapan bir çağrı geldiğinde yüklenir.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model():
    """multilingual-e5-large modelini bellek içi tek örnek olarak yükler."""
    import torch  # local import — testler bu modülü yüklediğinde torch zorunlu olmasın
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Gömme modeli yükleniyor: %s (device=%s)", settings.EMBEDDING_MODEL, device)
    model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
    logger.info(
        "Gömme modeli hazır",
        extra={
            "event": "embedding_model_loaded",
            "model": settings.EMBEDDING_MODEL,
            "dimension": model.get_sentence_embedding_dimension(),
            "device": device,
        },
    )
    return model


# ---------------------------------------------------------------------------
# Genel API
# ---------------------------------------------------------------------------

def embed_passage(text: str) -> list[float]:
    """Belge/parça gömesi — depolama yönü ('passage:' öneki)."""
    model = get_embedding_model()
    vector = model.encode(
        f"passage: {text}",
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Toplu belge gömeleri — ingestion sırasında verim için kullanılır."""
    model = get_embedding_model()
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Sorgu gömesi — arama yönü ('query:' öneki)."""
    model = get_embedding_model()
    vector = model.encode(
        f"query: {text}",
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()


def embedding_dimension() -> int:
    """Aktif modelin vektör boyutunu döndürür (collection metadata için)."""
    return get_embedding_model().get_sentence_embedding_dimension()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """İki normalize edilmiş vektör için kosinüs benzerliği (= iç çarpım)."""
    return float(sum(x * y for x, y in zip(a, b)))
