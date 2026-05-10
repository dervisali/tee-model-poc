"""
Chunklama stratejileri (Phase 1.2.B).

Üç strateji desteklenir; ingestion ve insert_new_document `strategy`
parametresi (varsayılan settings.CHUNKING_STRATEGY) ile seçer:

  - "paragraph"  : çift yeni satır + cümle bazlı dönüş; mevcut (varsayılan)
                   davranış. Türkçe regülatör belgesi gibi MADDE odaklı
                   metinlerde sağlam çalışır.
  - "semantic"   : Vertex AI embedding ile cümle gömeleri; bitişik
                   cümle benzerliği SEMANTIC_SIM_THRESHOLD altına düşünce
                   yeni chunk başlar. Anlamsal sınırları yakalar; küçük
                   topluluklarda paragraph'a kıyasla farklı sonuçlar verir.
                   Şubat 2026 7-strateji karşılaştırması paragraph'ı
                   semantic'ten üstün buldu (akademik metinde); o yüzden
                   varsayılan paragraph'ta kalır.
  - "fixed"      : sabit karakter boyutu + örtüşme. Ablation/benchmark
                   karşılaştırmaları için referans bazel olarak kullanışlı.

Tüm stratejiler PDR ile uyumludur: önce parent (300-800 karakter), sonra her
parent içinde child (50-200 karakter) bölmesi yapılır.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from src.config import settings


logger = logging.getLogger(__name__)


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# 1. Paragraph (varsayılan)
# ---------------------------------------------------------------------------

def paragraph_parent_chunks(text: str) -> list[str]:
    """Çift yeni satır birleştirme + cümle bazlı kapanış. Mevcut davranış."""
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pmin, pmax = settings.CHUNK_SIZE_PARENT_MIN, settings.CHUNK_SIZE_PARENT_MAX

    merged: list[str] = []
    buffer = ""
    for para in raw_paragraphs:
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if len(candidate) <= pmax:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = para
    if buffer:
        if merged and len(buffer) < pmin:
            merged[-1] += "\n\n" + buffer
        else:
            merged.append(buffer)

    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= pmax:
            final.append(chunk)
        else:
            sentences = _SENTENCE_END_RE.split(chunk)
            current = ""
            for s in sentences:
                candidate = (current + " " + s).strip() if current else s
                if len(candidate) <= pmax:
                    current = candidate
                else:
                    if current:
                        final.append(current)
                    current = s
            if current:
                final.append(current)

    return [c for c in final if c.strip()]


# ---------------------------------------------------------------------------
# 2. Semantic
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Türkçe metinler için yalın cümle ayırıcı."""
    # Önce paragrafları korumak için \n\n'leri tek boşluğa çevirmiyoruz; satır
    # bazlı bütünlüğü koruyacağız ama cümle sınırını yakalayacağız.
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def semantic_parent_chunks(text: str) -> list[str]:
    """
    Cümle gömeleri arasındaki kosinüs benzerliği eşiğinin altına düştüğü
    yerlerde yeni chunk başlatır. Parent boyut sınırları yine uygulanır.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    # Geç import — semantic strateji canlı Vertex AI embedding çağrısı yapar.
    from src.embeddings import embed_passages, cosine_similarity

    threshold = settings.SEMANTIC_SIM_THRESHOLD
    pmax = settings.CHUNK_SIZE_PARENT_MAX
    pmin = settings.CHUNK_SIZE_PARENT_MIN

    vectors = embed_passages(sentences)

    chunks: list[str] = []
    current_sentences: list[str] = [sentences[0]]
    current_vec = vectors[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(current_vec, vectors[i])
        candidate_text = " ".join(current_sentences + [sentences[i]])

        # Yüksek benzerlik VE boyut sınırı uygunsa devam et
        if sim >= threshold and len(candidate_text) <= pmax:
            current_sentences.append(sentences[i])
            current_vec = vectors[i]  # son cümlenin vektörü ile karşılaştırmaya devam
        else:
            # Mevcut chunk'ı kapat, yenisini başlat
            joined = " ".join(current_sentences)
            if chunks and len(joined) < pmin:
                chunks[-1] += " " + joined
            else:
                chunks.append(joined)
            current_sentences = [sentences[i]]
            current_vec = vectors[i]

    if current_sentences:
        joined = " ".join(current_sentences)
        if chunks and len(joined) < pmin:
            chunks[-1] += " " + joined
        else:
            chunks.append(joined)

    logger.info("Semantic chunklama: %d cümle → %d parent.", len(sentences), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# 3. Fixed (sabit karakter penceresi + örtüşme)
# ---------------------------------------------------------------------------

def fixed_parent_chunks(text: str, *, size: int | None = None, overlap: int = 50) -> list[str]:
    """Karakter bazlı sabit boyutlu kayan pencere. Ablation/benchmark için."""
    size = size or settings.CHUNK_SIZE_PARENT_MAX
    if size <= overlap:
        raise ValueError("size > overlap olmalı.")
    chunks: list[str] = []
    i = 0
    while i < len(text):
        chunk = text[i:i + size].strip()
        if chunk:
            chunks.append(chunk)
        i += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Strateji seçici
# ---------------------------------------------------------------------------

ParentChunker = Callable[[str], list[str]]


def get_parent_chunker(strategy: str) -> ParentChunker:
    """Verilen strateji adı için parent chunker fonksiyonunu döndürür."""
    if strategy == "paragraph":
        return paragraph_parent_chunks
    if strategy == "semantic":
        return semantic_parent_chunks
    if strategy == "fixed":
        return fixed_parent_chunks
    raise ValueError(
        f"Bilinmeyen chunking_strategy='{strategy}'. "
        f"Geçerli: paragraph | semantic | fixed."
    )
