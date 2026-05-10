"""
Contextual Chunk Enrichment — Anthropic, Eylül 2024.

Her child chunk gömülmeden ÖNCE, LLM'e tam belge ile chunk'ı verip 2-3
cümlelik bağlam özeti ürettiriliyoruz. Bu özet chunk'ın başına eklenir;
yalnızca eklenen metin gömme + BM25 yoluna girer. Orijinal chunk metadata'da
'original_text' alanında saklanır ve UI'da bu gösterilir.

Anthropic'in Eylül 2024 raporunda contextual embeddings + contextual BM25
birleşimi, top-20 retrieval başarısızlığını %5.7 → %2.9'a düşürdü (-%49).

Maliyet: gemma3:4b ile chunk başına ~5-10 saniye (CPU/MPS). Önbellek
(SHA1 hash → enriched_text) yeniden ingestion'da tekrar çağrı yapılmasını
engeller; veri seti değişmediği sürece tek seferlik ücrettir.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from src.config import settings
from src.llm import generate as llm_generate


logger = logging.getLogger(__name__)


CACHE_PATH = settings.CHROMA_DIR / "enrichment_cache.json"


# ---------------------------------------------------------------------------
# Cache yönetimi
# ---------------------------------------------------------------------------

def _load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Enrichment cache okunamadı: %s — sıfır cache ile devam.", exc)
        return {}


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def _cache_key(document_id: str, chunk_text: str) -> str:
    """document_id + chunk_text üzerinden deterministik anahtar."""
    digest = hashlib.sha1(f"{document_id}::{chunk_text}".encode("utf-8")).hexdigest()
    return digest


# ---------------------------------------------------------------------------
# LLM enrichment çağrısı
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
Aşağıda bir belge ve bu belgeden alınmış bir parça (chunk) yer alıyor.
Parçanın belge içindeki konumunu ve etrafındaki bağlamı 2-3 cümle ile
açıkla. Yalnızca belgede zaten var olan bilgileri kullan; yeni bilgi ekleme.
Çıktı yalnızca açıklama olsun; başlık, etiket veya markdown ekleme.

BELGE:
{document}

PARÇA:
{chunk}

BAĞLAM AÇIKLAMASI (2-3 cümle):"""


def _generate_context_summary(document_text: str, chunk_text: str) -> str:
    """Tek bir chunk için bağlam özeti üretir."""
    prompt = _PROMPT_TEMPLATE.format(document=document_text, chunk=chunk_text)
    raw = llm_generate(prompt=prompt)
    summary = raw.strip()
    # Bazı modeller başlık veya tırnak ekleyebilir; ilk paragrafı al.
    if "\n\n" in summary:
        summary = summary.split("\n\n", 1)[0].strip()
    # Aşırı uzunsa ilk 3 cümle ile sınırla.
    sentences = summary.replace("\n", " ").split(". ")
    if len(sentences) > 3:
        summary = ". ".join(sentences[:3]).rstrip(".") + "."
    return summary


def enrich_chunk_with_context(
    document_text: str,
    chunk_text: str,
    *,
    document_id: str = "",
    cache: dict[str, str] | None = None,
) -> str:
    """
    Tek chunk için zenginleştirilmiş metin döndürür.

    Format: "<bağlam özeti>\n\n<orijinal chunk>"

    cache verilirse, anahtar varsa LLM çağrısı yapılmaz.
    """
    key = _cache_key(document_id, chunk_text)
    if cache is not None and key in cache:
        return cache[key]

    summary = _generate_context_summary(document_text, chunk_text)
    enriched = f"{summary}\n\n{chunk_text}"

    if cache is not None:
        cache[key] = enriched
    return enriched


# ---------------------------------------------------------------------------
# Toplu API — ingestion'dan çağrılır
# ---------------------------------------------------------------------------

def enrich_chunks(
    document_text: str,
    chunks: list[str],
    *,
    document_id: str,
) -> tuple[list[str], dict]:
    """
    Bir kaynak belgenin tüm child chunk'ları için zenginleştirilmiş metinleri
    üretir. Cache hit/miss istatistiklerini de döndürür.

    Döner
    -----
    tuple[list[str], dict]
        - enriched_texts: chunks ile aynı sırada zenginleştirilmiş metinler
        - stats: {"hits": int, "misses": int, "total_seconds": float}
    """
    cache = _load_cache()
    enriched_list: list[str] = []
    hits = 0
    misses = 0
    started = time.perf_counter()

    for i, chunk_text in enumerate(chunks):
        key = _cache_key(document_id, chunk_text)
        if key in cache:
            enriched_list.append(cache[key])
            hits += 1
            continue
        try:
            enriched = enrich_chunk_with_context(
                document_text,
                chunk_text,
                document_id=document_id,
                cache=cache,
            )
        except Exception as exc:
            logger.warning(
                "Chunk %d/%d enrichment başarısız (%s); orijinal kullanılacak.",
                i + 1, len(chunks), exc,
            )
            enriched = chunk_text
        enriched_list.append(enriched)
        misses += 1

        if misses % 5 == 0:
            _save_cache(cache)
            logger.info("Enrichment ilerleme: %d/%d (cache hit=%d miss=%d)",
                        i + 1, len(chunks), hits, misses)

    _save_cache(cache)
    elapsed = time.perf_counter() - started
    stats = {"hits": hits, "misses": misses, "total_seconds": round(elapsed, 1)}
    logger.info(
        "Enrichment tamamlandı",
        extra={
            "event": "enrichment_complete",
            "document_id": document_id,
            **stats,
        },
    )
    return enriched_list, stats
