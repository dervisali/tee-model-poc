"""
Çoklu sorgu üretimi (multi-query retrieval / HyDE-style).

Bir sorgu farklı kelimelerle ifade edildiğinde retrieval bazen çok farklı
sonuç verir. Burada LLM, orijinal sorgu için 3 farklı (semantik olarak eş
ama leksikal olarak farklı) varyant üretir; retrieval her varyant için
çalıştırılır ve sonuçlar parent_id bazında deduplicate edilerek skorlarına
göre yeniden sıralanır.

Bu mission Phase 1.4 spec'ine göre default OFF; ENABLE_QUERY_REWRITING=true
yapıldığında devreye girer. gemma3:4b ile her çağrı ~5-15s sürdüğünden,
yalnızca kalite kritik üretim görevlerinde tercih edilmelidir.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from src.llm import generate as llm_generate


logger = logging.getLogger(__name__)


class QueryVariants(BaseModel):
    """3 sorgu varyantı için yapılandırılmış çıktı."""
    queries: list[str] = Field(min_length=1, max_length=8)


_REWRITE_PROMPT = """\
Aşağıdaki sorgu için 3 farklı arama sorgusu üret.
Her sorgu FARKLI kelimeler/eşanlamlılar kullanmalı ama AYNI bilgiyi aramalı.
Türkçe yaz. Yalnızca JSON döndür.

Orijinal sorgu: {original_query}

Format:
{{"queries": ["sorgu1", "sorgu2", "sorgu3"]}}"""


def generate_query_variants(original_query: str, n_variants: int = 3) -> list[str]:
    """
    Orijinal sorgu için n_variants Türkçe varyant üretir.

    Hata durumunda boş liste döner — çağıran kodun fallback davranışını
    seçmesi için (genelde orijinal sorguyla devam eder).
    """
    try:
        raw = llm_generate(
            prompt=_REWRITE_PROMPT.format(original_query=original_query),
            response_format=QueryVariants,
        )
        parsed = QueryVariants(**json.loads(raw))
        variants = [v.strip() for v in parsed.queries if v.strip()]
        # Orijinal ile birebir aynı olanları çıkar; uniq olanları döndür.
        seen = {original_query.strip().lower()}
        unique = []
        for v in variants:
            key = v.lower()
            if key not in seen:
                unique.append(v)
                seen.add(key)
        return unique[:n_variants]
    except Exception as exc:
        logger.warning("Query rewriting başarısız: %s", exc)
        return []


def multi_query_retrieve(
    original_query: str,
    *,
    top_k: int,
    source_filter: str | None = None,
    search_mode: str | None = None,
) -> dict:
    """
    Orijinal sorgu + 3 varyant ile retrieval yapar; sonuçları parent_id
    bazında deduplicate eder ve birleşik skora göre sıralar.

    Döner
    -----
    dict — {
      "original": str,
      "variants": list[str],
      "merged_results": list[dict]   # standart retrieve_context çıktı şeması
    }
    """
    # Geç import — döngüsel referansı önler.
    from src.retrieval import retrieve_context

    variants = generate_query_variants(original_query)
    queries_to_run = [original_query] + variants
    logger.info("Multi-query retrieval: %d sorgu çalıştırılıyor.", len(queries_to_run))

    # Her sorgu için retrieval çalıştır; tüm sonuçları topla.
    merged: dict[str, dict] = {}
    for q in queries_to_run:
        try:
            hits = retrieve_context(
                q,
                top_k=top_k,
                source_filter=source_filter,
                search_mode=search_mode,
            )
        except Exception as exc:
            logger.warning("Sorgu '%s' başarısız: %s", q, exc)
            continue
        for hit in hits:
            pid = hit.get("parent_id", "")
            if not pid:
                continue
            # En yüksek skoru / en düşük mesafeyi koru.
            existing = merged.get(pid)
            new_score = hit.get("score") if hit.get("score") is not None else (1.0 - hit.get("distance", 1.0))
            if existing is None or new_score > existing.get("_merge_score", -1.0):
                hit["_merge_score"] = new_score
                merged[pid] = hit

    ordered = sorted(merged.values(), key=lambda h: h.get("_merge_score", -1.0), reverse=True)[:top_k]
    for h in ordered:
        h.pop("_merge_score", None)

    return {
        "original": original_query,
        "variants": variants,
        "merged_results": ordered,
    }
