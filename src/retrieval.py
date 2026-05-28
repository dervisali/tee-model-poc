"""
TEE-Model retrieval katmanı — Parent Document Retrieval (PDR).

Akış:
  1. Sorgu, Vertex AI `gemini-embedding-001` ile gömülür.
  2. tee_children koleksiyonunda en yakın top_k*3 child aranır (over-fetch).
  3. Mesafe eşiğinin altındakiler, parent_id bazında deduplicate edilir
     (her parent için en yakın child saklanır).
  4. Parent metinleri parents.json'dan çözülür ve LLM'e gönderilmek üzere
     döndürülür.

Phase 1.3: Hybrid search (BM25 + dense, RRF füzyonu) ayrı bir sarmalayıcı
modülde uygulanır; bu modül yoğun (dense) yolu sağlar ve gerek duyuldukça
oradan çağrılır.
"""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path

import chromadb

from src.config import settings
from src.embeddings import embed_query


logger = logging.getLogger(__name__)


CHROMA_DIR = settings.CHROMA_DIR
PARENTS_JSON = CHROMA_DIR / "parents.json"
COLLECTION_NAME = settings.CHILD_COLLECTION_NAME


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _get_chroma_collection() -> chromadb.Collection:
    """Aktif child koleksiyonunu paylaşılan istemci üzerinden döndürür."""
    from src.chroma_client import get_child_collection
    return get_child_collection()


def _file_cache_key(path: Path) -> tuple[int, int]:
    """
    Return a cheap freshness key for file-backed in-memory caches.

    Re-ingestion replaces parents.json; including mtime and size lets the
    running Streamlit process see fresh data without a manual restart.
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=1)
def _load_parents(cache_key: tuple[int, int] | None = None) -> dict:
    """parents.json dosyasını yükler."""
    _ = cache_key
    if PARENTS_JSON.exists():
        return json.loads(PARENTS_JSON.read_text(encoding="utf-8"))
    return {}


def _load_current_parents() -> dict:
    """Load parents with freshness-aware cache, keeping old monkey-patches compatible."""
    try:
        return _load_parents(_file_cache_key(PARENTS_JSON))
    except TypeError:
        return _load_parents()


def _embed_query(query: str) -> list[float]:
    """Eski API ile uyumluluk takma adı (testler bunu import edebilir)."""
    return embed_query(query)


def _apply_reranking(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """
    Reranking açıksa adayları LLM yargıç ile yeniden sıralar; aksi halde
    chunks[:top_k] döner. Geç import — reranker yalnızca bayrak açıkken yüklenir.
    """
    if not settings.ENABLE_RERANKING:
        return chunks[:top_k]
    from src.reranker import rerank
    return rerank(query, chunks, top_n=top_k)


# ---------------------------------------------------------------------------
# Genel API
# ---------------------------------------------------------------------------

def retrieve_context(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
    distance_threshold: float | None = None,
    *,
    search_mode: str | None = None,
    alpha: float | None = None,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """
    Sorgu için en alakalı parent parçalarını PDR ile döndürür.

    search_mode:
      - "dense"  : yalnızca yoğun (eski davranış, mesafe eşiği uygulanır)
      - "sparse" : yalnızca BM25
      - "hybrid" : RRF füzyonu (alpha None ise) veya ağırlıklı füzyon
      - None     : settings.ENABLE_HYBRID_SEARCH true ise "hybrid", aksi halde "dense"

    Tüm modlarda parent_id bazında dedup uygulanır; her parent için en iyi
    child saklanır (dense'te en yakın, sparse/hybrid'te en yüksek skor).

    Geriye dönük uyum: distance_threshold yalnızca "dense" modunda anlamlıdır;
    diğer modlarda yok sayılır (skor tabanlı sıralama uygulanır).
    """
    top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
    if search_mode is None:
        search_mode = "hybrid" if settings.ENABLE_HYBRID_SEARCH else "dense"

    # Reranking açıksa daha geniş bir aday havuzu çek; yargıç top_k'ye kırpar.
    fetch_k = max(top_k, settings.RERANK_FETCH_K) if settings.ENABLE_RERANKING else top_k

    started = time.perf_counter()
    parents = _load_current_parents()
    parents_ms = (time.perf_counter() - started) * 1000
    if not parents:
        raise ValueError(
            "parents.json bulunamadı. Lütfen ingestion adımını tekrar çalıştırın."
        )

    if search_mode == "dense":
        dense_hits = _retrieve_dense(
            query, fetch_k, source_filter, distance_threshold, parents, metadata_filter
        )
        return _apply_reranking(query, dense_hits, top_k)

    # Phase 3.5: cross-lingual sorgu genişletme — yalnızca BM25 yolunu etkiler.
    # Dense yol orijinal sorgu üzerinden gider (multilingual embedding).
    bm25_query: str | None = None
    translation_ms = 0.0
    if settings.QUERY_LANGUAGE_AUTO_DETECT and search_mode in ("sparse", "hybrid"):
        from src.query_translator import detect_query_language, translate_query
        query_lang = detect_query_language(query)
        if query_lang != settings.CORPUS_PRIMARY_LANGUAGE:
            translation_started = time.perf_counter()
            bm25_query = translate_query(query, target_language=settings.CORPUS_PRIMARY_LANGUAGE)
            translation_ms = (time.perf_counter() - translation_started) * 1000
            logger.info(
                "Cross-lingual BM25",
                extra={
                    "event": "cross_lingual_translation",
                    "source_lang": query_lang,
                    "target_lang": settings.CORPUS_PRIMARY_LANGUAGE,
                    "translated_preview": bm25_query[:80],
                },
            )

    # Hibrit veya sparse — child seviyesinde arama, sonra parent_id'ye göre dedup
    from src.hybrid_search import hybrid_search_children  # geç import (döngüsel sorun yok)

    search_started = time.perf_counter()
    children = hybrid_search_children(
        query,
        top_k=fetch_k,
        source_filter=source_filter,
        search_mode=search_mode,
        alpha=alpha,
        bm25_query=bm25_query,
        metadata_filter=metadata_filter,
    )
    child_search_ms = (time.perf_counter() - search_started) * 1000
    if not children:
        raise ValueError("Sorgu için hiçbir sonuç döndürülmedi.")

    format_started = time.perf_counter()
    seen_parents: dict[str, dict] = {}
    for child in children:
        meta = child["metadata"]
        parent_id = meta.get("parent_id", "")
        if not parent_id or parent_id not in parents:
            continue
        # Hibritte yüksek skor iyi; mesafe ile uyumlu olsun diye distance = 1 - score saklanır
        score = child["score"]
        pseudo_distance = max(0.0, 1.0 - score) if score >= 0 else 1.0
        if parent_id not in seen_parents or score > seen_parents[parent_id]["score"]:
            seen_parents[parent_id] = {
                "child_text": child["document"],
                "child_index": meta.get("child_index", -1),
                "distance": round(pseudo_distance, 4),
                "score": round(score, 6),
                "dense_distance": child.get("dense_distance"),
                "bm25_score": child.get("bm25_score"),
                "parent_id": parent_id,
                "source": meta.get("doc_type", meta.get("source", "bilinmiyor")),
                "filename": meta.get("source_filename", meta.get("filename", "bilinmiyor")),
            }

    ordered = sorted(seen_parents.values(), key=lambda r: r["score"], reverse=True)[:fetch_k]

    formatted = []
    for item in ordered:
        parent_data = parents[item["parent_id"]]
        parent_text = parent_data["text"]
        formatted.append({
            "child_text": item["child_text"],
            "parent_text": parent_text,
            "text": parent_text,
            "source": item["source"],
            "filename": item["filename"],
            "parent_id": item["parent_id"],
            "child_index": item["child_index"],
            "distance": item["distance"],
            "score": item["score"],
            "dense_distance": item["dense_distance"],
            "bm25_score": item["bm25_score"],
            "search_mode": search_mode,
        })
    format_ms = (time.perf_counter() - format_started) * 1000
    total_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "Hibrit retrieval tamamlandı",
        extra={
            "event": "retrieval_complete",
            "query_len": len(query),
            "filter": source_filter or "all",
            "metadata_filter": bool(metadata_filter),
            "results": len(formatted),
            "search_mode": search_mode,
            "top_score": formatted[0]["score"] if formatted else None,
            "duration_ms": round(total_ms, 1),
            "parents_load_ms": round(parents_ms, 1),
            "translation_ms": round(translation_ms, 1),
            "child_search_ms": round(child_search_ms, 1),
            "parent_format_ms": round(format_ms, 1),
        },
    )
    return _apply_reranking(query, formatted, top_k)


def _retrieve_dense(
    query: str,
    top_k: int,
    source_filter: str | None,
    distance_threshold: float | None,
    parents: dict,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Eski yoğun-yalnız retrieval mantığı; geriye uyumluluk için korunur."""
    threshold = distance_threshold if distance_threshold is not None else settings.RETRIEVAL_DISTANCE_THRESHOLD

    started = time.perf_counter()
    collection = _get_chroma_collection()
    if collection.count() == 0:
        raise ValueError(
            "Vektör veritabanı boş. Lütfen önce 'Veritabanını Yenile' adımını çalıştırın."
        )

    embed_started = time.perf_counter()
    query_vector = _embed_query(query)
    embed_ms = (time.perf_counter() - embed_started) * 1000
    from src.metadata_filter import merge_where
    source_where = {"source": source_filter} if source_filter else None
    where_clause = merge_where(source_where, metadata_filter)
    query_kwargs: dict = {
        "query_embeddings": [query_vector],
        "n_results": min(top_k * 3, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_kwargs["where"] = where_clause

    try:
        chroma_started = time.perf_counter()
        results = collection.query(**query_kwargs)
        chroma_ms = (time.perf_counter() - chroma_started) * 1000
    except Exception as exc:
        raise RuntimeError(f"ChromaDB sorgu hatası: {exc}") from exc

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        raise ValueError("Sorgu için hiçbir sonuç döndürülmedi.")

    seen_parents: dict[str, dict] = {}
    for child_text, meta, dist in zip(documents, metadatas, distances):
        if float(dist) > threshold:
            continue
        parent_id = meta.get("parent_id", "")
        if not parent_id or parent_id not in parents:
            continue
        if parent_id not in seen_parents or float(dist) < seen_parents[parent_id]["distance"]:
            seen_parents[parent_id] = {
                "child_text": child_text,
                "child_index": meta.get("child_index", -1),
                "distance": round(float(dist), 4),
                "parent_id": parent_id,
                "source": meta.get("doc_type", meta.get("source", "bilinmiyor")),
                "filename": meta.get("source_filename", meta.get("filename", "bilinmiyor")),
            }

    ordered = sorted(seen_parents.values(), key=lambda r: r["distance"])[:top_k]

    formatted = []
    for item in ordered:
        parent_data = parents[item["parent_id"]]
        parent_text = parent_data["text"]
        formatted.append({
            "child_text": item["child_text"],
            "parent_text": parent_text,
            "text": parent_text,
            "source": item["source"],
            "filename": item["filename"],
            "parent_id": item["parent_id"],
            "child_index": item["child_index"],
            "distance": item["distance"],
            "search_mode": "dense",
        })
    total_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "Dense retrieval tamamlandı",
        extra={
            "event": "retrieval_complete",
            "query_len": len(query),
            "filter": source_filter or "all",
            "results": len(formatted),
            "search_mode": "dense",
            "top_distance": formatted[0]["distance"] if formatted else None,
            "duration_ms": round(total_ms, 1),
            "embedding_ms": round(embed_ms, 1),
            "chroma_query_ms": round(chroma_ms, 1),
        },
    )
    return formatted


def build_context_text(chunks: list[dict]) -> tuple[str, list[str]]:
    """
    Retrieved parent parçalarını LLM bağlam metnine dönüştürür.

    Döner
    -----
    tuple[str, list[str]]
        - context_text: kaynak etiketleri ile parent metinleri.
        - parent_ids: atıf kullanımı için.
    """
    parts: list[str] = []
    parent_ids: list[str] = []
    for i, chunk in enumerate(chunks):
        label = f"[Kaynak {i + 1}: {chunk['filename']} — {chunk['parent_id']}]"
        parts.append(f"{label}\n{chunk['parent_text']}")
        parent_ids.append(chunk["parent_id"])
    return "\n\n---\n\n".join(parts), parent_ids


def lookup_parent_context(parent_id: str) -> dict | None:
    """
    Verilen parent_id için parent metnini ve onun child'larını döndürür.
    Uzman onay paneli, kaynak alıntılarını göstermek için kullanır.
    """
    parents = _load_current_parents()
    if parent_id not in parents:
        return None
    parent_data = parents[parent_id]

    try:
        collection = _get_chroma_collection()
        results = collection.get(
            where={"parent_id": parent_id},
            include=["documents", "metadatas"],
        )
        children = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            children.append({
                "text": doc,
                "child_index": meta.get("child_index", 0),
            })
        children.sort(key=lambda c: c["child_index"])
    except Exception as exc:
        logger.warning("Child'lar alınamadı (%s): %s", parent_id, exc)
        children = []

    return {
        "parent_text": parent_data["text"],
        "children": children,
        "source": parent_data.get("doc_type", parent_data.get("source", "")),
        "filename": parent_data.get("source_filename", parent_data.get("filename", "")),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    test_query = "SGK kesintisi hesaplama"
    print(f"Test sorgusu: '{test_query}'\n")
    try:
        hits = retrieve_context(test_query, top_k=3)
        for i, hit in enumerate(hits, start=1):
            print(f"--- Sonuç {i} ---")
            print(f"Kaynak : {hit['source']} | Dosya: {hit['filename']} | Parent: {hit['parent_id']}")
            print(f"Mesafe : {hit['distance']}")
            print(f"Eşleşen child ({hit['child_index']}): {hit['child_text'][:120]}...")
            print(f"Parent metin : {hit['parent_text'][:300]}...")
            print()
    except (ValueError, RuntimeError) as exc:
        print(f"Hata: {exc}")
