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
    """Aktif child koleksiyonunu döndürür."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _load_parents() -> dict:
    """parents.json dosyasını yükler."""
    if PARENTS_JSON.exists():
        return json.loads(PARENTS_JSON.read_text(encoding="utf-8"))
    return {}


def _embed_query(query: str) -> list[float]:
    """Eski API ile uyumluluk takma adı (testler bunu import edebilir)."""
    return embed_query(query)


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

    parents = _load_parents()
    if not parents:
        raise ValueError(
            "parents.json bulunamadı. Lütfen ingestion adımını tekrar çalıştırın."
        )

    if search_mode == "dense":
        return _retrieve_dense(query, top_k, source_filter, distance_threshold, parents)

    # Phase 3.5: cross-lingual sorgu genişletme — yalnızca BM25 yolunu etkiler.
    # Dense yol orijinal sorgu üzerinden gider (multilingual embedding).
    bm25_query: str | None = None
    if settings.QUERY_LANGUAGE_AUTO_DETECT and search_mode in ("sparse", "hybrid"):
        from src.query_translator import detect_query_language, translate_query
        query_lang = detect_query_language(query)
        if query_lang != settings.CORPUS_PRIMARY_LANGUAGE:
            bm25_query = translate_query(query, target_language=settings.CORPUS_PRIMARY_LANGUAGE)
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

    children = hybrid_search_children(
        query,
        top_k=top_k,
        source_filter=source_filter,
        search_mode=search_mode,
        alpha=alpha,
        bm25_query=bm25_query,
    )
    if not children:
        raise ValueError("Sorgu için hiçbir sonuç döndürülmedi.")

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

    ordered = sorted(seen_parents.values(), key=lambda r: r["score"], reverse=True)[:top_k]

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

    logger.info(
        "Hibrit retrieval tamamlandı",
        extra={
            "event": "retrieval_complete",
            "query_len": len(query),
            "filter": source_filter or "all",
            "results": len(formatted),
            "search_mode": search_mode,
            "top_score": formatted[0]["score"] if formatted else None,
        },
    )
    return formatted


def _retrieve_dense(
    query: str,
    top_k: int,
    source_filter: str | None,
    distance_threshold: float | None,
    parents: dict,
) -> list[dict]:
    """Eski yoğun-yalnız retrieval mantığı; geriye uyumluluk için korunur."""
    threshold = distance_threshold if distance_threshold is not None else settings.RETRIEVAL_DISTANCE_THRESHOLD

    collection = _get_chroma_collection()
    if collection.count() == 0:
        raise ValueError(
            "Vektör veritabanı boş. Lütfen önce 'Veritabanını Yenile' adımını çalıştırın."
        )

    query_vector = _embed_query(query)
    where_clause = {"source": source_filter} if source_filter else None
    query_kwargs: dict = {
        "query_embeddings": [query_vector],
        "n_results": min(top_k * 3, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_kwargs["where"] = where_clause

    try:
        results = collection.query(**query_kwargs)
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

    logger.info(
        "Dense retrieval tamamlandı",
        extra={
            "event": "retrieval_complete",
            "query_len": len(query),
            "filter": source_filter or "all",
            "results": len(formatted),
            "search_mode": "dense",
            "top_distance": formatted[0]["distance"] if formatted else None,
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
    parents = _load_parents()
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
