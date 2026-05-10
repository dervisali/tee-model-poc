"""
Hibrit retrieval — yoğun (dense) + seyrek (BM25 sparse) füzyonu.

Türk mevzuat metinleri (örn. "ek gösterge katsayısı", "net aylığın 1/4'ü")
birebir terim eşleşmesine bağımlıdır; yoğun gömme bu eşleşmeleri sıkça
ıskalar. BM25 sözcük tabanlı eşleşme, bu tür terim aramalarında dense'i
tamamlar.

Bu modül child seviyesinde çalışır; sonuçlar `retrieval.retrieve_context`
tarafından parent_id bazında deduplicate edilir.

Füzyon yöntemleri:
1. Reciprocal Rank Fusion (RRF, Cormack et al. SIGIR 2009)
   score(d) = Σ 1 / (k + rank_i(d))
   Skor normalizasyonu gerektirmez; varsayılan yöntemdir (k=60).
2. Ağırlıklı (weighted) füzyon: alpha * dense + (1-alpha) * sparse,
   skorlar 0-1 aralığına normalize edilir. ENABLE_HYBRID_SEARCH false ise
   yine kullanılabilir; mission Phase 1.3 spec'i ile uyumludur.

BM25 indeksi ingestion zamanında inşa edilir ve diske pickle ile
kalıcılaştırılır. Yeniden ingestion'da indeks yeniden kurulur.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import settings
from src.embeddings import embed_query


logger = logging.getLogger(__name__)


BM25_INDEX_PATH = settings.CHROMA_DIR / "bm25_index.pkl"


# ---------------------------------------------------------------------------
# Türkçe odaklı basit tokenizasyon
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[\wçğıöşüÇĞİÖŞÜ]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """
    Hafif Türkçe tokenizasyon: küçük harf + Türkçe karakter farkındalığı.

    Stemmer KULLANILMAZ (gerek yok): e5 dense yolu zaten anlamsal benzerliği
    yakalar; BM25'in görevi birebir terim eşleşmesini güçlendirmektir. Ek
    morfolojik analiz hatası eklemek doğruluğu azaltır.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# İndeks veri yapısı
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Çocuk parçalar üzerinde BM25 indeksi.

    Disk formatı (pickle):
      {"ids": list[str], "tokens": list[list[str]],
       "documents": list[str], "metadatas": list[dict]}
    """

    def __init__(
        self,
        ids: list[str],
        tokens: list[list[str]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self.ids = ids
        self.tokens = tokens
        self.documents = documents
        self.metadatas = metadatas
        self.bm25 = BM25Okapi(tokens) if tokens else None

    def search(
        self,
        query: str,
        top_k: int,
        source_filter: str | None = None,
    ) -> list[tuple[str, float, str, dict]]:
        """
        Sorgu için en alakalı top_k kaydı (id, score, document, metadata)
        olarak döndürür. Source filtresi metadata üzerinden uygulanır.
        """
        if self.bm25 is None or not self.ids:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked: list[tuple[str, float, str, dict]] = []
        for idx, score in enumerate(scores):
            meta = self.metadatas[idx]
            if source_filter and meta.get("source") != source_filter:
                continue
            ranked.append((self.ids[idx], float(score), self.documents[idx], meta))

        ranked.sort(key=lambda r: r[1], reverse=True)
        return ranked[:top_k]

    # -- Persistence -------------------------------------------------------

    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids": self.ids,
            "tokens": self.tokens,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f)
        tmp.replace(path)
        logger.info("BM25 indeksi kaydedildi: %s (n=%d)", path, len(self.ids))

    @classmethod
    def load(cls, path: Path = BM25_INDEX_PATH) -> "BM25Index | None":
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
            return cls(
                ids=payload["ids"],
                tokens=payload["tokens"],
                documents=payload["documents"],
                metadatas=payload["metadatas"],
            )
        except Exception as exc:
            logger.warning("BM25 indeksi yüklenemedi: %s", exc)
            return None


# ---------------------------------------------------------------------------
# İndeks inşası — ingestion'dan çağrılır
# ---------------------------------------------------------------------------

def rebuild_bm25_index_from_collection() -> BM25Index | None:
    """
    Aktif tee_children koleksiyonundan BM25 indeksini yeniden kurar.
    Ingestion'dan sonra çağrılmalıdır.
    """
    import chromadb  # local to avoid circular ingestion import at module load
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    try:
        col = client.get_or_create_collection(
            name=settings.CHILD_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        logger.error("BM25 inşa: koleksiyon alınamadı: %s", exc)
        return None

    if col.count() == 0:
        logger.warning("BM25 inşa: koleksiyon boş, indeks atlandı.")
        return None

    data = col.get(include=["documents", "metadatas"])
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]
    tokens = [_tokenize(doc) for doc in documents]

    index = BM25Index(ids=ids, tokens=tokens, documents=documents, metadatas=metadatas)
    index.save()
    logger.info("BM25 indeksi inşa edildi: %d child.", len(ids))
    return index


# ---------------------------------------------------------------------------
# Yoğun (dense) yardımcı — child seviyesinde sıralı sonuç
# ---------------------------------------------------------------------------

def _dense_search_children(
    query: str,
    top_k: int,
    source_filter: str | None,
) -> list[tuple[str, float, str, dict]]:
    """ChromaDB üzerinden dense sıralama; (id, distance, doc, meta) listesi döner."""
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    col = client.get_or_create_collection(
        name=settings.CHILD_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if col.count() == 0:
        return []

    qvec = embed_query(query)
    where = {"source": source_filter} if source_filter else None
    kwargs = {
        "query_embeddings": [qvec],
        "n_results": min(top_k, col.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    return [(i, float(d), doc, meta) for i, d, doc, meta in zip(ids, dists, docs, metas)]


# ---------------------------------------------------------------------------
# Füzyon
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    rank_lists: list[list[str]],
    rrf_k: int,
) -> dict[str, float]:
    """
    RRF: score(d) = Σ 1 / (rrf_k + rank_i(d))
    Cormack, Clarke, Buettcher; SIGIR 2009. k=60 endüstri standardıdır.
    """
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, child_id in enumerate(ranks):
            scores[child_id] = scores.get(child_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


def _normalize_scores(values: list[float]) -> list[float]:
    """Min-max normalizasyon (0..1)."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-12:
        return [0.5 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


# ---------------------------------------------------------------------------
# Genel hibrit arama API'si
# ---------------------------------------------------------------------------

def hybrid_search_children(
    query: str,
    *,
    top_k: int,
    source_filter: str | None = None,
    search_mode: str = "hybrid",
    alpha: float | None = None,
    over_fetch: int = 3,
) -> list[dict]:
    """
    BM25 + dense hibrit arama; child seviyesinde sıralı sonuç döndürür.

    search_mode = "dense"  : yalnızca yoğun
                  "sparse" : yalnızca BM25
                  "hybrid" : füzyon (alpha None ise RRF, değilse ağırlıklı)

    Döner — her öğe: {id, document, metadata, score, dense_distance, bm25_score}
    """
    fetch_k = top_k * over_fetch

    dense_results = []
    bm25_results: list[tuple[str, float, str, dict]] = []

    if search_mode in ("dense", "hybrid"):
        dense_results = _dense_search_children(query, fetch_k, source_filter)

    if search_mode in ("sparse", "hybrid"):
        bm25_index = BM25Index.load()
        if bm25_index is None:
            logger.warning("BM25 indeksi yok; sparse atlanıyor. Ingestion'ı yeniden çalıştırın.")
        else:
            bm25_results = bm25_index.search(query, fetch_k, source_filter=source_filter)

    if search_mode == "dense":
        return [
            {
                "id": cid,
                "document": doc,
                "metadata": meta,
                "score": 1.0 - dist,  # cosine distance → similarity benzeri
                "dense_distance": dist,
                "bm25_score": None,
            }
            for cid, dist, doc, meta in dense_results[:top_k]
        ]

    if search_mode == "sparse":
        return [
            {
                "id": cid,
                "document": doc,
                "metadata": meta,
                "score": score,
                "dense_distance": None,
                "bm25_score": score,
            }
            for cid, score, doc, meta in bm25_results[:top_k]
        ]

    # --- hybrid füzyon ----------------------------------------------------

    dense_id_order = [r[0] for r in dense_results]
    bm25_id_order = [r[0] for r in bm25_results]

    dense_lookup = {r[0]: r for r in dense_results}
    bm25_lookup = {r[0]: r for r in bm25_results}

    if alpha is None:
        # RRF
        fused = _reciprocal_rank_fusion([dense_id_order, bm25_id_order], rrf_k=settings.HYBRID_RRF_K)
    else:
        dense_scores = [1.0 - r[1] for r in dense_results]
        bm25_scores = [r[1] for r in bm25_results]
        dense_norm = dict(zip(dense_id_order, _normalize_scores(dense_scores)))
        bm25_norm = dict(zip(bm25_id_order, _normalize_scores(bm25_scores)))
        fused = {}
        for cid in set(dense_id_order) | set(bm25_id_order):
            fused[cid] = alpha * dense_norm.get(cid, 0.0) + (1 - alpha) * bm25_norm.get(cid, 0.0)

    ordered_ids = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]

    output: list[dict] = []
    for cid in ordered_ids:
        d = dense_lookup.get(cid)
        b = bm25_lookup.get(cid)
        doc = (d or b)[2]
        meta = (d or b)[3]
        output.append({
            "id": cid,
            "document": doc,
            "metadata": meta,
            "score": round(fused[cid], 6),
            "dense_distance": d[1] if d else None,
            "bm25_score": b[1] if b else None,
        })

    logger.info(
        "Hibrit arama tamamlandı",
        extra={
            "event": "hybrid_search_complete",
            "search_mode": search_mode,
            "alpha": alpha,
            "dense_n": len(dense_results),
            "sparse_n": len(bm25_results),
            "fused_n": len(output),
        },
    )
    return output
