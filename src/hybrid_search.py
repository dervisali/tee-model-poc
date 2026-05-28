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
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import settings
from src.embeddings import embed_query


logger = logging.getLogger(__name__)


BM25_INDEX_PATH = settings.CHROMA_DIR / "bm25_index.pkl"


def _file_cache_key(path: Path) -> tuple[int, int]:
    """Cheap freshness key for file-backed caches."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


# ---------------------------------------------------------------------------
# Dile-duyarlı tokenizasyon
# ---------------------------------------------------------------------------
# Türkçe yol:   küçük harf + Türkçe karakter regex'i, stemmer YOK.
# Fransızca yol: lowercase + apostrof bölme (élision) + stop-words filtresi +
#                Snowball French stemmer.
#
# Aktif dil settings.CORPUS_PRIMARY_LANGUAGE üzerinden gelir; _tokenize'a açıkça da
# verilebilir. Tokenizer indeks inşası ve sorgu zamanı arasında SİMETRİK
# olmalıdır; aksi halde BM25 yanıltıcı sonuçlar verir.

_TURKISH_TOKEN_RE = re.compile(r"[\wçğıöşüÇĞİÖŞÜ]+", re.UNICODE)

# Fransızca harfler + rakamlar; aksanlı karakterler korunur (Snowball aksanı bekler).
_FRENCH_TOKEN_RE = re.compile(r"[a-zàâäéèêëïîôöùûüœæç0-9]+", re.IGNORECASE)
# Düz ve eğri apostrof — élision sınırı.
_APOSTROPHE_RE = re.compile(r"[’']")

# Sık karşılaşılan Fransızca fonksiyonel sözcükler. BM25 için sinyalsizdir.
_FRENCH_STOPWORDS: frozenset[str] = frozenset({
    # Tanımlıklar
    "le", "la", "les", "un", "une", "des", "du", "de", "d",
    "au", "aux", "à",
    # Zamirler
    "je", "j", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "m", "te", "se", "s", "lui", "leur", "leurs",
    "ce", "c", "ça", "cet", "cette", "ces",
    "qui", "que", "qu", "quoi", "dont", "où",
    "y", "en",
    # İyelik
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "notre", "nos", "votre", "vos",
    # Bağlaçlar / olumsuzlama
    "et", "ou", "ni", "mais", "donc", "or", "car",
    "si", "ne", "n", "pas", "plus", "moins",
    # Edatlar
    "pour", "par", "sur", "sous", "dans", "avec", "sans", "vers",
    "entre", "chez", "depuis", "pendant",
    # Yardımcı fiil sık formları
    "est", "sont", "était", "étaient", "sera", "seront", "été", "être",
    "a", "ont", "avait", "avaient", "aura", "auront", "avoir", "eu",
    "fait", "font", "faire",
    # Niceleyiciler / işaret sözcükleri
    "tout", "tous", "toute", "toutes", "même", "autre", "autres",
    "très", "trop", "comme", "comment", "quand",
})


@lru_cache(maxsize=1)
def _get_french_stemmer():
    """Snowball French stemmer'ı tek seferlik yükler. nltk veri indirme gerektirmez."""
    from nltk.stem.snowball import FrenchStemmer  # type: ignore[import-untyped]
    return FrenchStemmer()


def _tokenize_tr(text: str) -> list[str]:
    """Türkçe: küçük harf + Türkçe karakter farkındalığı. Stemmer yok."""
    return [t.lower() for t in _TURKISH_TOKEN_RE.findall(text)]


def _tokenize_fr(text: str) -> list[str]:
    """
    Fransızca: élision için apostrofu boşluğa çevir, küçük harfle tokenize et,
    stop-word'leri filtrele ve Snowball French stemmer uygula. Aksanlar korunur.
    """
    lowered = _APOSTROPHE_RE.sub(" ", text.lower())
    raw_tokens = _FRENCH_TOKEN_RE.findall(lowered)
    stemmer = _get_french_stemmer()
    out: list[str] = []
    for tok in raw_tokens:
        if len(tok) <= 1:
            continue
        if tok in _FRENCH_STOPWORDS:
            continue
        out.append(stemmer.stem(tok))
    return out


def _tokenize(text: str, language: str | None = None) -> list[str]:
    """Aktif dilin tokenizer'ına yönlendirir. `language` None ise settings'ten alır."""
    lang = (language or settings.CORPUS_PRIMARY_LANGUAGE).lower()
    if lang == "fr":
        return _tokenize_fr(text)
    return _tokenize_tr(text)


def _strip_accents(s: str) -> str:
    """NFKD normalize + birleştirici işaret stripping. Yardımcı (şu an iç kullanım)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


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
        language: str | None = None,
    ) -> list[tuple[str, float, str, dict]]:
        """
        Sorgu için en alakalı top_k kaydı (id, score, document, metadata)
        olarak döndürür. Source filtresi metadata üzerinden uygulanır.

        `language` None ise settings.CORPUS_PRIMARY_LANGUAGE; tokenizer indeks ile
        simetrik olmalıdır.
        """
        if self.bm25 is None or not self.ids:
            return []
        query_tokens = _tokenize(query, language=language)
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
    @lru_cache(maxsize=1)
    def load(
        cls,
        path: Path = BM25_INDEX_PATH,
        cache_key: tuple[int, int] | None = None,
    ) -> "BM25Index | None":
        _ = cache_key
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

def rebuild_bm25_index_from_collection(language: str | None = None) -> BM25Index | None:
    """
    Aktif tee_children koleksiyonundan BM25 indeksini yeniden kurar.
    Ingestion'dan sonra çağrılmalıdır.

    `language` None ise settings.CORPUS_PRIMARY_LANGUAGE kullanılır. İndeks pickle'ı
    dile pinlenmez (search ile aynı setting'i okuyacak); ancak log'a yazılır.
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

    active_language = (language or settings.CORPUS_PRIMARY_LANGUAGE).lower()
    data = col.get(include=["documents", "metadatas"])
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]
    tokens = [_tokenize(doc, language=active_language) for doc in documents]

    index = BM25Index(ids=ids, tokens=tokens, documents=documents, metadatas=metadatas)
    index.save()
    try:
        BM25Index.load.cache_clear()
    except AttributeError:
        BM25Index.load.__func__.cache_clear()
    logger.info("BM25 indeksi inşa edildi: %d child, language=%s.", len(ids), active_language)
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
    started = time.perf_counter()
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    col = client.get_or_create_collection(
        name=settings.CHILD_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if col.count() == 0:
        return []

    embed_started = time.perf_counter()
    qvec = embed_query(query)
    embed_ms = (time.perf_counter() - embed_started) * 1000
    where = {"source": source_filter} if source_filter else None
    kwargs = {
        "query_embeddings": [qvec],
        "n_results": min(top_k, col.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    chroma_started = time.perf_counter()
    results = col.query(**kwargs)
    chroma_ms = (time.perf_counter() - chroma_started) * 1000
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    output = [(i, float(d), doc, meta) for i, d, doc, meta in zip(ids, dists, docs, metas)]
    logger.info(
        "Dense child search tamamlandı",
        extra={
            "event": "dense_child_search_complete",
            "top_k": top_k,
            "results": len(output),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "embedding_ms": round(embed_ms, 1),
            "chroma_query_ms": round(chroma_ms, 1),
        },
    )
    return output


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
    bm25_query: str | None = None,
) -> list[dict]:
    """
    BM25 + dense hibrit arama; child seviyesinde sıralı sonuç döndürür.

    search_mode = "dense"  : yalnızca yoğun
                  "sparse" : yalnızca BM25
                  "hybrid" : füzyon (alpha None ise RRF, değilse ağırlıklı)

    bm25_query : Cross-lingual çeviri varyantı. None ise BM25 yolu da `query`
                 ile çalışır. Türkçe sorgu + Fransızca korpus durumunda
                 retrieve_context burayı çeviri ile besler; dense yine
                 orijinal sorgu üzerinden gider (multilingual embedding).

    Döner — her öğe: {id, document, metadata, score, dense_distance, bm25_score}
    """
    fetch_k = top_k * over_fetch
    sparse_query = bm25_query if bm25_query is not None else query

    dense_results = []
    bm25_results: list[tuple[str, float, str, dict]] = []
    dense_ms = 0.0
    bm25_load_ms = 0.0
    bm25_search_ms = 0.0

    if search_mode == "hybrid":
        # Run dense embedding concurrently with BM25 index load (independent I/O).
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_started = time.perf_counter()
            dense_future = executor.submit(_dense_search_children, query, fetch_k, source_filter)
            bm25_load_started = time.perf_counter()
            bm25_index = BM25Index.load(cache_key=_file_cache_key(BM25_INDEX_PATH))
            bm25_load_ms = (time.perf_counter() - bm25_load_started) * 1000
            dense_results = dense_future.result()
            dense_ms = (time.perf_counter() - dense_started) * 1000
        if bm25_index is None:
            logger.warning("BM25 indeksi yok; sparse atlanıyor. Ingestion'ı yeniden çalıştırın.")
        else:
            bm25_search_started = time.perf_counter()
            bm25_results = bm25_index.search(sparse_query, fetch_k, source_filter=source_filter)
            bm25_search_ms = (time.perf_counter() - bm25_search_started) * 1000
    elif search_mode == "dense":
        dense_started = time.perf_counter()
        dense_results = _dense_search_children(query, fetch_k, source_filter)
        dense_ms = (time.perf_counter() - dense_started) * 1000
    elif search_mode == "sparse":
        bm25_load_started = time.perf_counter()
        bm25_index = BM25Index.load(cache_key=_file_cache_key(BM25_INDEX_PATH))
        bm25_load_ms = (time.perf_counter() - bm25_load_started) * 1000
        if bm25_index is None:
            logger.warning("BM25 indeksi yok; sparse atlanıyor. Ingestion'ı yeniden çalıştırın.")
        else:
            bm25_search_started = time.perf_counter()
            bm25_results = bm25_index.search(sparse_query, fetch_k, source_filter=source_filter)
            bm25_search_ms = (time.perf_counter() - bm25_search_started) * 1000

    if search_mode == "dense":
        output = [
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
        logger.info(
            "Hibrit arama tamamlandı",
            extra={
                "event": "hybrid_search_complete",
                "search_mode": search_mode,
                "alpha": alpha,
                "dense_n": len(dense_results),
                "sparse_n": 0,
                "fused_n": len(output),
                "dense_ms": round(dense_ms, 1),
            },
        )
        return output

    if search_mode == "sparse":
        output = [
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
        logger.info(
            "Hibrit arama tamamlandı",
            extra={
                "event": "hybrid_search_complete",
                "search_mode": search_mode,
                "alpha": alpha,
                "dense_n": 0,
                "sparse_n": len(bm25_results),
                "fused_n": len(output),
                "bm25_load_ms": round(bm25_load_ms, 1),
                "bm25_search_ms": round(bm25_search_ms, 1),
            },
        )
        return output

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
            "dense_ms": round(dense_ms, 1),
            "bm25_load_ms": round(bm25_load_ms, 1),
            "bm25_search_ms": round(bm25_search_ms, 1),
        },
    )
    return output
