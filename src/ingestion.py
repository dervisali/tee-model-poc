"""
TEE-Model ingestion boru hattı — Parent Document Retrieval (PDR) baskısı.

İki seviyeli chunklama:
  - Parent (üst) parçaları (300-800 karakter): chroma_db/parents.json içine
    yazılır; LLM bağlamı olarak kullanılır.
  - Child (alt) parçaları (50-200 karakter): multilingual-e5-large ile gömülür
    ve ChromaDB 'tee_children' koleksiyonuna yazılır.

Her child kaydı, parent_id geri-referansı taşır. Retrieval, child seviyesinde
arama yapar; LLM'e ise parent metni gönderilir (daha geniş bağlam, daha iyi
yanıt).

Bu modül artık Google Gemini API'sine bağlı değildir; tamamen yerel
sentence-transformers gömme modeli üzerinden çalışır.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import chromadb

from src.config import settings
from src.embeddings import embed_passages, embedding_dimension


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geriye uyumluluk takma adları (testler bunları doğrudan import ediyor)
# ---------------------------------------------------------------------------
BASE_DIR = settings.BASE_DIR
CHROMA_DIR = settings.CHROMA_DIR
PARENTS_JSON = CHROMA_DIR / "parents.json"

CHILD_COLLECTION_NAME = settings.CHILD_COLLECTION_NAME
COLLECTION_NAME = CHILD_COLLECTION_NAME

# Eski tek-seviyeli chunker sabitleri — testler bunlara dayanıyor
MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 150

# PDR sabitleri — config'den okunur
PARENT_MAX_CHARS = settings.CHUNK_SIZE_PARENT_MAX
PARENT_MIN_CHARS = settings.CHUNK_SIZE_PARENT_MIN
CHILD_MAX_CHARS = settings.CHUNK_SIZE_CHILD
CHILD_MIN_CHARS = settings.CHUNK_SIZE_CHILD_MIN


# ---------------------------------------------------------------------------
# ChromaDB yardımcıları
# ---------------------------------------------------------------------------

def _get_child_collection() -> chromadb.Collection:
    """Aktif child koleksiyonunu döndürür (yoksa oluşturur)."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHILD_COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dim": embedding_dimension(),
        },
    )


def _get_chroma_collection() -> chromadb.Collection:
    """Eski adlandırma takma adı — child koleksiyonunu döner."""
    return _get_child_collection()


# ---------------------------------------------------------------------------
# Parent JSON deposu
# ---------------------------------------------------------------------------

def _load_parents() -> dict:
    """parents.json dosyasını yükler; yoksa boş sözlük döner."""
    if PARENTS_JSON.exists():
        return json.loads(PARENTS_JSON.read_text(encoding="utf-8"))
    return {}


def _save_parents(parents: dict) -> None:
    """parents sözlüğünü atomik olarak diske yazar."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PARENTS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PARENTS_JSON)


# ---------------------------------------------------------------------------
# Eski tek seviyeli chunker (geriye uyumluluk + testler için korunur)
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str) -> list[str]:
    """
    Tek seviyeli chunker — yalnızca testler ve geriye uyumluluk içindir.
    Yeni ingestion kodu PDR ikili stratejisini kullanır.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    merged: list[str] = []
    buffer = ""
    for para in raw_paragraphs:
        if buffer:
            buffer += " " + para
        else:
            buffer = para
        if len(buffer) >= MIN_CHUNK_CHARS:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] += " " + buffer
        else:
            merged.append(buffer)

    sentence_end = re.compile(r"(?<=[.!?])\s+")
    final_chunks: list[str] = []
    for chunk in merged:
        if len(chunk) <= MAX_CHUNK_CHARS:
            final_chunks.append(chunk)
        else:
            sentences = sentence_end.split(chunk)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 <= MAX_CHUNK_CHARS:
                    current = (current + " " + sentence).strip() if current else sentence
                else:
                    if current:
                        final_chunks.append(current)
                    current = sentence
            if current:
                final_chunks.append(current)

    result: list[str] = []
    for chunk in final_chunks:
        if chunk.strip():
            if result and len(chunk) < MIN_CHUNK_CHARS:
                result[-1] += " " + chunk
            else:
                result.append(chunk)

    if OVERLAP_CHARS > 0 and len(result) > 1:
        overlapped = [result[0]]
        for i in range(1, len(result)):
            prev_tail = result[i - 1][-OVERLAP_CHARS:]
            overlapped.append(prev_tail + " " + result[i])
        result = overlapped

    return result


# ---------------------------------------------------------------------------
# PDR iki seviyeli chunker
# ---------------------------------------------------------------------------

def _split_into_parent_chunks(text: str) -> list[str]:
    """
    Metni büyük parent parçalarına ayırır (~PARENT_MAX_CHARS hedef).
    Parent'lar arasında örtüşme yoktur; her biri bağımsız bir LLM bağlam
    penceresidir.
    """
    sentence_end = re.compile(r"(?<=[.!?])\s+")
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    merged: list[str] = []
    buffer = ""
    for para in raw_paragraphs:
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if len(candidate) <= PARENT_MAX_CHARS:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = para
    if buffer:
        if merged and len(buffer) < PARENT_MIN_CHARS:
            merged[-1] += "\n\n" + buffer
        else:
            merged.append(buffer)

    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= PARENT_MAX_CHARS:
            final.append(chunk)
        else:
            sentences = sentence_end.split(chunk)
            current = ""
            for s in sentences:
                candidate = (current + " " + s).strip() if current else s
                if len(candidate) <= PARENT_MAX_CHARS:
                    current = candidate
                else:
                    if current:
                        final.append(current)
                    current = s
            if current:
                final.append(current)

    return [c for c in final if c.strip()]


def _split_into_child_chunks(parent_text: str) -> list[str]:
    """
    Parent parçasını gömme için küçük child parçalarına ayırır.
    Her child, parent_id geri-referansını metadata'da taşır.
    """
    sentence_end = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_end.split(parent_text)

    children: list[str] = []
    current = ""
    for s in sentences:
        candidate = (current + " " + s).strip() if current else s
        if len(candidate) <= CHILD_MAX_CHARS:
            current = candidate
        else:
            if current:
                children.append(current)
            if len(s) > CHILD_MAX_CHARS:
                words = s.split()
                sub = ""
                for w in words:
                    sub_cand = (sub + " " + w).strip() if sub else w
                    if len(sub_cand) <= CHILD_MAX_CHARS:
                        sub = sub_cand
                    else:
                        if sub:
                            children.append(sub)
                        sub = w
                current = sub
            else:
                current = s
    if current:
        children.append(current)

    result: list[str] = []
    for child in children:
        if result and len(child) < CHILD_MIN_CHARS:
            result[-1] += " " + child
        else:
            result.append(child)

    return [c for c in result if c.strip()]


# ---------------------------------------------------------------------------
# Kaynak dosya yükleme
# ---------------------------------------------------------------------------

def _load_source_files() -> list[dict]:
    """
    Mevzuat ve anonimleştirilmiş tacit transkriptini yükler.
    Tacit dosyası yoksa, ham transkripten anonimleştirme ile üretir.
    """
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from src.anonymizer import anonymize_text  # noqa: WPS433

    explicit_path = settings.DATA_DIR / "explicit_mevzuat.txt"
    raw_tacit_path = settings.DATA_DIR / "tacit_interview_raw.txt"
    clean_tacit_path = settings.PROCESSED_DIR / "tacit_interview_clean.txt"

    sources: list[dict] = []

    if not explicit_path.exists():
        raise FileNotFoundError(f"Mevzuat dosyası bulunamadı: {explicit_path}")
    sources.append({
        "source": "explicit",
        "filename": explicit_path.name,
        "text": explicit_path.read_text(encoding="utf-8"),
    })

    if not clean_tacit_path.exists():
        if not raw_tacit_path.exists():
            raise FileNotFoundError(f"Ham transkript bulunamadı: {raw_tacit_path}")
        raw_text = raw_tacit_path.read_text(encoding="utf-8")
        result = anonymize_text(raw_text)
        clean_tacit_path.parent.mkdir(parents=True, exist_ok=True)
        clean_tacit_path.write_text(result["anonymized_text"], encoding="utf-8")
        logger.info("Transkript anonimleştirildi: %d kayıt maskelendi.", len(result["mask_log"]))

    tacit_text = clean_tacit_path.read_text(encoding="utf-8")
    if "\n---\n" in tacit_text:
        tacit_text = tacit_text.split("\n---\n", 1)[1].strip()
    sources.append({
        "source": "tacit",
        "filename": clean_tacit_path.name,
        "text": tacit_text,
    })

    return sources


# ---------------------------------------------------------------------------
# Ana ingestion
# ---------------------------------------------------------------------------

def run_ingestion() -> dict:
    """
    Tam PDR ingestion boru hattı.

    Her kaynak için:
      1. Parent parçalarına böl (parents.json'a yaz).
      2. Her parent'ı child'lara böl.
      3. Tüm child'ları toplu olarak (batch) gömele ve ChromaDB'ye yaz.

    Yerel e5-large modeli kullanıldığı için gömme adımı toplu yürütülür;
    eski Gemini kodundaki sleep(1) rate-limit gecikmesi tamamen kaldırılmıştır.
    """
    child_col = _get_child_collection()
    sources = _load_source_files()

    parents: dict = {}
    pending_texts: list[str] = []
    pending_ids: list[str] = []
    pending_metas: list[dict] = []
    explicit_children = 0
    tacit_children = 0

    for source_info in sources:
        stem = Path(source_info["filename"]).stem
        parent_chunks = _split_into_parent_chunks(source_info["text"])
        logger.info(
            "%s için %d parent chunk üretildi.",
            source_info["source"], len(parent_chunks),
        )

        for p_idx, parent_text in enumerate(parent_chunks):
            parent_id = f"{stem}_p{p_idx}"
            parents[parent_id] = {
                "text": parent_text,
                "source": source_info["source"],
                "filename": source_info["filename"],
                "parent_index": p_idx,
                "char_count": len(parent_text),
            }

            child_chunks = _split_into_child_chunks(parent_text)
            for c_idx, child_text in enumerate(child_chunks):
                child_id = f"{stem}_p{p_idx}_c{c_idx}"
                pending_texts.append(child_text)
                pending_ids.append(child_id)
                pending_metas.append({
                    "source": source_info["source"],
                    "filename": source_info["filename"],
                    "child_index": c_idx,
                    "parent_id": parent_id,
                    "char_count": len(child_text),
                    "chunking_strategy": settings.CHUNKING_STRATEGY,
                })
                if source_info["source"] == "explicit":
                    explicit_children += 1
                else:
                    tacit_children += 1

    _save_parents(parents)
    logger.info("parents.json kaydedildi: %d parent.", len(parents))

    if pending_ids:
        logger.info("Toplu gömme başlıyor: %d child.", len(pending_ids))
        embeddings = embed_passages(pending_texts)
        child_col.upsert(
            ids=pending_ids,
            embeddings=embeddings,
            documents=pending_texts,
            metadatas=pending_metas,
        )
        logger.info("ChromaDB'ye yazıldı: %d child.", len(pending_ids))

    # Hibrit aramayı destekleyen BM25 indeksi: dense yazımı bittikten sonra
    # her zaman yeniden inşa edilir (tutarlılık için).
    from src.hybrid_search import rebuild_bm25_index_from_collection
    rebuild_bm25_index_from_collection()

    summary = {
        "total_chunks": explicit_children + tacit_children,
        "explicit_chunks": explicit_children,
        "tacit_chunks": tacit_children,
        "collection_size": child_col.count(),
    }
    logger.info(
        "Ingestion özeti: toplam=%d explicit=%d tacit=%d db=%d",
        summary["total_chunks"], summary["explicit_chunks"],
        summary["tacit_chunks"], summary["collection_size"],
    )
    return summary


def clear_and_reingest() -> dict:
    """tee_children koleksiyonunu, parents.json'u ve BM25 indeksini silip baştan ingest eder."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection(CHILD_COLLECTION_NAME)
    except Exception as exc:
        logger.warning("Koleksiyon silinemedi: %s", exc)

    if PARENTS_JSON.exists():
        PARENTS_JSON.unlink()
        logger.info("parents.json silindi.")

    from src.hybrid_search import BM25_INDEX_PATH
    if BM25_INDEX_PATH.exists():
        BM25_INDEX_PATH.unlink()
        logger.info("BM25 indeksi silindi.")

    return run_ingestion()


def get_ingestion_status() -> dict:
    """Mevcut koleksiyonun boyutunu ve kaynak dağılımını döner."""
    try:
        col = _get_child_collection()
        total = col.count()
        if total == 0:
            return {"collection_size": 0, "explicit_chunks": 0, "tacit_chunks": 0}
        explicit = col.get(where={"source": "explicit"})
        tacit = col.get(where={"source": "tacit"})
        return {
            "collection_size": total,
            "explicit_chunks": len(explicit["ids"]),
            "tacit_chunks": len(tacit["ids"]),
        }
    except Exception as exc:
        logger.error("Durum sorgulanamadı: %s", exc)
        return {"collection_size": 0, "explicit_chunks": 0, "tacit_chunks": 0}


def insert_new_document(filepath: str, source_type: str) -> dict:
    """
    Mevcut koleksiyona tek bir yeni belge ekler. Var olan veriyi silmez.

    Mission Phase 1.2 buraya yeni `chunking_strategy` parametresi eklenecek;
    şimdilik mevcut paragraph davranışı korunmaktadır.
    """
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from src.anonymizer import anonymize_text  # noqa: WPS433

    doc_path = Path(filepath)
    if not doc_path.exists():
        raise FileNotFoundError(f"Belge bulunamadı: {doc_path}")

    raw_text = doc_path.read_text(encoding="utf-8")
    anon = anonymize_text(raw_text)
    clean_text = anon["anonymized_text"]
    logger.info(
        "Yeni belge alındı: %s — %d karakter, %d PII maskelendi.",
        doc_path.name, len(raw_text), len(anon["mask_log"]),
    )

    stem = doc_path.stem
    parent_chunks = _split_into_parent_chunks(clean_text)

    child_col = _get_child_collection()
    parents = _load_parents()

    pending_texts: list[str] = []
    pending_ids: list[str] = []
    pending_metas: list[dict] = []

    for p_idx, parent_text in enumerate(parent_chunks):
        parent_id = f"{source_type}_{stem}_p{p_idx}"
        parents[parent_id] = {
            "text": parent_text,
            "source": source_type,
            "filename": doc_path.name,
            "parent_index": p_idx,
            "char_count": len(parent_text),
        }
        for c_idx, child_text in enumerate(_split_into_child_chunks(parent_text)):
            child_id = f"{source_type}_{stem}_p{p_idx}_c{c_idx}"
            pending_texts.append(child_text)
            pending_ids.append(child_id)
            pending_metas.append({
                "source": source_type,
                "filename": doc_path.name,
                "child_index": c_idx,
                "parent_id": parent_id,
                "char_count": len(child_text),
                "chunking_strategy": settings.CHUNKING_STRATEGY,
            })

    _save_parents(parents)

    if pending_ids:
        embeddings = embed_passages(pending_texts)
        child_col.upsert(
            ids=pending_ids,
            embeddings=embeddings,
            documents=pending_texts,
            metadatas=pending_metas,
        )

    # BM25 indeksi her insert sonrası yeniden inşa edilir.
    from src.hybrid_search import rebuild_bm25_index_from_collection
    rebuild_bm25_index_from_collection()

    summary = {
        "new_chunks": len(pending_ids),
        "filename": doc_path.name,
        "collection_size": child_col.count(),
    }
    logger.info("Ekleme özeti: new=%d total=%d", summary["new_chunks"], summary["collection_size"])
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run_ingestion()
    print("\nYükleme Özeti:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
