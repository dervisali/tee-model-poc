"""
RAG ingestion pipeline for TEE-Model POC — Parent Document Retrieval edition.

Two-level chunking strategy:
  - Parent chunks (600-800 chars): stored in chroma_db/parents.json for context lookup
  - Child chunks  (150-200 chars): embedded and stored in ChromaDB collection 'tee_children'

Each child carries a parent_id reference. Retrieval embeds the query against children,
then returns the corresponding parent text to the LLM for richer context.
"""

import os
import re
import json
import time
import logging
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
PARENTS_JSON = CHROMA_DIR / "parents.json"

# Collection names
CHILD_COLLECTION_NAME = "tee_children"
COLLECTION_NAME = CHILD_COLLECTION_NAME          # backward-compat alias used by tests

# Legacy chunk constants — kept for backward compat (tests import them directly)
MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 150

# PDR chunk constants
PARENT_MAX_CHARS = 800
PARENT_MIN_CHARS = 300
CHILD_MAX_CHARS = 200
CHILD_MIN_CHARS = 50


# ---------------------------------------------------------------------------
# Client / collection helpers
# ---------------------------------------------------------------------------

def _get_genai_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY ortam değişkeni bulunamadı. Lütfen .env dosyasını kontrol edin."
        )
    return genai.Client(api_key=api_key)


def _get_child_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHILD_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _get_chroma_collection() -> chromadb.Collection:
    """Backward-compat alias — returns the child (embedded) collection."""
    return _get_child_collection()


# ---------------------------------------------------------------------------
# Parent JSON store helpers
# ---------------------------------------------------------------------------

def _load_parents() -> dict:
    """Load the parent lookup table from disk. Returns {} if not yet created."""
    if PARENTS_JSON.exists():
        return json.loads(PARENTS_JSON.read_text(encoding="utf-8"))
    return {}


def _save_parents(parents: dict) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    PARENTS_JSON.write_text(
        json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Chunking — legacy (kept for backward compat / tests)
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str) -> list[str]:
    """
    Original single-level chunker. Kept for backward compatibility and tests.
    New ingestion code uses _split_into_parent_chunks / _split_into_child_chunks.
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

    sentence_end = re.compile(r'(?<=[.!?])\s+')
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
# Chunking — PDR two-level
# ---------------------------------------------------------------------------

def _split_into_parent_chunks(text: str) -> list[str]:
    """
    Split text into large parent chunks (target PARENT_MAX_CHARS).
    No overlap — each parent is an independent context window for the LLM.
    """
    sentence_end = re.compile(r'(?<=[.!?])\s+')
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Merge short paragraphs up to PARENT_MAX_CHARS
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

    # Split any oversized parents at sentence boundaries
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
    Further split a parent chunk into small child chunks (target CHILD_MAX_CHARS).
    Children are what get embedded; they carry a parent_id back-reference.
    """
    sentence_end = re.compile(r'(?<=[.!?])\s+')
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
            # Single sentence longer than CHILD_MAX_CHARS — split at word boundary
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

    # Merge tiny tails into the previous child
    result: list[str] = []
    for child in children:
        if result and len(child) < CHILD_MIN_CHARS:
            result[-1] += " " + child
        else:
            result.append(child)

    return [c for c in result if c.strip()]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_chunk(client: genai.Client, chunk_text: str) -> list[float]:
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk_text,
            config={"task_type": "retrieval_document"},
        )
        return response.embeddings[0].values
    except Exception as exc:
        raise RuntimeError(
            f"Embedding API hatası: {exc}\nChunk (ilk 80 karakter): {chunk_text[:80]}"
        ) from exc


# ---------------------------------------------------------------------------
# Source file loading
# ---------------------------------------------------------------------------

def _load_source_files() -> list[dict]:
    import sys as _sys
    if str(BASE_DIR) not in _sys.path:
        _sys.path.insert(0, str(BASE_DIR))
    from src.anonymizer import anonymize_text

    explicit_path = BASE_DIR / "data" / "explicit_mevzuat.txt"
    raw_tacit_path = BASE_DIR / "data" / "tacit_interview_raw.txt"
    clean_tacit_path = BASE_DIR / "processed" / "tacit_interview_clean.txt"

    sources = []

    if not explicit_path.exists():
        raise FileNotFoundError(f"Mevzuat dosyası bulunamadı: {explicit_path}")
    sources.append({
        "source": "explicit",
        "filename": explicit_path.name,
        "text": explicit_path.read_text(encoding="utf-8"),
    })
    logger.info("Mevzuat dosyası yüklendi: %s", explicit_path.name)

    if not clean_tacit_path.exists():
        logger.info("Anonimleştirilmiş transkript bulunamadı, oluşturuluyor...")
        if not raw_tacit_path.exists():
            raise FileNotFoundError(f"Ham transkript bulunamadı: {raw_tacit_path}")
        raw_text = raw_tacit_path.read_text(encoding="utf-8")
        result = anonymize_text(raw_text)
        clean_tacit_path.parent.mkdir(parents=True, exist_ok=True)
        clean_tacit_path.write_text(result["anonymized_text"], encoding="utf-8")
        logger.info("Transkript anonimleştirildi: %d kayıt maskelendi.", len(result["mask_log"]))
    else:
        logger.info("Mevcut anonimleştirilmiş transkript kullanılıyor: %s", clean_tacit_path.name)

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
# Main ingestion — two-collection PDR strategy
# ---------------------------------------------------------------------------

def run_ingestion() -> dict:
    """
    Full PDR ingestion pipeline.

    For each source document:
      1. Split into parent chunks (600-800 chars) — stored in parents.json
      2. Split each parent into child chunks (150-200 chars) — embedded into tee_children
      3. Each child metadata includes parent_id for lookup at retrieval time

    Returns dict with total_chunks, explicit_chunks, tacit_chunks, collection_size.
    """
    client = _get_genai_client()
    child_col = _get_child_collection()

    sources = _load_source_files()
    parents: dict = {}

    child_ids: list[str] = []
    child_embeddings: list[list[float]] = []
    child_documents: list[str] = []
    child_metadatas: list[dict] = []

    explicit_children = 0
    tacit_children = 0

    for source_info in sources:
        stem = Path(source_info["filename"]).stem
        parent_chunks = _split_into_parent_chunks(source_info["text"])

        logger.info(
            "'%s' için %d parent chunk oluşturuldu.",
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

                logger.info(
                    "Embedding: %s parent=%d child=%d/%d",
                    source_info["source"], p_idx, c_idx + 1, len(child_chunks),
                )

                try:
                    vector = _embed_chunk(client, child_text)
                except RuntimeError as exc:
                    logger.error("Child chunk atlandı: %s", exc)
                    continue

                child_ids.append(child_id)
                child_embeddings.append(vector)
                child_documents.append(child_text)
                child_metadatas.append({
                    "source": source_info["source"],
                    "filename": source_info["filename"],
                    "child_index": c_idx,
                    "parent_id": parent_id,
                    "char_count": len(child_text),
                })

                if source_info["source"] == "explicit":
                    explicit_children += 1
                else:
                    tacit_children += 1

                time.sleep(1)  # Rate limit: 5 req/min on free tier

    _save_parents(parents)
    logger.info("parents.json kaydedildi: %d parent chunk.", len(parents))

    if child_ids:
        child_col.upsert(
            ids=child_ids,
            embeddings=child_embeddings,
            documents=child_documents,
            metadatas=child_metadatas,
        )
        logger.info("tee_children'a %d child chunk yazıldı.", len(child_ids))
    else:
        logger.warning("Yazılacak child chunk bulunamadı.")

    collection_size = child_col.count()
    summary = {
        "total_chunks": explicit_children + tacit_children,
        "explicit_chunks": explicit_children,
        "tacit_chunks": tacit_children,
        "collection_size": collection_size,
    }

    logger.info(
        "Yükleme özeti — Toplam: %d | Mevzuat: %d | Tacit: %d | DB boyutu: %d",
        summary["total_chunks"], summary["explicit_chunks"],
        summary["tacit_chunks"], summary["collection_size"],
    )

    return summary


def clear_and_reingest() -> dict:
    """Drop tee_children collection and parents.json, then re-run full ingestion."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection(CHILD_COLLECTION_NAME)
        logger.info("tee_children koleksiyonu silindi.")
    except Exception as exc:
        logger.warning("Koleksiyon silinemedi: %s", exc)

    if PARENTS_JSON.exists():
        PARENTS_JSON.unlink()
        logger.info("parents.json silindi.")

    return run_ingestion()


def get_ingestion_status() -> dict:
    """Return current collection metadata without re-ingesting."""
    try:
        col = _get_child_collection()
        total = col.count()
        if total == 0:
            return {"collection_size": 0, "explicit_chunks": 0, "tacit_chunks": 0}

        explicit_results = col.get(where={"source": "explicit"})
        tacit_results = col.get(where={"source": "tacit"})

        return {
            "collection_size": total,
            "explicit_chunks": len(explicit_results["ids"]),
            "tacit_chunks": len(tacit_results["ids"]),
        }
    except Exception as exc:
        logger.error("Durum sorgulanamadı: %s", exc)
        return {"collection_size": 0, "explicit_chunks": 0, "tacit_chunks": 0}


def insert_new_document(filepath: str, source_type: str) -> dict:
    """
    Insert a single new document using the PDR two-level strategy.
    Appends to the existing tee_children collection and parents.json.
    Does NOT wipe existing data.
    """
    import sys as _sys
    if str(BASE_DIR) not in _sys.path:
        _sys.path.insert(0, str(BASE_DIR))
    from src.anonymizer import anonymize_text

    doc_path = Path(filepath)
    if not doc_path.exists():
        raise FileNotFoundError(f"Belge bulunamadı: {doc_path}")

    raw_text = doc_path.read_text(encoding="utf-8")
    logger.info("Yeni belge okundu: %s (%d karakter)", doc_path.name, len(raw_text))

    anon_result = anonymize_text(raw_text)
    clean_text = anon_result["anonymized_text"]
    logger.info("Anonimleştirme: %d kayıt maskelendi.", len(anon_result["mask_log"]))

    stem = doc_path.stem
    parent_chunks = _split_into_parent_chunks(clean_text)
    logger.info("%d parent chunk oluşturuldu: %s", len(parent_chunks), doc_path.name)

    client = _get_genai_client()
    child_col = _get_child_collection()
    parents = _load_parents()

    child_ids: list[str] = []
    child_embeddings: list[list[float]] = []
    child_documents: list[str] = []
    child_metadatas: list[dict] = []

    for p_idx, parent_text in enumerate(parent_chunks):
        parent_id = f"{source_type}_{stem}_p{p_idx}"

        parents[parent_id] = {
            "text": parent_text,
            "source": source_type,
            "filename": doc_path.name,
            "parent_index": p_idx,
            "char_count": len(parent_text),
        }

        child_chunks = _split_into_child_chunks(parent_text)

        for c_idx, child_text in enumerate(child_chunks):
            child_id = f"{source_type}_{stem}_p{p_idx}_c{c_idx}"

            logger.info(
                "Embedding: %s parent=%d child=%d/%d",
                doc_path.name, p_idx, c_idx + 1, len(child_chunks),
            )

            try:
                vector = _embed_chunk(client, child_text)
            except RuntimeError as exc:
                logger.error("Child chunk atlandı: %s", exc)
                continue

            child_ids.append(child_id)
            child_embeddings.append(vector)
            child_documents.append(child_text)
            child_metadatas.append({
                "source": source_type,
                "filename": doc_path.name,
                "child_index": c_idx,
                "parent_id": parent_id,
                "char_count": len(child_text),
            })

            time.sleep(1)

    _save_parents(parents)

    if child_ids:
        child_col.upsert(
            ids=child_ids,
            embeddings=child_embeddings,
            documents=child_documents,
            metadatas=child_metadatas,
        )
        logger.info("tee_children'a %d yeni child chunk eklendi.", len(child_ids))
    else:
        logger.warning("Hiç child chunk eklenemedi.")

    collection_size = child_col.count()
    summary = {
        "new_chunks": len(child_ids),
        "filename": doc_path.name,
        "collection_size": collection_size,
    }

    logger.info(
        "Ekleme özeti — Yeni child: %d | Toplam koleksiyon: %d",
        summary["new_chunks"], summary["collection_size"],
    )

    return summary


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    summary = run_ingestion()
    print("\nYükleme Özeti:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
