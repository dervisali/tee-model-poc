"""
RAG ingestion pipeline for TEE-Model POC.

Loads explicit regulation document and anonymized tacit interview transcript,
chunks them, generates embeddings via gemini-embedding-001, and stores
everything in a persistent ChromaDB collection named 'tee_knowledge_base'.
"""

import os
import re
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
COLLECTION_NAME = "tee_knowledge_base"

MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 800   # Lowered from 1200: dense regulatory paragraphs now split
OVERLAP_CHARS = 150     # into single-topic chunks, improving embedding precision.

# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def _get_genai_client() -> genai.Client:
    """Initialise and return the google-genai client using GOOGLE_API_KEY."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY ortam değişkeni bulunamadı. "
            "Lütfen .env dosyasını kontrol edin."
        )
    return genai.Client(api_key=api_key)


def _get_chroma_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str) -> list[str]:
    """
    Split text into semantically coherent chunks.

    Primary boundary: double newline (paragraph break).
    - Chunks shorter than MIN_CHUNK_CHARS are merged with the next one.
    - Chunks longer than MAX_CHUNK_CHARS are split at the nearest sentence
      boundary (period/exclamation/question mark followed by whitespace).

    Parameters
    ----------
    text : str
        Full document text.

    Returns
    -------
    list[str]
        List of non-empty chunk strings.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Merge short paragraphs
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

    # Split oversized chunks at sentence boundaries
    final_chunks: list[str] = []
    sentence_end = re.compile(r'(?<=[.!?])\s+')

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

    # Post-process: merge any sub-minimum chunks into their predecessor
    result: list[str] = []
    for chunk in final_chunks:
        if chunk.strip():
            if result and len(chunk) < MIN_CHUNK_CHARS:
                result[-1] += " " + chunk
            else:
                result.append(chunk)

    # Add overlap: prepend the tail of the previous chunk to each chunk
    if OVERLAP_CHARS > 0 and len(result) > 1:
        overlapped = [result[0]]
        for i in range(1, len(result)):
            prev_tail = result[i - 1][-OVERLAP_CHARS:]
            overlapped.append(prev_tail + " " + result[i])
        result = overlapped

    return result


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_chunk(client: genai.Client, chunk_text: str) -> list[float]:
    """
    Generate an embedding vector for a single text chunk.

    Uses gemini-embedding-001 with task_type='retrieval_document'.
    Raises RuntimeError on API failure.

    Parameters
    ----------
    client : genai.Client
        Authenticated google-genai client.
    chunk_text : str
        Text to embed.

    Returns
    -------
    list[float]
        Embedding vector.
    """
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
# Main ingestion
# ---------------------------------------------------------------------------

def _load_source_files() -> list[dict]:
    """
    Load explicit regulation document and anonymized tacit transcript.

    If the anonymized transcript does not yet exist, runs the anonymizer first.

    Returns
    -------
    list[dict]
        Each entry has keys "source", "filename", "text".
    """
    import sys as _sys
    if str(BASE_DIR) not in _sys.path:
        _sys.path.insert(0, str(BASE_DIR))
    from src.anonymizer import anonymize_text  # local import to avoid circular deps

    explicit_path = BASE_DIR / "data" / "explicit_mevzuat.txt"
    raw_tacit_path = BASE_DIR / "data" / "tacit_interview_raw.txt"
    clean_tacit_path = BASE_DIR / "processed" / "tacit_interview_clean.txt"

    sources = []

    # --- Explicit document ---
    if not explicit_path.exists():
        raise FileNotFoundError(f"Mevzuat dosyası bulunamadı: {explicit_path}")
    sources.append({
        "source": "explicit",
        "filename": explicit_path.name,
        "text": explicit_path.read_text(encoding="utf-8"),
    })
    logger.info("Mevzuat dosyası yüklendi: %s", explicit_path.name)

    # --- Tacit transcript (anonymize on demand) ---
    if not clean_tacit_path.exists():
        logger.info("Anonimleştirilmiş transkript bulunamadı, oluşturuluyor...")
        if not raw_tacit_path.exists():
            raise FileNotFoundError(f"Ham transkript bulunamadı: {raw_tacit_path}")
        raw_text = raw_tacit_path.read_text(encoding="utf-8")
        result = anonymize_text(raw_text)
        clean_tacit_path.parent.mkdir(parents=True, exist_ok=True)
        clean_tacit_path.write_text(result["anonymized_text"], encoding="utf-8")
        logger.info(
            "Transkript anonimleştirildi ve kaydedildi: %d kayıt maskelendi.",
            len(result["mask_log"]),
        )
    else:
        logger.info("Mevcut anonimleştirilmiş transkript kullanılıyor: %s", clean_tacit_path.name)

    tacit_text = clean_tacit_path.read_text(encoding="utf-8")
    # Strip document header: everything up to and including the first "---" separator
    if "\n---\n" in tacit_text:
        tacit_text = tacit_text.split("\n---\n", 1)[1].strip()
    sources.append({
        "source": "tacit",
        "filename": clean_tacit_path.name,
        "text": tacit_text,
    })

    return sources


def run_ingestion() -> dict:
    """
    Full RAG ingestion pipeline.

    Steps:
      1. Load source files (explicit + tacit).
      2. Chunk each document.
      3. Generate embeddings (with 1-second sleep between calls for rate limit).
      4. Upsert chunks + metadata + vectors into ChromaDB.
      5. Return a summary dict.

    Returns
    -------
    dict with keys:
        - "total_chunks": int
        - "explicit_chunks": int
        - "tacit_chunks": int
        - "collection_size": int
    """
    client = _get_genai_client()
    collection = _get_chroma_collection()

    sources = _load_source_files()

    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    explicit_count = 0
    tacit_count = 0
    global_idx = 0

    for source_info in sources:
        chunks = _split_into_chunks(source_info["text"])
        logger.info(
            "'%s' kaynağı için %d chunk oluşturuldu.",
            source_info["source"],
            len(chunks),
        )

        for local_idx, chunk_text in enumerate(chunks):
            chunk_id = f"{source_info['source']}_{local_idx}"
            metadata = {
                "source": source_info["source"],
                "filename": source_info["filename"],
                "chunk_index": local_idx,
                "char_count": len(chunk_text),
            }

            logger.info(
                "Embedding oluşturuluyor: kaynak=%s chunk=%d/%d",
                source_info["source"],
                local_idx + 1,
                len(chunks),
            )

            try:
                vector = _embed_chunk(client, chunk_text)
            except RuntimeError as exc:
                logger.error("Chunk atlandı (embedding hatası): %s", exc)
                continue

            all_ids.append(chunk_id)
            all_embeddings.append(vector)
            all_documents.append(chunk_text)
            all_metadatas.append(metadata)

            if source_info["source"] == "explicit":
                explicit_count += 1
            else:
                tacit_count += 1

            global_idx += 1

            # Rate limit: 5 req/min on free tier
            time.sleep(1)

    if all_ids:
        collection.upsert(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_documents,
            metadatas=all_metadatas,
        )
        logger.info("ChromaDB'ye %d chunk yazıldı.", len(all_ids))
    else:
        logger.warning("Yazılacak chunk bulunamadı.")

    collection_size = collection.count()
    summary = {
        "total_chunks": explicit_count + tacit_count,
        "explicit_chunks": explicit_count,
        "tacit_chunks": tacit_count,
        "collection_size": collection_size,
    }

    logger.info(
        "Yükleme özeti — Toplam: %d | Mevzuat: %d | Tacit: %d | DB boyutu: %d",
        summary["total_chunks"],
        summary["explicit_chunks"],
        summary["tacit_chunks"],
        summary["collection_size"],
    )

    return summary


def clear_and_reingest() -> dict:
    """
    Drop the existing ChromaDB collection and re-run the full ingestion pipeline.

    Returns
    -------
    dict
        Same summary dict as run_ingestion().
    """
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        logger.info("Mevcut koleksiyon silindi: %s", COLLECTION_NAME)
    except Exception as exc:
        logger.warning("Koleksiyon silinemedi (belki yoktu): %s", exc)

    return run_ingestion()


def get_ingestion_status() -> dict:
    """
    Return the current ChromaDB collection metadata without re-ingesting.

    Returns
    -------
    dict with keys "collection_size", "explicit_chunks", "tacit_chunks".
    Returns zeros if collection does not exist or is empty.
    """
    try:
        collection = _get_chroma_collection()
        total = collection.count()
        if total == 0:
            return {"collection_size": 0, "explicit_chunks": 0, "tacit_chunks": 0}

        explicit_results = collection.get(where={"source": "explicit"})
        tacit_results = collection.get(where={"source": "tacit"})

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
    Insert a single new document into the EXISTING ChromaDB collection.

    Does NOT wipe the collection. Suitable for incremental knowledge-base updates.

    Steps:
      1. Read the file from disk.
      2. Run it through the anonymizer.
      3. Chunk it with _split_into_chunks().
      4. Embed each chunk via the Gemini API.
      5. Upsert into the existing tee_knowledge_base collection.
      6. Return a summary dict.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the document file.
    source_type : str
        Label to store in the "source" metadata field (e.g. "explicit", "tacit").

    Returns
    -------
    dict with keys:
        - "new_chunks": int      — number of chunks successfully embedded and added.
        - "filename": str        — the file's base name.
        - "collection_size": int — total chunks in the collection after insertion.
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

    # Anonymize
    anon_result = anonymize_text(raw_text)
    clean_text = anon_result["anonymized_text"]
    logger.info(
        "Anonimleştirme tamamlandı: %d kayıt maskelendi.", len(anon_result["mask_log"])
    )

    # Chunk
    chunks = _split_into_chunks(clean_text)
    logger.info("%d chunk oluşturuldu: %s", len(chunks), doc_path.name)

    # Embed and collect
    client = _get_genai_client()
    collection = _get_chroma_collection()

    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict] = []

    for local_idx, chunk_text in enumerate(chunks):
        # Use stem+index for unique IDs that don't collide with existing chunks
        chunk_id = f"{source_type}_{doc_path.stem}_{local_idx}"
        metadata = {
            "source": source_type,
            "filename": doc_path.name,
            "chunk_index": local_idx,
            "char_count": len(chunk_text),
        }

        logger.info(
            "Embedding oluşturuluyor: %s chunk %d/%d",
            doc_path.name,
            local_idx + 1,
            len(chunks),
        )

        try:
            vector = _embed_chunk(client, chunk_text)
        except RuntimeError as exc:
            logger.error("Chunk atlandı (embedding hatası): %s", exc)
            continue

        all_ids.append(chunk_id)
        all_embeddings.append(vector)
        all_documents.append(chunk_text)
        all_metadatas.append(metadata)

        time.sleep(1)  # Rate limit: 5 req/min on free tier

    if all_ids:
        collection.upsert(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_documents,
            metadatas=all_metadatas,
        )
        logger.info("ChromaDB'ye %d yeni chunk eklendi.", len(all_ids))
    else:
        logger.warning("Hiç chunk eklenemedi.")

    collection_size = collection.count()

    summary = {
        "new_chunks": len(all_ids),
        "filename": doc_path.name,
        "collection_size": collection_size,
    }

    logger.info(
        "Ekleme özeti — Yeni chunk: %d | Toplam koleksiyon boyutu: %d",
        summary["new_chunks"],
        summary["collection_size"],
    )

    return summary


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    summary = run_ingestion()
    print("\nYükleme Özeti:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
