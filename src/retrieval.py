"""
Retrieval layer for TEE-Model POC.

Embeds user queries with gemini-embedding-001 and fetches the most semantically
relevant chunks from the ChromaDB vector store.
"""

import os
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
    """Return the persistent ChromaDB collection. Raises if it does not exist."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_query(client: genai.Client, query: str) -> list[float]:
    """
    Embed a user query for similarity search.

    Uses gemini-embedding-001 with task_type='retrieval_query'.

    Parameters
    ----------
    client : genai.Client
        Authenticated google-genai client.
    query : str
        The user's natural-language query string.

    Returns
    -------
    list[float]
        Embedding vector for the query.

    Raises
    ------
    RuntimeError
        On API failure.
    """
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config={"task_type": "retrieval_query"},
        )
        return response.embeddings[0].values
    except Exception as exc:
        raise RuntimeError(f"Sorgu embedding hatası: {exc}") from exc


def retrieve_context(
    query: str,
    top_k: int = 5,
    source_filter: str = None,
    distance_threshold: float = 0.7,
) -> list[dict]:
    """
    Retrieve the most semantically relevant document chunks for a query.

    Parameters
    ----------
    query : str
        Natural-language query (Turkish).
    top_k : int
        Maximum number of results to return. Defaults to 5.
    source_filter : str or None
        Restrict results to "explicit", "tacit", or None for both.

    Returns
    -------
    list[dict]
        Each item contains:
            - "text": str          — chunk text
            - "source": str        — "explicit" or "tacit"
            - "filename": str      — source file name
            - "chunk_index": int   — position in original document
            - "distance": float    — cosine distance (lower = more similar)

    Raises
    ------
    ValueError
        If the collection is empty or no results are found.
    RuntimeError
        On embedding API failure.
    """
    collection = _get_chroma_collection()
    if collection.count() == 0:
        raise ValueError(
            "Vektör veritabanı boş. Lütfen önce veri yükleme işlemini çalıştırın."
        )

    client = _get_genai_client()
    query_vector = _embed_query(client, query)

    where_clause = {"source": source_filter} if source_filter else None

    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results": top_k,
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
        raise ValueError(
            "Sorgu için hiçbir sonuç döndürülmedi. "
            "Vektör veritabanı boş. Lütfen önce veri yükleme işlemini çalıştırın."
        )

    formatted = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        if float(dist) > distance_threshold:
            continue
        formatted.append(
            {
                "text": doc,
                "source": meta.get("source", "bilinmiyor"),
                "filename": meta.get("filename", "bilinmiyor"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": round(float(dist), 4),
            }
        )

    logger.info(
        "Sorgu: '%s' | Filtre: %s | Sonuç sayısı: %d | En iyi mesafe: %.4f",
        query[:80],
        source_filter or "tümü",
        len(formatted),
        formatted[0]["distance"] if formatted else 0.0,
    )

    return formatted


def build_context_text(chunks: list[dict]) -> tuple[str, list[int]]:
    """
    Format retrieved chunks into a single context string for LLM prompts.

    Parameters
    ----------
    chunks : list[dict]
        Output of retrieve_context().

    Returns
    -------
    tuple[str, list[int]]
        - context_text: formatted context block
        - chunk_indices: list of chunk_index values for citation
    """
    parts = []
    indices = []
    for i, chunk in enumerate(chunks):
        label = (
            f"[Kaynak {i + 1}: {chunk['filename']} — Chunk {chunk['chunk_index']}]"
        )
        parts.append(f"{label}\n{chunk['text']}")
        indices.append(chunk["chunk_index"])

    return "\n\n---\n\n".join(parts), indices


if __name__ == "__main__":
    # Quick smoke test
    test_query = "maaş hesaplama adımları"
    print(f"Test sorgusu: '{test_query}'\n")
    try:
        hits = retrieve_context(test_query, top_k=3)
        for i, hit in enumerate(hits, start=1):
            print(f"--- Sonuç {i} ---")
            print(f"Kaynak : {hit['source']} | Dosya: {hit['filename']} | Chunk: {hit['chunk_index']}")
            print(f"Mesafe : {hit['distance']}")
            print(f"Metin  : {hit['text'][:200]}...")
            print()
    except (ValueError, RuntimeError) as exc:
        print(f"Hata: {exc}")
