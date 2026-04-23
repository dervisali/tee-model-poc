"""
Retrieval layer for TEE-Model POC — Parent Document Retrieval edition.

Query flow:
  1. Embed the user query with gemini-embedding-001
  2. Search 'tee_children' collection for the top-k most similar child chunks
  3. For each child hit, fetch its parent chunk from parents.json using parent_id
  4. Deduplicate: multiple children sharing the same parent yield only one result
  5. Return parent texts to generators — richer context than raw child chunks
"""

import os
import json
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
COLLECTION_NAME = "tee_children"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_genai_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY ortam değişkeni bulunamadı. Lütfen .env dosyasını kontrol edin."
        )
    return genai.Client(api_key=api_key)


def _get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _load_parents() -> dict:
    """Load the parent lookup table from parents.json."""
    if PARENTS_JSON.exists():
        return json.loads(PARENTS_JSON.read_text(encoding="utf-8"))
    return {}


def _embed_query(client: genai.Client, query: str) -> list[float]:
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config={"task_type": "retrieval_query"},
        )
        return response.embeddings[0].values
    except Exception as exc:
        raise RuntimeError(f"Sorgu embedding hatası: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_context(
    query: str,
    top_k: int = 5,
    source_filter: str = None,
    distance_threshold: float = 0.7,
) -> list[dict]:
    """
    Retrieve the most relevant parent chunks for a query using PDR.

    Steps:
      1. Embed query and search tee_children for top_k*2 candidates
         (over-fetch before dedup)
      2. Filter by distance_threshold
      3. For each child hit, look up the corresponding parent in parents.json
      4. Deduplicate: keep the closest child per unique parent_id
      5. Return up to top_k parent-level results

    Parameters
    ----------
    query : str
        Natural-language query (Turkish).
    top_k : int
        Maximum number of deduplicated parent results to return.
    source_filter : str or None
        Restrict to "explicit", "tacit", or None for both.
    distance_threshold : float
        Maximum cosine distance for a child chunk to qualify.

    Returns
    -------
    list[dict]
        Each item contains:
            "child_text"  — the small chunk that matched the query
            "parent_text" — the larger context chunk sent to the LLM
            "text"        — alias for parent_text (backward compat)
            "source"      — "explicit" or "tacit"
            "filename"    — source file name
            "parent_id"   — parent chunk identifier
            "child_index" — position of child within its parent
            "distance"    — cosine distance of the matched child (lower = better)
    """
    collection = _get_chroma_collection()
    if collection.count() == 0:
        raise ValueError(
            "Vektör veritabanı boş. Lütfen önce veri yükleme işlemini çalıştırın."
        )

    parents = _load_parents()
    if not parents:
        raise ValueError(
            "parents.json bulunamadı. Lütfen önce veri yükleme işlemini çalıştırın."
        )

    client = _get_genai_client()
    query_vector = _embed_query(client, query)

    where_clause = {"source": source_filter} if source_filter else None

    query_kwargs = {
        "query_embeddings": [query_vector],
        "n_results": min(top_k * 3, collection.count()),  # over-fetch for dedup
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

    # Filter by threshold, then deduplicate by parent_id (keep closest child)
    seen_parents: dict[str, dict] = {}
    for child_text, meta, dist in zip(documents, metadatas, distances):
        if float(dist) > distance_threshold:
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
                "source": meta.get("source", "bilinmiyor"),
                "filename": meta.get("filename", "bilinmiyor"),
            }

    # Sort by distance, take top_k, attach parent text
    ordered = sorted(seen_parents.values(), key=lambda r: r["distance"])[:top_k]

    formatted = []
    for item in ordered:
        parent_data = parents[item["parent_id"]]
        parent_text = parent_data["text"]
        formatted.append({
            "child_text": item["child_text"],
            "parent_text": parent_text,
            "text": parent_text,           # backward-compat alias
            "source": item["source"],
            "filename": item["filename"],
            "parent_id": item["parent_id"],
            "child_index": item["child_index"],
            "distance": item["distance"],
        })

    logger.info(
        "Sorgu: '%s' | Filtre: %s | Sonuç: %d parent | En iyi mesafe: %.4f",
        query[:80],
        source_filter or "tümü",
        len(formatted),
        formatted[0]["distance"] if formatted else 0.0,
    )

    return formatted


def build_context_text(chunks: list[dict]) -> tuple[str, list]:
    """
    Format retrieved parent chunks into a context string for LLM prompts.

    Returns
    -------
    tuple[str, list]
        - context_text: formatted context block using parent_text
        - parent_ids: list of parent_id strings for citation
    """
    parts = []
    parent_ids = []
    for i, chunk in enumerate(chunks):
        label = (
            f"[Kaynak {i + 1}: {chunk['filename']} — {chunk['parent_id']}]"
        )
        parts.append(f"{label}\n{chunk['parent_text']}")
        parent_ids.append(chunk["parent_id"])

    return "\n\n---\n\n".join(parts), parent_ids


def lookup_parent_context(parent_id: str) -> dict | None:
    """
    Return parent text and its child chunks for a given parent_id.
    Used by the expert approval panel to display source citations.

    Returns None if parent_id is not found.
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
        logger.warning("Child chunk'lar alınamadı (parent_id=%s): %s", parent_id, exc)
        children = []

    return {
        "parent_text": parent_data["text"],
        "children": children,
        "source": parent_data.get("source", ""),
        "filename": parent_data.get("filename", ""),
    }


if __name__ == "__main__":
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
