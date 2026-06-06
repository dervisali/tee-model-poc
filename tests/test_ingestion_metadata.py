"""Ingestion metadata guards."""

from __future__ import annotations

import chromadb

from src.ingestion import _sanitize_chroma_metadata


def test_sanitize_chroma_metadata_drops_none_and_keeps_scalars():
    raw = {
        "source_filename": "Introduction_criteres.docx",
        "page": None,
        "child_index": 0,
        "enriched": False,
        "score": 0.8,
    }

    clean = _sanitize_chroma_metadata(raw)

    assert "page" not in clean
    assert clean == {
        "source_filename": "Introduction_criteres.docx",
        "child_index": 0,
        "enriched": False,
        "score": 0.8,
    }


def test_sanitized_metadata_is_accepted_by_chromadb(tmp_path):
    col = chromadb.PersistentClient(path=str(tmp_path)).get_or_create_collection(
        "tee_children",
        metadata={"hnsw:space": "cosine"},
    )
    meta = _sanitize_chroma_metadata({
        "parent_id": "docx_par0",
        "page": None,
        "source_filename": "sample.docx",
    })

    col.add(
        ids=["docx_par0_c0"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["exemple"],
        metadatas=[meta],
    )

    assert col.count() == 1
