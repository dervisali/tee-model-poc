"""Tests for corpus readiness checks."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from src.readiness_check import check_readiness


def test_check_readiness_uses_explicit_chroma_dir(tmp_path):
    missing = tmp_path / "explicit_missing_chroma"

    result = check_readiness(run_smoke=False, chroma_dir=missing)

    assert result.ok is False
    assert str(missing) in result.failures[0]


def test_check_readiness_counts_collection_from_explicit_chroma_dir(monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma_db_enriched"
    chroma_dir.mkdir()
    (chroma_dir / "parents.json").write_text(json.dumps({"p1": {"text": "source"}}), encoding="utf-8")
    (chroma_dir / "bm25_index.pkl").write_bytes(b"index")

    calls = {}

    class FakeCollection:
        name = "tee_children"

        def count(self):
            return 3

    class FakeClient:
        def __init__(self, *, path):
            calls["path"] = path

        def get_collection(self, name):
            calls["collection_name"] = name
            return FakeCollection()

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=FakeClient))

    result = check_readiness(run_smoke=False, chroma_dir=chroma_dir)

    assert result.ok is True
    assert result.child_count == 3
    assert result.parent_count == 1
    assert calls["path"] == str(chroma_dir)
    assert calls["collection_name"] == "tee_children"
