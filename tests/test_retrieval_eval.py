"""Birim testleri — deterministik kaynak-recall yardımcıları (Phase 2).

Saf küme matematiği; API çağrısı yapılmaz.
"""
from __future__ import annotations

from src import evaluator


def test_source_recall_perfect():
    assert evaluator._source_recall(["a.pdf", "b.pdf"], ["a.pdf", "b.pdf", "c.pdf"]) == 1.0


def test_source_recall_partial():
    assert evaluator._source_recall(["a.pdf", "b.pdf"], ["a.pdf", "x.pdf"]) == 0.5


def test_source_recall_zero():
    assert evaluator._source_recall(["z.pdf"], ["a.pdf"]) == 0.0


def test_source_recall_empty_gold_is_one():
    assert evaluator._source_recall([], ["a.pdf"]) == 1.0


def test_source_precision_basic():
    assert evaluator._source_precision(["a.pdf"], ["a.pdf", "b.pdf"]) == 0.5


def test_source_precision_empty_retrieved_is_zero():
    assert evaluator._source_precision(["a.pdf"], []) == 0.0


def test_hit_at_k_true_and_false():
    assert evaluator._hit_at_k(["a.pdf"], ["x.pdf", "a.pdf"]) == 1.0
    assert evaluator._hit_at_k(["a.pdf"], ["x.pdf", "y.pdf"]) == 0.0


def test_basename_strips_path():
    # expected_sources gerçek korpusta yol içermez ama eşleme yine de yol-güvenli olmalı.
    assert evaluator._source_recall(["a.pdf"], ["sub/dir/a.pdf"]) == 1.0


def test_retrieved_sources_prefers_filename():
    hits = [
        {"filename": "a.pdf", "parent_id": "p1"},
        {"source_file": "b.pdf", "parent_id": "p2"},
        {"source": "c.pdf", "parent_id": "p3"},
    ]
    assert evaluator._retrieved_sources(hits) == ["a.pdf", "b.pdf", "c.pdf"]
