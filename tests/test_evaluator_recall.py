"""Birim testleri — deterministik recall@k raporlayıcı (Phase 2).

recall@k matematiğini ve kova (bucket) toplamasını MOCK_MODE altında,
retrieve_context monkeypatch'lenerek doğrular. API çağrısı yapılmaz.
"""
from __future__ import annotations

import json

import pytest

from src import evaluator


@pytest.fixture
def synthetic_questions():
    return [
        {
            "id": "s1",
            "question": "Soru bir",
            "expected_sources": ["a.pdf"],
            "split": "dev",
            "language": "tr",
            "type": "factoid",
            "category": "grille",
            "level": "B2",
        },
        {
            "id": "s2",
            "question": "Question deux",
            "expected_sources": ["b.pdf", "c.pdf"],
            "split": "held_out",
            "language": "fr",
            "type": "definition",
            "category": "descripteur",
            "level": "A1",
        },
        {
            # answerable=False → recall'a katılmamalı
            "id": "s3",
            "question": "Cevapsız",
            "expected_sources": [],
            "answerable": False,
            "split": "dev",
            "language": "tr",
            "type": "factoid",
            "category": "autre",
            "level": "general",
        },
    ]


def _fake_retrieve_factory(mapping):
    def _fake(query, top_k=None, **kwargs):
        names = mapping.get(query, [])[: (top_k or len(mapping.get(query, [])))]
        return [{"filename": f, "parent_id": f"p_{f}", "parent_text": "x"} for f in names]
    return _fake


def test_recall_math_perfect_and_partial(synthetic_questions):
    # s1: a.pdf at rank 1 → recall@k=1.0 for all k; hit@1=1
    # s2: gold {b,c}; retrieve b at rank1, c at rank4 → recall@1=0.5, recall@5=1.0
    fake = _fake_retrieve_factory({
        "Soru bir": ["a.pdf", "z.pdf"],
        "Question deux": ["b.pdf", "x.pdf", "y.pdf", "c.pdf"],
    })
    raw = evaluator.evaluate_retrieval(synthetic_questions, ks=[1, 3, 5], retrieve_fn=fake)
    by_id = {r["id"]: r for r in raw["per_question"]}

    # s3 excluded (answerable=False)
    assert set(by_id) == {"s1", "s2"}

    assert by_id["s1"]["recall@1"] == 1.0
    assert by_id["s1"]["hit@1"] == 1.0

    assert by_id["s2"]["recall@1"] == 0.5  # only b among top-1
    assert by_id["s2"]["recall@3"] == 0.5  # c still at rank 4
    assert by_id["s2"]["recall@5"] == 1.0  # both b and c within top-5
    assert by_id["s2"]["hit@1"] == 1.0


def test_aggregate_overall_and_buckets(synthetic_questions):
    fake = _fake_retrieve_factory({
        "Soru bir": ["a.pdf"],
        "Question deux": ["b.pdf", "c.pdf"],
    })
    raw = evaluator.evaluate_retrieval(synthetic_questions, ks=[1, 3], retrieve_fn=fake)
    agg = evaluator._aggregate(raw["per_question"], [1, 3])

    # overall recall@3 = mean(1.0, 1.0) = 1.0
    assert agg["overall"]["n"] == 2
    assert agg["overall"]["recall@3"] == 1.0
    # recall@1 = mean(s1=1.0, s2=0.5) = 0.75
    assert agg["overall"]["recall@1"] == 0.75

    # language buckets present and split correctly
    assert agg["buckets"]["language"]["tr"]["n"] == 1
    assert agg["buckets"]["language"]["fr"]["n"] == 1
    assert agg["buckets"]["split"]["dev"]["n"] == 1
    assert agg["buckets"]["split"]["held_out"]["n"] == 1
    # doc_type bucket uses the question's category
    assert "grille" in agg["buckets"]["doc_type"]
    assert "descripteur" in agg["buckets"]["doc_type"]


def test_run_recall_report_writes_artifacts(synthetic_questions, tmp_path, monkeypatch):
    fake = _fake_retrieve_factory({
        "Soru bir": ["a.pdf"],
        "Question deux": ["b.pdf", "c.pdf"],
    })
    monkeypatch.setattr(evaluator.settings, "RAGAS_RESULTS_DIR", tmp_path)
    report = evaluator.run_recall_report(
        synthetic_questions, ks=[1, 3, 5], retrieve_fn=fake, save=True
    )
    md = report["metadata"]
    assert md["n_questions_evaluated"] == 2
    assert md["m3_recall_at_k"] is not None
    # artifacts on disk
    jsons = list(tmp_path.glob("recall_baseline_*.json"))
    mds = list(tmp_path.glob("recall_baseline_*.md"))
    assert len(jsons) == 1 and len(mds) == 1
    loaded = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert loaded["aggregate"]["overall"]["n"] == 2


def test_run_recall_report_no_save(synthetic_questions):
    fake = _fake_retrieve_factory({"Soru bir": ["a.pdf"], "Question deux": ["b.pdf"]})
    report = evaluator.run_recall_report(synthetic_questions, ks=[1], retrieve_fn=fake, save=False)
    assert "artifact_json" not in report["metadata"]
    assert report["metadata"]["n_questions_evaluated"] == 2


def test_retrieval_failure_is_handled(synthetic_questions):
    def boom(query, top_k=None, **kwargs):
        raise RuntimeError("no chroma")
    raw = evaluator.evaluate_retrieval(synthetic_questions, ks=[1, 3], retrieve_fn=boom)
    # every answerable question scores 0 recall but does not crash
    for r in raw["per_question"]:
        assert r["recall@1"] == 0.0
        assert r["hit@3"] == 0.0
