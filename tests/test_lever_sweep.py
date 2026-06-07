"""Tests for non-destructive retrieval lever sweep summaries."""

from scripts.lever_sweep import _config_specs, _summarize


def test_sweep_includes_full_recall_mode():
    names = [name for name, _, _ in _config_specs()]

    assert "full_recall_mode_no_rerank" in names
    assert "full_recall_mode" in names


def test_summarize_reports_m3_and_top3_buckets():
    report = {
        "metadata": {
            "m3_k": 3,
            "m3_recall_at_k": 0.78,
            "m3_hit_at_k": 0.86,
        },
        "aggregate": {
            "overall": {
                "recall@3": 0.78,
                "recall@5": 0.83,
                "recall@10": 0.90,
                "hit@3": 0.86,
                "hit@5": 0.90,
            },
            "buckets": {
                "language": {
                    "tr": {"recall@3": 0.90, "recall@5": 0.92},
                    "fr": {"recall@3": 0.66, "recall@5": 0.75},
                },
                "doc_type": {
                    "grille": {"recall@3": 0.70, "recall@5": 0.85},
                    "descripteur": {"recall@3": 0.50, "recall@5": 0.56},
                },
                "level": {
                    "B2": {"recall@3": 0.25, "recall@5": 0.50},
                },
            },
        },
    }

    summary = _summarize(report)

    assert summary["m3_k"] == 3
    assert summary["m3_recall"] == 0.78
    assert summary["m3_hit"] == 0.86
    assert summary["m3_passed"] is False
    assert summary["recall@3"] == 0.78
    assert summary["tr@3"] == 0.90
    assert summary["grille@3"] == 0.70
    assert summary["descripteur@3"] == 0.50
    assert summary["B2@3"] == 0.25
    assert summary["recall@5"] == 0.83
