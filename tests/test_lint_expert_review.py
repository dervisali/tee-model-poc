"""Tests for expert-review CSV linting."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.lint_expert_review import lint_review


FIELDNAMES = [
    "id",
    "validation_decision",
    "expert_corrected_sources",
    "expert_corrected_ground_truth",
    "expert_notes",
]


def _write_eval(path):
    path.write_text(
        json.dumps({
            "questions": [
                {"id": "q1"},
                {"id": "q2"},
            ]
        }),
        encoding="utf-8",
    )


def _write_corpus(path):
    path.write_text(
        json.dumps({
            "files": {
                "manuel-exacor.pdf": {"child_count": 1},
                "Diaporama_rôle_EC_VF.pptx": {"child_count": 1},
            }
        }),
        encoding="utf-8",
    )


def _write_review(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_lint_review_accepts_completed_valid_review(tmp_path):
    eval_path = tmp_path / "eval.json"
    corpus_path = tmp_path / "corpus.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    _write_review(review_path, [
        {
            "id": "q1",
            "validation_decision": "VALIDATED",
            "expert_corrected_sources": "",
            "expert_corrected_ground_truth": "",
        },
        {
            "id": "q2",
            "validation_decision": "FIX_SOURCE",
            "expert_corrected_sources": "manuel-exacor.pdf; Diaporama_rôle_EC_VF.pptx",
            "expert_corrected_ground_truth": "",
        },
    ])

    result = lint_review(
        eval_path=eval_path,
        review_path=review_path,
        corpus_files_path=corpus_path,
        require_complete=True,
    )

    assert result.ok is True
    assert result.errors == []


def test_lint_review_reports_missing_decision_when_complete_required(tmp_path):
    eval_path = tmp_path / "eval.json"
    corpus_path = tmp_path / "corpus.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    _write_review(review_path, [
        {"id": "q1", "validation_decision": ""},
        {"id": "q2", "validation_decision": "VALIDATED"},
    ])

    result = lint_review(
        eval_path=eval_path,
        review_path=review_path,
        corpus_files_path=corpus_path,
        require_complete=True,
    )

    assert result.ok is False
    assert any("missing validation_decision" in error for error in result.errors)


def test_lint_review_reports_unknown_corrected_source(tmp_path):
    eval_path = tmp_path / "eval.json"
    corpus_path = tmp_path / "corpus.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    _write_review(review_path, [
        {
            "id": "q1",
            "validation_decision": "FIX_SOURCE",
            "expert_corrected_sources": "not-in-corpus.pdf",
        },
        {"id": "q2", "validation_decision": "VALIDATED"},
    ])

    result = lint_review(
        eval_path=eval_path,
        review_path=review_path,
        corpus_files_path=corpus_path,
        require_complete=True,
    )

    assert result.ok is False
    assert any("not in corpus_files.json" in error for error in result.errors)


def test_lint_review_rejects_ai_grounded_notes_as_expert_evidence(tmp_path):
    eval_path = tmp_path / "eval.json"
    corpus_path = tmp_path / "corpus.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    _write_review(review_path, [
        {
            "id": "q1",
            "validation_decision": "VALIDATED",
            "expert_notes": "AI-grounded against extracted chunks",
        },
        {"id": "q2", "validation_decision": "VALIDATED"},
    ])

    result = lint_review(
        eval_path=eval_path,
        review_path=review_path,
        corpus_files_path=corpus_path,
        require_complete=True,
    )

    assert result.ok is False
    assert any(
        "line 2 id=q1: expert_notes indicate non-expert/AI review evidence: ai-grounded" in error
        for error in result.errors
    )


def test_lint_review_reports_unknown_retained_source_for_validated_row(tmp_path):
    eval_path = tmp_path / "eval.json"
    corpus_path = tmp_path / "corpus.json"
    review_path = tmp_path / "review.csv"
    eval_path.write_text(
        json.dumps({
            "questions": [
                {"id": "q1", "expected_sources": ["not-in-corpus.pdf"]},
                {"id": "q2", "expected_sources": ["manuel-exacor.pdf"]},
            ]
        }),
        encoding="utf-8",
    )
    _write_corpus(corpus_path)
    _write_review(review_path, [
        {"id": "q1", "validation_decision": "VALIDATED"},
        {"id": "q2", "validation_decision": "FIX_ANSWER", "expert_corrected_ground_truth": "fixed answer"},
    ])

    result = lint_review(
        eval_path=eval_path,
        review_path=review_path,
        corpus_files_path=corpus_path,
        require_complete=True,
    )

    assert result.ok is False
    assert any("line 2 id=q1: source not in corpus_files.json: not-in-corpus.pdf" in error for error in result.errors)


def test_lint_review_reports_duplicate_eval_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    corpus_path = tmp_path / "corpus.json"
    review_path = tmp_path / "review.csv"
    eval_path.write_text(
        json.dumps({"questions": [{"id": "q1"}, {"id": "q1"}]}),
        encoding="utf-8",
    )
    _write_corpus(corpus_path)
    _write_review(review_path, [{"id": "q1", "validation_decision": "VALIDATED"}])

    result = lint_review(
        eval_path=eval_path,
        review_path=review_path,
        corpus_files_path=corpus_path,
        require_complete=True,
    )

    assert result.ok is False
    assert "Eval JSON contains duplicate IDs: q1" in result.errors


def test_lint_review_reports_blank_eval_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    corpus_path = tmp_path / "corpus.json"
    review_path = tmp_path / "review.csv"
    eval_path.write_text(
        json.dumps({"questions": [{"id": "q1"}, {"id": ""}, {}]}),
        encoding="utf-8",
    )
    _write_corpus(corpus_path)
    _write_review(review_path, [{"id": "q1", "validation_decision": "VALIDATED"}])

    result = lint_review(
        eval_path=eval_path,
        review_path=review_path,
        corpus_files_path=corpus_path,
        require_complete=True,
    )

    assert result.ok is False
    assert "Eval JSON contains blank or missing IDs" in result.errors


def test_lint_review_header_only_csv_reports_missing_eval_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    corpus_path = tmp_path / "corpus.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    _write_review(review_path, [])

    result = lint_review(
        eval_path=eval_path,
        review_path=review_path,
        corpus_files_path=corpus_path,
        require_complete=True,
    )

    assert result.ok is False
    assert result.rows == 0
    assert "Review CSV missing eval IDs: q1, q2" in result.errors
    assert not any("Missing required columns" in error for error in result.errors)


def test_lint_review_empty_csv_reports_missing_required_columns(tmp_path):
    eval_path = tmp_path / "eval.json"
    corpus_path = tmp_path / "corpus.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    review_path.write_text("", encoding="utf-8")

    result = lint_review(
        eval_path=eval_path,
        review_path=review_path,
        corpus_files_path=corpus_path,
        require_complete=True,
    )

    assert result.ok is False
    assert any("Missing required columns" in error for error in result.errors)


def test_lint_review_cli_accepts_eval_path_alias(tmp_path):
    eval_path = tmp_path / "eval.json"
    corpus_path = tmp_path / "corpus.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    _write_review(review_path, [
        {"id": "q1", "validation_decision": "VALIDATED"},
        {"id": "q2", "validation_decision": "VALIDATED"},
    ])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.lint_expert_review",
            "--eval-path",
            str(eval_path),
            "--review",
            str(review_path),
            "--corpus-files",
            str(corpus_path),
            "--require-complete",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '"ok": true' in result.stdout
