"""Tests for expert-validation packet export."""

from __future__ import annotations

import csv
import json

import pytest

from scripts.prepare_expert_validation import export_review_csv, export_source_catalog


def test_export_source_catalog_from_corpus_files(tmp_path):
    corpus = tmp_path / "corpus_files.json"
    output = tmp_path / "source_catalog.csv"
    corpus.write_text(
        json.dumps({
            "files": {
                "B2_Grille_PO.pdf": {
                    "child_count": 12,
                    "doc_type": [["grille", 12]],
                    "level": [["B2", 12]],
                    "skill": [["PO", 12]],
                }
            }
        }),
        encoding="utf-8",
    )

    count = export_source_catalog(corpus, output)
    with output.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    assert count == 1
    assert rows[0]["filename"] == "B2_Grille_PO.pdf"
    assert rows[0]["doc_type"] == "grille:12"
    assert rows[0]["level"] == "B2:12"
    assert rows[0]["skill"] == "PO:12"


def test_export_review_csv_refuses_to_overwrite_existing_expert_work(tmp_path):
    questions = tmp_path / "questions.json"
    output = tmp_path / "review.csv"
    questions.write_text(
        json.dumps({
            "questions": [
                {
                    "id": "d01",
                    "question": "Question?",
                    "ground_truth": "Answer.",
                    "expected_sources": ["manuel-exacor.pdf"],
                }
            ]
        }),
        encoding="utf-8",
    )
    output.write_text(
        "id,validation_decision,expert_corrected_sources,expert_corrected_ground_truth,expert_notes\n"
        "d01,VALIDATED,,,,\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing expert review work"):
        export_review_csv(questions, output)


def test_export_review_csv_allows_force_overwrite_existing_expert_work(tmp_path):
    questions = tmp_path / "questions.json"
    output = tmp_path / "review.csv"
    questions.write_text(
        json.dumps({
            "questions": [
                {
                    "id": "d01",
                    "question": "Question?",
                    "ground_truth": "Answer.",
                    "expected_sources": ["manuel-exacor.pdf"],
                }
            ]
        }),
        encoding="utf-8",
    )
    output.write_text(
        "id,validation_decision,expert_corrected_sources,expert_corrected_ground_truth,expert_notes\n"
        "d01,VALIDATED,,,,\n",
        encoding="utf-8",
    )

    count = export_review_csv(questions, output, force_overwrite=True)

    with output.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert count == 1
    assert rows[0]["validation_decision"] == ""
