"""Tests for applying DELF expert review decisions."""

from __future__ import annotations

import csv
import json

import pytest

from scripts.apply_expert_validation import apply_review, default_manifest_path


FIELDNAMES = [
    "id",
    "validation_decision",
    "expert_corrected_sources",
    "expert_corrected_ground_truth",
    "expert_notes",
]


def _write_eval(path):
    payload = {
        "questions": [
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT",
                "expected_sources": ["old.pdf"],
                "ground_truth": "old answer",
            },
            {
                "id": "q2",
                "answerable": True,
                "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT",
                "expected_sources": ["keep.pdf"],
                "ground_truth": "keep answer",
            },
            {
                "id": "q3",
                "answerable": True,
                "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT",
                "expected_sources": ["drop.pdf"],
                "ground_truth": "drop answer",
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_review(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_corpus(path):
    path.write_text(
        json.dumps({"files": {"keep.pdf": {}, "a.pdf": {}, "b.pdf": {}}}),
        encoding="utf-8",
    )


def test_apply_review_validates_fixes_and_drops(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    output_path = tmp_path / "validated.json"
    corpus_path = tmp_path / "corpus_files.json"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    _write_review(review_path, [
        {
            "id": "q1",
            "validation_decision": "FIX_BOTH",
            "expert_corrected_sources": "a.pdf; b.pdf",
            "expert_corrected_ground_truth": "new answer",
            "expert_notes": "better official source",
        },
        {
            "id": "q2",
            "validation_decision": "VALIDATED",
            "expert_corrected_sources": "",
            "expert_corrected_ground_truth": "",
            "expert_notes": "",
        },
        {
            "id": "q3",
            "validation_decision": "DROP",
            "expert_corrected_sources": "",
            "expert_corrected_ground_truth": "",
            "expert_notes": "ambiguous",
        },
    ])

    counts = apply_review(
        eval_path,
        review_path,
        output_path,
        strict=True,
        corpus_files_path=corpus_path,
    )
    out = json.loads(output_path.read_text(encoding="utf-8"))["questions"]
    by_id = {item["id"]: item for item in out}

    assert counts == {"validated": 2, "fixed": 1, "dropped": 1, "unchanged": 0}
    assert by_id["q1"]["expected_sources"] == ["a.pdf", "b.pdf"]
    assert by_id["q1"]["ground_truth"] == "new answer"
    assert by_id["q1"]["validation_status"] == "VALIDATED_BY_DELF_EXPERT"
    assert by_id["q2"]["validation_status"] == "VALIDATED_BY_DELF_EXPERT"
    assert by_id["q3"]["answerable"] is False
    assert by_id["q3"]["validation_status"] == "DROPPED_BY_DELF_EXPERT"

    manifest = json.loads(default_manifest_path(output_path).read_text(encoding="utf-8"))
    assert manifest["strict"] is True
    assert manifest["output_eval_path"] == str(output_path)
    assert manifest["counts"] == counts
    assert manifest["output_eval_sha256"]
    assert manifest["corpus_files_path"] == str(corpus_path)
    assert manifest["corpus_files_sha256"]


def test_apply_review_strict_requires_decision(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_review(review_path, [{"id": "q1", "validation_decision": ""}])

    with pytest.raises(SystemExit, match="Missing validation_decision"):
        apply_review(eval_path, review_path, tmp_path / "out.json", strict=True)


def test_apply_review_rejects_missing_required_columns(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    output_path = tmp_path / "out.json"
    _write_eval(eval_path)
    review_path.write_text("id,validation_decision\nq1,VALIDATED\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Missing required columns"):
        apply_review(eval_path, review_path, output_path, strict=True)
    assert not output_path.exists()


def test_apply_review_rejects_ai_grounded_notes_as_expert_evidence(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    output_path = tmp_path / "out.json"
    _write_eval(eval_path)
    _write_review(review_path, [
        {
            "id": "q1",
            "validation_decision": "VALIDATED",
            "expert_notes": "AI-grounded from retrieved snippets",
        },
        {"id": "q2", "validation_decision": "VALIDATED"},
        {"id": "q3", "validation_decision": "DROP"},
    ])

    with pytest.raises(SystemExit, match="q1 expert_notes indicate non-expert/AI review evidence: ai-grounded"):
        apply_review(eval_path, review_path, output_path, strict=True)
    assert not output_path.exists()


def test_apply_review_rejects_empty_review_csv(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    output_path = tmp_path / "out.json"
    _write_eval(eval_path)
    review_path.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="Missing required columns"):
        apply_review(eval_path, review_path, output_path, strict=True)
    assert not output_path.exists()


def test_apply_review_refuses_to_overwrite_input_eval(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_review(review_path, [{"id": "q1", "validation_decision": "VALIDATED"}])

    with pytest.raises(SystemExit, match="--output must not overwrite the input eval JSON"):
        apply_review(eval_path, review_path, eval_path, strict=True)


def test_apply_review_refuses_to_overwrite_review_or_corpus_catalog(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    corpus_path = tmp_path / "corpus_files.json"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    _write_review(review_path, [{"id": "q1", "validation_decision": "VALIDATED"}])

    with pytest.raises(SystemExit, match="--output must not overwrite the expert review CSV"):
        apply_review(eval_path, review_path, review_path, strict=True)
    with pytest.raises(SystemExit, match="--output must not overwrite the corpus catalog"):
        apply_review(
            eval_path,
            review_path,
            corpus_path,
            strict=True,
            corpus_files_path=corpus_path,
        )


def test_apply_review_strict_requires_all_eval_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_review(review_path, [
        {"id": "q1", "validation_decision": "VALIDATED"},
        {"id": "q2", "validation_decision": "VALIDATED"},
    ])

    with pytest.raises(SystemExit, match="Review CSV missing eval IDs: q3"):
        apply_review(eval_path, review_path, tmp_path / "out.json", strict=True)


def test_apply_review_rejects_duplicate_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    _write_eval(eval_path)
    _write_review(review_path, [
        {"id": "q1", "validation_decision": "VALIDATED"},
        {"id": "q1", "validation_decision": "DROP"},
        {"id": "q2", "validation_decision": "VALIDATED"},
        {"id": "q3", "validation_decision": "DROP"},
    ])

    with pytest.raises(SystemExit, match="Duplicate id in review CSV: q1"):
        apply_review(eval_path, review_path, tmp_path / "out.json", strict=True)


def test_apply_review_rejects_duplicate_eval_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    eval_path.write_text(
        json.dumps({
            "questions": [
                {"id": "q1", "answerable": True, "validation_status": "UNVALIDATED"},
                {"id": "q1", "answerable": True, "validation_status": "UNVALIDATED"},
            ]
        }),
        encoding="utf-8",
    )
    _write_review(review_path, [{"id": "q1", "validation_decision": "VALIDATED"}])

    with pytest.raises(SystemExit, match="Eval JSON contains duplicate IDs: q1"):
        apply_review(eval_path, review_path, tmp_path / "out.json", strict=True)


def test_apply_review_rejects_blank_eval_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    eval_path.write_text(
        json.dumps({
            "questions": [
                {"id": "q1", "answerable": True, "validation_status": "UNVALIDATED"},
                {"id": "", "answerable": True, "validation_status": "UNVALIDATED"},
                {"answerable": True, "validation_status": "UNVALIDATED"},
            ]
        }),
        encoding="utf-8",
    )
    _write_review(review_path, [{"id": "q1", "validation_decision": "VALIDATED"}])

    with pytest.raises(SystemExit, match="Eval JSON contains blank or missing IDs"):
        apply_review(eval_path, review_path, tmp_path / "out.json", strict=True)


def test_apply_review_rejects_validated_row_with_source_outside_corpus(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    output_path = tmp_path / "out.json"
    corpus_path = tmp_path / "corpus_files.json"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    _write_review(review_path, [
        {"id": "q1", "validation_decision": "FIX_SOURCE", "expert_corrected_sources": "a.pdf"},
        {"id": "q2", "validation_decision": "VALIDATED"},
        {"id": "q3", "validation_decision": "VALIDATED"},
    ])

    with pytest.raises(SystemExit, match="q3 marked VALIDATED but retained source is not in corpus_files.json"):
        apply_review(
            eval_path,
            review_path,
            output_path,
            strict=True,
            corpus_files_path=corpus_path,
        )
    assert not output_path.exists()


def test_apply_review_rejects_corrected_source_outside_corpus(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    output_path = tmp_path / "out.json"
    corpus_path = tmp_path / "corpus_files.json"
    _write_eval(eval_path)
    _write_corpus(corpus_path)
    _write_review(review_path, [
        {"id": "q1", "validation_decision": "FIX_SOURCE", "expert_corrected_sources": "missing.pdf"},
        {"id": "q2", "validation_decision": "VALIDATED"},
        {"id": "q3", "validation_decision": "DROP"},
    ])

    with pytest.raises(SystemExit, match="q1 marked FIX_SOURCE but corrected source is not in corpus_files.json"):
        apply_review(
            eval_path,
            review_path,
            output_path,
            strict=True,
            corpus_files_path=corpus_path,
        )
    assert not output_path.exists()


def test_apply_review_supports_top_level_list_json(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    output_path = tmp_path / "validated.json"
    eval_path.write_text(
        json.dumps([
            {"id": "q1", "answerable": True, "validation_status": "UNVALIDATED"}
        ]),
        encoding="utf-8",
    )
    _write_review(review_path, [{"id": "q1", "validation_decision": "VALIDATED"}])

    apply_review(eval_path, review_path, output_path, strict=True)
    out = json.loads(output_path.read_text(encoding="utf-8"))

    assert out[0]["validation_status"] == "VALIDATED_BY_DELF_EXPERT"
