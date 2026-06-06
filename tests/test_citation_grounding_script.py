"""Unit tests for the citation grounding measurement harness."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from scripts.measure_citation_grounding import (
    BASELINE_CHROMA_DIR,
    main,
    select_questions,
    summarize,
    validate_certification_eval_questions,
    validate_certification_inputs,
)


def test_select_questions_can_require_validated():
    questions = [
        {"id": "a", "answerable": True, "validation_status": "VALIDATED_BY_DELF_EXPERT"},
        {"id": "b", "answerable": True, "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT"},
        {"id": "c", "answerable": False, "validation_status": "VALIDATED_BY_DELF_EXPERT"},
    ]

    selected = select_questions(questions, require_validated=True, max_questions=None)

    assert [q["id"] for q in selected] == ["a"]


def test_select_questions_respects_max_questions():
    questions = [
        {"id": "a", "answerable": True},
        {"id": "b", "answerable": True},
    ]

    selected = select_questions(questions, require_validated=False, max_questions=1)

    assert [q["id"] for q in selected] == ["a"]


def test_summarize_counts_citation_passes_and_errors():
    rows = [
        {"citation_passed": True, "error": None, "safety": {"grounding_allowed": True}},
        {"citation_passed": False, "error": None, "safety": {"grounding_allowed": False}},
        {"citation_passed": False, "error": "boom", "safety": {}},
    ]

    out = summarize(rows)

    assert out["n"] == 3
    assert out["citation_passed"] == 1
    assert out["citation_pass_rate"] == 1 / 3
    assert out["no_error"] == 2
    assert out["grounding_allowed"] == 2


def test_certification_inputs_require_full_validated_nonbaseline_run(monkeypatch, tmp_path):
    chroma_dir = tmp_path / "chroma_db_enriched"
    calls = {}

    def fake_check_readiness(*, run_smoke, chroma_dir):
        calls["run_smoke"] = run_smoke
        calls["chroma_dir"] = chroma_dir
        return SimpleNamespace(ok=True, failures=[])

    monkeypatch.setattr("src.readiness_check.check_readiness", fake_check_readiness)

    validate_certification_inputs(
        certification_run_id="run-1",
        require_validated=True,
        max_questions=None,
        chroma_dir=chroma_dir,
    )

    assert calls["run_smoke"] is False
    assert calls["chroma_dir"] == chroma_dir.resolve(strict=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"require_validated": False}, "--certification-run-id requires --require-validated"),
        ({"max_questions": 5}, "must cover the full validated eval set"),
        ({"chroma_dir": None}, "--certification-run-id requires --chroma-dir"),
    ],
)
def test_certification_inputs_reject_incomplete_m4_evidence(tmp_path, kwargs, message):
    params = {
        "certification_run_id": "run-1",
        "require_validated": True,
        "max_questions": None,
        "chroma_dir": tmp_path / "chroma_db_enriched",
    }
    params.update(kwargs)

    with pytest.raises(SystemExit, match=message):
        validate_certification_inputs(**params)


def test_certification_inputs_reject_baseline_chroma_dir(tmp_path):
    with pytest.raises(SystemExit, match="baseline chroma_db"):
        validate_certification_inputs(
            certification_run_id="run-1",
            require_validated=True,
            max_questions=None,
            chroma_dir=tmp_path / "chroma_db",
        )


def test_certification_inputs_reject_chroma_dir_inside_baseline():
    with pytest.raises(SystemExit, match="inside baseline chroma_db"):
        validate_certification_inputs(
            certification_run_id="run-1",
            require_validated=True,
            max_questions=None,
            chroma_dir=BASELINE_CHROMA_DIR / "nested_m4",
        )


def test_certification_eval_questions_reject_unvalidated_answerable_rows():
    questions = [
        {"id": "d01", "answerable": True, "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT"},
        {"id": "d02", "answerable": False, "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT"},
    ]

    with pytest.raises(SystemExit, match="eval file contains unvalidated answerable items: d01"):
        validate_certification_eval_questions(questions, certification_run_id="run-1")


def test_certification_inputs_reject_unready_chroma_dir(monkeypatch, tmp_path):
    def fake_check_readiness(*, run_smoke, chroma_dir):
        return SimpleNamespace(
            ok=False,
            failures=[f"parents.json yok: '{chroma_dir / 'parents.json'}'."],
        )

    monkeypatch.setattr("src.readiness_check.check_readiness", fake_check_readiness)

    with pytest.raises(SystemExit, match="CHROMA_DIR is not ready"):
        validate_certification_inputs(
            certification_run_id="run-1",
            require_validated=True,
            max_questions=None,
            chroma_dir=tmp_path / "chroma_db_enriched",
        )


def test_main_sets_chroma_env_before_certification_readiness(monkeypatch, tmp_path):
    eval_path = tmp_path / "eval.json"
    chroma_dir = tmp_path / "chroma_db_enriched"
    eval_path.write_text(
        json.dumps({
            "questions": [
                {
                    "id": "q1",
                    "answerable": True,
                    "validation_status": "VALIDATED_BY_DELF_EXPERT",
                }
            ]
        }),
        encoding="utf-8",
    )
    calls = {}

    def fake_check_readiness(*, run_smoke, chroma_dir):
        calls["run_smoke"] = run_smoke
        calls["chroma_dir"] = chroma_dir
        calls["env_chroma_dir"] = os.environ.get("CHROMA_DIR")
        calls["env_eval_path"] = os.environ.get("RAGAS_TEST_SET_PATH")
        return SimpleNamespace(ok=True, failures=[])

    monkeypatch.setitem(
        sys.modules,
        "src.readiness_check",
        SimpleNamespace(check_readiness=fake_check_readiness),
    )
    monkeypatch.setattr(sys, "argv", [
        "measure_citation_grounding",
        "--eval-path",
        str(eval_path),
        "--chroma-dir",
        str(chroma_dir),
        "--require-validated",
        "--certification-run-id",
        "run-1",
        "--preflight",
    ])

    assert main() == 0
    assert calls["run_smoke"] is False
    assert calls["chroma_dir"] == chroma_dir.resolve(strict=False)
    assert calls["env_chroma_dir"] == str(chroma_dir)
    assert calls["env_eval_path"] == str(eval_path)
