"""Tests for packaging the expert validation packet."""

from __future__ import annotations

import json
import zipfile

from scripts.package_expert_validation import DEFAULT_FILES, REPO, create_packet


def test_create_packet_writes_zip_with_manifest(tmp_path):
    output = tmp_path / "packet.zip"
    payload = create_packet([
        REPO / "evaluation" / "delf_questions_expert_review.csv",
        REPO / "evaluation" / "EXPERT_VALIDATION_GUIDE.md",
    ], output)

    assert output.exists()
    assert payload["sha256"]
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        assert "evaluation/delf_questions_expert_review.csv" in names
        assert "evaluation/EXPERT_VALIDATION_GUIDE.md" in names
        manifest = json.loads(
            zf.read("evaluation/delf_expert_validation_packet_manifest.json")
        )
    assert manifest["files"][0]["sha256"]


def test_default_packet_files_include_eval_and_corpus_provenance():
    names = {path.relative_to(REPO).as_posix() for path in DEFAULT_FILES}

    assert "evaluation/delf_questions.json" in names
    assert "evaluation/corpus_files.json" in names
    assert "evaluation/delf_questions_expert_review.csv" in names
    assert "evaluation/delf_source_catalog.csv" in names
