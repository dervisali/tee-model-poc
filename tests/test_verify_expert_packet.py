"""Tests for expert-validation packet verification."""

from __future__ import annotations

import json
import zipfile

from scripts.package_expert_validation import REPO, create_packet
from scripts.verify_expert_packet import MANIFEST_ARCNAME, verify_packet


def test_verify_packet_accepts_matching_manifest(tmp_path):
    packet = tmp_path / "packet.zip"
    create_packet([
        REPO / "evaluation" / "delf_questions_expert_review.csv",
        REPO / "evaluation" / "EXPERT_VALIDATION_GUIDE.md",
    ], packet)

    result = verify_packet(packet)

    assert result["ok"] is True
    assert result["packet_sha256"]
    assert result["failures"] == []


def test_verify_packet_rejects_tampered_file(tmp_path):
    packet = tmp_path / "packet.zip"
    create_packet([
        REPO / "evaluation" / "delf_questions_expert_review.csv",
        REPO / "evaluation" / "EXPERT_VALIDATION_GUIDE.md",
    ], packet)

    with zipfile.ZipFile(packet) as zf:
        manifest = json.loads(zf.read(MANIFEST_ARCNAME).decode("utf-8"))
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "evaluation/delf_questions_expert_review.csv",
            "id,validation_decision,expert_corrected_sources,expert_corrected_ground_truth\nq1,VALIDATED,,\n",
        )
        zf.writestr("evaluation/EXPERT_VALIDATION_GUIDE.md", "# changed\n")
        zf.writestr(MANIFEST_ARCNAME, json.dumps(manifest))

    result = verify_packet(tampered)

    assert result["ok"] is False
    assert "evaluation/delf_questions_expert_review.csv sha256 mismatch" in result["failures"]
    assert "evaluation/EXPERT_VALIDATION_GUIDE.md sha256 mismatch" in result["failures"]


def test_verify_packet_rejects_missing_manifest(tmp_path):
    packet = tmp_path / "packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("evaluation/delf_questions_expert_review.csv", "id\nq1\n")

    result = verify_packet(packet)

    assert result["ok"] is False
    assert f"manifest missing: {MANIFEST_ARCNAME}" in result["failures"]
