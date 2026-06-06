"""Tests for the guarded certification pipeline helpers."""

from __future__ import annotations

import json
import sys
from argparse import Namespace

import pytest

from scripts.run_certification_pipeline import (
    _write_pipeline_deploy_plan,
    _write_pipeline_snapshot_plan,
    _write_summary,
    main,
)


def test_write_summary_creates_artifact():
    path = _write_summary({"ok": True})
    data = json.loads(path.read_text(encoding="utf-8"))

    assert path.name.startswith("certification_pipeline_")
    assert path.name.endswith(".json")
    assert data["ok"] is True
    path.unlink()


def test_write_pipeline_deploy_plan_uses_pipeline_results_dir(tmp_path):
    args = Namespace(
        deploy_project="p",
        deploy_image="img:abc",
        deploy_service_account="sa@p.iam.gserviceaccount.com",
        deploy_gcs_bucket="bucket",
        deploy_invoker="group:team@org.test",
        deploy_service="svc",
        deploy_region="us-central1",
        deploy_location=None,
        snapshot_prefix="custom-prefix",
        certification_run_id="run-1",
        results_dir=tmp_path,
    )

    path = _write_pipeline_deploy_plan(args)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == tmp_path
    assert path.name.startswith("cloud_run_deploy_plan_")
    assert data["metadata"]["service"] == "svc"
    assert data["metadata"]["snapshot_prefix"] == "custom-prefix"
    assert data["env_vars"]["GCS_SNAPSHOT_PREFIX"] == "custom-prefix"
    assert data["metadata"]["invoker_member"] == "group:team@org.test"
    assert data["metadata"]["certification_run_id"] == "run-1"
    assert data["safety_invariants"]["private_cloud_run"] is True


def test_write_pipeline_deploy_plan_requires_all_deploy_args(tmp_path):
    args = Namespace(
        deploy_project="p",
        deploy_image=None,
        deploy_service_account="sa@p.iam.gserviceaccount.com",
        deploy_gcs_bucket="bucket",
        deploy_invoker="group:team@org.test",
        deploy_service="svc",
        deploy_region="us-central1",
        deploy_location=None,
        snapshot_prefix="tee-corpus",
        certification_run_id="run-1",
        results_dir=tmp_path,
    )

    with pytest.raises(SystemExit, match="--write-deploy-plan requires --deploy-image"):
        _write_pipeline_deploy_plan(args)


def test_write_pipeline_snapshot_plan_uses_deploy_bucket_and_run_id(tmp_path):
    chroma_dir = tmp_path / "chroma_db_enriched"
    chroma_dir.mkdir()
    (chroma_dir / "parents.json").write_text('{"p1": {"text": "ok"}}', encoding="utf-8")
    (chroma_dir / "bm25_index.pkl").write_bytes(b"bm25")
    (chroma_dir / "chroma.sqlite3").write_bytes(b"SQLite format 3\x00" + b"\x01" * 256)
    index_dir = chroma_dir / "abc-collection"
    index_dir.mkdir()
    for filename in (
        "data_level0.bin",
        "header.bin",
        "length.bin",
        "link_lists.bin",
        "index_metadata.pickle",
    ):
        (index_dir / filename).write_bytes(b"index")
    args = Namespace(
        chroma_dir=chroma_dir,
        snapshot_gcs_bucket=None,
        deploy_gcs_bucket="bucket",
        snapshot_prefix="tee-corpus",
        certification_run_id="run-1",
        results_dir=tmp_path,
    )

    path = _write_pipeline_snapshot_plan(args)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == tmp_path
    assert path.name.startswith("corpus_snapshot_publish_plan_")
    assert data["metadata"]["gcs_bucket"] == "bucket"
    assert data["metadata"]["snapshot_prefix"] == "tee-corpus"
    assert data["metadata"]["certification_run_id"] == "run-1"
    assert data["chroma_dir_readiness"]["ok"] is True


def test_write_pipeline_snapshot_plan_requires_bucket(tmp_path):
    args = Namespace(
        chroma_dir=tmp_path / "chroma_db_enriched",
        snapshot_gcs_bucket=None,
        deploy_gcs_bucket=None,
        snapshot_prefix="tee-corpus",
        certification_run_id="run-1",
        results_dir=tmp_path,
    )

    with pytest.raises(SystemExit, match="--write-snapshot-plan requires"):
        _write_pipeline_snapshot_plan(args)


def test_pipeline_does_not_write_deploy_artifacts_when_review_incomplete(tmp_path, monkeypatch):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    corpus_path = tmp_path / "corpus_files.json"
    validated_path = tmp_path / "validated.json"
    results_dir = tmp_path / "results"

    eval_path.write_text(json.dumps({"questions": [{"id": "q1"}]}), encoding="utf-8")
    corpus_path.write_text(json.dumps({"files": {"manuel-exacor.pdf": {}}}), encoding="utf-8")
    review_path.write_text(
        "id,validation_decision,expert_corrected_sources,expert_corrected_ground_truth\n"
        "q1,,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "run_certification_pipeline",
        "--eval-path",
        str(eval_path),
        "--review",
        str(review_path),
        "--corpus-files",
        str(corpus_path),
        "--validated-output",
        str(validated_path),
        "--results-dir",
        str(results_dir),
        "--run-enriched-ingest",
        "--measure-citations",
        "--write-deploy-plan",
        "--write-snapshot-plan",
        "--deploy-project",
        "woven-operative-491610-u6",
        "--deploy-image",
        "us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
        "--deploy-service-account",
        "delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
        "--deploy-gcs-bucket",
        "delf-corpus-prod",
        "--deploy-invoker",
        "group:reviewers@example.org",
    ])

    assert main() == 2
    assert not list(results_dir.glob("cloud_run_deploy_plan_*.json"))
    assert not list(results_dir.glob("corpus_snapshot_publish_plan_*.json"))


def test_pipeline_refuses_deploy_artifacts_without_live_gates(tmp_path, monkeypatch):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    corpus_path = tmp_path / "corpus_files.json"
    validated_path = tmp_path / "validated.json"
    results_dir = tmp_path / "results"

    eval_path.write_text(json.dumps({"questions": [{"id": "q1"}]}), encoding="utf-8")
    corpus_path.write_text(json.dumps({"files": {"manuel-exacor.pdf": {}}}), encoding="utf-8")
    review_path.write_text(
        "id,validation_decision,expert_corrected_sources,expert_corrected_ground_truth\n"
        "q1,VALIDATED,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "run_certification_pipeline",
        "--eval-path",
        str(eval_path),
        "--review",
        str(review_path),
        "--corpus-files",
        str(corpus_path),
        "--validated-output",
        str(validated_path),
        "--results-dir",
        str(results_dir),
        "--write-deploy-plan",
        "--deploy-project",
        "woven-operative-491610-u6",
        "--deploy-image",
        "us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
        "--deploy-service-account",
        "delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
        "--deploy-gcs-bucket",
        "delf-corpus-prod",
        "--deploy-invoker",
        "group:reviewers@example.org",
    ])

    with pytest.raises(SystemExit, match="require --run-enriched-ingest and --measure-citations"):
        main()
    assert not results_dir.exists()


def test_pipeline_skips_deploy_artifacts_when_live_gates_do_not_pass(tmp_path, monkeypatch):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    corpus_path = tmp_path / "corpus_files.json"
    validated_path = tmp_path / "validated.json"
    results_dir = tmp_path / "results"
    chroma_dir = tmp_path / "chroma_db_enriched"

    eval_path.write_text(
        json.dumps({
            "questions": [
                {
                    "id": "q1",
                    "answerable": True,
                    "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT",
                    "expected_sources": ["manuel-exacor.pdf"],
                    "ground_truth": "answer",
                }
            ]
        }),
        encoding="utf-8",
    )
    corpus_path.write_text(json.dumps({"files": {"manuel-exacor.pdf": {}}}), encoding="utf-8")
    review_path.write_text(
        "id,validation_decision,expert_corrected_sources,expert_corrected_ground_truth\n"
        "q1,VALIDATED,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.run_certification_pipeline._run", lambda cmd: None)
    monkeypatch.setattr(sys, "argv", [
        "run_certification_pipeline",
        "--eval-path",
        str(eval_path),
        "--review",
        str(review_path),
        "--corpus-files",
        str(corpus_path),
        "--validated-output",
        str(validated_path),
        "--chroma-dir",
        str(chroma_dir),
        "--results-dir",
        str(results_dir),
        "--run-enriched-ingest",
        "--measure-citations",
        "--write-deploy-plan",
        "--write-snapshot-plan",
        "--deploy-project",
        "woven-operative-491610-u6",
        "--deploy-image",
        "us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
        "--deploy-service-account",
        "delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
        "--deploy-gcs-bucket",
        "delf-corpus-prod",
        "--deploy-invoker",
        "group:reviewers@example.org",
    ])

    assert main() == 2
    assert validated_path.exists()
    assert not list(results_dir.glob("cloud_run_deploy_plan_*.json"))
    assert not list(results_dir.glob("corpus_snapshot_publish_plan_*.json"))
    summaries = list(results_dir.glob("certification_pipeline_*.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["deployment_artifacts_skipped"]["failed_gates"] == [
        "m3_recall",
        "m4_citation_grounding",
    ]


def test_pipeline_skips_deploy_artifacts_when_live_gates_are_from_stale_run_id(
    tmp_path,
    monkeypatch,
):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    corpus_path = tmp_path / "corpus_files.json"
    validated_path = tmp_path / "validated.json"
    results_dir = tmp_path / "results"
    chroma_dir = tmp_path / "chroma_db_enriched"

    eval_path.write_text(
        json.dumps({
            "questions": [
                {
                    "id": "q1",
                    "answerable": True,
                    "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT",
                    "expected_sources": ["manuel-exacor.pdf"],
                    "ground_truth": "answer",
                }
            ]
        }),
        encoding="utf-8",
    )
    corpus_path.write_text(json.dumps({"files": {"manuel-exacor.pdf": {}}}), encoding="utf-8")
    review_path.write_text(
        "id,validation_decision,expert_corrected_sources,expert_corrected_ground_truth\n"
        "q1,VALIDATED,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.run_certification_pipeline._run", lambda cmd: None)
    monkeypatch.setattr(
        "scripts.run_certification_pipeline.enriched_status",
        lambda results: {
            "artifact": "old-m3.json",
            "passed": True,
            "certification_run_id": "old-run",
        },
    )
    monkeypatch.setattr(
        "scripts.run_certification_pipeline.citation_status",
        lambda results: {
            "artifact": "old-m4.json",
            "passed": True,
            "certification_run_id": "old-run",
        },
    )
    monkeypatch.setattr(sys, "argv", [
        "run_certification_pipeline",
        "--certification-run-id",
        "current-run",
        "--eval-path",
        str(eval_path),
        "--review",
        str(review_path),
        "--corpus-files",
        str(corpus_path),
        "--validated-output",
        str(validated_path),
        "--chroma-dir",
        str(chroma_dir),
        "--results-dir",
        str(results_dir),
        "--run-enriched-ingest",
        "--measure-citations",
        "--write-deploy-plan",
        "--write-snapshot-plan",
        "--deploy-project",
        "woven-operative-491610-u6",
        "--deploy-image",
        "us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
        "--deploy-service-account",
        "delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
        "--deploy-gcs-bucket",
        "delf-corpus-prod",
        "--deploy-invoker",
        "group:reviewers@example.org",
    ])

    assert main() == 2
    assert not list(results_dir.glob("cloud_run_deploy_plan_*.json"))
    assert not list(results_dir.glob("corpus_snapshot_publish_plan_*.json"))
    summaries = list(results_dir.glob("certification_pipeline_*.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["deployment_artifacts_skipped"]["failed_gates"] == []
    assert summary["deployment_artifacts_skipped"]["run_id_mismatches"] == [
        "m3_recall",
        "m4_citation_grounding",
    ]
    assert summary["deployment_artifacts_skipped"]["expected_certification_run_id"] == "current-run"


def test_pipeline_skips_deploy_artifacts_when_live_gate_artifacts_are_stale_paths(
    tmp_path,
    monkeypatch,
):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    corpus_path = tmp_path / "corpus_files.json"
    validated_path = tmp_path / "validated.json"
    results_dir = tmp_path / "results"
    chroma_dir = tmp_path / "chroma_db_enriched"
    results_dir.mkdir()

    eval_path.write_text(
        json.dumps({
            "questions": [
                {
                    "id": "q1",
                    "answerable": True,
                    "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT",
                    "expected_sources": ["manuel-exacor.pdf"],
                    "ground_truth": "answer",
                }
            ]
        }),
        encoding="utf-8",
    )
    corpus_path.write_text(json.dumps({"files": {"manuel-exacor.pdf": {}}}), encoding="utf-8")
    review_path.write_text(
        "id,validation_decision,expert_corrected_sources,expert_corrected_ground_truth\n"
        "q1,VALIDATED,,\n",
        encoding="utf-8",
    )
    (results_dir / "enriched_experiment_20260530T000000Z.json").write_text(
        json.dumps({
            "metadata": {
                "eval_path": str(tmp_path / "old_validated.json"),
                "experiment_chroma_dir": str(tmp_path / "old_chroma_db_enriched"),
                "certification_run_id": "current-run",
            },
            "preflight": {
                "eval_path": str(tmp_path / "old_validated.json"),
            },
            "recall_with_rerank": {
                "n_questions_evaluated": 1,
            },
        }),
        encoding="utf-8",
    )
    (results_dir / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps({
            "metadata": {
                "eval_path": str(tmp_path / "old_validated.json"),
                "chroma_dir": str(tmp_path / "old_chroma_db_enriched"),
                "questions_selected": 1,
                "certification_run_id": "current-run",
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.run_certification_pipeline._run", lambda cmd: None)
    monkeypatch.setattr(
        "scripts.run_certification_pipeline.enriched_status",
        lambda results: {
            "artifact": "m3.json",
            "passed": True,
            "certification_run_id": "current-run",
        },
    )
    monkeypatch.setattr(
        "scripts.run_certification_pipeline.citation_status",
        lambda results: {
            "artifact": "m4.json",
            "passed": True,
            "certification_run_id": "current-run",
        },
    )
    monkeypatch.setattr(sys, "argv", [
        "run_certification_pipeline",
        "--certification-run-id",
        "current-run",
        "--eval-path",
        str(eval_path),
        "--review",
        str(review_path),
        "--corpus-files",
        str(corpus_path),
        "--validated-output",
        str(validated_path),
        "--chroma-dir",
        str(chroma_dir),
        "--results-dir",
        str(results_dir),
        "--run-enriched-ingest",
        "--measure-citations",
        "--write-deploy-plan",
        "--write-snapshot-plan",
        "--deploy-project",
        "woven-operative-491610-u6",
        "--deploy-image",
        "us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
        "--deploy-service-account",
        "delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
        "--deploy-gcs-bucket",
        "delf-corpus-prod",
        "--deploy-invoker",
        "group:reviewers@example.org",
    ])

    assert main() == 2
    assert not list(results_dir.glob("cloud_run_deploy_plan_*.json"))
    assert not list(results_dir.glob("corpus_snapshot_publish_plan_*.json"))
    summaries = list(results_dir.glob("certification_pipeline_*.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["deployment_artifacts_skipped"]["failed_gates"] == []
    assert summary["deployment_artifacts_skipped"]["run_id_mismatches"] == []
    assert "M3 eval_path must match current validated output before deploy artifacts" in (
        summary["deployment_artifacts_skipped"]["alignment_failures"]
    )
    assert "M4 chroma_dir must match current chroma_dir before deploy artifacts" in (
        summary["deployment_artifacts_skipped"]["alignment_failures"]
    )
