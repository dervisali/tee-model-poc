"""Tests for certification status artifact inspection."""

from __future__ import annotations

import json
import csv
from pathlib import Path

from scripts.apply_expert_validation import default_manifest_path
from scripts.certification_status import (
    REQUIRED_DOCKERIGNORE_PATTERNS,
    _sha256,
    citation_status,
    certification_status,
    container_image_status,
    enriched_status,
    eval_status,
)
from scripts.render_cloud_run_deploy import DeployConfig, build_deploy_plan
from scripts.render_snapshot_publish_plan import SnapshotPublishConfig, build_snapshot_publish_plan
from scripts.run_enriched_retrieval_experiment import _recall_gate_summary


def _validated_eval_items(n: int = 20) -> list[dict]:
    return [
        {
            "id": f"q{i + 1}",
            "answerable": True,
            "validation_status": "VALIDATED_BY_DELF_EXPERT",
            "expected_sources": [f"source-{i + 1}.pdf"],
        }
        for i in range(n)
    ]


def _write_corpus_files(path, questions):
    files = {}
    for item in questions:
        for source in item.get("expected_sources", []):
            files[source] = {}
    path.write_text(json.dumps({"files": files}), encoding="utf-8")


def _write_validation_manifest(eval_path, *, strict=True, counts=None, corpus_files_path=None):
    questions = json.loads(eval_path.read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions.get("questions", [])
    input_eval_path = eval_path.with_name("delf_questions.json")
    input_eval_path.write_text(
        json.dumps({"questions": questions, "source": "pre_expert_review"}),
        encoding="utf-8",
    )
    review_path = eval_path.with_name("review.csv")
    review_path.write_text(
        "id,validation_decision,expert_corrected_sources,expert_corrected_ground_truth\n"
        + "\n".join(f"{item.get('id')},VALIDATED,," for item in questions),
        encoding="utf-8",
    )
    default_counts = {
        "validated": sum(
            1
            for item in questions
            if item.get("answerable", True)
            and str(item.get("validation_status", "")).startswith("VALIDATED")
        ),
        "fixed": 0,
        "dropped": sum(
            1
            for item in questions
            if str(item.get("validation_status", "")).startswith("DROPPED")
        ),
        "unchanged": 0,
    }
    manifest = {
        "created_at": "2026-05-30T00:00:00+00:00",
        "strict": strict,
        "input_eval_path": str(input_eval_path),
        "review_path": str(review_path),
        "output_eval_path": str(eval_path),
        "input_eval_sha256": _sha256(input_eval_path),
        "review_sha256": _sha256(review_path),
        "output_eval_sha256": _sha256(eval_path),
        "counts": counts or default_counts,
    }
    if corpus_files_path is not None:
        manifest["corpus_files_path"] = str(corpus_files_path)
        manifest["corpus_files_sha256"] = _sha256(corpus_files_path)
    default_manifest_path(eval_path).write_text(json.dumps(manifest), encoding="utf-8")


def _controlled_enriched_artifact(
    m3: float = 0.91,
    *,
    eval_path: str = "/tmp/delf_questions.validated.json",
    chroma_dir: str = "/tmp/chroma_db_enriched",
    n: int = 20,
    run_id: str = "run-1",
) -> dict:
    return {
        "metadata": {
            "experiment_chroma_dir": chroma_dir,
            "eval_path": eval_path,
            "contextual_enrichment": True,
            "chunking_strategy": "paragraph",
            "embedding_dimension": 3072,
            "cross_lingual_bm25": False,
            "certification_run_id": run_id,
        },
        "preflight": {
            "documents_total": 56,
            "documents_nonempty": 56,
            "document_failures": [],
            "eval_path": eval_path,
            "eval_questions": n,
            "eval_unvalidated": 0,
        },
        "experiment_db_readiness": {
            "ok": True,
            "failures": [],
            "parent_count": 1302,
            "child_count": 4950,
        },
        "baseline_integrity": {
            "ok": True,
            "baseline_chroma_dir": "/tmp/chroma_db",
            "before": {
                "path": "/tmp/chroma_db",
                "present": True,
                "file_count": 4,
                "total_bytes": 143000000,
                "sha256": "abc123",
            },
            "after": {
                "path": "/tmp/chroma_db",
                "present": True,
                "file_count": 4,
                "total_bytes": 143000000,
                "sha256": "abc123",
            },
        },
        "recall_with_rerank": {
            "m3_recall_at_k": m3,
            "m3_k": 3,
            "n_questions_evaluated": n,
        },
    }


def _write_controlled_enriched_artifact(results, artifact, name="enriched_experiment_20260530T000000Z.json"):
    recall = artifact.get("recall_with_rerank") or artifact.get("recall_no_rerank") or {}
    metadata = artifact.get("metadata", {})
    n = recall.get("n_questions_evaluated", 0)
    m3 = recall.get("m3_recall_at_k")
    m3_k = recall.get("m3_k", 3)
    report_path = results / f"recall_for_{name.removesuffix('.json')}.json"
    md_path = results / f"recall_for_{name.removesuffix('.json')}.md"
    report = {
        "metadata": {
            "n_questions_evaluated": n,
            "m3_k": m3_k,
            "m3_recall_at_k": m3,
            "test_set": metadata.get("eval_path"),
        },
        "aggregate": {"overall": {"n": n, f"recall@{m3_k}": m3}},
        "per_question": [{"id": f"q{i + 1}"} for i in range(n)],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    md_path.write_text("# recall\n", encoding="utf-8")
    recall["artifact_json"] = str(report_path)
    recall["artifact_md"] = str(md_path)
    out = results / name
    out.write_text(json.dumps(artifact), encoding="utf-8")
    return out


def _write_controlled_snapshot_plan(
    results,
    *,
    chroma_dir: str = "/tmp/chroma_db_enriched",
    gcs_bucket: str = "delf-corpus-prod",
    snapshot_prefix: str = "tee-corpus",
    run_id: str = "run-1",
):
    chroma_path = Path(chroma_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)
    (chroma_path / "parents.json").write_text('{"p1": {"text": "ok"}}', encoding="utf-8")
    (chroma_path / "bm25_index.pkl").write_bytes(b"bm25")
    plan = build_snapshot_publish_plan(
        SnapshotPublishConfig(
            chroma_dir=chroma_dir,
            gcs_bucket=gcs_bucket,
            snapshot_prefix=snapshot_prefix,
            certification_run_id=run_id,
        )
    )
    (results / "corpus_snapshot_publish_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    return plan


def _controlled_citation_artifact(
    rate: float = 0.95,
    *,
    eval_path: str = "/tmp/delf_questions.validated.json",
    chroma_dir: str = "/tmp/chroma_db_enriched",
    n: int = 20,
    run_id: str = "run-1",
) -> dict:
    passed = int(rate * n)
    return {
        "metadata": {
            "eval_path": eval_path,
            "chroma_dir": chroma_dir,
            "questions_total": n,
            "questions_selected": n,
            "require_validated": True,
            "max_questions": None,
            "certification_run_id": run_id,
        },
        "summary": {
            "n": n,
            "citation_passed": passed,
            "citation_pass_rate": rate,
            "no_error": n,
            "no_error_rate": 1.0,
            "grounding_allowed": n,
            "grounding_allowed_rate": 1.0,
        },
        "per_question": [
            {
                "id": f"q{i + 1}",
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "citation_passed": i < passed,
                "error": None,
                "safety": {"grounding_allowed": True},
            }
            for i in range(n)
        ],
    }


def test_recall_gate_summary_marks_m3_pass():
    report = {
        "metadata": {
            "artifact_json": "x.json",
            "artifact_md": "x.md",
            "n_questions_evaluated": 50,
            "m3_k": 3,
            "m3_recall_at_k": 0.91,
            "m3_hit_at_k": 0.96,
        },
        "aggregate": {
            "overall": {
                "recall@1": 0.5,
                "recall@3": 0.91,
                "recall@5": 0.91,
                "recall@10": 0.97,
            }
        },
    }

    out = _recall_gate_summary(report)

    assert out["m3_passed"] is True
    assert out["recall_at_5"] == 0.91


def test_certification_status_requires_all_production_gates(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps(_validated_eval_items()),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    _write_controlled_enriched_artifact(
        results,
        _controlled_enriched_artifact(m3=0.9, eval_path=str(eval_path)),
    )
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(_controlled_citation_artifact(rate=0.95, eval_path=str(eval_path))),
        encoding="utf-8",
    )
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    assert out["certified"] is True
    assert out["gates"]["deployment_safety"]["passed"] is True
    assert out["gates"]["snapshot_publication"]["passed"] is True
    assert out["gates"]["container_image"]["passed"] is True
    assert out["next_actions"] == []


def test_container_image_status_rejects_root_runtime(tmp_path):
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "ENTRYPOINT [\"./entrypoint.sh\"]\n",
        encoding="utf-8",
    )
    (tmp_path / ".dockerignore").write_text(
        "\n".join(sorted(REQUIRED_DOCKERIGNORE_PATTERNS)),
        encoding="utf-8",
    )

    out = container_image_status(tmp_path)

    assert out["passed"] is False
    assert "Dockerfile must switch to non-root USER app before ENTRYPOINT" in out["failures"]


def test_container_image_status_rejects_missing_context_exclusions(tmp_path):
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "ENV HF_HOME=/tmp/hf-cache XDG_CACHE_HOME=/tmp/.cache\n"
        "RUN useradd --system app && mkdir -p /tmp/chroma_db\n"
        "USER app\n"
        "ENTRYPOINT [\"./entrypoint.sh\"]\n",
        encoding="utf-8",
    )
    (tmp_path / ".dockerignore").write_text(".env\nchroma_db/\n", encoding="utf-8")

    out = container_image_status(tmp_path)

    assert out["passed"] is False
    assert any(".dockerignore missing production exclusions" in failure for failure in out["failures"])


def test_certification_status_fails_when_eval_unvalidated(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {"id": "q1", "answerable": True, "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT"}
        ]),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    assert out["gates"]["eval_validation"]["unvalidated_answerable"] == 1
    assert out["gates"]["eval_validation"]["unvalidated_answerable_ids"] == ["q1"]
    assert out["next_actions"][0]["gate"] == "eval_validation"
    assert out["next_actions"][0]["owner"] == "DELF/DALF expert"
    assert out["next_actions"][0]["current_gap"]["unvalidated_answerable"] == 1
    assert "scripts.lint_expert_review" in out["next_actions"][0]["commands"][0]


def test_eval_status_surfaces_non_expert_review_preflight(tmp_path):
    eval_path = tmp_path / "eval.json"
    review_path = tmp_path / "review.csv"
    corpus_path = tmp_path / "corpus_files.json"
    questions = [
        {
            "id": "q1",
            "answerable": True,
            "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT",
            "expected_sources": ["source.pdf"],
        }
    ]
    eval_path.write_text(json.dumps(questions), encoding="utf-8")
    _write_corpus_files(corpus_path, questions)
    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "validation_decision",
                "expert_corrected_sources",
                "expert_corrected_ground_truth",
                "expert_notes",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "id": "q1",
            "validation_decision": "VALIDATED",
            "expert_notes": "AI-grounded draft",
        })

    out = eval_status(eval_path, corpus_path, review_path)

    assert out["passed"] is False
    assert "current expert review CSV lint failed" in out["failures"]
    assert out["review_preflight"]["ok"] is False
    assert any("non-expert/AI review evidence" in error for error in out["review_preflight"]["errors"])


def test_certification_status_rejects_validated_eval_without_manifest(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    assert "validation manifest missing" in out["gates"]["eval_validation"]["failures"]


def test_enriched_status_rejects_experiment_dir_inside_baseline(tmp_path):
    results = tmp_path
    artifact_path = results / "enriched_experiment_99991231T235959Z.json"
    recall_path = results / "recall_for_unsafe_baseline_nested.json"
    recall_md = results / "recall_for_unsafe_baseline_nested.md"
    recall_path.write_text(
        json.dumps({
            "metadata": {
                "n_questions_evaluated": 1,
                "m3_k": 3,
                "m3_recall_at_k": 0.91,
                "test_set": "/tmp/delf_questions.validated.json",
            },
            "aggregate": {"overall": {"n": 1, "recall@3": 0.91}},
            "per_question": [{"id": "q1"}],
        }),
        encoding="utf-8",
    )
    recall_md.write_text("# recall\n", encoding="utf-8")
    artifact_path.write_text(
        json.dumps(_controlled_enriched_artifact(
            m3=0.91,
            eval_path="/tmp/delf_questions.validated.json",
            chroma_dir=str(Path.cwd() / "chroma_db" / "nested_experiment"),
            n=1,
        )),
        encoding="utf-8",
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["recall_with_rerank"]["artifact_json"] = str(recall_path)
    artifact["recall_with_rerank"]["artifact_md"] = str(recall_md)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    try:
        out = enriched_status(results)
        assert out["passed"] is False
        assert "experiment_chroma_dir must not be inside the baseline chroma_db" in out["failures"]
    finally:
        artifact_path.unlink(missing_ok=True)
        recall_path.unlink(missing_ok=True)
        recall_md.unlink(missing_ok=True)


def test_certification_status_rejects_manifest_count_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items(n=2)), encoding="utf-8")
    _write_validation_manifest(
        eval_path,
        counts={"validated": 1, "fixed": 2, "dropped": 2, "unchanged": 1},
    )

    out = certification_status(eval_path, results)

    failures = out["gates"]["eval_validation"]["failures"]
    assert "validation manifest validated count must match eval file" in failures
    assert "validation manifest dropped count must match eval file" in failures
    assert "validation manifest unchanged count must be zero" in failures
    assert "validation manifest fixed count cannot exceed validated count" in failures
    assert "validation manifest validated+dropped count must equal eval total" in failures


def test_certification_status_rejects_review_hash_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items(n=2)), encoding="utf-8")
    _write_validation_manifest(eval_path)
    eval_path.with_name("review.csv").write_text("tampered\n", encoding="utf-8")

    out = certification_status(eval_path, results)

    failures = out["gates"]["eval_validation"]["failures"]
    assert "validation manifest review sha256 mismatch" in failures


def test_certification_status_rejects_input_eval_hash_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items(n=2)), encoding="utf-8")
    _write_validation_manifest(eval_path)
    eval_path.with_name("delf_questions.json").write_text("tampered\n", encoding="utf-8")

    out = certification_status(eval_path, results)

    failures = out["gates"]["eval_validation"]["failures"]
    assert "validation manifest input eval sha256 mismatch" in failures


def test_certification_status_rejects_duplicate_eval_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source-a.pdf"],
            },
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source-b.pdf"],
            },
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)

    out = certification_status(eval_path, results)

    failures = out["gates"]["eval_validation"]["failures"]
    assert "eval file contains duplicate IDs: q1" in failures


def test_certification_status_rejects_blank_eval_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source-a.pdf"],
            },
            {
                "id": "",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source-b.pdf"],
            },
            {
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source-c.pdf"],
            },
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)

    out = certification_status(eval_path, results)

    failures = out["gates"]["eval_validation"]["failures"]
    assert "eval file contains blank or missing IDs" in failures


def test_certification_status_rejects_manifest_without_corpus_catalog_binding(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    questions = _validated_eval_items(n=2)
    corpus_path = tmp_path / "corpus_files.json"
    eval_path.write_text(json.dumps(questions), encoding="utf-8")
    _write_corpus_files(corpus_path, questions)
    _write_validation_manifest(eval_path)

    out = certification_status(eval_path, results, corpus_path)

    failures = out["gates"]["eval_validation"]["failures"]
    assert "validation manifest corpus_files_path must match corpus catalog" in failures
    assert "validation manifest corpus_files sha256 mismatch" in failures


def test_certification_status_rejects_corpus_catalog_hash_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    questions = _validated_eval_items(n=2)
    corpus_path = tmp_path / "corpus_files.json"
    eval_path.write_text(json.dumps(questions), encoding="utf-8")
    _write_corpus_files(corpus_path, questions)
    _write_validation_manifest(eval_path, corpus_files_path=corpus_path)
    corpus_path.write_text(json.dumps({"files": {"other.pdf": {}}}), encoding="utf-8")

    out = certification_status(eval_path, results, corpus_path)

    failures = out["gates"]["eval_validation"]["failures"]
    assert "validation manifest corpus_files sha256 mismatch" in failures
    assert any("has sources outside corpus_files.json" in failure for failure in failures)


def test_certification_status_rejects_eval_sources_outside_corpus_catalog(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    questions = [
        {
            "id": "q1",
            "answerable": True,
            "validation_status": "VALIDATED_BY_DELF_EXPERT",
            "expected_sources": ["not-in-corpus.pdf"],
        }
    ]
    corpus_path = tmp_path / "corpus_files.json"
    eval_path.write_text(json.dumps(questions), encoding="utf-8")
    corpus_path.write_text(json.dumps({"files": {"source.pdf": {}}}), encoding="utf-8")
    _write_validation_manifest(eval_path, corpus_files_path=corpus_path)

    out = certification_status(eval_path, results, corpus_path)

    failures = out["gates"]["eval_validation"]["failures"]
    assert "eval item q1 has sources outside corpus_files.json: not-in-corpus.pdf" in failures


def test_certification_status_requires_deployment_safety_artifact(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    _write_controlled_enriched_artifact(
        results,
        _controlled_enriched_artifact(m3=0.91, eval_path=str(eval_path)),
    )
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(_controlled_citation_artifact(rate=0.96, eval_path=str(eval_path))),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    assert out["gates"]["deployment_safety"]["reason"] == "missing cloud_run_deploy_plan artifact"


def test_certification_status_rejects_placeholder_deploy_plan(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    plan["metadata"]["image"] = "us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:SHA"
    plan["metadata"]["gcs_bucket"] = "DELF_CORPUS_BUCKET"
    plan["metadata"]["snapshot_prefix"] = "REAL_SNAPSHOT_PREFIX"
    plan["metadata"]["invoker_member"] = "group:delf-examiners@example.org"
    plan["env_vars"]["GCS_BUCKET"] = "DELF_CORPUS_BUCKET"
    plan["env_vars"]["GCS_SNAPSHOT_PREFIX"] = "REAL_SNAPSHOT_PREFIX"
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    failures = out["gates"]["deployment_safety"]["failures"]
    assert "image is missing or placeholder" in failures
    assert "gcs_bucket is missing or placeholder" in failures
    assert "snapshot_prefix is missing or placeholder" in failures
    assert "GCS_BUCKET env is missing or placeholder" in failures
    assert "GCS_SNAPSHOT_PREFIX env is missing or placeholder" in failures
    assert "invoker_member is missing or placeholder" in failures


def test_certification_status_rejects_mutable_deploy_image_tag(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    plan["metadata"]["image"] = "us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:latest"
    plan["deploy_command"][plan["deploy_command"].index("--image") + 1] = plan["metadata"]["image"]
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    failures = out["gates"]["deployment_safety"]["failures"]
    assert "image must include an immutable digest or non-latest tag" in failures


def test_certification_status_rejects_cross_project_deploy_refs(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    plan["metadata"]["image"] = "us-central1-docker.pkg.dev/other-project/apps/delf:abc123"
    plan["deploy_command"][plan["deploy_command"].index("--image") + 1] = plan["metadata"]["image"]
    plan["metadata"]["service_account"] = "delf-runtime@other-project.iam.gserviceaccount.com"
    plan["deploy_command"][plan["deploy_command"].index("--service-account") + 1] = plan["metadata"]["service_account"]
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    failures = out["gates"]["deployment_safety"]["failures"]
    assert "image Artifact Registry project must match deploy project" in failures
    assert "service_account project must match deploy project" in failures


def test_certification_status_rejects_malformed_deploy_identities(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    plan["metadata"]["service_account"] = "delf-runtime"
    plan["deploy_command"][plan["deploy_command"].index("--service-account") + 1] = "delf-runtime"
    plan["metadata"]["invoker_member"] = "delf-examiners@org.test"
    plan["invoker_command"][plan["invoker_command"].index("--member") + 1] = "delf-examiners@org.test"
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    failures = out["gates"]["deployment_safety"]["failures"]
    assert "service_account must be a service account email" in failures
    assert "invoker_member must be an IAM member with group:, user:, serviceAccount:, or domain:" in failures


def test_certification_status_rejects_deploy_plan_without_runtime_iam(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    plan["runtime_iam_commands"] = []
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    failures = out["gates"]["deployment_safety"]["failures"]
    assert "runtime IAM command missing roles/aiplatform.user" in failures
    assert "runtime IAM command missing roles/storage.objectViewer" in failures


def test_certification_status_rejects_runtime_iam_for_wrong_member_or_bucket(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    plan["runtime_iam_commands"] = [
        [
            "gcloud",
            "projects",
            "add-iam-policy-binding",
            "woven-operative-491610-u6",
            "--member",
            "serviceAccount:other@woven-operative-491610-u6.iam.gserviceaccount.com",
            "--role",
            "roles/aiplatform.user",
        ],
        [
            "gcloud",
            "storage",
            "buckets",
            "add-iam-policy-binding",
            "gs://other-bucket",
            "--member",
            "serviceAccount:other@woven-operative-491610-u6.iam.gserviceaccount.com",
            "--role",
            "roles/storage.objectViewer",
        ],
    ]
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results, chroma_dir="/tmp/chroma_db_enriched_a")

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    failures = out["gates"]["deployment_safety"]["failures"]
    assert "Vertex runtime IAM command must target deploy service account" in failures
    assert "GCS runtime IAM command must target deploy service account" in failures
    assert "GCS runtime IAM command must target corpus bucket" in failures


def test_certification_status_rejects_deploy_plan_command_metadata_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    image_idx = plan["deploy_command"].index("--image") + 1
    plan["deploy_command"][image_idx] = "us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:other"
    invoker_idx = plan["invoker_command"].index("--member") + 1
    plan["invoker_command"][invoker_idx] = "group:other@org.test"
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    failures = out["gates"]["deployment_safety"]["failures"]
    assert "deploy command --image must match metadata image" in failures
    assert "invoker command member must match metadata invoker_member" in failures


def test_certification_status_rejects_deploy_plan_env_metadata_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    plan["env_vars"]["GCS_BUCKET"] = "other-corpus-bucket"
    plan["env_vars"]["GCS_SNAPSHOT_PREFIX"] = "other-prefix"
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results, chroma_dir="/tmp/chroma_db_enriched_a")

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    failures = out["gates"]["deployment_safety"]["failures"]
    assert "deploy command env GCS_BUCKET must match deploy-plan env_vars" in failures
    assert "deploy command env GCS_SNAPSHOT_PREFIX must match deploy-plan env_vars" in failures
    assert "GCS_BUCKET env must match metadata gcs_bucket" in failures
    assert "GCS_SNAPSHOT_PREFIX env must match metadata snapshot_prefix" in failures


def test_certification_status_rejects_deploy_bucket_uri_in_runtime_fields(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    plan["metadata"]["gcs_bucket"] = "gs://delf-corpus-prod"
    plan["env_vars"]["GCS_BUCKET"] = "gs://delf-corpus-prod"
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    failures = out["gates"]["deployment_safety"]["failures"]
    assert "gcs_bucket must be a bucket name, not a gs:// URI" in failures
    assert "GCS_BUCKET env must be a bucket name, not a gs:// URI" in failures


def test_certification_status_rejects_undersized_cloud_run_resources(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    plan["deploy_command"][plan["deploy_command"].index("--memory") + 1] = "1024Mi"
    plan["deploy_command"][plan["deploy_command"].index("--cpu") + 1] = "1"
    plan["deploy_command"][plan["deploy_command"].index("--timeout") + 1] = "120"
    plan["deploy_command"][plan["deploy_command"].index("--min-instances") + 1] = "0"
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    failures = out["gates"]["deployment_safety"]["failures"]
    assert "deploy command memory must match resource_settings" in failures
    assert "deploy command cpu must match resource_settings" in failures
    assert "deploy command timeout must match resource_settings" in failures
    assert "deploy command min_instances must match resource_settings" in failures
    assert "Cloud Run memory must be at least 2Gi" in failures
    assert "Cloud Run CPU must be at least 2" in failures
    assert "Cloud Run timeout must be at least 300 seconds" in failures
    assert "Cloud Run min-instances must be at least 1" in failures


def test_certification_status_rejects_missing_snapshot_publish_artifact(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")

    out = certification_status(eval_path, results)

    assert out["gates"]["snapshot_publication"]["reason"] == "missing corpus_snapshot_publish_plan artifact"


def test_certification_status_rejects_bad_snapshot_publish_plan(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")
    plan = _write_controlled_snapshot_plan(results)
    plan["command"] = ["python", "-m", "src.other"]
    plan["metadata"]["chroma_dir"] = "/tmp/chroma_db"
    plan["metadata"]["gcs_bucket"] = "DELF_CORPUS_BUCKET"
    plan["metadata"]["snapshot_prefix"] = "REAL_SNAPSHOT_PREFIX"
    plan["env_vars"]["GCS_BUCKET"] = "other-bucket"
    plan["env_vars"]["GCS_SNAPSHOT_PREFIX"] = "other-prefix"
    plan["env_vars"]["CHROMA_DIR"] = "/tmp/other_chroma"
    (results / "corpus_snapshot_publish_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    failures = out["gates"]["snapshot_publication"]["failures"]
    assert "snapshot publish command must run python -m scripts.publish_corpus_snapshot" in failures
    assert "snapshot publish GCS_SNAPSHOT_PREFIX must be 'tee-corpus'" in failures
    assert "snapshot publish gcs_bucket is missing or placeholder" in failures
    assert "snapshot publish snapshot_prefix is missing or placeholder" in failures
    assert "snapshot publish chroma_dir must not be baseline chroma_db" in failures
    assert "snapshot publish CHROMA_DIR env must match metadata chroma_dir" in failures
    assert "snapshot publish GCS_BUCKET env must match metadata gcs_bucket" in failures
    assert "snapshot publish GCS_SNAPSHOT_PREFIX env must match metadata snapshot_prefix" in failures
    assert "snapshot publish readiness path must match metadata chroma_dir" in failures


def test_certification_status_rejects_nested_baseline_snapshot_publish_plan(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")
    plan = _write_controlled_snapshot_plan(results)
    nested = str(Path.cwd() / "chroma_db" / "nested_snapshot")
    plan["metadata"]["chroma_dir"] = nested
    plan["env_vars"]["CHROMA_DIR"] = nested
    plan["chroma_dir_readiness"]["path"] = nested
    (results / "corpus_snapshot_publish_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    failures = out["gates"]["snapshot_publication"]["failures"]
    assert "snapshot publish chroma_dir must not be inside baseline chroma_db" in failures


def test_certification_status_rejects_snapshot_plan_without_readiness_evidence(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")
    plan = _write_controlled_snapshot_plan(results)
    plan.pop("chroma_dir_readiness")
    plan["safety_invariants"].pop("chroma_dir_ready")
    (results / "corpus_snapshot_publish_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    failures = out["gates"]["snapshot_publication"]["failures"]
    assert "snapshot publish chroma_dir_readiness must pass" in failures
    assert "snapshot publish safety invariant chroma_dir_ready must be true" in failures


def test_certification_status_rejects_snapshot_bucket_uri_in_runtime_fields(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")
    plan = _write_controlled_snapshot_plan(results)
    plan["metadata"]["gcs_bucket"] = "gs://delf-corpus-prod"
    plan["env_vars"]["GCS_BUCKET"] = "gs://delf-corpus-prod"
    (results / "corpus_snapshot_publish_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    failures = out["gates"]["snapshot_publication"]["failures"]
    assert "snapshot publish gcs_bucket must be a bucket name, not a gs:// URI" in failures
    assert "snapshot publish GCS_BUCKET env must be a bucket name, not a gs:// URI" in failures


def test_certification_status_rejects_naked_m3_recall_artifact(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    (results / "enriched_experiment_20260530T000000Z.json").write_text(
        json.dumps({"recall_with_rerank": {"m3_recall_at_k": 0.95, "m3_k": 3}}),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    failures = out["gates"]["m3_recall"]["failures"]
    assert "contextual_enrichment must be True" in failures
    assert "experiment DB readiness must pass" in failures
    assert "preflight documents_total must be positive" in failures


def test_certification_status_rejects_baseline_chroma_m3_artifact(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {
                "id": "q1",
                "answerable": True,
                "validation_status": "VALIDATED_BY_DELF_EXPERT",
                "expected_sources": ["source.pdf"],
            }
        ]),
        encoding="utf-8",
    )
    _write_validation_manifest(eval_path)
    artifact = _controlled_enriched_artifact(m3=0.95)
    artifact["metadata"]["experiment_chroma_dir"] = "/tmp/chroma_db"
    _write_controlled_enriched_artifact(results, artifact)

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    assert "experiment_chroma_dir must not be the baseline chroma_db" in out["gates"]["m3_recall"]["failures"]


def test_certification_status_rejects_m3_recall_artifact_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")
    _write_validation_manifest(eval_path)
    artifact = _controlled_enriched_artifact(m3=0.95, eval_path=str(eval_path))
    _write_controlled_enriched_artifact(results, artifact)
    report_path = results / "recall_for_enriched_experiment_20260530T000000Z.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["metadata"]["m3_recall_at_k"] = 0.50
    report["aggregate"]["overall"]["recall@3"] = 0.50
    report["per_question"].pop()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    out = certification_status(eval_path, results)

    failures = out["gates"]["m3_recall"]["failures"]
    assert "M3 recall artifact per_question row count must match summary" in failures
    assert "M3 recall artifact m3_recall_at_k must match summary" in failures
    assert "M3 recall artifact aggregate recall must match summary" in failures


def test_certification_status_rejects_m3_recall_duplicate_or_blank_question_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items(n=2)), encoding="utf-8")
    _write_validation_manifest(eval_path)
    artifact = _controlled_enriched_artifact(m3=0.95, eval_path=str(eval_path), n=2)
    _write_controlled_enriched_artifact(results, artifact)
    report_path = results / "recall_for_enriched_experiment_20260530T000000Z.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["per_question"][0]["id"] = ""
    report["per_question"][1]["id"] = ""
    report_path.write_text(json.dumps(report), encoding="utf-8")

    out = certification_status(eval_path, results)

    failures = out["gates"]["m3_recall"]["failures"]
    assert "M3 per_question IDs must be non-empty" in failures
    assert "M3 recall artifact per_question row count must match summary" not in failures


def test_certification_status_rejects_m3_recall_non_object_rows(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items(n=2)), encoding="utf-8")
    _write_validation_manifest(eval_path)
    artifact = _controlled_enriched_artifact(m3=0.95, eval_path=str(eval_path), n=2)
    _write_controlled_enriched_artifact(results, artifact)
    report_path = results / "recall_for_enriched_experiment_20260530T000000Z.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["per_question"][1] = "q2"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    out = certification_status(eval_path, results)

    failures = out["gates"]["m3_recall"]["failures"]
    assert "M3 per_question rows must be objects" in failures
    assert "M3 per_question IDs must be non-empty" in failures
    assert "M3 recall artifact per_question row count must match summary" not in failures


def test_certification_status_rejects_m3_when_baseline_integrity_changed(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")
    artifact = _controlled_enriched_artifact(eval_path=str(eval_path))
    artifact["baseline_integrity"]["ok"] = False
    artifact["baseline_integrity"]["after"] = {
        "path": "/tmp/chroma_db",
        "present": True,
        "file_count": 4,
        "total_bytes": 143000001,
        "sha256": "changed",
    }
    _write_controlled_enriched_artifact(results, artifact)

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    assert "baseline chroma_db integrity must be unchanged" in out["gates"]["m3_recall"]["failures"]


def test_certification_status_rejects_naked_citation_artifact(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {"id": "q1", "answerable": True, "validation_status": "VALIDATED_BY_DELF_EXPERT"}
        ]),
        encoding="utf-8",
    )
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps({"summary": {"citation_pass_rate": 0.96, "n": 1}}),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    failures = out["gates"]["m4_citation_grounding"]["failures"]
    assert "citation run must require validated eval items" in failures
    assert "citation chroma_dir missing" in failures
    assert "per_question row count must equal summary n" in failures


def test_certification_status_rejects_baseline_chroma_citation_artifact(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {"id": "q1", "answerable": True, "validation_status": "VALIDATED_BY_DELF_EXPERT"}
        ]),
        encoding="utf-8",
    )
    artifact = _controlled_citation_artifact(rate=0.96)
    artifact["metadata"]["chroma_dir"] = "/tmp/chroma_db"
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    assert "citation chroma_dir must not be the baseline chroma_db" in out["gates"]["m4_citation_grounding"]["failures"]


def test_certification_status_rejects_nested_baseline_chroma_citation_artifact(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {"id": "q1", "answerable": True, "validation_status": "VALIDATED_BY_DELF_EXPERT"}
        ]),
        encoding="utf-8",
    )
    artifact = _controlled_citation_artifact(rate=0.96)
    artifact["metadata"]["chroma_dir"] = str(Path.cwd() / "chroma_db" / "nested_m4")
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    assert "citation chroma_dir must not be inside the baseline chroma_db" in (
        out["gates"]["m4_citation_grounding"]["failures"]
    )


def test_certification_status_rejects_citation_summary_row_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {"id": "q1", "answerable": True, "validation_status": "VALIDATED_BY_DELF_EXPERT"}
        ]),
        encoding="utf-8",
    )
    artifact = _controlled_citation_artifact(rate=1.0, n=2)
    artifact["per_question"][0]["citation_passed"] = False
    artifact["per_question"][1]["error"] = "boom"
    artifact["per_question"][1]["safety"]["grounding_allowed"] = False
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    failures = out["gates"]["m4_citation_grounding"]["failures"]
    assert "citation_passed summary must match per_question rows" in failures
    assert "no_error summary must match per_question rows" in failures
    assert "grounding_allowed summary must match per_question rows" in failures
    assert "citation_pass_rate summary must match per_question rows" in failures
    assert "no_error_rate summary must match per_question rows" in failures
    assert "grounding_allowed_rate summary must match per_question rows" in failures


def test_certification_status_rejects_citation_duplicate_or_blank_question_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items(n=2)), encoding="utf-8")
    artifact = _controlled_citation_artifact(rate=1.0, n=2)
    artifact["per_question"][1]["id"] = "q1"
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    failures = out["gates"]["m4_citation_grounding"]["failures"]
    assert "M4 per_question IDs must be unique" in failures
    assert "per_question row count must equal summary n" not in failures


def test_certification_status_rejects_citation_non_object_rows(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items(n=2)), encoding="utf-8")
    artifact = _controlled_citation_artifact(rate=1.0, n=2)
    artifact["per_question"][1] = "q2"
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    failures = out["gates"]["m4_citation_grounding"]["failures"]
    assert "M4 per_question rows must be objects" in failures
    assert "M4 per_question IDs must be non-empty" in failures
    assert "per_question row count must equal summary n" not in failures


def test_certification_status_rejects_citation_artifact_with_errors(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(
        json.dumps([
            {"id": "q1", "answerable": True, "validation_status": "VALIDATED_BY_DELF_EXPERT"}
        ]),
        encoding="utf-8",
    )
    artifact = _controlled_citation_artifact(rate=1.0)
    artifact["summary"]["no_error_rate"] = 0.0
    artifact["per_question"][0]["error"] = "boom"
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    out = certification_status(eval_path, results)

    failures = out["gates"]["m4_citation_grounding"]["failures"]
    assert "citation run must have no generation errors" in failures
    assert "per_question contains errored rows: ['q1']" in failures


def test_certification_status_rejects_m3_m4_eval_path_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")
    _write_controlled_enriched_artifact(
        results,
        _controlled_enriched_artifact(eval_path=str(eval_path)),
    )
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(_controlled_citation_artifact(eval_path="/tmp/other_eval.json")),
        encoding="utf-8",
    )
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results, chroma_dir="/tmp/chroma_db_enriched_a")

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    assert "M4 eval_path must match certification eval_path" in out["gates"]["artifact_consistency"]["failures"]


def test_certification_status_rejects_m3_m4_chroma_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")
    _write_controlled_enriched_artifact(
        results,
        _controlled_enriched_artifact(eval_path=str(eval_path), chroma_dir="/tmp/chroma_db_enriched_a"),
    )
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(_controlled_citation_artifact(eval_path=str(eval_path), chroma_dir="/tmp/chroma_db_enriched_b")),
        encoding="utf-8",
    )
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results, chroma_dir="/tmp/chroma_db_enriched_a")

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    assert "M3 experiment_chroma_dir must match M4 chroma_dir" in out["gates"]["artifact_consistency"]["failures"]


def test_certification_status_rejects_m3_m4_question_id_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items(n=2)), encoding="utf-8")
    _write_validation_manifest(eval_path)
    _write_controlled_enriched_artifact(
        results,
        _controlled_enriched_artifact(eval_path=str(eval_path), n=2),
    )
    report_path = results / "recall_for_enriched_experiment_20260530T000000Z.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["per_question"][1]["id"] = "q999"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    citation = _controlled_citation_artifact(eval_path=str(eval_path), n=2)
    citation["per_question"][1]["id"] = "q999"
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(citation),
        encoding="utf-8",
    )
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    failures = out["gates"]["artifact_consistency"]["failures"]
    assert "M3 per_question IDs must match answerable sourced eval IDs" in failures
    assert "M4 per_question IDs must match answerable sourced eval IDs" in failures


def test_certification_status_rejects_duplicate_or_blank_artifact_question_ids(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items(n=2)), encoding="utf-8")
    _write_validation_manifest(eval_path)
    _write_controlled_enriched_artifact(
        results,
        _controlled_enriched_artifact(eval_path=str(eval_path), n=2),
    )
    report_path = results / "recall_for_enriched_experiment_20260530T000000Z.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["per_question"][1]["id"] = "q1"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    citation = _controlled_citation_artifact(eval_path=str(eval_path), n=2)
    citation["per_question"][0]["id"] = ""
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(citation),
        encoding="utf-8",
    )
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-1",
        )
    )
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results)

    out = certification_status(eval_path, results)

    failures = out["gates"]["artifact_consistency"]["failures"]
    assert "M3 per_question IDs must be unique" in failures
    assert "M3 per_question IDs must match answerable sourced eval IDs" in failures
    assert "M4 per_question IDs must be non-empty" in failures
    assert "M4 per_question IDs must match answerable sourced eval IDs" in failures


def test_certification_status_rejects_run_id_mismatch(tmp_path):
    eval_path = tmp_path / "eval.json"
    results = tmp_path / "results"
    results.mkdir()
    eval_path.write_text(json.dumps(_validated_eval_items()), encoding="utf-8")
    _write_controlled_enriched_artifact(
        results,
        _controlled_enriched_artifact(eval_path=str(eval_path), run_id="run-a"),
    )
    (results / "citation_grounding_20260530T000000Z.json").write_text(
        json.dumps(_controlled_citation_artifact(eval_path=str(eval_path), run_id="run-b")),
        encoding="utf-8",
    )
    plan = build_deploy_plan(
        DeployConfig(
            project="woven-operative-491610-u6",
            image="us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf:abc123",
            service_account="delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
            invoker="group:delf-examiners@org.test",
            certification_run_id="run-a",
        )
    )
    (results / "cloud_run_deploy_plan_20260530T000000Z.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    _write_controlled_snapshot_plan(results, run_id="run-a")

    out = certification_status(eval_path, results)

    assert out["certified"] is False
    assert "M3, M4, deploy, and snapshot publish certification_run_id values must match" in out["gates"]["artifact_consistency"]["failures"]
