"""
Guarded end-to-end retrieval certification pipeline.

Default mode is intentionally cheap: lint the expert CSV and apply it to a
validated JSON. The expensive live steps are opt-in:
  --run-enriched-ingest   rebuilds chroma_db_enriched with contextual enrichment
  --measure-citations     runs live chatbot generation for M4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts.apply_expert_validation import apply_review
from scripts.certification_status import certification_status, citation_status, enriched_status
from scripts.lint_expert_review import lint_review
from scripts.render_cloud_run_deploy import DeployConfig, build_deploy_plan, write_deploy_plan
from scripts.render_snapshot_publish_plan import (
    SnapshotPublishConfig,
    build_snapshot_publish_plan,
    write_snapshot_publish_plan,
)


REPO = Path(__file__).resolve().parent.parent
DEFAULT_EVAL = REPO / "evaluation" / "delf_questions.json"
DEFAULT_REVIEW = REPO / "evaluation" / "delf_questions_expert_review.csv"
DEFAULT_VALIDATED = REPO / "evaluation" / "delf_questions.validated.json"
DEFAULT_CORPUS_FILES = REPO / "evaluation" / "corpus_files.json"
DEFAULT_RESULTS = REPO / "evaluation" / "results"
DEFAULT_CHROMA_ENRICHED = REPO / "chroma_db_enriched"


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO), check=True)


def _write_summary(payload: dict, results_dir: Path = DEFAULT_RESULTS) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = results_dir / f"certification_pipeline_{ts}_{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _latest_json(results_dir: Path, pattern: str) -> dict | None:
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def _norm_path(path: Path | str | None) -> str:
    if not path:
        return ""
    return str(Path(path).expanduser().resolve(strict=False))


def _pre_deploy_alignment_failures(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    expected_eval = _norm_path(args.validated_output)
    expected_chroma = _norm_path(args.chroma_dir)
    m3 = _latest_json(args.results_dir, "enriched_experiment_*.json")
    m4 = _latest_json(args.results_dir, "citation_grounding_*.json")

    if not m3:
        failures.append("M3 artifact missing for pre-deploy alignment")
    else:
        m3_meta = m3.get("metadata", {})
        m3_preflight = m3.get("preflight", {})
        if _norm_path(m3_meta.get("eval_path")) != expected_eval:
            failures.append("M3 eval_path must match current validated output before deploy artifacts")
        if _norm_path(m3_preflight.get("eval_path")) != expected_eval:
            failures.append("M3 preflight eval_path must match current validated output before deploy artifacts")
        if _norm_path(m3_meta.get("experiment_chroma_dir")) != expected_chroma:
            failures.append("M3 experiment_chroma_dir must match current chroma_dir before deploy artifacts")

    if not m4:
        failures.append("M4 artifact missing for pre-deploy alignment")
    else:
        m4_meta = m4.get("metadata", {})
        if _norm_path(m4_meta.get("eval_path")) != expected_eval:
            failures.append("M4 eval_path must match current validated output before deploy artifacts")
        if _norm_path(m4_meta.get("chroma_dir")) != expected_chroma:
            failures.append("M4 chroma_dir must match current chroma_dir before deploy artifacts")

    if m3 and m4:
        m3_recall = m3.get("recall_with_rerank") or m3.get("recall_no_rerank") or {}
        if m3_recall.get("n_questions_evaluated") != m4.get("metadata", {}).get("questions_selected"):
            failures.append("M3/M4 evaluated question counts must match before deploy artifacts")
    return failures


def _write_pipeline_deploy_plan(args: argparse.Namespace) -> Path:
    missing = [
        flag for flag, value in {
            "--deploy-project": args.deploy_project,
            "--deploy-image": args.deploy_image,
            "--deploy-service-account": args.deploy_service_account,
            "--deploy-gcs-bucket": args.deploy_gcs_bucket,
            "--deploy-invoker": args.deploy_invoker,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit("--write-deploy-plan requires " + ", ".join(missing))

    plan = build_deploy_plan(
        DeployConfig(
            project=args.deploy_project,
            image=args.deploy_image,
            service_account=args.deploy_service_account,
            gcs_bucket=args.deploy_gcs_bucket,
            service=args.deploy_service,
            region=args.deploy_region,
            location=args.deploy_location,
            invoker=args.deploy_invoker,
            snapshot_prefix=args.snapshot_prefix,
            certification_run_id=args.certification_run_id,
        )
    )
    return write_deploy_plan(plan, args.results_dir)


def _write_pipeline_snapshot_plan(args: argparse.Namespace) -> Path:
    bucket = args.snapshot_gcs_bucket or args.deploy_gcs_bucket
    if not bucket:
        raise SystemExit("--write-snapshot-plan requires --snapshot-gcs-bucket or --deploy-gcs-bucket")
    plan = build_snapshot_publish_plan(
        SnapshotPublishConfig(
            chroma_dir=str(args.chroma_dir),
            gcs_bucket=bucket,
            snapshot_prefix=args.snapshot_prefix,
            certification_run_id=args.certification_run_id,
        )
    )
    return write_snapshot_publish_plan(plan, args.results_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run guarded retrieval certification pipeline.")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--validated-output", type=Path, default=DEFAULT_VALIDATED)
    parser.add_argument("--corpus-files", type=Path, default=DEFAULT_CORPUS_FILES)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_ENRICHED)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-enriched-ingest", action="store_true")
    parser.add_argument("--measure-citations", action="store_true")
    parser.add_argument("--write-deploy-plan", action="store_true")
    parser.add_argument("--write-snapshot-plan", action="store_true")
    parser.add_argument("--snapshot-gcs-bucket")
    parser.add_argument("--snapshot-prefix", default="tee-corpus")
    parser.add_argument("--deploy-project")
    parser.add_argument("--deploy-image")
    parser.add_argument("--deploy-service-account")
    parser.add_argument("--deploy-gcs-bucket")
    parser.add_argument("--deploy-invoker")
    parser.add_argument("--deploy-service", default="delf-dalf-examiner-assistant")
    parser.add_argument("--deploy-region", default="us-central1")
    parser.add_argument("--deploy-location")
    parser.add_argument(
        "--certification-run-id",
        help="Optional stable ID to stamp all artifacts from the same certification attempt.",
    )
    parser.add_argument(
        "--allow-incomplete-review",
        action="store_true",
        help="Do not require all validation_decision cells. Useful only for dry diagnostics.",
    )
    args = parser.parse_args()
    args.certification_run_id = args.certification_run_id or (
        "cert-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    )
    if (args.write_deploy_plan or args.write_snapshot_plan) and (
        not args.run_enriched_ingest or not args.measure_citations
    ):
        raise SystemExit(
            "--write-deploy-plan/--write-snapshot-plan require "
            "--run-enriched-ingest and --measure-citations in the guarded pipeline. "
            "Use the standalone renderers for the manual path."
        )

    require_complete = not args.allow_incomplete_review
    lint = lint_review(
        eval_path=args.eval_path,
        review_path=args.review,
        corpus_files_path=args.corpus_files,
        require_complete=require_complete,
    )
    summary: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "certification_run_id": args.certification_run_id,
        "lint": {
            "ok": lint.ok,
            "rows": lint.rows,
            "errors": lint.errors,
            "warnings": lint.warnings,
        },
        "paths": {
            "eval_path": str(args.eval_path),
            "review": str(args.review),
            "validated_output": str(args.validated_output),
            "chroma_dir": str(args.chroma_dir),
            "results_dir": str(args.results_dir),
        },
    }
    if not lint.ok:
        out = _write_summary(summary, args.results_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"pipeline summary: {out}")
        return 2

    if not require_complete and lint.warnings:
        summary["diagnostic_only"] = True
        summary["reason"] = "Review is incomplete; validated JSON was not written."
        out = _write_summary(summary, args.results_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"pipeline summary: {out}")
        return 2

    counts = apply_review(
        args.eval_path,
        args.review,
        args.validated_output,
        strict=require_complete,
        corpus_files_path=args.corpus_files,
    )
    summary["apply_review"] = counts

    if args.run_enriched_ingest:
        _run([
            sys.executable,
            "-m",
            "scripts.run_enriched_retrieval_experiment",
            "--eval-path",
            str(args.validated_output),
            "--chroma-dir",
            str(args.chroma_dir),
            "--results-dir",
            str(args.results_dir),
            "--certification-run-id",
            args.certification_run_id,
            "--require-validated",
            "--run-ingest",
            "--force",
            "--evaluate",
        ])

    if args.measure_citations:
        _run([
            sys.executable,
            "-m",
            "scripts.measure_citation_grounding",
            "--eval-path",
            str(args.validated_output),
            "--chroma-dir",
            str(args.chroma_dir),
            "--results-dir",
            str(args.results_dir),
            "--certification-run-id",
            args.certification_run_id,
            "--require-validated",
        ])

    if args.write_deploy_plan or args.write_snapshot_plan:
        pre_deploy_gates = {
            "m3_recall": enriched_status(args.results_dir),
            "m4_citation_grounding": citation_status(args.results_dir),
        }
        summary["pre_deploy_gate_status"] = pre_deploy_gates
        failed = [name for name, gate in pre_deploy_gates.items() if not gate.get("passed")]
        run_id_mismatches = [
            name
            for name, gate in pre_deploy_gates.items()
            if gate.get("passed") and gate.get("certification_run_id") != args.certification_run_id
        ]
        alignment_failures = _pre_deploy_alignment_failures(args)
        if failed or run_id_mismatches or alignment_failures:
            summary["deployment_artifacts_skipped"] = {
                "reason": (
                    "M3/M4 gates must pass for this certification_run_id before "
                    "deployment artifacts are written."
                ),
                "failed_gates": failed,
                "run_id_mismatches": run_id_mismatches,
                "alignment_failures": alignment_failures,
                "expected_certification_run_id": args.certification_run_id,
            }
            out = _write_summary(summary, args.results_dir)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"pipeline summary: {out}")
            return 2

    if args.write_deploy_plan:
        deploy_plan = _write_pipeline_deploy_plan(args)
        summary["deploy_plan"] = str(deploy_plan)
    if args.write_snapshot_plan:
        snapshot_plan = _write_pipeline_snapshot_plan(args)
        summary["snapshot_publish_plan"] = str(snapshot_plan)

    status = certification_status(args.validated_output, args.results_dir, args.corpus_files)
    summary["certification_status"] = status
    out = _write_summary(summary, args.results_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"pipeline summary: {out}")
    return 0 if status["certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
