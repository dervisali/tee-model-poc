"""
Report current retrieval-certification gate status from eval + artifacts.

This script does not run ingestion, retrieval, or generation. It only inspects
current files and tells whether the evidence required by the production gate is
present:
  - all answerable eval items expert-validated,
  - latest enriched experiment has M3 recall@3 >= 90%,
  - latest citation-grounding artifact has citation pass rate >= 95%,
  - latest Cloud Run deploy plan proves private, GCS-backed serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.apply_expert_validation import _item_sources, _load_corpus_basenames, _missing_corpus_sources
from scripts.lint_expert_review import lint_review


REPO = Path(__file__).resolve().parent.parent
DEFAULT_EVAL = REPO / "evaluation" / "delf_questions.json"
DEFAULT_VALIDATED_EVAL = REPO / "evaluation" / "delf_questions.validated.json"
DEFAULT_REVIEW = REPO / "evaluation" / "delf_questions_expert_review.csv"


def default_eval_path() -> Path:
    """Certify the expert-validated eval when it exists, else the raw eval.

    The certification target is the validated set produced by
    ``scripts.apply_expert_validation --strict``. Until that file exists the
    gate falls back to the raw eval, which correctly fails (no manifest).
    """
    return DEFAULT_VALIDATED_EVAL if DEFAULT_VALIDATED_EVAL.exists() else DEFAULT_EVAL
DEFAULT_RESULTS = REPO / "evaluation" / "results"
DEFAULT_CORPUS_FILES = REPO / "evaluation" / "corpus_files.json"

REQUIRED_DOCKERIGNORE_PATTERNS = {
    ".env",
    ".agents/",
    ".codex/",
    ".playwright-cli/",
    ".playwright-mcp/",
    "chroma_db/",
    "chroma_db_enriched/",
    "new_docs/",
    "new_docs_subset/",
    "evaluation/",
    "logs/",
    "output/",
    "screenshots/",
    "tests/",
    "*.zip",
}


def _load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", data) if isinstance(data, dict) else data


def _latest(results_dir: Path, pattern: str) -> Path | None:
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _load_latest_json(results_dir: Path, pattern: str) -> tuple[Path | None, dict | None]:
    path = _latest(results_dir, pattern)
    if path is None:
        return None, None
    return path, json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact_path(value: str | None, *, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _norm_path(value: str | None) -> str | None:
    if not value:
        return None
    return str(Path(value).expanduser().resolve(strict=False))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dockerignore_patterns(path: Path) -> set[str]:
    patterns: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.add(line)
    return patterns


def _validation_manifest_path(eval_path: Path) -> Path:
    return eval_path.with_suffix(".validation_manifest.json")


def _is_validated(item: dict) -> bool:
    return str(item.get("validation_status", "")).startswith("VALIDATED")


def _is_dropped(item: dict) -> bool:
    return str(item.get("validation_status", "")).startswith("DROPPED")


def _review_preflight_status(
    *,
    eval_path: Path,
    review_path: Path,
    corpus_files_path: Path,
) -> dict:
    if not review_path.exists():
        return {
            "path": str(review_path),
            "checked": False,
            "ok": False,
            "errors": ["review CSV missing"],
            "warnings": [],
        }
    try:
        result = lint_review(
            eval_path=eval_path,
            review_path=review_path,
            corpus_files_path=corpus_files_path,
            require_complete=True,
        )
        return {
            "path": str(review_path),
            "checked": True,
            "ok": result.ok,
            "rows": result.rows,
            "errors": result.errors,
            "warnings": result.warnings,
        }
    except Exception as exc:  # noqa: BLE001 - lint evidence must be explicit
        return {
            "path": str(review_path),
            "checked": True,
            "ok": False,
            "errors": [f"review CSV lint failed: {type(exc).__name__}: {exc}"],
            "warnings": [],
        }


def eval_status(
    eval_path: Path,
    corpus_files_path: Path | None = None,
    review_path: Path | None = None,
) -> dict:
    questions = _load_questions(eval_path)
    question_ids = [str(item.get("id", "")).strip() for item in questions]
    answerable = [q for q in questions if q.get("answerable", True)]
    validated = [q for q in answerable if _is_validated(q)]
    unvalidated = [q for q in answerable if not _is_validated(q)]
    dropped = [q for q in questions if _is_dropped(q)]
    failures: list[str] = []
    review_preflight: dict | None = None
    corpus_names: set[str] | None = None
    if corpus_files_path is not None:
        if not corpus_files_path.exists():
            failures.append("corpus_files catalog missing")
        else:
            try:
                corpus_names = _load_corpus_basenames(corpus_files_path)
            except Exception as exc:  # noqa: BLE001 - malformed source catalog is a no-go
                failures.append(f"corpus_files catalog unreadable: {type(exc).__name__}: {exc}")
    if any(not qid for qid in question_ids):
        failures.append("eval file contains blank or missing IDs")
    duplicate_ids = sorted(qid for qid, count in Counter(question_ids).items() if qid and count > 1)
    if duplicate_ids:
        failures.append(
            "eval file contains duplicate IDs: "
            + ", ".join(duplicate_ids[:10])
            + (" ..." if len(duplicate_ids) > 10 else "")
        )
    if review_path is not None and corpus_files_path is not None:
        review_preflight = _review_preflight_status(
            eval_path=eval_path,
            review_path=review_path,
            corpus_files_path=corpus_files_path,
        )
        if not review_preflight["ok"]:
            failures.append("current expert review CSV lint failed")
    manifest_path = _validation_manifest_path(eval_path)
    manifest: dict | None = None
    if not manifest_path.exists():
        failures.append("validation manifest missing")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("strict") is not True:
                failures.append("validation manifest must be strict")
            if _norm_path(str(manifest.get("output_eval_path", ""))) != _norm_path(str(eval_path)):
                failures.append("validation manifest output_eval_path must match eval path")
            if manifest.get("output_eval_sha256") != _sha256(eval_path):
                failures.append("validation manifest output sha256 mismatch")
            input_eval_path = Path(str(manifest.get("input_eval_path", ""))).expanduser()
            if not input_eval_path.is_absolute():
                input_eval_path = input_eval_path.resolve(strict=False)
            if not input_eval_path.exists():
                failures.append("validation manifest input_eval_path must exist")
            elif manifest.get("input_eval_sha256") != _sha256(input_eval_path):
                failures.append("validation manifest input eval sha256 mismatch")
            review_path = Path(str(manifest.get("review_path", ""))).expanduser()
            if not review_path.is_absolute():
                review_path = review_path.resolve(strict=False)
            if not review_path.exists():
                failures.append("validation manifest review_path must exist")
            elif manifest.get("review_sha256") != _sha256(review_path):
                failures.append("validation manifest review sha256 mismatch")
            if corpus_files_path is not None:
                if _norm_path(str(manifest.get("corpus_files_path", ""))) != _norm_path(str(corpus_files_path)):
                    failures.append("validation manifest corpus_files_path must match corpus catalog")
                if corpus_files_path.exists() and manifest.get("corpus_files_sha256") != _sha256(corpus_files_path):
                    failures.append("validation manifest corpus_files sha256 mismatch")
            counts = manifest.get("counts", {})
            if counts.get("validated", 0) + counts.get("dropped", 0) <= 0:
                failures.append("validation manifest must record validated or dropped rows")
            if counts.get("validated") != len(validated):
                failures.append("validation manifest validated count must match eval file")
            if counts.get("dropped") != len(dropped):
                failures.append("validation manifest dropped count must match eval file")
            if counts.get("unchanged", 0) != 0:
                failures.append("validation manifest unchanged count must be zero")
            if counts.get("fixed", 0) > counts.get("validated", 0):
                failures.append("validation manifest fixed count cannot exceed validated count")
            if counts.get("validated", 0) + counts.get("dropped", 0) != len(questions):
                failures.append("validation manifest validated+dropped count must equal eval total")
        except Exception as exc:  # noqa: BLE001 - malformed manifest is a no-go
            failures.append(f"validation manifest unreadable: {type(exc).__name__}: {exc}")

    if not (len(answerable) > 0 and len(validated) == len(answerable)):
        failures.append("not all answerable eval items are expert-validated")
    if corpus_names is not None:
        for item in answerable:
            qid = str(item.get("id", "<missing>")).strip() or "<missing>"
            missing_sources = _missing_corpus_sources(_item_sources(item), corpus_names)
            if missing_sources:
                failures.append(
                    f"eval item {qid} has sources outside corpus_files.json: "
                    + ", ".join(missing_sources[:5])
                )

    return {
        "path": str(eval_path),
        "manifest": str(manifest_path),
        "total": len(questions),
        "answerable": len(answerable),
        "validated_answerable": len(validated),
        "unvalidated_answerable": len(unvalidated),
        "unvalidated_answerable_ids": [
            str(item.get("id", "")).strip() or "<missing>"
            for item in unvalidated[:10]
        ],
        "dropped": len(dropped),
        "passed": not failures,
        "failures": failures,
        "review_preflight": review_preflight,
    }


def enriched_status(results_dir: Path) -> dict:
    path, data = _load_latest_json(results_dir, "enriched_experiment_*.json")
    if path is None or data is None:
        return {"artifact": None, "passed": False, "reason": "missing enriched_experiment artifact"}
    recall = data.get("recall_with_rerank") or data.get("recall_no_rerank") or {}
    m3 = recall.get("m3_recall_at_k")
    m3_k = recall.get("m3_k")
    metadata = data.get("metadata", {})
    preflight = data.get("preflight", {})
    readiness = data.get("experiment_db_readiness", {})
    baseline_integrity = data.get("baseline_integrity", {})
    failures: list[str] = []

    if m3 is None:
        failures.append("m3_recall_at_k missing")
    elif m3 < 0.90:
        failures.append("m3_recall_at_k below 0.90")
    if m3_k != 3:
        failures.append("M3 must be measured at k=3")

    expected_metadata = {
        "contextual_enrichment": True,
        "chunking_strategy": "paragraph",
        "embedding_dimension": 3072,
        "cross_lingual_bm25": False,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            failures.append(f"{key} must be {expected!r}")

    experiment_dir = str(metadata.get("experiment_chroma_dir", ""))
    if not experiment_dir:
        failures.append("experiment_chroma_dir missing")
    else:
        experiment_path = Path(experiment_dir).expanduser().resolve(strict=False)
        baseline_path = (REPO / "chroma_db").resolve(strict=False)
        repo_path = REPO.resolve(strict=False)
        if experiment_path.name == "chroma_db" or experiment_path == baseline_path:
            failures.append("experiment_chroma_dir must not be the baseline chroma_db")
        if _is_relative_to(experiment_path, baseline_path):
            failures.append("experiment_chroma_dir must not be inside the baseline chroma_db")
        if _is_relative_to(repo_path, experiment_path) or _is_relative_to(baseline_path, experiment_path):
            failures.append("experiment_chroma_dir must not be a parent of the repo or baseline chroma_db")

    total = preflight.get("documents_total")
    nonempty = preflight.get("documents_nonempty")
    doc_failures = preflight.get("document_failures")
    if not isinstance(total, int) or total <= 0:
        failures.append("preflight documents_total must be positive")
    if nonempty != total:
        failures.append("preflight documents_nonempty must equal documents_total")
    if doc_failures:
        failures.append("preflight document_failures must be empty")
    if preflight.get("eval_unvalidated") != 0:
        failures.append("preflight eval_unvalidated must be 0")

    if readiness.get("ok") is not True:
        failures.append("experiment DB readiness must pass")
    if not readiness.get("parent_count"):
        failures.append("experiment DB parent_count missing or zero")
    if not readiness.get("child_count"):
        failures.append("experiment DB child_count missing or zero")
    if baseline_integrity.get("ok") is not True:
        failures.append("baseline chroma_db integrity must be unchanged")
    if not baseline_integrity.get("before") or not baseline_integrity.get("after"):
        failures.append("baseline integrity before/after fingerprints required")

    recall_artifact = _resolve_artifact_path(recall.get("artifact_json"), base_dir=results_dir)
    if recall_artifact is None:
        failures.append("M3 recall artifact_json missing")
    elif not recall_artifact.exists():
        failures.append("M3 recall artifact_json does not exist")
    else:
        try:
            recall_report = json.loads(recall_artifact.read_text(encoding="utf-8"))
            recall_meta = recall_report.get("metadata", {})
            recall_overall = recall_report.get("aggregate", {}).get("overall", {})
            recall_rows = recall_report.get("per_question", [])
            recall_n = recall.get("n_questions_evaluated")
            if recall_meta.get("n_questions_evaluated") != recall_n:
                failures.append("M3 recall artifact n_questions_evaluated must match summary")
            if recall_overall.get("n") != recall_n:
                failures.append("M3 recall artifact aggregate n must match summary")
            if isinstance(recall_rows, list) and len(recall_rows) != recall_n:
                failures.append("M3 recall artifact per_question row count must match summary")
            elif not isinstance(recall_rows, list):
                failures.append("M3 recall artifact per_question must be a list")
            else:
                _append_row_shape_failures("M3", recall_rows, failures)
                m3_row_ids = _row_ids(recall_rows)
                if m3_row_ids is not None:
                    _append_row_id_failures("M3", m3_row_ids, failures)
            if recall_meta.get("m3_k") != m3_k:
                failures.append("M3 recall artifact m3_k must match summary")
            if recall_meta.get("m3_recall_at_k") != m3:
                failures.append("M3 recall artifact m3_recall_at_k must match summary")
            if m3_k is not None and recall_overall.get(f"recall@{m3_k}") != m3:
                failures.append("M3 recall artifact aggregate recall must match summary")
            if _norm_path(str(recall_meta.get("test_set", ""))) != _norm_path(str(metadata.get("eval_path", ""))):
                failures.append("M3 recall artifact test_set must match experiment eval_path")
        except Exception as exc:  # noqa: BLE001 - malformed evidence is a no-go
            failures.append(f"M3 recall artifact unreadable: {type(exc).__name__}: {exc}")

    recall_md = _resolve_artifact_path(recall.get("artifact_md"), base_dir=results_dir)
    if recall_md is None:
        failures.append("M3 recall artifact_md missing")
    elif not recall_md.exists():
        failures.append("M3 recall artifact_md does not exist")

    return {
        "artifact": str(path),
        "m3_recall_at_k": m3,
        "m3_k": m3_k,
        "certification_run_id": metadata.get("certification_run_id"),
        "passed": not failures,
        "failures": failures,
    }


def citation_status(results_dir: Path) -> dict:
    path, data = _load_latest_json(results_dir, "citation_grounding_*.json")
    if path is None or data is None:
        return {"artifact": None, "passed": False, "reason": "missing citation_grounding artifact"}
    metadata = data.get("metadata", {})
    summary = data.get("summary", {})
    rows = data.get("per_question", [])
    rate = summary.get("citation_pass_rate")
    failures: list[str] = []

    if rate is None:
        failures.append("citation_pass_rate missing")
    elif rate < 0.95:
        failures.append("citation_pass_rate below 0.95")

    n = summary.get("n")
    if not isinstance(n, int) or n <= 0:
        failures.append("citation summary n must be positive")
    if n != metadata.get("questions_selected"):
        failures.append("summary n must equal metadata questions_selected")
    if not isinstance(rows, list) or len(rows) != n:
        failures.append("per_question row count must equal summary n")

    if metadata.get("require_validated") is not True:
        failures.append("citation run must require validated eval items")
    if metadata.get("max_questions") is not None:
        failures.append("citation run must cover the full validated eval set")

    chroma_dir = str(metadata.get("chroma_dir", ""))
    if not chroma_dir:
        failures.append("citation chroma_dir missing")
    else:
        chroma_path = Path(chroma_dir).expanduser().resolve(strict=False)
        baseline_path = (REPO / "chroma_db").resolve(strict=False)
        repo_path = REPO.resolve(strict=False)
        if chroma_path.name == "chroma_db" or chroma_path == baseline_path:
            failures.append("citation chroma_dir must not be the baseline chroma_db")
        if _is_relative_to(chroma_path, baseline_path):
            failures.append("citation chroma_dir must not be inside the baseline chroma_db")
        if _is_relative_to(repo_path, chroma_path) or _is_relative_to(baseline_path, chroma_path):
            failures.append("citation chroma_dir must not be a parent of the repo or baseline chroma_db")

    if summary.get("no_error_rate") != 1.0:
        failures.append("citation run must have no generation errors")
    if summary.get("grounding_allowed_rate") != 1.0:
        failures.append("grounding gate must allow every evaluated answer")

    if isinstance(rows, list):
        _append_row_shape_failures("M4", rows, failures)
        m4_row_ids = _row_ids(rows)
        if m4_row_ids is not None:
            _append_row_id_failures("M4", m4_row_ids, failures)
        dict_rows = [row for row in rows if isinstance(row, dict)]
        row_citation_passed = sum(1 for row in dict_rows if row.get("citation_passed") is True)
        row_no_error = sum(1 for row in dict_rows if not row.get("error"))
        row_grounding_allowed = sum(
            1 for row in dict_rows
            if row.get("safety", {}).get("grounding_allowed") is not False
        )
        row_count = len(rows)
        if summary.get("citation_passed") != row_citation_passed:
            failures.append("citation_passed summary must match per_question rows")
        if summary.get("no_error") != row_no_error:
            failures.append("no_error summary must match per_question rows")
        if summary.get("grounding_allowed") != row_grounding_allowed:
            failures.append("grounding_allowed summary must match per_question rows")
        if row_count:
            expected_citation_rate = row_citation_passed / row_count
            expected_no_error_rate = row_no_error / row_count
            expected_grounding_rate = row_grounding_allowed / row_count
            if rate != expected_citation_rate:
                failures.append("citation_pass_rate summary must match per_question rows")
            if summary.get("no_error_rate") != expected_no_error_rate:
                failures.append("no_error_rate summary must match per_question rows")
            if summary.get("grounding_allowed_rate") != expected_grounding_rate:
                failures.append("grounding_allowed_rate summary must match per_question rows")

        unvalidated = [
            str(row.get("id", idx + 1))
            for idx, row in enumerate(dict_rows)
            if not _is_validated(row)
        ]
        errored = [
            str(row.get("id", idx + 1))
            for idx, row in enumerate(dict_rows)
            if row.get("error")
        ]
        if unvalidated:
            failures.append(f"per_question contains unvalidated rows: {unvalidated[:5]}")
        if errored:
            failures.append(f"per_question contains errored rows: {errored[:5]}")

    return {
        "artifact": str(path),
        "citation_pass_rate": rate,
        "n": n,
        "certification_run_id": metadata.get("certification_run_id"),
        "passed": not failures,
        "failures": failures,
    }


def _placeholder(value: str | None) -> bool:
    if value is None:
        return True
    text = value.strip()
    if not text:
        return True
    upper = text.upper()
    return any(
        token in upper
        for token in ("PLACEHOLDER", "YOUR_", "REAL_", "EXAMPLE.", "DELF_CORPUS_BUCKET", ":SHA", "<", ">")
    )


def _bucket_uri(value: str | None) -> bool:
    return bool(value and value.strip().startswith("gs://"))


def _flag_value(command: list, flag: str) -> str | None:
    try:
        idx = command.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(command):
        return None
    return str(command[idx + 1])


def _parse_env_arg(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    pairs: dict[str, str] = {}
    for part in value.split(","):
        if "=" not in part:
            continue
        key, item_value = part.split("=", 1)
        pairs[key] = item_value
    return pairs


def _image_reference_pinned(image: str) -> bool:
    last_segment = image.rsplit("/", 1)[-1]
    if "@sha256:" in image:
        return True
    if ":" not in last_segment:
        return False
    return last_segment.rsplit(":", 1)[-1].strip().lower() != "latest"


def _artifact_registry_project(image: str) -> str | None:
    parts = image.split("/")
    if len(parts) >= 2 and parts[0].endswith(".pkg.dev"):
        return parts[1]
    return None


def _service_account_project(service_account: str) -> str | None:
    suffix = ".iam.gserviceaccount.com"
    if "@" not in service_account or not service_account.endswith(suffix):
        return None
    return service_account.split("@", 1)[1][:-len(suffix)]


def _valid_service_account_email(service_account: str) -> bool:
    return "@" in service_account and service_account.endswith(".iam.gserviceaccount.com")


def _valid_invoker_member(invoker: str | None) -> bool:
    if not invoker:
        return False
    return invoker.startswith(("group:", "user:", "serviceAccount:", "domain:"))


def _row_ids(rows: object) -> list[str] | None:
    if not isinstance(rows, list):
        return None
    return [str(row.get("id", "")) if isinstance(row, dict) else "" for row in rows]


def _append_row_shape_failures(prefix: str, rows: list, failures: list[str]) -> None:
    if any(not isinstance(row, dict) for row in rows):
        failures.append(f"{prefix} per_question rows must be objects")


def _append_row_id_failures(prefix: str, row_ids: list[str], failures: list[str]) -> None:
    if any(not qid.strip() for qid in row_ids):
        failures.append(f"{prefix} per_question IDs must be non-empty")
    duplicate_ids = sorted(qid for qid, count in Counter(row_ids).items() if qid.strip() and count > 1)
    if duplicate_ids:
        failures.append(f"{prefix} per_question IDs must be unique")


def _memory_to_mib(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    try:
        if text.endswith("Gi"):
            return int(text[:-2]) * 1024
        if text.endswith("Mi"):
            return int(text[:-2])
    except ValueError:
        return None
    return None


def deployment_status(results_dir: Path) -> dict:
    path, data = _load_latest_json(results_dir, "cloud_run_deploy_plan_*.json")
    if path is None or data is None:
        return {"artifact": None, "passed": False, "reason": "missing cloud_run_deploy_plan artifact"}
    metadata = data.get("metadata", {})
    env = data.get("env_vars", {})
    deploy_command = data.get("deploy_command") or []
    invoker_command = data.get("invoker_command")
    runtime_iam_commands = data.get("runtime_iam_commands") or []
    failures: list[str] = []

    if "--no-allow-unauthenticated" not in deploy_command:
        failures.append("Cloud Run deploy does not include --no-allow-unauthenticated")
    if "--allow-unauthenticated" in deploy_command:
        failures.append("Cloud Run deploy includes unsafe --allow-unauthenticated")

    required_env = {
        "PERSISTENCE_BACKEND": "gcs",
        "SYNC_ON_STARTUP": "true",
        "REQUIRE_CORPUS_ON_STARTUP": "true",
        "MOCK_MODE": "false",
        "INFERENCE_BACKEND": "vertex",
        "EMBEDDING_DIMENSION": "3072",
        "CHROMA_DIR": "/tmp/chroma_db",
        "GCS_SNAPSHOT_PREFIX": "tee-corpus",
    }
    for key, expected in required_env.items():
        if str(env.get(key)) != expected:
            failures.append(f"{key} must be {expected!r}")

    for key in ("project", "image", "service_account", "gcs_bucket", "snapshot_prefix"):
        if _placeholder(str(metadata.get(key, ""))):
            failures.append(f"{key} is missing or placeholder")
    if _placeholder(str(env.get("GCS_BUCKET", ""))):
        failures.append("GCS_BUCKET env is missing or placeholder")
    if _placeholder(str(env.get("GCS_SNAPSHOT_PREFIX", ""))):
        failures.append("GCS_SNAPSHOT_PREFIX env is missing or placeholder")
    if _bucket_uri(str(metadata.get("gcs_bucket", ""))):
        failures.append("gcs_bucket must be a bucket name, not a gs:// URI")
    if _bucket_uri(str(env.get("GCS_BUCKET", ""))):
        failures.append("GCS_BUCKET env must be a bucket name, not a gs:// URI")

    service = str(metadata.get("service", ""))
    project = str(metadata.get("project", ""))
    region = str(metadata.get("region", ""))
    image = str(metadata.get("image", ""))
    service_account = str(metadata.get("service_account", ""))
    gcs_bucket = str(metadata.get("gcs_bucket", ""))
    snapshot_prefix = str(metadata.get("snapshot_prefix", ""))
    if not deploy_command[:3] == ["gcloud", "run", "deploy"] or len(deploy_command) < 4:
        failures.append("deploy command must be a gcloud run deploy command")
    elif service and deploy_command[3] != service:
        failures.append("deploy command service must match metadata service")
    if image and not _placeholder(image) and not _image_reference_pinned(image):
        failures.append("image must include an immutable digest or non-latest tag")
    image_project = _artifact_registry_project(image)
    if image_project is not None and project and not _placeholder(project) and image_project != project:
        failures.append("image Artifact Registry project must match deploy project")
    service_account_project = _service_account_project(service_account)
    if (
        service_account_project is not None
        and project
        and not _placeholder(project)
        and service_account_project != project
    ):
        failures.append("service_account project must match deploy project")
    if service_account and not _placeholder(service_account) and not _valid_service_account_email(service_account):
        failures.append("service_account must be a service account email")

    deploy_expected_flags = {
        "--project": ("project", project),
        "--region": ("region", region),
        "--image": ("image", image),
        "--service-account": ("service_account", service_account),
    }
    for flag, (field, expected) in deploy_expected_flags.items():
        if expected and not _placeholder(expected) and _flag_value(deploy_command, flag) != expected:
            failures.append(f"deploy command {flag} must match metadata {field}")

    command_env = _parse_env_arg(_flag_value(deploy_command, "--set-env-vars"))
    for key, expected in env.items():
        if str(command_env.get(key)) != str(expected):
            failures.append(f"deploy command env {key} must match deploy-plan env_vars")
    metadata_env = {
        "GOOGLE_CLOUD_PROJECT": ("project", project),
        "GCS_BUCKET": ("gcs_bucket", gcs_bucket),
        "GCS_SNAPSHOT_PREFIX": ("snapshot_prefix", snapshot_prefix),
    }
    for key, (field, expected) in metadata_env.items():
        if expected and not _placeholder(expected) and str(env.get(key)) != expected:
            failures.append(f"{key} env must match metadata {field}")

    resource_settings = data.get("resource_settings", {})
    memory = _flag_value(deploy_command, "--memory")
    cpu = _flag_value(deploy_command, "--cpu")
    timeout = _flag_value(deploy_command, "--timeout")
    min_instances = _flag_value(deploy_command, "--min-instances")
    expected_resources = {
        "memory": memory,
        "cpu": cpu,
        "timeout": timeout,
        "min_instances": int(min_instances) if str(min_instances or "").isdigit() else min_instances,
    }
    for key, actual in expected_resources.items():
        if resource_settings and str(resource_settings.get(key)) != str(actual):
            failures.append(f"deploy command {key} must match resource_settings")
    if (_memory_to_mib(memory) or 0) < 2048:
        failures.append("Cloud Run memory must be at least 2Gi")
    try:
        if float(cpu or "0") < 2:
            failures.append("Cloud Run CPU must be at least 2")
    except ValueError:
        failures.append("Cloud Run CPU must be numeric")
    try:
        if int(timeout or "0") < 300:
            failures.append("Cloud Run timeout must be at least 300 seconds")
    except ValueError:
        failures.append("Cloud Run timeout must be an integer number of seconds")
    try:
        if int(min_instances or "0") < 1:
            failures.append("Cloud Run min-instances must be at least 1")
    except ValueError:
        failures.append("Cloud Run min-instances must be an integer")

    invoker = metadata.get("invoker_member")
    if _placeholder(str(invoker or "")):
        failures.append("invoker_member is missing or placeholder")
    elif not _valid_invoker_member(str(invoker)):
        failures.append("invoker_member must be an IAM member with group:, user:, serviceAccount:, or domain:")
    if invoker in {"allUsers", "allAuthenticatedUsers"}:
        failures.append("invoker_member must not be public")
    if not invoker_command:
        failures.append("roles/run.invoker binding command is missing")
    elif "roles/run.invoker" not in invoker_command:
        failures.append("invoker command does not grant roles/run.invoker")
    elif str(invoker) not in invoker_command:
        failures.append("invoker command member must match metadata invoker_member")
    elif service and service not in invoker_command:
        failures.append("invoker command service must match metadata service")
    elif project and project not in invoker_command:
        failures.append("invoker command project must match metadata project")
    elif region and region not in invoker_command:
        failures.append("invoker command region must match metadata region")

    runtime_member = f"serviceAccount:{service_account}"
    flat_runtime_commands = [
        [str(part) for part in command]
        for command in runtime_iam_commands
        if isinstance(command, list)
    ]
    vertex_commands = [
        command
        for command in flat_runtime_commands
        if "roles/aiplatform.user" in command
    ]
    storage_commands = [
        command
        for command in flat_runtime_commands
        if "roles/storage.objectViewer" in command
    ]

    if not vertex_commands:
        failures.append("runtime IAM command missing roles/aiplatform.user")
    if not storage_commands:
        failures.append("runtime IAM command missing roles/storage.objectViewer")
    if vertex_commands and not any(runtime_member in command for command in vertex_commands):
        failures.append("Vertex runtime IAM command must target deploy service account")
    if vertex_commands and not any(str(metadata.get("project", "")) in command for command in vertex_commands):
        failures.append("Vertex runtime IAM command must target deploy project")
    if storage_commands and not any(runtime_member in command for command in storage_commands):
        failures.append("GCS runtime IAM command must target deploy service account")
    bucket_resource = gcs_bucket if gcs_bucket.startswith("gs://") else f"gs://{gcs_bucket}"
    if storage_commands and not any(bucket_resource in command for command in storage_commands):
        failures.append("GCS runtime IAM command must target corpus bucket")

    return {
        "artifact": str(path),
        "passed": not failures,
        "failures": failures,
        "service": metadata.get("service"),
        "project": metadata.get("project"),
        "region": metadata.get("region"),
        "invoker_member": invoker,
    }


def snapshot_publication_status(results_dir: Path) -> dict:
    path, data = _load_latest_json(results_dir, "corpus_snapshot_publish_plan_*.json")
    if path is None or data is None:
        return {"artifact": None, "passed": False, "reason": "missing corpus_snapshot_publish_plan artifact"}
    metadata = data.get("metadata", {})
    env = data.get("env_vars", {})
    command = data.get("command") or []
    readiness = data.get("chroma_dir_readiness", {})
    failures: list[str] = []

    if command != ["python", "-m", "scripts.publish_corpus_snapshot"]:
        failures.append("snapshot publish command must run python -m scripts.publish_corpus_snapshot")

    required_env = {
        "PERSISTENCE_BACKEND": "gcs",
        "GCS_SNAPSHOT_PREFIX": "tee-corpus",
        "EMBEDDING_DIMENSION": "3072",
    }
    for key, expected in required_env.items():
        if str(env.get(key)) != expected:
            failures.append(f"snapshot publish {key} must be {expected!r}")

    for key in ("gcs_bucket", "snapshot_prefix", "chroma_dir"):
        if _placeholder(str(metadata.get(key, ""))):
            failures.append(f"snapshot publish {key} is missing or placeholder")
    if _placeholder(str(env.get("GCS_BUCKET", ""))):
        failures.append("snapshot publish GCS_BUCKET env is missing or placeholder")
    if _placeholder(str(env.get("GCS_SNAPSHOT_PREFIX", ""))):
        failures.append("snapshot publish GCS_SNAPSHOT_PREFIX env is missing or placeholder")
    if _bucket_uri(str(metadata.get("gcs_bucket", ""))):
        failures.append("snapshot publish gcs_bucket must be a bucket name, not a gs:// URI")
    if _bucket_uri(str(env.get("GCS_BUCKET", ""))):
        failures.append("snapshot publish GCS_BUCKET env must be a bucket name, not a gs:// URI")

    chroma_dir = str(metadata.get("chroma_dir", ""))
    if chroma_dir:
        chroma_path = Path(chroma_dir).expanduser().resolve(strict=False)
        baseline_path = (REPO / "chroma_db").resolve(strict=False)
        repo_path = REPO.resolve(strict=False)
        if chroma_path.name == "chroma_db" or chroma_path == baseline_path:
            failures.append("snapshot publish chroma_dir must not be baseline chroma_db")
        if _is_relative_to(chroma_path, baseline_path):
            failures.append("snapshot publish chroma_dir must not be inside baseline chroma_db")
        if chroma_path == repo_path:
            failures.append("snapshot publish chroma_dir must not be repository root")
        if _is_relative_to(repo_path, chroma_path) or _is_relative_to(baseline_path, chroma_path):
            failures.append("snapshot publish chroma_dir must not be a parent of the repo or baseline chroma_db")
    if str(env.get("CHROMA_DIR")) != chroma_dir:
        failures.append("snapshot publish CHROMA_DIR env must match metadata chroma_dir")
    if str(env.get("GCS_BUCKET")) != str(metadata.get("gcs_bucket", "")):
        failures.append("snapshot publish GCS_BUCKET env must match metadata gcs_bucket")
    if str(env.get("GCS_SNAPSHOT_PREFIX")) != str(metadata.get("snapshot_prefix", "")):
        failures.append("snapshot publish GCS_SNAPSHOT_PREFIX env must match metadata snapshot_prefix")
    if not isinstance(readiness, dict) or readiness.get("ok") is not True:
        failures.append("snapshot publish chroma_dir_readiness must pass")
    elif _norm_path(str(readiness.get("path", ""))) != _norm_path(chroma_dir):
        failures.append("snapshot publish readiness path must match metadata chroma_dir")
    else:
        chroma_sqlite = str(readiness.get("chroma_sqlite", ""))
        hnsw_dirs = readiness.get("hnsw_index_dirs")
        if not chroma_sqlite:
            failures.append("snapshot publish readiness must include chroma_sqlite")
        elif _norm_path(chroma_sqlite) != _norm_path(str(Path(chroma_dir) / "chroma.sqlite3")):
            failures.append("snapshot publish chroma_sqlite must be inside metadata chroma_dir")
        if not isinstance(hnsw_dirs, list) or not hnsw_dirs:
            failures.append("snapshot publish readiness must include at least one HNSW index directory")
        else:
            chroma_path = Path(chroma_dir).expanduser().resolve(strict=False)
            outside_hnsw_dirs = [
                item
                for item in hnsw_dirs
                if not _is_relative_to(Path(str(item)).expanduser().resolve(strict=False), chroma_path)
            ]
            if outside_hnsw_dirs:
                failures.append("snapshot publish HNSW index dirs must be inside metadata chroma_dir")
    invariants = data.get("safety_invariants", {})
    if invariants.get("chroma_dir_ready") is not True:
        failures.append("snapshot publish safety invariant chroma_dir_ready must be true")

    return {
        "artifact": str(path),
        "passed": not failures,
        "failures": failures,
        "chroma_dir": chroma_dir,
        "gcs_bucket": metadata.get("gcs_bucket"),
        "snapshot_prefix": metadata.get("snapshot_prefix"),
    }


def artifact_consistency_status(eval_path: Path, results_dir: Path) -> dict:
    m3_path, m3_data = _load_latest_json(results_dir, "enriched_experiment_*.json")
    m4_path, m4_data = _load_latest_json(results_dir, "citation_grounding_*.json")
    deploy_path, deploy_data = _load_latest_json(results_dir, "cloud_run_deploy_plan_*.json")
    snapshot_path, snapshot_data = _load_latest_json(results_dir, "corpus_snapshot_publish_plan_*.json")
    failures: list[str] = []

    if m3_path is None or m3_data is None:
        failures.append("missing enriched_experiment artifact")
    if m4_path is None or m4_data is None:
        failures.append("missing citation_grounding artifact")
    if deploy_path is None or deploy_data is None:
        failures.append("missing cloud_run_deploy_plan artifact")
    if snapshot_path is None or snapshot_data is None:
        failures.append("missing corpus_snapshot_publish_plan artifact")
    if failures:
        return {
            "passed": False,
            "failures": failures,
            "m3_artifact": str(m3_path) if m3_path else None,
            "m4_artifact": str(m4_path) if m4_path else None,
            "deploy_artifact": str(deploy_path) if deploy_path else None,
            "snapshot_artifact": str(snapshot_path) if snapshot_path else None,
        }

    expected_eval = _norm_path(str(eval_path))
    questions = _load_questions(eval_path)
    eval_total = len(questions)
    evaluated_questions = [
        q for q in questions
        if q.get("answerable", True) and q.get("expected_sources")
    ]
    eval_evaluated = len(evaluated_questions)
    eval_evaluated_ids = {str(q.get("id", "")) for q in evaluated_questions}
    m3_meta = m3_data.get("metadata", {})
    m3_preflight = m3_data.get("preflight", {})
    m4_meta = m4_data.get("metadata", {})
    deploy_meta = deploy_data.get("metadata", {})
    snapshot_meta = snapshot_data.get("metadata", {})
    m3_eval = _norm_path(str(m3_meta.get("eval_path", "")))
    m3_preflight_eval = _norm_path(str(m3_preflight.get("eval_path", "")))
    m4_eval = _norm_path(str(m4_meta.get("eval_path", "")))
    m3_chroma = _norm_path(str(m3_meta.get("experiment_chroma_dir", "")))
    m4_chroma = _norm_path(str(m4_meta.get("chroma_dir", "")))
    snapshot_chroma = _norm_path(str(snapshot_meta.get("chroma_dir", "")))

    if m3_eval != expected_eval:
        failures.append("M3 eval_path must match certification eval_path")
    if m3_preflight_eval != expected_eval:
        failures.append("M3 preflight eval_path must match certification eval_path")
    if m4_eval != expected_eval:
        failures.append("M4 eval_path must match certification eval_path")
    if m3_chroma != m4_chroma:
        failures.append("M3 experiment_chroma_dir must match M4 chroma_dir")
    if m3_chroma != snapshot_chroma:
        failures.append("M3 experiment_chroma_dir must match snapshot publish chroma_dir")
    if str(deploy_meta.get("gcs_bucket", "")) != str(snapshot_meta.get("gcs_bucket", "")):
        failures.append("deploy gcs_bucket must match snapshot publish gcs_bucket")
    if str(deploy_meta.get("snapshot_prefix", "")) != str(snapshot_meta.get("snapshot_prefix", "")):
        failures.append("deploy snapshot_prefix must match snapshot publish snapshot_prefix")

    recall = m3_data.get("recall_with_rerank") or m3_data.get("recall_no_rerank") or {}
    m3_n = recall.get("n_questions_evaluated")
    m4_n = m4_meta.get("questions_selected")
    if m3_n != m4_n:
        failures.append("M3 n_questions_evaluated must match M4 questions_selected")
    if m3_preflight.get("eval_questions") != m4_meta.get("questions_total"):
        failures.append("M3 preflight eval_questions must match M4 questions_total")
    if m3_preflight.get("eval_questions") != eval_total:
        failures.append("artifact eval question count must match certification eval file")
    if m3_n != eval_evaluated:
        failures.append("M3 n_questions_evaluated must match answerable sourced eval count")
    if m4_n != eval_evaluated:
        failures.append("M4 questions_selected must match answerable sourced eval count")

    recall_artifact = _resolve_artifact_path(recall.get("artifact_json"), base_dir=results_dir)
    if recall_artifact and recall_artifact.exists():
        try:
            recall_report = json.loads(recall_artifact.read_text(encoding="utf-8"))
            m3_row_ids = _row_ids(recall_report.get("per_question", []))
            if m3_row_ids is None:
                failures.append("M3 per_question must be a list")
            else:
                _append_row_shape_failures("M3", recall_report.get("per_question", []), failures)
                _append_row_id_failures("M3", m3_row_ids, failures)
                if set(m3_row_ids) != eval_evaluated_ids:
                    failures.append("M3 per_question IDs must match answerable sourced eval IDs")
        except Exception as exc:  # noqa: BLE001 - malformed evidence is a no-go
            failures.append(f"M3 per_question IDs unreadable: {type(exc).__name__}: {exc}")

    m4_row_ids = _row_ids(m4_data.get("per_question", []))
    if m4_row_ids is None:
        failures.append("M4 per_question must be a list")
    else:
        _append_row_shape_failures("M4", m4_data.get("per_question", []), failures)
        _append_row_id_failures("M4", m4_row_ids, failures)
        if set(m4_row_ids) != eval_evaluated_ids:
            failures.append("M4 per_question IDs must match answerable sourced eval IDs")

    m3_run_id = m3_meta.get("certification_run_id")
    m4_run_id = m4_meta.get("certification_run_id")
    deploy_run_id = deploy_meta.get("certification_run_id")
    snapshot_run_id = snapshot_meta.get("certification_run_id")
    if not m3_run_id:
        failures.append("M3 certification_run_id missing")
    if not m4_run_id:
        failures.append("M4 certification_run_id missing")
    if not deploy_run_id:
        failures.append("deploy certification_run_id missing")
    if not snapshot_run_id:
        failures.append("snapshot publish certification_run_id missing")
    run_ids = [m3_run_id, m4_run_id, deploy_run_id, snapshot_run_id]
    if all(run_ids) and len(set(run_ids)) != 1:
        failures.append("M3, M4, deploy, and snapshot publish certification_run_id values must match")

    return {
        "passed": not failures,
        "failures": failures,
        "m3_artifact": str(m3_path),
        "m4_artifact": str(m4_path),
        "deploy_artifact": str(deploy_path),
        "snapshot_artifact": str(snapshot_path),
        "eval_path": expected_eval,
        "chroma_dir": m3_chroma,
        "eval_questions": eval_total,
        "eval_evaluated": eval_evaluated,
        "certification_run_id": m3_run_id,
    }


def container_image_status(repo: Path = REPO) -> dict:
    dockerfile = repo / "Dockerfile"
    dockerignore = repo / ".dockerignore"
    failures: list[str] = []
    dockerfile_text = ""
    dockerignore_patterns: set[str] = set()

    if not dockerfile.exists():
        failures.append("Dockerfile missing")
    else:
        dockerfile_text = dockerfile.read_text(encoding="utf-8")
        if "USER app" not in dockerfile_text:
            failures.append("Dockerfile must switch to non-root USER app before ENTRYPOINT")
        elif "ENTRYPOINT" in dockerfile_text and dockerfile_text.index("USER app") > dockerfile_text.index("ENTRYPOINT"):
            failures.append("Dockerfile USER app must appear before ENTRYPOINT")
        if "useradd --system" not in dockerfile_text:
            failures.append("Dockerfile must create an unprivileged app user")
        if "HF_HOME=/tmp/hf-cache" not in dockerfile_text:
            failures.append("Dockerfile must keep HF_HOME in writable /tmp")
        if "XDG_CACHE_HOME=/tmp/.cache" not in dockerfile_text:
            failures.append("Dockerfile must keep XDG_CACHE_HOME in writable /tmp")
        if "/tmp/chroma_db" not in dockerfile_text:
            failures.append("Dockerfile must prepare /tmp/chroma_db for Cloud Run corpus restore")

    if not dockerignore.exists():
        failures.append(".dockerignore missing")
    else:
        dockerignore_patterns = _dockerignore_patterns(dockerignore)
        missing = sorted(REQUIRED_DOCKERIGNORE_PATTERNS - dockerignore_patterns)
        if missing:
            failures.append(".dockerignore missing production exclusions: " + ", ".join(missing))
        for runtime_path in ("src/", "app.py", "entrypoint.sh", "requirements.txt"):
            if runtime_path in dockerignore_patterns:
                failures.append(f".dockerignore must not exclude runtime file/path: {runtime_path}")

    return {
        "passed": not failures,
        "failures": failures,
        "dockerfile": str(dockerfile),
        "dockerignore": str(dockerignore),
        "non_root_user": "USER app" in dockerfile_text,
        "required_exclusions_present": REQUIRED_DOCKERIGNORE_PATTERNS <= dockerignore_patterns,
    }


def next_actions(gates: dict) -> list[dict]:
    """Return ordered human/operator actions needed to satisfy failed gates."""
    actions: list[dict] = []
    eval_gate = gates["eval_validation"]
    m3_gate = gates["m3_recall"]
    m4_gate = gates["m4_citation_grounding"]
    deploy_gate = gates["deployment_safety"]
    snapshot_gate = gates["snapshot_publication"]
    container_gate = gates["container_image"]

    if not eval_gate["passed"]:
        actions.append({
            "gate": "eval_validation",
            "owner": "DELF/DALF expert",
            "reason": "Expert validation is required before M3/M4 can be certified.",
            "evidence_needed": (
                "Completed DELF/DALF expert review CSV with a valid validation_decision for every "
                "answerable row, no AI/LLM-grounded provenance markers, then a strict validation manifest."
            ),
            "current_gap": {
                "unvalidated_answerable": eval_gate.get("unvalidated_answerable"),
                "sample_ids": eval_gate.get("unvalidated_answerable_ids", []),
            },
            "commands": [
                "python3 -m scripts.lint_expert_review --review evaluation/delf_questions_expert_review.csv --json evaluation/delf_questions.json --corpus-files evaluation/corpus_files.json --require-complete",
                "python3 -m scripts.apply_expert_validation --review evaluation/delf_questions_expert_review.csv --output evaluation/delf_questions.validated.json --strict",
            ],
        })

    if not m3_gate["passed"]:
        commands = [
            "python3 -m scripts.run_enriched_retrieval_experiment --preflight",
            "python3 -m scripts.run_enriched_retrieval_experiment --eval-path evaluation/delf_questions.validated.json --require-validated --certification-run-id \"$CERT_RUN_ID\" --run-ingest --force --evaluate",
        ]
        actions.append({
            "gate": "m3_recall",
            "owner": "engineering",
            "reason": "M3 requires an enriched, non-baseline retrieval run with recall@3 >= 0.90.",
            "evidence_needed": "evaluation/results/enriched_experiment_*.json plus linked recall JSON/Markdown artifacts.",
            "blocked_by": ["eval_validation"] if not eval_gate["passed"] else [],
            "commands": commands,
        })

    if not m4_gate["passed"]:
        actions.append({
            "gate": "m4_citation_grounding",
            "owner": "engineering",
            "reason": "M4 requires citation grounding >= 0.95 over the full validated eval set.",
            "evidence_needed": "evaluation/results/citation_grounding_*.json from the same validated eval and enriched Chroma dir.",
            "blocked_by": [
                gate
                for gate, passed in (
                    ("eval_validation", eval_gate["passed"]),
                    ("m3_recall", m3_gate["passed"]),
                )
                if not passed
            ],
            "commands": [
                "python3 -m scripts.measure_citation_grounding --eval-path evaluation/delf_questions.validated.json --chroma-dir chroma_db_enriched --require-validated --certification-run-id \"$CERT_RUN_ID\"",
            ],
        })

    deploy_or_snapshot_blockers = [
        gate
        for gate, passed in (
            ("eval_validation", eval_gate["passed"]),
            ("m3_recall", m3_gate["passed"]),
            ("m4_citation_grounding", m4_gate["passed"]),
        )
        if not passed
    ]
    if not snapshot_gate["passed"]:
        actions.append({
            "gate": "snapshot_publication",
            "owner": "engineering",
            "reason": "Production serving needs a ready, publishable GCS-backed corpus snapshot plan.",
            "evidence_needed": "evaluation/results/corpus_snapshot_publish_plan_*.json for the certified enriched Chroma dir.",
            "blocked_by": deploy_or_snapshot_blockers,
            "commands": [
                "python3 -m scripts.render_snapshot_publish_plan --chroma-dir chroma_db_enriched --gcs-bucket REAL_CORPUS_BUCKET --snapshot-prefix tee-corpus --certification-run-id \"$CERT_RUN_ID\" --write-plan",
            ],
        })
    if not deploy_gate["passed"]:
        actions.append({
            "gate": "deployment_safety",
            "owner": "engineering",
            "reason": "Production serving must be private, authenticated, pinned-image, and GCS-backed.",
            "evidence_needed": "evaluation/results/cloud_run_deploy_plan_*.json with real project, image, bucket, service account, and invoker values.",
            "blocked_by": deploy_or_snapshot_blockers,
            "commands": [
                "python3 -m scripts.render_cloud_run_deploy --project woven-operative-491610-u6 --region us-central1 --image us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf-dalf-assistant:REAL_IMAGE_TAG --service-account delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com --gcs-bucket REAL_CORPUS_BUCKET --invoker group:REAL_APPROVED_GROUP@YOUR_DOMAIN --certification-run-id \"$CERT_RUN_ID\" --write-plan",
            ],
        })
    if not container_gate["passed"]:
        actions.append({
            "gate": "container_image",
            "owner": "engineering",
            "reason": "Container evidence must prove a non-root runtime and safe build context.",
            "evidence_needed": "Hardened Dockerfile and .dockerignore accepted by container_image_status.",
            "commands": [
                "python3 -m scripts.certification_status",
            ],
        })
    return actions


def certification_status(
    eval_path: Path,
    results_dir: Path,
    corpus_files_path: Path | None = None,
    review_path: Path | None = None,
    repo: Path = REPO,
) -> dict:
    gates = {
        "eval_validation": eval_status(eval_path, corpus_files_path, review_path),
        "m3_recall": enriched_status(results_dir),
        "m4_citation_grounding": citation_status(results_dir),
        "artifact_consistency": artifact_consistency_status(eval_path, results_dir),
        "deployment_safety": deployment_status(results_dir),
        "snapshot_publication": snapshot_publication_status(results_dir),
        "container_image": container_image_status(repo),
    }
    certified = all(gate["passed"] for gate in gates.values())
    return {
        "certified": certified,
        "gates": gates,
        "next_actions": [] if certified else next_actions(gates),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect retrieval certification status.")
    parser.add_argument("--eval-path", type=Path, default=None)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--corpus-files", type=Path, default=DEFAULT_CORPUS_FILES)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()

    eval_path = args.eval_path.expanduser() if args.eval_path is not None else default_eval_path()

    status = certification_status(
        eval_path,
        args.results_dir.expanduser(),
        args.corpus_files.expanduser(),
        args.review.expanduser(),
        args.repo.expanduser(),
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
