"""
Render a production-safe Cloud Run deploy command for the DELF/DALF assistant.

The script is intentionally a renderer, not an executor. It makes the dangerous
serving choices explicit and testable:
  - no public unauthenticated access,
  - GCS-backed corpus snapshot sync on startup,
  - hard failure when the corpus is not ready,
  - Vertex AI backend with MOCK_MODE disabled.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SERVICE = "delf-dalf-examiner-assistant"
DEFAULT_REGION = "us-central1"
DEFAULT_MEMORY = "2Gi"
DEFAULT_CPU = "2"
DEFAULT_TIMEOUT = "300"
DEFAULT_CHROMA_DIR = "/tmp/chroma_db"
DEFAULT_SNAPSHOT_PREFIX = "tee-corpus"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "results"


@dataclass(frozen=True)
class DeployConfig:
    project: str
    image: str
    service_account: str
    gcs_bucket: str
    service: str = DEFAULT_SERVICE
    region: str = DEFAULT_REGION
    memory: str = DEFAULT_MEMORY
    cpu: str = DEFAULT_CPU
    timeout: str = DEFAULT_TIMEOUT
    min_instances: int = 1
    chroma_dir: str = DEFAULT_CHROMA_DIR
    snapshot_prefix: str = DEFAULT_SNAPSHOT_PREFIX
    generation_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 3072
    location: str | None = None
    invoker: str | None = None
    certification_run_id: str | None = None

    def env_vars(self) -> dict[str, str]:
        location = self.location or self.region
        return {
            "GOOGLE_CLOUD_PROJECT": self.project,
            "GOOGLE_CLOUD_LOCATION": location,
            "INFERENCE_BACKEND": "vertex",
            "MOCK_MODE": "false",
            "PERSISTENCE_BACKEND": "gcs",
            "GCS_BUCKET": self.gcs_bucket,
            "GCS_SNAPSHOT_PREFIX": self.snapshot_prefix,
            "SYNC_ON_STARTUP": "true",
            "REQUIRE_CORPUS_ON_STARTUP": "true",
            "CHROMA_DIR": self.chroma_dir,
            "GENERATION_MODEL": self.generation_model,
            "EMBEDDING_MODEL": self.embedding_model,
            "EMBEDDING_DIMENSION": str(self.embedding_dimension),
            "ENABLE_CONTEXTUAL_ENRICHMENT": "false",
            "ENABLE_HYBRID_SEARCH": "true",
            "ENABLE_CROSS_LINGUAL_BM25": "false",
        }


def _require_non_empty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise SystemExit(f"{name} is required")


def _is_placeholder(value: str | None) -> bool:
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


def _reject_placeholder(name: str, value: str | None) -> None:
    if _is_placeholder(value):
        raise SystemExit(f"{name} must be a real non-placeholder value")


def _reject_gs_uri_bucket(name: str, value: str) -> None:
    if value.strip().startswith("gs://"):
        raise SystemExit(f"{name} must be a bucket name, not a gs:// URI")


def _validate_image_reference(image: str) -> None:
    last_segment = image.rsplit("/", 1)[-1]
    if "@sha256:" in image:
        return
    if ":" not in last_segment:
        raise SystemExit("--image must include an immutable digest or a non-latest tag")
    tag = last_segment.rsplit(":", 1)[-1].strip().lower()
    if tag == "latest":
        raise SystemExit("--image must not use the mutable latest tag")


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


def _validate_service_account_email(service_account: str) -> None:
    if "@" not in service_account or not service_account.endswith(".iam.gserviceaccount.com"):
        raise SystemExit("--service-account must be a service account email ending in .iam.gserviceaccount.com")


def _validate_project_scoped_refs(config: DeployConfig) -> None:
    _validate_service_account_email(config.service_account)
    image_project = _artifact_registry_project(config.image)
    if image_project is not None and image_project != config.project:
        raise SystemExit("--image Artifact Registry project must match --project")
    service_account_project = _service_account_project(config.service_account)
    if service_account_project is not None and service_account_project != config.project:
        raise SystemExit("--service-account project must match --project")


def _validate_invoker_member(invoker: str) -> None:
    allowed_prefixes = ("group:", "user:", "serviceAccount:", "domain:")
    if not invoker.startswith(allowed_prefixes):
        raise SystemExit("--invoker must be an IAM member such as group:, user:, serviceAccount:, or domain:")


def _env_arg(env_vars: dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in sorted(env_vars.items()))


def _memory_to_mib(value: str) -> int | None:
    text = value.strip()
    if text.endswith("Gi"):
        return int(text[:-2]) * 1024
    if text.endswith("Mi"):
        return int(text[:-2])
    return None


def _validate_serving_shape(config: DeployConfig) -> None:
    memory_mib = _memory_to_mib(config.memory)
    if memory_mib is None or memory_mib < 2048:
        raise SystemExit("--memory must be at least 2Gi for corpus snapshot restore")
    try:
        cpu = float(config.cpu)
    except ValueError as exc:
        raise SystemExit("--cpu must be numeric") from exc
    if cpu < 2:
        raise SystemExit("--cpu must be at least 2 for corpus snapshot restore")
    try:
        timeout = int(config.timeout)
    except ValueError as exc:
        raise SystemExit("--timeout must be seconds as an integer") from exc
    if timeout < 300:
        raise SystemExit("--timeout must be at least 300 seconds")
    if config.min_instances < 1:
        raise SystemExit("--min-instances must be at least 1 for production serving")


def render_deploy_command(config: DeployConfig) -> list[str]:
    _require_non_empty("--project", config.project)
    _require_non_empty("--image", config.image)
    _require_non_empty("--service-account", config.service_account)
    _require_non_empty("--gcs-bucket", config.gcs_bucket)
    _require_non_empty("--snapshot-prefix", config.snapshot_prefix)
    _reject_placeholder("--project", config.project)
    _reject_placeholder("--image", config.image)
    _reject_placeholder("--service-account", config.service_account)
    _reject_placeholder("--gcs-bucket", config.gcs_bucket)
    _reject_placeholder("--snapshot-prefix", config.snapshot_prefix)
    _reject_gs_uri_bucket("--gcs-bucket", config.gcs_bucket)
    _validate_image_reference(config.image)
    _validate_project_scoped_refs(config)
    if config.embedding_dimension != 3072:
        raise SystemExit("EMBEDDING_DIMENSION must remain 3072 unless the corpus is fully re-ingested")
    _validate_serving_shape(config)

    return [
        "gcloud", "run", "deploy", config.service,
        "--project", config.project,
        "--region", config.region,
        "--image", config.image,
        "--service-account", config.service_account,
        "--memory", config.memory,
        "--cpu", config.cpu,
        "--timeout", config.timeout,
        "--min-instances", str(config.min_instances),
        "--set-env-vars", _env_arg(config.env_vars()),
        "--no-allow-unauthenticated",
    ]


def render_invoker_command(config: DeployConfig) -> list[str] | None:
    if not config.invoker:
        return None
    _reject_placeholder("--invoker", config.invoker)
    _validate_invoker_member(config.invoker)
    if config.invoker in {"allUsers", "allAuthenticatedUsers"}:
        raise SystemExit("--invoker must not be public")
    return [
        "gcloud", "run", "services", "add-iam-policy-binding", config.service,
        "--project", config.project,
        "--region", config.region,
        "--member", config.invoker,
        "--role", "roles/run.invoker",
    ]


def render_runtime_iam_commands(config: DeployConfig) -> list[list[str]]:
    _require_non_empty("--project", config.project)
    _require_non_empty("--service-account", config.service_account)
    _require_non_empty("--gcs-bucket", config.gcs_bucket)
    _validate_service_account_email(config.service_account)
    bucket = config.gcs_bucket
    if not bucket.startswith("gs://"):
        bucket = f"gs://{bucket}"
    member = f"serviceAccount:{config.service_account}"
    return [
        [
            "gcloud", "projects", "add-iam-policy-binding", config.project,
            "--member", member,
            "--role", "roles/aiplatform.user",
        ],
        [
            "gcloud", "storage", "buckets", "add-iam-policy-binding", bucket,
            "--member", member,
            "--role", "roles/storage.objectViewer",
        ],
    ]


def build_deploy_plan(config: DeployConfig) -> dict:
    deploy_command = render_deploy_command(config)
    invoker_command = render_invoker_command(config)
    runtime_iam_commands = render_runtime_iam_commands(config)
    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "service": config.service,
            "project": config.project,
            "region": config.region,
            "image": config.image,
            "service_account": config.service_account,
            "gcs_bucket": config.gcs_bucket,
            "snapshot_prefix": config.snapshot_prefix,
            "invoker_member": config.invoker,
            "certification_run_id": config.certification_run_id,
        },
        "deploy_command": deploy_command,
        "invoker_command": invoker_command,
        "runtime_iam_commands": runtime_iam_commands,
        "resource_settings": {
            "memory": config.memory,
            "cpu": config.cpu,
            "timeout": config.timeout,
            "min_instances": config.min_instances,
        },
        "env_vars": config.env_vars(),
        "safety_invariants": {
            "private_cloud_run": "--no-allow-unauthenticated" in deploy_command,
            "gcs_persistence": config.env_vars().get("PERSISTENCE_BACKEND") == "gcs",
            "startup_requires_corpus": config.env_vars().get("REQUIRE_CORPUS_ON_STARTUP") == "true",
            "mock_mode_disabled": config.env_vars().get("MOCK_MODE") == "false",
            "embedding_dimension_locked": config.embedding_dimension == 3072,
            "explicit_invoker_binding": invoker_command is not None,
            "runtime_vertex_access": any("roles/aiplatform.user" in command for command in runtime_iam_commands),
            "runtime_gcs_read_access": any("roles/storage.objectViewer" in command for command in runtime_iam_commands),
            "serving_shape": (
                (_memory_to_mib(config.memory) or 0) >= 2048
                and float(config.cpu) >= 2
                and int(config.timeout) >= 300
                and config.min_instances >= 1
            ),
        },
    }


def write_deploy_plan(plan: dict, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = results_dir / f"cloud_run_deploy_plan_{ts}.json"
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render production Cloud Run deploy commands.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--gcs-bucket", required=True)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--location")
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--cpu", default=DEFAULT_CPU)
    parser.add_argument("--timeout", default=DEFAULT_TIMEOUT)
    parser.add_argument("--min-instances", type=int, default=1)
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--snapshot-prefix", default=DEFAULT_SNAPSHOT_PREFIX)
    parser.add_argument("--generation-model", default="gemini-2.5-flash")
    parser.add_argument("--embedding-model", default="gemini-embedding-001")
    parser.add_argument("--embedding-dimension", type=int, default=3072)
    parser.add_argument(
        "--invoker",
        help="Optional IAM member to grant roles/run.invoker, e.g. group:examiners@example.org",
    )
    parser.add_argument("--certification-run-id")
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="Write a machine-checkable deployment plan artifact under evaluation/results/.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    config = DeployConfig(
        project=args.project,
        image=args.image,
        service_account=args.service_account,
        gcs_bucket=args.gcs_bucket,
        service=args.service,
        region=args.region,
        location=args.location,
        memory=args.memory,
        cpu=args.cpu,
        timeout=args.timeout,
        min_instances=args.min_instances,
        chroma_dir=args.chroma_dir,
        snapshot_prefix=args.snapshot_prefix,
        generation_model=args.generation_model,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        invoker=args.invoker,
        certification_run_id=args.certification_run_id,
    )

    plan = build_deploy_plan(config)

    print("# Grant runtime service account access")
    for command in plan["runtime_iam_commands"]:
        print(_shell_join(command))
    print()
    print("# Deploy authenticated Cloud Run service")
    print(_shell_join(plan["deploy_command"]))
    print()
    print("# Grant access to an approved user/group only")
    invoker_command = plan["invoker_command"]
    if invoker_command:
        print(_shell_join(invoker_command))
    else:
        print("# Re-run with --invoker group:YOUR_GROUP@example.org to render the IAM binding.")
    if args.write_plan:
        print()
        print(f"# Deploy plan artifact: {write_deploy_plan(plan, args.results_dir.expanduser())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
