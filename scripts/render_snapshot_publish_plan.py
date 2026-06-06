"""
Render a production corpus snapshot publication plan.

This is intentionally a renderer, not an uploader. The actual snapshot publish is
performed by scripts.publish_corpus_snapshot, which publishes an already-certified
local corpus without re-ingesting it. The plan artifact makes the certified corpus
source, destination bucket/prefix, and embedding settings machine-checkable before
Cloud Run deploy.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO / "evaluation" / "results"
DEFAULT_CHROMA_DIR = REPO / "chroma_db_enriched"
BASELINE_CHROMA_DIR = REPO / "chroma_db"
DEFAULT_SNAPSHOT_PREFIX = "tee-corpus"


@dataclass(frozen=True)
class SnapshotPublishConfig:
    chroma_dir: str
    gcs_bucket: str
    snapshot_prefix: str = DEFAULT_SNAPSHOT_PREFIX
    embedding_dimension: int = 3072
    certification_run_id: str | None = None

    def env_vars(self) -> dict[str, str]:
        return {
            "PERSISTENCE_BACKEND": "gcs",
            "GCS_BUCKET": self.gcs_bucket,
            "GCS_SNAPSHOT_PREFIX": self.snapshot_prefix,
            "CHROMA_DIR": self.chroma_dir,
            "EMBEDDING_DIMENSION": str(self.embedding_dimension),
        }


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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_chroma_dir_target(chroma_path: Path) -> None:
    baseline = BASELINE_CHROMA_DIR.resolve(strict=False)
    repo = REPO.resolve(strict=False)
    if chroma_path.name == "chroma_db" or chroma_path == baseline:
        raise SystemExit("--chroma-dir must not be the baseline chroma_db")
    if _is_relative_to(chroma_path, baseline):
        raise SystemExit("--chroma-dir must not be inside the baseline chroma_db")
    if chroma_path == repo:
        raise SystemExit("--chroma-dir must not be the repository root")
    if _is_relative_to(repo, chroma_path) or _is_relative_to(baseline, chroma_path):
        raise SystemExit("--chroma-dir must not be a parent of the repository or baseline chroma_db")


def render_publish_command() -> list[str]:
    return ["python", "-m", "scripts.publish_corpus_snapshot"]


def _chroma_dir_readiness(chroma_dir: Path) -> dict:
    failures: list[str] = []
    if not chroma_dir.exists():
        failures.append(f"CHROMA_DIR does not exist: {chroma_dir}")
    elif not chroma_dir.is_dir():
        failures.append(f"CHROMA_DIR is not a directory: {chroma_dir}")
    if chroma_dir.is_symlink():
        failures.append(f"CHROMA_DIR must not be a symlink: {chroma_dir}")
    for filename in ("parents.json", "bm25_index.pkl"):
        path = chroma_dir / filename
        if not path.exists():
            failures.append(f"{filename} missing: {path}")
        elif path.stat().st_size == 0:
            failures.append(f"{filename} is empty: {path}")
    return {
        "ok": not failures,
        "failures": failures,
        "path": str(chroma_dir),
        "parents_json": str(chroma_dir / "parents.json"),
        "bm25_index": str(chroma_dir / "bm25_index.pkl"),
    }


def build_snapshot_publish_plan(config: SnapshotPublishConfig) -> dict:
    _reject_placeholder("--gcs-bucket", config.gcs_bucket)
    _reject_placeholder("--snapshot-prefix", config.snapshot_prefix)
    _reject_gs_uri_bucket("--gcs-bucket", config.gcs_bucket)
    chroma_path = Path(config.chroma_dir).expanduser().resolve(strict=False)
    _validate_chroma_dir_target(chroma_path)
    chroma_dir = str(chroma_path)
    if config.embedding_dimension != 3072:
        raise SystemExit("EMBEDDING_DIMENSION must remain 3072 unless the corpus is fully re-ingested")
    readiness = _chroma_dir_readiness(chroma_path)
    if not readiness["ok"]:
        raise SystemExit(
            "Cannot render snapshot publish plan for incomplete CHROMA_DIR:\n"
            + "\n".join(readiness["failures"])
        )

    normalized = SnapshotPublishConfig(
        chroma_dir=chroma_dir,
        gcs_bucket=config.gcs_bucket,
        snapshot_prefix=config.snapshot_prefix,
        embedding_dimension=config.embedding_dimension,
        certification_run_id=config.certification_run_id,
    )
    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chroma_dir": chroma_dir,
            "gcs_bucket": config.gcs_bucket,
            "snapshot_prefix": config.snapshot_prefix,
            "certification_run_id": config.certification_run_id,
        },
        "command": render_publish_command(),
        "env_vars": normalized.env_vars(),
        "chroma_dir_readiness": readiness,
        "safety_invariants": {
            "gcs_persistence": True,
            "non_baseline_chroma_dir": Path(chroma_dir).name != "chroma_db",
            "embedding_dimension_locked": config.embedding_dimension == 3072,
            "snapshot_prefix_present": not _is_placeholder(config.snapshot_prefix),
            "chroma_dir_ready": readiness["ok"],
        },
    }


def write_snapshot_publish_plan(plan: dict, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = results_dir / f"corpus_snapshot_publish_plan_{ts}.json"
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render production corpus snapshot publication plan.")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--gcs-bucket", required=True)
    parser.add_argument("--snapshot-prefix", default=DEFAULT_SNAPSHOT_PREFIX)
    parser.add_argument("--embedding-dimension", type=int, default=3072)
    parser.add_argument("--certification-run-id")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    plan = build_snapshot_publish_plan(
        SnapshotPublishConfig(
            chroma_dir=args.chroma_dir,
            gcs_bucket=args.gcs_bucket,
            snapshot_prefix=args.snapshot_prefix,
            embedding_dimension=args.embedding_dimension,
            certification_run_id=args.certification_run_id,
        )
    )

    print("# Publish certified corpus snapshot")
    for key, value in sorted(plan["env_vars"].items()):
        print(f"export {key}={shlex.quote(str(value))}")
    print(_shell_join(plan["command"]))
    if args.write_plan:
        print()
        print(f"# Snapshot publish plan artifact: {write_snapshot_publish_plan(plan, args.results_dir.expanduser())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
