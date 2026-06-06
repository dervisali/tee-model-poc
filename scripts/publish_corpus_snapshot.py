"""
Publish an already-certified local Chroma corpus snapshot to GCS.

Unlike ``python -m src.ingestion``, this command does not rebuild or mutate the
corpus. It verifies the existing ``CHROMA_DIR`` and then publishes exactly that
directory via ``src.persistence.publish_snapshot``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
BASELINE_CHROMA_DIR = REPO / "chroma_db"


def _is_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    text = value.strip()
    if not text:
        return True
    upper = text.upper()
    return any(
        token in upper
        for token in ("PLACEHOLDER", "YOUR_", "REAL_", "EXAMPLE.", "DELF_CORPUS_BUCKET", "<", ">")
    )


def _reject_placeholder(name: str, value: str | None) -> None:
    if _is_placeholder(value):
        raise SystemExit(f"{name} must be a real non-placeholder value")


def _reject_gs_uri_bucket(bucket: str) -> None:
    if bucket.strip().startswith("gs://"):
        raise SystemExit("--gcs-bucket/GCS_BUCKET must be a bucket name, not a gs:// URI")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_publish_chroma_dir(chroma_dir: Path) -> Path:
    resolved = chroma_dir.expanduser().resolve(strict=False)
    repo = REPO.resolve(strict=False)
    baseline = BASELINE_CHROMA_DIR.resolve(strict=False)
    if resolved == repo:
        raise SystemExit(f"Refusing to publish repository root as CHROMA_DIR: {resolved}")
    if resolved.name == "chroma_db" or resolved == baseline:
        raise SystemExit(f"Refusing to publish baseline CHROMA_DIR as certified snapshot: {resolved}")
    if _is_relative_to(resolved, baseline):
        raise SystemExit(f"Refusing to publish a path inside baseline CHROMA_DIR: {resolved}")
    if _is_relative_to(repo, resolved) or _is_relative_to(baseline, resolved):
        raise SystemExit(
            f"Refusing to publish a parent of the repository or baseline CHROMA_DIR: {resolved}"
        )
    if chroma_dir.is_symlink() or resolved.is_symlink():
        raise SystemExit(f"Refusing to publish symlinked CHROMA_DIR: {chroma_dir}")
    return resolved


def publish_existing_corpus(
    *,
    chroma_dir: Path,
    bucket: str,
    prefix: str,
    snapshot_id: str | None = None,
) -> dict:
    from src.persistence import publish_snapshot
    from src.readiness_check import check_readiness

    _reject_placeholder("--gcs-bucket/GCS_BUCKET", bucket)
    _reject_placeholder("--snapshot-prefix/GCS_SNAPSHOT_PREFIX", prefix)
    _reject_gs_uri_bucket(bucket)
    resolved_chroma_dir = _resolve_publish_chroma_dir(chroma_dir)
    readiness = check_readiness(run_smoke=False, chroma_dir=resolved_chroma_dir)
    if not readiness.ok:
        raise SystemExit(
            "Refusing to publish corpus snapshot; CHROMA_DIR is not ready:\n"
            + "\n".join(readiness.failures)
        )

    snapshot_object = publish_snapshot(
        resolved_chroma_dir,
        bucket=bucket,
        prefix=prefix,
        child_count=readiness.child_count,
        parent_count=readiness.parent_count,
        snapshot_id=snapshot_id,
    )
    return {
        "snapshot_object": snapshot_object,
        "chroma_dir": str(resolved_chroma_dir),
        "gcs_bucket": bucket,
        "snapshot_prefix": prefix,
        "child_count": readiness.child_count,
        "parent_count": readiness.parent_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish certified local corpus snapshot to GCS.")
    parser.add_argument("--chroma-dir", type=Path)
    parser.add_argument("--gcs-bucket")
    parser.add_argument("--snapshot-prefix")
    parser.add_argument("--snapshot-id")
    args = parser.parse_args()

    from src.config import settings

    chroma_dir = args.chroma_dir or settings.CHROMA_DIR
    bucket = args.gcs_bucket or settings.GCS_BUCKET
    prefix = args.snapshot_prefix or settings.GCS_SNAPSHOT_PREFIX
    if not bucket:
        raise SystemExit("--gcs-bucket or GCS_BUCKET is required")
    if not prefix:
        raise SystemExit("--snapshot-prefix or GCS_SNAPSHOT_PREFIX is required")

    result = publish_existing_corpus(
        chroma_dir=chroma_dir,
        bucket=bucket,
        prefix=prefix,
        snapshot_id=args.snapshot_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
