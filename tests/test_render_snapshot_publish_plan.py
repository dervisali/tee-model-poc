import json

import pytest

from scripts.render_snapshot_publish_plan import (
    SnapshotPublishConfig,
    build_snapshot_publish_plan,
    write_snapshot_publish_plan,
)


def _make_ready_chroma_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "parents.json").write_text('{"p1": {"text": "ok"}}', encoding="utf-8")
    (path / "bm25_index.pkl").write_bytes(b"bm25")


def test_build_snapshot_publish_plan_records_certified_source_and_destination(tmp_path):
    chroma_dir = tmp_path / "chroma_db_enriched"
    _make_ready_chroma_dir(chroma_dir)
    plan = build_snapshot_publish_plan(
        SnapshotPublishConfig(
            chroma_dir=str(chroma_dir),
            gcs_bucket="delf-corpus-prod",
            certification_run_id="run-1",
        )
    )

    assert plan["metadata"]["chroma_dir"] == str(chroma_dir.resolve())
    assert plan["metadata"]["gcs_bucket"] == "delf-corpus-prod"
    assert plan["metadata"]["snapshot_prefix"] == "tee-corpus"
    assert plan["metadata"]["certification_run_id"] == "run-1"
    assert plan["command"] == ["python", "-m", "scripts.publish_corpus_snapshot"]
    assert plan["env_vars"]["PERSISTENCE_BACKEND"] == "gcs"
    assert plan["env_vars"]["GCS_BUCKET"] == "delf-corpus-prod"
    assert plan["env_vars"]["GCS_SNAPSHOT_PREFIX"] == "tee-corpus"
    assert plan["env_vars"]["CHROMA_DIR"] == str(chroma_dir.resolve())
    assert plan["env_vars"]["EMBEDDING_DIMENSION"] == "3072"
    assert plan["safety_invariants"]["non_baseline_chroma_dir"] is True
    assert plan["safety_invariants"]["chroma_dir_ready"] is True
    assert plan["chroma_dir_readiness"]["ok"] is True

    out = write_snapshot_publish_plan(plan, tmp_path)
    assert out.name.startswith("corpus_snapshot_publish_plan_")
    assert json.loads(out.read_text(encoding="utf-8"))["metadata"]["certification_run_id"] == "run-1"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gcs_bucket": "DELF_CORPUS_BUCKET"}, "--gcs-bucket must be a real non-placeholder value"),
        ({"gcs_bucket": "gs://delf-corpus-prod"}, "--gcs-bucket must be a bucket name, not a gs:// URI"),
        ({"snapshot_prefix": "REAL_SNAPSHOT_PREFIX"}, "--snapshot-prefix must be a real non-placeholder value"),
        ({"chroma_dir": "/tmp/chroma_db"}, "--chroma-dir must not be the baseline chroma_db"),
        ({"embedding_dimension": 1024}, "EMBEDDING_DIMENSION must remain 3072"),
    ],
)
def test_build_snapshot_publish_plan_rejects_unsafe_inputs(tmp_path, kwargs, message):
    chroma_dir = tmp_path / "chroma_db_enriched"
    _make_ready_chroma_dir(chroma_dir)
    params = {
        "chroma_dir": str(chroma_dir),
        "gcs_bucket": "delf-corpus-prod",
    }
    params.update(kwargs)

    with pytest.raises(SystemExit, match=message):
        build_snapshot_publish_plan(SnapshotPublishConfig(**params))


def test_build_snapshot_publish_plan_rejects_incomplete_chroma_dir(tmp_path):
    chroma_dir = tmp_path / "chroma_db_enriched"
    chroma_dir.mkdir()

    with pytest.raises(SystemExit, match="Cannot render snapshot publish plan for incomplete CHROMA_DIR"):
        build_snapshot_publish_plan(
            SnapshotPublishConfig(
                chroma_dir=str(chroma_dir),
                gcs_bucket="delf-corpus-prod",
            )
        )
