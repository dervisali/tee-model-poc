"""Tests for non-mutating corpus snapshot publication wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.publish_corpus_snapshot import BASELINE_CHROMA_DIR, REPO, publish_existing_corpus


def test_publish_existing_corpus_uses_readiness_counts(monkeypatch, tmp_path):
    calls = {}

    def fake_check_readiness(*, run_smoke, chroma_dir):
        calls["run_smoke"] = run_smoke
        calls["readiness_chroma_dir"] = chroma_dir
        return SimpleNamespace(ok=True, failures=[], child_count=42, parent_count=7)

    def fake_publish_snapshot(chroma_dir, *, bucket, prefix, child_count, parent_count, snapshot_id):
        calls.update({
            "chroma_dir": chroma_dir,
            "bucket": bucket,
            "prefix": prefix,
            "child_count": child_count,
            "parent_count": parent_count,
            "snapshot_id": snapshot_id,
        })
        return "tee-corpus/snap/corpus.tar.gz"

    monkeypatch.setattr("src.readiness_check.check_readiness", fake_check_readiness)
    monkeypatch.setattr("src.persistence.publish_snapshot", fake_publish_snapshot)
    chroma_dir = tmp_path / "chroma_db_enriched"

    result = publish_existing_corpus(
        chroma_dir=chroma_dir,
        bucket="bucket",
        prefix="tee-corpus",
        snapshot_id="snap",
    )

    assert calls["run_smoke"] is False
    assert calls["readiness_chroma_dir"] == chroma_dir.resolve(strict=False)
    assert calls["chroma_dir"] == chroma_dir.resolve(strict=False)
    assert calls["child_count"] == 42
    assert calls["parent_count"] == 7
    assert calls["snapshot_id"] == "snap"
    assert result["snapshot_object"] == "tee-corpus/snap/corpus.tar.gz"
    assert result["chroma_dir"] == str(chroma_dir.resolve(strict=False))


def test_publish_existing_corpus_refuses_unready_chroma(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.readiness_check.check_readiness",
        lambda *, run_smoke, chroma_dir: SimpleNamespace(
            ok=False,
            failures=[f"missing parents in {chroma_dir}"],
            child_count=0,
            parent_count=0,
        ),
    )

    with pytest.raises(SystemExit, match="Refusing to publish corpus snapshot"):
        publish_existing_corpus(
            chroma_dir=tmp_path / "chroma_db_enriched",
            bucket="bucket",
            prefix="tee-corpus",
        )


def test_publish_existing_corpus_rejects_gs_uri_bucket(tmp_path):
    with pytest.raises(SystemExit, match="must be a bucket name, not a gs:// URI"):
        publish_existing_corpus(
            chroma_dir=tmp_path / "chroma_db_enriched",
            bucket="gs://bucket",
            prefix="tee-corpus",
        )


@pytest.mark.parametrize(
    ("bucket", "prefix"),
    [
        ("DELF_CORPUS_BUCKET", "tee-corpus"),
        ("bucket", "REAL_SNAPSHOT_PREFIX"),
        ("", "tee-corpus"),
    ],
)
def test_publish_existing_corpus_rejects_placeholder_destination(bucket, prefix, tmp_path):
    with pytest.raises(SystemExit, match="real non-placeholder value"):
        publish_existing_corpus(
            chroma_dir=tmp_path / "chroma_db_enriched",
            bucket=bucket,
            prefix=prefix,
        )


def test_publish_existing_corpus_rejects_baseline_chroma_dir(tmp_path):
    with pytest.raises(SystemExit, match="baseline CHROMA_DIR"):
        publish_existing_corpus(
            chroma_dir=tmp_path / "chroma_db",
            bucket="bucket",
            prefix="tee-corpus",
        )


def test_publish_existing_corpus_rejects_symlinked_chroma_dir(tmp_path):
    target = tmp_path / "real_chroma"
    target.mkdir()
    link = tmp_path / "chroma_db_enriched"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(SystemExit, match="symlinked CHROMA_DIR"):
        publish_existing_corpus(
            chroma_dir=link,
            bucket="bucket",
            prefix="tee-corpus",
        )


def test_publish_existing_corpus_rejects_path_inside_baseline_chroma():
    with pytest.raises(SystemExit, match="inside baseline CHROMA_DIR"):
        publish_existing_corpus(
            chroma_dir=BASELINE_CHROMA_DIR / "nested_publish",
            bucket="bucket",
            prefix="tee-corpus",
        )


def test_publish_existing_corpus_rejects_parent_of_repo():
    with pytest.raises(SystemExit, match="parent of the repository or baseline CHROMA_DIR"):
        publish_existing_corpus(
            chroma_dir=REPO.parent,
            bucket="bucket",
            prefix="tee-corpus",
        )
