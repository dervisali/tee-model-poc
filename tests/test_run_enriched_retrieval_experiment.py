import json

from scripts.run_enriched_retrieval_experiment import (
    BASELINE_CHROMA_DIR,
    REPO,
    _baseline_integrity,
    _directory_fingerprint,
    _experiment_db_readiness,
    _prepare_reingest_target,
    _refuse_unsafe_target,
    _write_experiment_summary,
)


def test_experiment_db_readiness_reports_missing_directory(tmp_path):
    readiness = _experiment_db_readiness(tmp_path / "missing")

    assert readiness["ok"] is False
    assert any("CHROMA_DIR does not exist" in failure for failure in readiness["failures"])
    assert any("parents.json missing" in failure for failure in readiness["failures"])
    assert any("bm25_index.pkl missing" in failure for failure in readiness["failures"])


def test_experiment_db_readiness_rejects_partial_directory(tmp_path):
    chroma_dir = tmp_path / "chroma_db_enriched"
    chroma_dir.mkdir()
    (chroma_dir / "parents.json").write_text(json.dumps({"p1": {"text": "ok"}}), encoding="utf-8")

    readiness = _experiment_db_readiness(chroma_dir)

    assert readiness["ok"] is False
    assert readiness["parent_count"] == 1
    assert any("bm25_index.pkl missing" in failure for failure in readiness["failures"])
    assert any("tee_children collection" in failure for failure in readiness["failures"])


def test_write_experiment_summary_uses_custom_results_dir(tmp_path):
    path = _write_experiment_summary({"ok": True}, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == tmp_path
    assert path.name.startswith("enriched_experiment_")
    assert data["ok"] is True


def test_directory_fingerprint_detects_changes(tmp_path):
    corpus = tmp_path / "chroma_db"
    corpus.mkdir()
    (corpus / "parents.json").write_text("{}", encoding="utf-8")

    before = _directory_fingerprint(corpus)
    (corpus / "parents.json").write_text('{"changed": true}', encoding="utf-8")
    after = _directory_fingerprint(corpus)

    assert before["present"] is True
    assert before["sha256"] != after["sha256"]
    assert _baseline_integrity(before, after)["ok"] is False


def test_directory_fingerprint_absent_directory_is_stable(tmp_path):
    missing = tmp_path / "missing"
    before = _directory_fingerprint(missing)
    after = _directory_fingerprint(missing)

    assert before["present"] is False
    assert _baseline_integrity(before, after)["ok"] is True


def test_refuse_unsafe_target_requires_force_for_existing_experiment(tmp_path):
    chroma_dir = tmp_path / "chroma_db_enriched"
    chroma_dir.mkdir()

    try:
        _refuse_unsafe_target(chroma_dir, force=False)
    except SystemExit as exc:
        assert "Use --force" in str(exc)
    else:
        raise AssertionError("existing experiment directory should require --force")


def test_prepare_reingest_target_clears_existing_experiment_directory(tmp_path):
    chroma_dir = tmp_path / "chroma_db_enriched"
    chroma_dir.mkdir()
    stale = chroma_dir / "stale.sqlite"
    stale.write_text("old data", encoding="utf-8")

    result = _prepare_reingest_target(chroma_dir, force=True)

    assert result["cleared_existing_directory"] is True
    assert result["chroma_dir"] == str(chroma_dir.resolve())
    assert not chroma_dir.exists()
    assert not stale.exists()


def test_prepare_reingest_target_rejects_symlink(tmp_path):
    target = tmp_path / "real_chroma"
    target.mkdir()
    link = tmp_path / "linked_chroma"
    link.symlink_to(target, target_is_directory=True)

    try:
        _prepare_reingest_target(link, force=True)
    except SystemExit as exc:
        assert "symlinked experiment target" in str(exc)
    else:
        raise AssertionError("symlinked experiment target should be refused")


def test_prepare_reingest_target_rejects_path_inside_baseline_chroma():
    nested = BASELINE_CHROMA_DIR / "nested_experiment"

    try:
        _prepare_reingest_target(nested, force=True)
    except SystemExit as exc:
        assert "inside baseline CHROMA_DIR" in str(exc)
    else:
        raise AssertionError("nested baseline experiment target should be refused")


def test_prepare_reingest_target_rejects_parent_of_repo():
    parent = REPO.parent

    try:
        _prepare_reingest_target(parent, force=True)
    except SystemExit as exc:
        assert "parent of the repository or baseline DB" in str(exc)
    else:
        raise AssertionError("parent of repository should be refused")
