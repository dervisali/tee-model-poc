"""Production Docker build-context hygiene tests."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _dockerignore_patterns() -> set[str]:
    patterns: set[str] = set()
    for raw in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.add(line)
    return patterns


def test_dockerignore_excludes_local_corpus_and_certification_artifacts():
    patterns = _dockerignore_patterns()

    required = {
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

    assert required <= patterns


def test_dockerignore_keeps_runtime_source_available():
    patterns = _dockerignore_patterns()

    assert "src/" not in patterns
    assert "app.py" not in patterns
    assert "entrypoint.sh" not in patterns
    assert "requirements.txt" not in patterns
