"""Production Dockerfile hardening tests."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _dockerfile_text() -> str:
    return (REPO / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_runs_as_non_root_user():
    text = _dockerfile_text()

    assert "useradd --system" in text
    assert "USER app" in text
    assert "ENTRYPOINT" in text
    assert text.index("USER app") < text.index("ENTRYPOINT")


def test_dockerfile_prepares_writable_runtime_paths():
    text = _dockerfile_text()

    assert "HF_HOME=/tmp/hf-cache" in text
    assert "XDG_CACHE_HOME=/tmp/.cache" in text
    assert "mkdir -p /tmp/chroma_db /tmp/hf-cache /tmp/.cache /app/logs" in text
    assert "chown -R app:app /app /home/app /tmp/chroma_db /tmp/hf-cache /tmp/.cache" in text
