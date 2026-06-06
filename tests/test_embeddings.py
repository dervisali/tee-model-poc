"""Tests for embedding batching controls."""

from src import embeddings


def test_embed_passages_uses_configured_batch_size_and_sleep(monkeypatch):
    calls: list[list[str]] = []
    sleeps: list[float] = []

    monkeypatch.setattr(embeddings.settings, "EMBED_BATCH_SIZE", 2)
    monkeypatch.setattr(embeddings.settings, "EMBED_SLEEP_BETWEEN_BATCHES", 0.25)
    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

    def fake_embed(texts, *, task_type):
        calls.append(list(texts))
        assert task_type == "RETRIEVAL_DOCUMENT"
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(embeddings, "_embed", fake_embed)

    out = embeddings.embed_passages(["a", "bb", "ccc", "dddd", "eeeee"])

    assert calls == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert sleeps == [0.25, 0.25]
    assert out == [[1.0], [2.0], [3.0], [4.0], [5.0]]
