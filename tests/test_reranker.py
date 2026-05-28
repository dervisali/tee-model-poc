"""
LLM reranker birim testleri.

src.llm.generate monkeypatch edilir — gerçek LLM çağrısı yapılmaz. Passthrough
koşulları, yeniden sıralama ve hata dayanıklılığı (passthrough) doğrulanır.
"""

from __future__ import annotations

import src.reranker as reranker_mod
from src.reranker import rerank


def _candidates(n: int) -> list[dict]:
    return [{"parent_text": f"doc {i}", "parent_id": str(i)} for i in range(n)]


def _ids(chunks: list[dict]) -> list[str]:
    return [c["parent_id"] for c in chunks]


class TestPassthrough:
    def test_bayrak_kapali_passthrough(self, monkeypatch):
        monkeypatch.setattr(reranker_mod.settings, "ENABLE_RERANKING", False)
        cands = _candidates(8)
        out = rerank("q", cands, top_n=3)
        assert _ids(out) == ["0", "1", "2"]

    def test_aday_sayisi_top_n_alti_passthrough(self, monkeypatch):
        monkeypatch.setattr(reranker_mod.settings, "ENABLE_RERANKING", True)
        # 3 aday, top_n=5 → sıralamanın anlamı yok, olduğu gibi döner.
        cands = _candidates(3)
        out = rerank("q", cands, top_n=5)
        assert _ids(out) == ["0", "1", "2"]

    def test_bos_aday(self, monkeypatch):
        monkeypatch.setattr(reranker_mod.settings, "ENABLE_RERANKING", True)
        assert rerank("q", [], top_n=5) == []


class TestReorder:
    def test_yargic_skoruna_gore_yeniden_siralar(self, monkeypatch):
        monkeypatch.setattr(reranker_mod.settings, "ENABLE_RERANKING", True)

        # Yargıç: 5. adayı en yüksek, 1. adayı en düşük puanlasın.
        def fake_generate(*args, **kwargs):
            return (
                '{"scores": ['
                '{"index": 1, "score": 1.0},'
                '{"index": 2, "score": 2.0},'
                '{"index": 3, "score": 3.0},'
                '{"index": 4, "score": 4.0},'
                '{"index": 5, "score": 9.0}'
                ']}'
            )

        monkeypatch.setattr("src.llm.generate", fake_generate)
        cands = _candidates(5)
        out = rerank("q", cands, top_n=3)
        # En yüksek skor index 5 (parent_id "4"), sonra 4 ("3"), sonra 3 ("2").
        assert _ids(out) == ["4", "3", "2"]

    def test_puanlanmayan_aday_sona_duser_ama_elenmez(self, monkeypatch):
        monkeypatch.setattr(reranker_mod.settings, "ENABLE_RERANKING", True)

        # Yalnızca tek aday puanlanır; geri kalanlar -1 ile sona düşer.
        def fake_generate(*args, **kwargs):
            return '{"scores": [{"index": 4, "score": 8.0}]}'

        monkeypatch.setattr("src.llm.generate", fake_generate)
        cands = _candidates(5)
        out = rerank("q", cands, top_n=3)
        assert out[0]["parent_id"] == "3"  # index 4 → parent_id "3"
        assert len(out) == 3


class TestResilience:
    def test_llm_hatasi_passthrough(self, monkeypatch):
        monkeypatch.setattr(reranker_mod.settings, "ENABLE_RERANKING", True)

        def boom(*args, **kwargs):
            raise RuntimeError("vertex down")

        monkeypatch.setattr("src.llm.generate", boom)
        cands = _candidates(8)
        out = rerank("q", cands, top_n=3)
        assert _ids(out) == ["0", "1", "2"]  # füzyon sırası korunur

    def test_bozuk_json_passthrough(self, monkeypatch):
        monkeypatch.setattr(reranker_mod.settings, "ENABLE_RERANKING", True)
        monkeypatch.setattr("src.llm.generate", lambda *a, **k: "not json at all")
        cands = _candidates(8)
        out = rerank("q", cands, top_n=3)
        assert _ids(out) == ["0", "1", "2"]
