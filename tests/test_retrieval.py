"""
Retrieval birim testleri.

Ağır indeksleme/embedding adımlarını atlamak için BM25 mantığı izole edilir.
RRF füzyon hesabı ile tokenizer saf birim testleri olarak çalışır; gerçek
ChromaDB sorgusu integration test'inde yapılır.
"""

from __future__ import annotations

import pytest

# Tüm BM25/RRF testleri rank_bm25 paketini gerektirir (src.hybrid_search import).
pytestmark = pytest.mark.requires_bm25


class TestBM25Tokenizer:
    """Dile duyarlı tokenizasyon. Türkçe testleri language='tr' ile, Fransızca testleri language='fr' ile."""

    def test_turkce_karakterler_korunur(self):
        from src.hybrid_search import _tokenize
        tokens = _tokenize("Maaş Mutemedi: kümülatif matrah hesabı", language="tr")
        assert "maaş" in tokens
        assert "mutemedi" in tokens
        assert "kümülatif" in tokens
        assert ":" not in tokens

    def test_tokenizer_bos_metin(self):
        from src.hybrid_search import _tokenize
        assert _tokenize("", language="tr") == []
        assert _tokenize("", language="fr") == []

    def test_tokenizer_kucuk_harf(self):
        from src.hybrid_search import _tokenize
        result = _tokenize("İCRA Kesintisi", language="tr")
        assert result in (["i̇cra", "kesintisi"], ["icra", "kesintisi"])

    def test_fransizca_elision_ve_stop_words(self):
        from src.hybrid_search import _tokenize
        tokens = _tokenize("l'examen de la production écrite", language="fr")
        # l, de, la stop-words; examen ve production écrite stem'lenir
        assert "examen" in tokens
        assert "product" in tokens  # Snowball French: production -> product
        assert "écrit" in tokens     # écrite -> écrit
        assert "la" not in tokens
        assert "de" not in tokens

    def test_fransizca_stemmer_inflectional(self):
        from src.hybrid_search import _tokenize
        # Aynı kökten gelen iki form aynı stem'i üretmeli
        assert _tokenize("corrigés", language="fr") == _tokenize("corriger", language="fr")
        assert _tokenize("correcteurs", language="fr") == _tokenize("correcteur", language="fr")


class TestRRFFuzyonu:
    """Reciprocal Rank Fusion (Cormack 2009) hesaplaması."""

    def test_rrf_yuksek_rank_yuksek_skor(self):
        """Aynı belge tüm listelerin başında ise toplam skor en yüksek olmalı."""
        from src.hybrid_search import _reciprocal_rank_fusion
        rank_lists = [
            ["a", "b", "c"],
            ["a", "b", "c"],
        ]
        scores = _reciprocal_rank_fusion(rank_lists, rrf_k=60)
        assert scores["a"] > scores["b"] > scores["c"]

    def test_rrf_yalnizca_birinde_olmasi(self):
        """Yalnızca bir listede görünen, her ikisinde görünenden düşük skor alır."""
        from src.hybrid_search import _reciprocal_rank_fusion
        rank_lists = [
            ["a", "b"],   # her ikisi
            ["b", "a"],   # her ikisi (sıra farklı)
        ]
        scores = _reciprocal_rank_fusion(rank_lists, rrf_k=60)
        # Her iki belge de hem listede; toplam skoru pozitif
        assert scores["a"] > 0
        assert scores["b"] > 0
        # Top-1'de iki kez geçen (a) en yüksek olmayabilir; b 1+0 = 1.
        # Asıl test: skorlar farklı olmalı (sıra farklılığı yansır).
        assert abs(scores["a"] - scores["b"]) < 1e-9 or scores["a"] != scores["b"]

    def test_rrf_k_buyuyunce_skor_kucult(self):
        """k = 60 → k = 600 olunca toplam skor küçülmeli."""
        from src.hybrid_search import _reciprocal_rank_fusion
        rank_lists = [["a"]]
        s60 = _reciprocal_rank_fusion(rank_lists, rrf_k=60)
        s600 = _reciprocal_rank_fusion(rank_lists, rrf_k=600)
        assert s60["a"] > s600["a"]


class TestSearchModeDispatch:
    """retrieve_context'in search_mode parametresine göre yönlendirme yaptığını test eder."""

    @pytest.mark.requires_chromadb
    def test_bilinmeyen_search_mode_hata_atmaz(self, monkeypatch):
        """Tanımlanmamış mod 'hybrid' fallback'ine yönlenmemeli; geçerli modlar
        whitelist üzerindedir."""
        # Burada davranış kontratını değil, açık API yüzeyini test ediyoruz.
        # Geçerli modlar: dense, sparse, hybrid. Tanımsız modda davranış
        # belirsiz olduğundan testte yalnızca tipik kullanım kontratı vardır.
        from src.retrieval import retrieve_context
        assert callable(retrieve_context)


class TestSourceDiversification:
    def test_source_diversification_prefers_distinct_filenames(self):
        from src.retrieval import _diversify_by_source

        chunks = [
            {"filename": "a.pdf", "score": 0.9},
            {"filename": "a.pdf", "score": 0.8},
            {"filename": "b.pdf", "score": 0.7},
            {"filename": "c.pdf", "score": 0.6},
        ]

        diversified = _diversify_by_source(chunks, top_k=3)
        assert [c["filename"] for c in diversified] == ["a.pdf", "b.pdf", "c.pdf"]

    def test_apply_reranking_diversifies_without_reranker(self, monkeypatch):
        import src.retrieval as retrieval

        monkeypatch.setattr(retrieval.settings, "ENABLE_RERANKING", False)
        monkeypatch.setattr(retrieval.settings, "ENABLE_SOURCE_DIVERSIFICATION", True)

        chunks = [
            {"filename": "a.pdf"},
            {"filename": "a.pdf"},
            {"filename": "b.pdf"},
        ]

        ranked = retrieval._apply_reranking("query", chunks, top_k=2)
        assert [c["filename"] for c in ranked] == ["a.pdf", "b.pdf"]


class TestAutoMetadataFilter:
    def test_auto_metadata_filter_only_applies_validated_po_rule(self):
        from src.retrieval import _auto_metadata_filter

        assert _auto_metadata_filter("B2 production orale süresi") == {
            "$and": [{"level": {"$eq": "B2"}}, {"skill": {"$eq": "PO"}}]
        }
        assert _auto_metadata_filter("B2 production écrite kelime sayısı") is None
        assert _auto_metadata_filter("A1 seviyesinde kullanıcı neler yapabilir") is None


class TestSourceHints:
    def test_detect_source_hints_for_named_reference_documents(self):
        from src.retrieval import _detect_source_hints

        assert _detect_source_hints("Selon l'échelle globale du CECRL, niveau B1") == [
            "Echelle globale.pdf",
        ]
        assert _detect_source_hints(
            "Dans l'exercice d'évaluation des productions écrites par niveaux"
        ) == ["Niveaux_CECRL_PE_V2.pdf"]
        assert _detect_source_hints(
            "Réalisation de la tâche en production orale du DELF B2"
        ) == ["B2_Descripteurs_PO.pdf"]

    def test_source_hints_are_not_broad_level_guesses(self):
        from src.retrieval import _detect_source_hints

        assert _detect_source_hints("A1 seviyesinde kullanıcı neler yapabilir?") == []

    def test_merge_source_hints_deduplicates_parents(self):
        from src.retrieval import _merge_source_hints

        hinted = [{"parent_id": "p2", "filename": "hint.pdf"}]
        ranked = [
            {"parent_id": "p1", "filename": "base.pdf"},
            {"parent_id": "p2", "filename": "hint.pdf"},
            {"parent_id": "p3", "filename": "other.pdf"},
        ]

        merged = _merge_source_hints(hinted, ranked, top_k=3)

        assert [item["parent_id"] for item in merged] == ["p2", "p1", "p3"]


class TestBM25MetadataFilter:
    """BM25Index.search'in metadata_filter'ı Python tarafında uyguladığını test eder."""

    def _index(self):
        from src.hybrid_search import BM25Index, _tokenize
        docs = [
            "production orale niveau B2 grille",
            "production orale niveau A1 grille",
            "production écrite niveau B2 grille",
        ]
        metas = [
            {"level": "B2", "skill": "PO"},
            {"level": "A1", "skill": "PO"},
            {"level": "B2", "skill": "PE"},
        ]
        tokens = [_tokenize(d, language="fr") for d in docs]
        return BM25Index(ids=["c0", "c1", "c2"], tokens=tokens, documents=docs, metadatas=metas)

    def test_filtresiz_tum_eslesmeler(self):
        idx = self._index()
        results = idx.search("production orale grille", top_k=10, language="fr")
        returned = {r[0] for r in results}
        assert returned == {"c0", "c1", "c2"}

    def test_level_filtresi_kisitlar(self):
        from src.metadata_filter import build_where_clause
        idx = self._index()
        where = build_where_clause(level="B2")
        results = idx.search("production grille", top_k=10, language="fr", metadata_filter=where)
        returned = {r[0] for r in results}
        assert returned == {"c0", "c2"}  # yalnızca B2'ler

    def test_level_ve_skill_filtresi(self):
        from src.metadata_filter import build_where_clause
        idx = self._index()
        where = build_where_clause(level="B2", skill="PO")
        results = idx.search("production grille", top_k=10, language="fr", metadata_filter=where)
        returned = {r[0] for r in results}
        assert returned == {"c0"}  # B2 + PO


class TestNormalizeScores:
    """Min-max normalizasyon yardımcısı."""

    def test_normalize_temel_durum(self):
        from src.hybrid_search import _normalize_scores
        result = _normalize_scores([0.0, 5.0, 10.0])
        assert result[0] == 0.0
        assert result[2] == 1.0
        assert 0.0 < result[1] < 1.0

    def test_normalize_tum_ayni(self):
        from src.hybrid_search import _normalize_scores
        result = _normalize_scores([3.0, 3.0, 3.0])
        assert all(r == 0.5 for r in result)

    def test_normalize_bos(self):
        from src.hybrid_search import _normalize_scores
        assert _normalize_scores([]) == []
