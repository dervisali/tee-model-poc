"""
Phase 3.5 — cross-lingual query expansion için birim testleri.

Canlı Vertex çağrısı yapılmaz: testler MOCK_MODE=true (conftest) altında
çalışır ve translate_query deterministik mock dize döner.
"""

from __future__ import annotations

import pytest


class TestDetectQueryLanguage:
    """Heuristic dil tespiti — Türkçe sinyalleri yakalamalı, aksi halde 'fr'."""

    def test_turkish_unique_chars(self):
        from src.query_translator import detect_query_language
        assert detect_query_language("B2 yazılı üretim grillesinde kohezyon") == "tr"
        assert detect_query_language("ş ğ ı") == "tr"

    def test_turkish_suffix(self):
        from src.query_translator import detect_query_language
        # No Turkish-unique chars, but suffix gives it away.
        assert detect_query_language("ders verilecek") == "tr"
        assert detect_query_language("öğrenciler kompozisyonda hata yapıyor") == "tr"

    def test_french_query(self):
        from src.query_translator import detect_query_language
        assert detect_query_language("où la grille PE B2 définit-elle le critère") == "fr"
        assert detect_query_language("correction de la production écrite") == "fr"

    def test_short_ambiguous_defaults_fr(self):
        from src.query_translator import detect_query_language
        # No Turkish signals → corpus-language default (fr).
        assert detect_query_language("grille") == "fr"
        assert detect_query_language("A1 niveau") == "fr"

    def test_empty(self):
        from src.query_translator import detect_query_language
        # Empty falls back to corpus primary language.
        assert detect_query_language("") in ("fr", "tr")


class TestTranslateQuery:
    """translate_query — MOCK_MODE altında deterministik."""

    def setup_method(self):
        from src.query_translator import translate_query
        translate_query.cache_clear()

    def test_same_language_is_noop(self):
        from src.query_translator import translate_query
        result = translate_query("production écrite cohérence", target_language="fr")
        assert result == "production écrite cohérence"

    def test_tr_to_fr_returns_mock_marker(self):
        from src.query_translator import translate_query
        result = translate_query("B2 yazılı üretim kohezyon", target_language="fr")
        assert result.startswith("[MOCK-FR]")
        assert "B2 yazılı üretim kohezyon" in result

    def test_empty_query_passes_through(self):
        from src.query_translator import translate_query
        assert translate_query("", target_language="fr") == ""

    def test_cache_hit_same_input(self):
        from src.query_translator import translate_query
        r1 = translate_query("kohezyon kriteri", target_language="fr")
        r2 = translate_query("kohezyon kriteri", target_language="fr")
        assert r1 == r2  # deterministic + cached


class TestRetrieveContextCrossLingual:
    """retrieve_context Türkçe sorgu için BM25'e çeviri varyantı geçirmeli."""

    @pytest.mark.requires_chromadb
    def test_tr_query_routes_translation_to_bm25(self, monkeypatch):
        """
        Phase 3.5 contract testi: retrieve_context Türkçe sorgu aldığında
        hybrid_search_children'a bm25_query parametresinin geçtiğini
        doğrula. Gerçek embed/BM25/Chroma çağrısı yok — sahte iç fonksiyonlar.
        """
        from src import retrieval as retrieval_mod
        from src.config import settings

        # Pretend parents.json has one entry; retrieve_context will look up
        # the matched parent_id and format the result.
        monkeypatch.setattr(retrieval_mod, "_load_parents", lambda: {"x_par0": {"text": "FR parent text"}})

        captured = {}

        def fake_hybrid_search_children(query, *, top_k, source_filter, search_mode, alpha, bm25_query=None, **kwargs):
            captured["query"] = query
            captured["bm25_query"] = bm25_query
            return [{
                "id": "x_par0_c0",
                "document": "child text",
                "metadata": {"parent_id": "x_par0", "child_index": 0,
                             "source": "fake", "filename": "fake.pdf"},
                "score": 0.9,
                "dense_distance": 0.1,
                "bm25_score": 1.5,
            }]

        import src.hybrid_search as hs_mod
        monkeypatch.setattr(hs_mod, "hybrid_search_children", fake_hybrid_search_children)

        # Auto-detect must be on, corpus must be fr (default).
        assert settings.QUERY_LANGUAGE_AUTO_DETECT is True
        assert settings.CORPUS_PRIMARY_LANGUAGE == "fr"

        out = retrieval_mod.retrieve_context(
            "B2 yazılı üretim kohezyon kriteri",
            top_k=1,
            search_mode="hybrid",
        )

        # Dense path uses original Turkish query
        assert captured["query"] == "B2 yazılı üretim kohezyon kriteri"
        # BM25 path gets translated variant (MOCK_MODE deterministic mock)
        assert captured["bm25_query"] is not None
        assert captured["bm25_query"].startswith("[MOCK-FR]")
        assert len(out) == 1
