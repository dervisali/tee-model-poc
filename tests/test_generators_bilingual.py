"""
Phase 8 — bilingual üretici testleri.

MOCK_MODE altında her üreticinin TR ve FR varyantları farklı fixture döner
ve şema uyumlu kalır.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_translator_cache():
    """Generator chain'i query_translator'a dokunabilir; her test temiz cache."""
    from src.query_translator import translate_query
    translate_query.cache_clear()
    yield
    translate_query.cache_clear()


class TestProcessMapBilingual:
    def test_tr_fixture_returns_steps(self):
        from src.generators import generate_process_map
        out = generate_process_map(language="tr")
        assert "steps" in out
        assert len(out["steps"]) >= 1
        assert "_confidence" in out
        # TR fixture has Turkish content
        assert any("Aday" in s["baslik"] or "kadro" in s["baslik"].lower() for s in out["steps"])

    def test_fr_fixture_returns_french_content(self):
        from src.generators import generate_process_map
        out = generate_process_map(language="fr")
        assert "steps" in out
        assert "_confidence" in out
        # FR fixture has French content
        assert any("é" in s["baslik"] or "Vérifier" in s["baslik"] for s in out["steps"])

    def test_default_language_reads_settings(self):
        from src.config import settings
        from src.generators import generate_process_map
        out = generate_process_map()  # no explicit language
        assert "steps" in out
        # Should match the configured default (TR by default)
        assert settings.OUTPUT_LANGUAGE in ("tr", "fr")


class TestErrorCardsBilingual:
    def test_tr_and_fr_both_yield_cards(self):
        from src.generators import generate_error_cards
        tr = generate_error_cards(language="tr")
        fr = generate_error_cards(language="fr")
        assert "hata_kartlari" in tr and len(tr["hata_kartlari"]) >= 1
        assert "hata_kartlari" in fr and len(fr["hata_kartlari"]) >= 1
        # FR cards should look French
        assert any("é" in c["hata"] or "correcteur" in c["hata"].lower() for c in fr["hata_kartlari"])


class TestGlossaryBilingual:
    def test_tr_and_fr_terms(self):
        from src.generators import generate_glossary
        tr = generate_glossary(language="tr")
        fr = generate_glossary(language="fr")
        assert "terimler" in tr
        assert "terimler" in fr
        # FR glossary has DELF technical terms
        assert any("Grille" in t["terim"] or "Descripteur" in t["terim"] for t in fr["terimler"])


class TestSimulationBilingual:
    def test_tr_and_fr_simulation(self):
        from src.generators import generate_simulation_scenario
        tr = generate_simulation_scenario(language="tr")
        fr = generate_simulation_scenario(language="fr")
        # Both have schema-required fields
        for sim in (tr, fr):
            assert "senaryo_basligi" in sim
            assert "secenekler" in sim
            assert any(opt["dogru_mu"] for opt in sim["secenekler"])
        # FR has French scenario text
        assert any(c in fr["senaryo_basligi"] for c in "éèàç")


class TestPromptRegistry:
    def test_each_generator_has_tr_and_fr_prompts(self):
        from src.generators import _GENERATOR_PROMPTS
        expected_keys = {"process_map", "error_cards", "glossary", "simulation"}
        assert set(_GENERATOR_PROMPTS.keys()) == expected_keys
        for key, by_lang in _GENERATOR_PROMPTS.items():
            assert set(by_lang.keys()) == {"tr", "fr"}, f"{key} missing a language"
            for lang in ("tr", "fr"):
                assert "query" in by_lang[lang]
                assert "instruction" in by_lang[lang]
                assert by_lang[lang]["query"].strip()
                assert by_lang[lang]["instruction"].strip()
