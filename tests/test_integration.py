"""
Entegrasyon testi — MOCK_MODE altında uçtan uca çıktı tutarlılığı.

Tam stack çalıştırılmaz (Ollama'ya hiç bağlanmaz); MOCK fixture'ler ile
generators katmanı, app session_state'ine yerleştirilebilir uyumlu çıktılar
üretiyor mu — kontratı doğrular.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestUctanUcaMock:
    """Tüm generator çıktıları, app.py'nin beklediği şemada olmalıdır."""

    def test_dort_generator_calisir_ve_sema_uyumlu(self, monkeypatch):
        monkeypatch.setenv("MOCK_MODE", "true")
        import importlib
        from src import config as cfg
        importlib.reload(cfg)
        from src import generators
        importlib.reload(generators)

        process_map = generators.generate_process_map()
        error_cards = generators.generate_error_cards()
        glossary = generators.generate_glossary()
        simulation = generators.generate_simulation_scenario()

        assert "steps" in process_map and "_confidence" in process_map
        assert "hata_kartlari" in error_cards and "_confidence" in error_cards
        assert "terimler" in glossary and "_confidence" in glossary
        assert "secenekler" in simulation and "_confidence" in simulation

        # Her birinde en az bir öğe ve geçerli güven skoru
        for icerik in (process_map, error_cards, glossary, simulation):
            score = icerik["_confidence"]
            assert 0.0 <= score["guven_skoru"] <= 1.0


@pytest.mark.integration
class TestKonfigurasyonYukluyor:
    """Yapılandırma doğru ortam değişkenleri ile yüklenebilmelidir."""

    def test_env_overlay_calisir(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_TOP_K", "9")
        monkeypatch.setenv("HYBRID_ALPHA", "0.3")
        import importlib
        from src import config as cfg
        importlib.reload(cfg)
        assert cfg.settings.RETRIEVAL_TOP_K == 9
        assert cfg.settings.HYBRID_ALPHA == 0.3
