"""
Üretici fonksiyonların MOCK_MODE altında çalıştığını ve şema-uyumlu çıktı
döndürdüğünü test eder.

MOCK_MODE=true'da hiçbir Vertex AI çağrısı yapılmaz; sabit fixture döner.
Bu testler 'pip install requirements' yapılmadan da geçer (yalnızca
src.generators ve onun bağımlılıklarını içerir; ağır LLM yolu çalışmaz).
"""

from __future__ import annotations

import os

import pytest


# Tüm generator testleri için MOCK_MODE açık olmalı (conftest da set ediyor).
@pytest.fixture(autouse=True)
def _ensure_mock_mode(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "true")
    # settings cache'lendiği için reload edilmez; ama generators MOCK_MODE'u
    # her çağrıda settings.MOCK_MODE üzerinden okur — env değişikliği etkili.


def _reload_settings():
    """settings singleton'ını yenile (env değişikliği sonrası)."""
    import importlib
    from src import config as cfg_mod
    importlib.reload(cfg_mod)


class TestProcessMapMock:
    """generate_process_map MOCK fixture'i şema uyumlu döndürmelidir."""

    def test_mock_donus_steps_listesi(self):
        _reload_settings()
        from src.generators import generate_process_map
        sonuc = generate_process_map()
        assert "steps" in sonuc
        assert isinstance(sonuc["steps"], list)
        assert len(sonuc["steps"]) >= 1

    def test_mock_step_alanlari_mevcut(self):
        _reload_settings()
        from src.generators import generate_process_map
        sonuc = generate_process_map()
        ilk = sonuc["steps"][0]
        for alan in ("adim_no", "baslik", "giris", "cikis", "karar_noktasi", "risk", "kontrol"):
            assert alan in ilk

    def test_mock_confidence_eklenir(self):
        """Phase 2.2 — MOCK fixture'ler de _confidence içermeli."""
        _reload_settings()
        from src.generators import generate_process_map
        sonuc = generate_process_map()
        assert "_confidence" in sonuc
        score = sonuc["_confidence"]
        assert 0.0 <= score["guven_skoru"] <= 1.0
        assert score["uzman_onay_tavsiyesi"] in ("hizli_inceleme", "detayli_inceleme", "reddet")


class TestErrorCardsMock:
    def test_mock_donus_kart_listesi(self):
        _reload_settings()
        from src.generators import generate_error_cards
        sonuc = generate_error_cards()
        assert "hata_kartlari" in sonuc
        assert isinstance(sonuc["hata_kartlari"], list)

    def test_kart_alanlari_ascii(self):
        """Pydantic field rename: kok_neden ve tespit_yontemi (ASCII)."""
        _reload_settings()
        from src.generators import generate_error_cards
        sonuc = generate_error_cards()
        ilk = sonuc["hata_kartlari"][0]
        assert "kok_neden" in ilk
        assert "tespit_yontemi" in ilk


class TestGlossaryMock:
    def test_mock_donus_terim_listesi(self):
        _reload_settings()
        from src.generators import generate_glossary
        sonuc = generate_glossary()
        assert "terimler" in sonuc
        assert len(sonuc["terimler"]) >= 1
        assert "tanim" in sonuc["terimler"][0]


class TestSimulationMock:
    def test_mock_donus_secenek_listesi(self):
        _reload_settings()
        from src.generators import generate_simulation_scenario
        sonuc = generate_simulation_scenario()
        assert "secenekler" in sonuc
        assert len(sonuc["secenekler"]) == 3

    def test_dogru_secenek_var(self):
        """Tam olarak bir seçenek dogru_mu=True olmalıdır."""
        _reload_settings()
        from src.generators import generate_simulation_scenario
        sonuc = generate_simulation_scenario()
        dogru = [s for s in sonuc["secenekler"] if s["dogru_mu"]]
        assert len(dogru) == 1
