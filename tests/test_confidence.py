"""
Güven skoru rozet rengi ve eşik mantığı testleri.

Skoru üretmek LLM çağrısı gerektirir; burada yalnızca skor → rozet/tavsiye
dönüşümü ve fallback davranışı test edilir.
"""

from __future__ import annotations

import pytest

from src.confidence import _badge_color, _badge_text


class TestRozetRengi:
    """guven_skoru → rozet_renk dönüşümü."""

    @pytest.mark.parametrize("skor,renk", [
        (1.00, "green"),
        (0.85, "green"),
        (0.84, "yellow"),
        (0.60, "yellow"),
        (0.59, "red"),
        (0.00, "red"),
    ])
    def test_skor_rozet_rengi(self, skor: float, renk: str):
        assert _badge_color(skor) == renk


class TestRozetMetni:
    """Skor → Türkçe rozet metni."""

    def test_yuksek_guven(self):
        assert _badge_text(0.95) == "Yüksek Güven"

    def test_orta_guven(self):
        assert "Orta" in _badge_text(0.70)

    def test_dusuk_guven(self):
        assert "Düşük" in _badge_text(0.30)


class TestFallback:
    """ENABLE_CONFIDENCE_SCORING=false veya kaynak chunk yok → güvenli fallback."""

    def test_devre_disi(self, monkeypatch):
        """ENABLE_CONFIDENCE_SCORING kapalıysa skor 0 fakat hata=False döner."""
        monkeypatch.setenv("ENABLE_CONFIDENCE_SCORING", "false")
        # settings reload
        import importlib
        from src import config as cfg
        importlib.reload(cfg)
        from src import confidence
        importlib.reload(confidence)
        sonuc = confidence.score_generated_content(
            generated_json={"steps": []},
            source_chunks=[{"parent_id": "x", "parent_text": "y"}],
            content_type="process_map",
        )
        assert sonuc.get("atlandi") is True
        assert sonuc["uzman_onay_tavsiyesi"] in ("hizli_inceleme", "detayli_inceleme", "reddet")

    def test_kaynak_chunk_yok(self, monkeypatch):
        """source_chunks=[] verildiğinde fallback dönmelidir."""
        monkeypatch.setenv("ENABLE_CONFIDENCE_SCORING", "true")
        import importlib
        from src import config as cfg
        importlib.reload(cfg)
        from src import confidence
        importlib.reload(confidence)
        sonuc = confidence.score_generated_content(
            generated_json={"steps": []},
            source_chunks=[],
            content_type="process_map",
        )
        assert sonuc["guven_skoru"] == 0.0
        assert sonuc["uzman_onay_tavsiyesi"] == "detayli_inceleme"
