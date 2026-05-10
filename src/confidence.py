"""
Üretilen içerik için ikinci geçiş güven skorlaması (Phase 2.2).

Mission spec'e uygun olarak: her üretici fonksiyon (process_map / error_cards
/ glossary / simulation) çıktısı, kaynak chunk'lara karşı bir ikinci LLM
çağrısı ile değerlendirilir. Üretilen JSON'daki her iddia, sağlanan parent
metinlere karşı doğrulanır; halüsinasyon riskini sayısal olarak görünür kılar.

Çıktı sözleşmesi:
{
  "guven_skoru": 0.0-1.0,
  "desteklenen_iddialar": int,
  "desteklenmeyen_iddialar": int,
  "desteklenmeyen_liste": [str, ...],
  "uzman_onay_tavsiyesi": "hizli_inceleme" | "detayli_inceleme" | "reddet"
}

Uzman paneli rozet renkleri (app.py tarafından kullanılır):
  >= 0.85 → yeşil "Yüksek Güven"
  0.60-0.85 → sarı "Orta Güven — İncele"
  < 0.60 → kırmızı "Düşük Güven — Dikkatle İncele"
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.config import settings
from src.llm import generate as llm_generate


logger = logging.getLogger(__name__)


class ConfidenceScore(BaseModel):
    """Tek bir üretici çıktısı için güven skoru."""

    guven_skoru: float = Field(ge=0.0, le=1.0)
    desteklenen_iddialar: int = Field(ge=0)
    desteklenmeyen_iddialar: int = Field(ge=0)
    desteklenmeyen_liste: list[str] = Field(default_factory=list)
    uzman_onay_tavsiyesi: Literal["hizli_inceleme", "detayli_inceleme", "reddet"]


_PROMPT = """\
Sen bir kamu kurumu eğitim materyali kalite denetçisisin.
Aşağıdaki eğitim içeriği, verilen kaynak belgelerden üretilmiştir.
Her iddiayı kaynak belgelerle karşılaştır ve şunları belirle:
- Kaç iddia kaynak belgelerle DOĞRUDAN destekleniyor?
- Kaç iddia kaynak belgelerde YOK veya açıkça FARKLILAŞIYOR?
- Genel güven skorunu 0.0-1.0 aralığında ver.

İÇERİK TÜRÜ: {content_type}

ÜRETİLEN İÇERİK:
{generated_json}

KAYNAK BELGELER:
{source_chunks}

YALNIZCA aşağıdaki şemada JSON döndür; başka hiçbir şey yazma:
{{
  "guven_skoru": <0.0-1.0 arası ondalık>,
  "desteklenen_iddialar": <tam sayı>,
  "desteklenmeyen_iddialar": <tam sayı>,
  "desteklenmeyen_liste": [<en kritik 3 desteklenmeyen iddia, kısa Türkçe>],
  "uzman_onay_tavsiyesi": "hizli_inceleme" | "detayli_inceleme" | "reddet"
}}

Tavsiye eşikleri:
- guven_skoru >= 0.85 → "hizli_inceleme"
- 0.60-0.85 → "detayli_inceleme"
- < 0.60 → "reddet"
"""


_DEFAULT_FALLBACK: dict = {
    "guven_skoru": 0.0,
    "desteklenen_iddialar": 0,
    "desteklenmeyen_iddialar": 0,
    "desteklenmeyen_liste": ["Güven skorlaması başarısız oldu; manuel inceleme zorunludur."],
    "uzman_onay_tavsiyesi": "detayli_inceleme",
    "hata": True,
}


def _format_chunks(chunks: list[dict]) -> str:
    """Kaynak chunk listesini düz metne dönüştürür (LLM tüketimi için)."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        label = f"[{i}] parent_id={c.get('parent_id', '?')} | kaynak={c.get('source', '?')}"
        text = c.get("parent_text") or c.get("text") or ""
        blocks.append(f"{label}\n{text}")
    return "\n\n---\n\n".join(blocks)


def score_generated_content(
    generated_json: dict,
    source_chunks: list[dict],
    content_type: str,
) -> dict:
    """
    Üretilen JSON içeriğine bir güven skoru atar.

    Parametreler
    -----------
    generated_json : dict
        Üretici fonksiyonun döndürdüğü içerik (örn. ProcessMapSchema dump'ı).
    source_chunks : list[dict]
        retrieve_context'in döndürdüğü parent chunk listesi.
    content_type : str
        "process_map" | "error_cards" | "glossary" | "simulation"

    Döner
    -----
    dict — ConfidenceScore alanları + bayrak alanları (örn. "rozet_renk").
    """
    if not settings.ENABLE_CONFIDENCE_SCORING:
        return {**_DEFAULT_FALLBACK, "hata": False, "atlandi": True}

    if not source_chunks:
        logger.warning("Güven skoru: kaynak chunk yok; fallback döndürülüyor.")
        return _DEFAULT_FALLBACK

    try:
        content_str = json.dumps(generated_json, ensure_ascii=False, indent=2)[:6000]
    except Exception as exc:
        logger.warning("Güven skoru: içerik serialize edilemedi: %s", exc)
        return _DEFAULT_FALLBACK

    chunks_str = _format_chunks(source_chunks)[:8000]
    prompt = _PROMPT.format(
        content_type=content_type,
        generated_json=content_str,
        source_chunks=chunks_str,
    )

    try:
        raw = llm_generate(prompt=prompt, response_format=ConfidenceScore)
        parsed = json.loads(raw)
        validated = ConfidenceScore(**parsed)
    except Exception as exc:
        logger.warning("Güven skoru parse edilemedi: %s", exc)
        return _DEFAULT_FALLBACK

    score = validated.model_dump()
    score["rozet_renk"] = _badge_color(score["guven_skoru"])
    score["rozet_metin"] = _badge_text(score["guven_skoru"])
    score["hata"] = False
    return score


def _badge_color(skor: float) -> str:
    if skor >= 0.85:
        return "green"
    if skor >= 0.60:
        return "yellow"
    return "red"


def _badge_text(skor: float) -> str:
    if skor >= 0.85:
        return "Yüksek Güven"
    if skor >= 0.60:
        return "Orta Güven — İncele"
    return "Düşük Güven — Dikkatle İncele"
