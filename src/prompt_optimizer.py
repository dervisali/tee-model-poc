"""
Prompt Optimizer — autoresearch ilhamlı otonom değerlendirme döngüsü.

Karpathy autoresearch projesinden esinlenmiştir:
  autoresearch : train.py değiştir → 5 dk eğit → val_bpb ölç → en iyiyi sakla
  Burada      : sorgu + talimat değiştir → içerik üret → LLM hakemiyle puanla
                 → en iyiyi optimized_prompts.json'a sakla

Cloud branch notu: Tüm model çağrıları `src.llm.generate` üzerinden Vertex AI
Gemini'ye yönlendirilir. Hakem (judge) prompt'u modele agnostiktir.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from src.config import settings
from src.generators import _build_grounded_prompt
from src.llm import generate as llm_generate


logger = logging.getLogger(__name__)


OPTIMIZED_PROMPTS_PATH = settings.BASE_DIR / "optimized_prompts.json"


# ---------------------------------------------------------------------------
# Varyantlar — her üretici için 3 (varsayılan + 2 alternatif)
# ---------------------------------------------------------------------------

VARIANTS: dict[str, list[dict]] = {
    "process_map": [
        {"name": "Varsayılan", "query": "DELF değerlendirme süreci sınavcı-düzeltici protokolü iş akışı", "extra_hint": ""},
        {"name": "Faz + sorumlu",
         "query": "sınavcı-düzeltici habilitasyon hazırlık bireysel düzeltme çift düzeltme finalizasyon",
         "extra_hint": "Her adımda fazı (Hazırlık / Bireysel Düzeltme / Çift Düzeltme / Finalizasyon) ve sorumlu rolü açıkça belirt."},
        {"name": "Risk + atipik",
         "query": "atipik kopya istisnai durum puan uyuşmazlığı delibération karar riski",
         "extra_hint": "Her adımın risk alanına atipik kopya veya puan uyuşmazlığı gibi olası istisnaları ekle."},
    ],
    "error_cards": [
        {"name": "Varsayılan", "query": "sınavcı değerlendirme hataları yanlış puanlama DELF", "extra_hint": ""},
        {"name": "Halo + önyargı",
         "query": "halo etkisi önyargı tutarsız puanlama güvenilirlik arası değerlendirici",
         "extra_hint": "Her hatanın nasıl önleneceğine dair somut pratik adım ekle."},
        {"name": "Grille uyumu",
         "query": "grille kriterleri yanlış yorumlama bant tanımlayıcı seviye karışıklığı",
         "extra_hint": "Her hatanın hangi grille kriteri veya tanımlayıcısıyla ilgili olduğunu belirt."},
    ],
    "glossary": [
        {"name": "Varsayılan", "query": "DELF CECRL terimleri descripteur grille kriterler bant", "extra_hint": ""},
        {"name": "Kullanım + hata",
         "query": "DELF terim yanlış anlama seviye karışıklığı uygulama hatası",
         "extra_hint": "Terimin yanlış anlaşılmasından kaynaklanan yaygın hatayı kullanim_ornegi alanına ekle."},
        {"name": "Beceri + seviye",
         "query": "PE PO CE CO becerisi A1 A2 B1 B2 C1 seviye tanımlayıcıları",
         "extra_hint": "Her terimin hangi beceri (PE/PO/CE/CO) ve seviye (A1-C1) bağlamında kullanıldığını belirt."},
    ],
    "simulation": [
        {"name": "Varsayılan", "query": "kritik değerlendirme kararı atipik kopya ikilemli durum", "extra_hint": ""},
        {"name": "Puan uyuşmazlığı",
         "query": "iki düzeltici arasında puan farkı uzlaşma prosedürü eşik",
         "extra_hint": "Senaryo iki sınavcı arasındaki puan uyuşmazlığının çözüm sürecini test etsin."},
        {"name": "Sınır seviye",
         "query": "sınır puan geçme kalma kararı A2 B1 B2 eşik grille bant",
         "extra_hint": "Senaryo bir adayın seviye eşiğinde olduğu ve kararın güç olduğu durumu ele alsın."},
    ],
}

GENERATOR_LABELS = {
    "process_map": "Süreç Haritası",
    "error_cards": "Hata Kartları",
    "glossary": "Terim Sözlüğü",
    "simulation": "Simülasyon Senaryosu",
}


_INSTRUCTION_TEMPLATES: dict[str, str] = {
    "process_map": (
        "Yukarıdaki bağlam belgelerine dayanarak DELF/DALF sınavcı-düzeltici değerlendirme "
        "sürecinin kapsamlı haritasını oluştur. En az 10, en fazla 15 adım üret.\n"
        "EXTRA_HINT\n"
        "Sadece geçerli JSON döndür. kaynak_chunk_indeksleri alanına kullandığın "
        "parent_id değerlerini yaz."
    ),
    "error_cards": (
        "Yukarıdaki bağlam belgelerine dayanarak yeni DELF/DALF sınavcı-düzelticisinin "
        "sık yaptığı değerlendirme hatalarını listele.\n"
        "EXTRA_HINT\n"
        "Sadece geçerli JSON döndür. kaynak_chunk_indeksleri alanına kullandığın "
        "parent_id değerlerini yaz."
    ),
    "glossary": (
        "Yukarıdaki bağlam belgelerine dayanarak DELF/DALF sınavcıları için "
        "CECRL/DELF terim sözlüğü oluştur.\n"
        "EXTRA_HINT\n"
        "Sadece geçerli JSON döndür."
    ),
    "simulation": (
        "Yukarıdaki bağlam belgelerine dayanarak DELF/DALF sınavcı-düzelticileri için "
        "etkileşimli bir değerlendirme karar senaryosu oluştur.\n"
        "EXTRA_HINT\n"
        "Doğru cevap seçeneğinde dogru_mu = true olmalı. Sadece geçerli JSON döndür."
    ),
}


def _build_instruction(generator_key: str, extra_hint: str) -> str:
    """EXTRA_HINT yer tutucusunu doldurur."""
    template = _INSTRUCTION_TEMPLATES[generator_key]
    return template.replace("EXTRA_HINT", extra_hint or "").strip()


# ---------------------------------------------------------------------------
# Hakem (judge)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
Sen bir kamu kurumu eğitim içeriği değerlendirme uzmanısın.
Aşağıdaki yapay zeka tarafından üretilen içeriği üç kriter üzerinden 0-10 arasında puanla:

1. DAYANDIRMA (grounding): İçerik kaynak chunk indekslerine dayandırılmış mı? Atıflar makul mü?
2. TAMLAMA (completeness): Tüm zorunlu JSON alanları eksiksiz ve anlamlı biçimde dolu mu?
3. PRATIK FAYDALILIK (usefulness): İçerik yeni bir DELF/DALF sınavcı-düzelticisi için gerçekten öğretici mi?

ÜRETİLEN İÇERİK:
{content}

YALNIZCA şu JSON formatında yanıt ver, başka hiçbir şey yazma:
{{"grounding": <0-10>, "completeness": <0-10>, "usefulness": <0-10>, "reasoning": "<kısa Türkçe gerekçe>"}}
"""


def _score_output(content: dict) -> dict:
    """LLM hakemi ile içerik puanlama."""
    content_str = json.dumps(content, ensure_ascii=False, indent=2)[:3000]
    prompt = _JUDGE_PROMPT.format(content=content_str)
    try:
        raw = llm_generate(prompt=prompt)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        scores = json.loads(cleaned)
        scores["total"] = (
            scores.get("grounding", 0)
            + scores.get("completeness", 0)
            + scores.get("usefulness", 0)
        )
        return scores
    except Exception as exc:
        logger.warning("Puanlama başarısız: %s", exc)
        return {
            "grounding": 0,
            "completeness": 0,
            "usefulness": 0,
            "total": 0,
            "reasoning": f"Puanlama hatası: {exc}",
        }


# ---------------------------------------------------------------------------
# Optimizasyon döngüsü
# ---------------------------------------------------------------------------

def run_optimization(
    generator_key: str,
    progress_callback: Callable | None = None,
) -> list[dict]:
    """
    Tüm prompt varyantlarını sırayla çalıştırır, her birini hakem ile puanlar
    ve toplam skora göre sıralı liste döndürür (en iyisi başta).
    """
    if generator_key not in VARIANTS:
        raise ValueError(f"Bilinmeyen generator: {generator_key}")

    variants = VARIANTS[generator_key]
    results: list[dict] = []

    for i, variant in enumerate(variants):
        name = variant["name"]
        query = variant["query"]
        extra_hint = variant["extra_hint"]
        instruction = _build_instruction(generator_key, extra_hint)
        logger.info("Varyant %d/%d: %s", i + 1, len(variants), name)

        if progress_callback:
            progress_callback(i, name, "generating")

        try:
            system, user_prompt, _ = _build_grounded_prompt(query, instruction, top_k=8)
            raw = llm_generate(prompt=user_prompt, system=system)
            try:
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                output = json.loads(cleaned)
            except Exception as exc:
                output = {"hata": str(exc), "ham_cikti": raw}
        except Exception as exc:
            logger.error("Varyant '%s' üretim hatası: %s", name, exc)
            output = {"hata": str(exc)}

        if progress_callback:
            progress_callback(i, name, "scoring")

        if "hata" in output:
            scores = {
                "grounding": 0,
                "completeness": 0,
                "usefulness": 0,
                "total": 0,
                "reasoning": output.get("hata", "Üretim başarısız"),
            }
        else:
            scores = _score_output(output)

        results.append({
            "name": name,
            "query": query,
            "extra_hint": extra_hint,
            "instruction": instruction,
            "output": output,
            "scores": scores,
            "total_score": scores.get("total", 0),
        })

        logger.info(
            "  → grounding=%s completeness=%s usefulness=%s total=%s",
            scores.get("grounding"),
            scores.get("completeness"),
            scores.get("usefulness"),
            scores.get("total"),
        )

    results.sort(key=lambda r: r["total_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Kalıcı saklama
# ---------------------------------------------------------------------------

def save_best_prompt(generator_key: str, best_result: dict) -> None:
    """En iyi varyantı optimized_prompts.json dosyasına yazar."""
    existing = load_optimized_prompts()
    existing[generator_key] = {
        "name": best_result["name"],
        "query": best_result["query"],
        "instruction": best_result["instruction"],
        "scores": best_result["scores"],
    }
    OPTIMIZED_PROMPTS_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Kaydedildi: %s → %s", generator_key, best_result["name"])


def load_optimized_prompts() -> dict:
    """optimized_prompts.json içeriğini yükler."""
    if not OPTIMIZED_PROMPTS_PATH.exists():
        return {}
    try:
        return json.loads(OPTIMIZED_PROMPTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
