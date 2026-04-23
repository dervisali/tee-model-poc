"""
Prompt Optimizer — autoresearch-inspired autonomous evaluation loop for TEE-Model POC.

Inspired by github.com/karpathy/autoresearch:
  autoresearch : modify train.py  → train 5 min  → measure val_bpb   → keep/discard
  Here         : modify query+instruction → generate content → score with LLM judge → keep/discard

For each generator, tests multiple (query, instruction) variants, scores each with a
Gemini LLM judge on grounding, completeness, and usefulness, then persists the best
variant to optimized_prompts.json so generators pick it up automatically on next run.
"""

import json
import logging
from pathlib import Path

from src.generators import _get_genai_client, _call_llm, _build_grounded_prompt, _parse_json_with_retry

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OPTIMIZED_PROMPTS_PATH = BASE_DIR / "optimized_prompts.json"

# ---------------------------------------------------------------------------
# Prompt variants — 3 per generator (current default + 2 alternatives)
# ---------------------------------------------------------------------------

VARIANTS: dict[str, list[dict]] = {
    "process_map": [
        {
            "name": "Varsayılan",
            "query": "maaş hesaplama adımları süreç akışı prosedür",
            "extra_hint": "",
        },
        {
            "name": "Kontrol odaklı",
            "query": "bordro hazırlama işlem sırası kontrol listesi onay adım",
            "extra_hint": "Her adımda kontrol noktasını ve sorumlu kişiyi açıkça belirt.",
        },
        {
            "name": "Mevzuat + takvim",
            "query": "mutemet aylık görev takvimi mevzuat son tarih sorumluluk",
            "extra_hint": "Her adım için yasal dayanak veya kritik son tarihi ekle.",
        },
    ],
    "error_cards": [
        {
            "name": "Varsayılan",
            "query": "sık yapılan hatalar yanlış uygulama kaçırılan adım",
            "extra_hint": "",
        },
        {
            "name": "Tacit + önlem",
            "query": "deneyimli mutemet uyarı tuzak dikkat pratik ipucu",
            "extra_hint": "Her hatanın nasıl önleneceğine dair pratik adım ekle.",
        },
        {
            "name": "Yasal sonuç",
            "query": "yasal yaptırım ceza sorumluluk idari işlem hata sonucu",
            "extra_hint": "Her hatanın yasal ve idari sonuçlarını dogru_uygulama alanında belirt.",
        },
    ],
    "glossary": [
        {
            "name": "Varsayılan",
            "query": "kuruma özgü terimler teknik kavramlar kısaltmalar",
            "extra_hint": "",
        },
        {
            "name": "Kullanım + hata",
            "query": "maaş bordro vergi kesinti tazminat teknik terim kullanım",
            "extra_hint": "Terimin yanlış anlaşılmasından kaynaklanan yaygın hatayı kullanim_ornegi alanına ekle.",
        },
        {
            "name": "Form + belge",
            "query": "resmi form belge kısaltma SGK vergi dairesi bildirim",
            "extra_hint": "Her terimin hangi form veya belgede geçtiğini belirt.",
        },
    ],
    "simulation": [
        {
            "name": "Varsayılan",
            "query": "kritik karar noktası yüksek hata riski zor durum",
            "extra_hint": "",
        },
        {
            "name": "Ay ortası",
            "query": "ay ortası işe giriş çıkış kısmi maaş SGK bildirim",
            "extra_hint": "Senaryo ay ortası başlayan veya ayrılan personel durumunu ele alsın.",
        },
        {
            "name": "Yasal limit",
            "query": "yasal limit icra kesintisi net aylık dörtte bir vergi dilimi",
            "extra_hint": "Senaryo bir yasal sınırın aşılma riskini içeren kararı test etsin.",
        },
    ],
}

GENERATOR_LABELS = {
    "process_map": "Süreç Haritası",
    "error_cards": "Hata Kartları",
    "glossary": "Terim Sözlüğü",
    "simulation": "Simülasyon Senaryosu",
}

# ---------------------------------------------------------------------------
# Instruction templates — EXTRA_HINT replaced at runtime
# ---------------------------------------------------------------------------

_INSTRUCTION_TEMPLATES: dict[str, str] = {
    "process_map": (
        "Yukarıdaki bağlam belgelerine dayanarak maaş mutemetliği süreç haritasını oluştur.\n"
        "EXTRA_HINT\n"
        "Aşağıdaki JSON şemasını kullan ve BAŞKA HİÇBİR ŞEY yazma:\n\n"
        '{\n'
        '  "steps": [\n'
        '    {\n'
        '      "adim_no": 1,\n'
        '      "baslik": "...",\n'
        '      "giris": "...",\n'
        '      "cikis": "...",\n'
        '      "karar_noktasi": "...",\n'
        '      "risk": "...",\n'
        '      "kontrol": "...",\n'
        '      "kaynak_chunk_indeksleri": [0, 2]\n'
        '    }\n'
        '  ]\n'
        '}'
    ),
    "error_cards": (
        "Yukarıdaki bağlam belgelerine dayanarak yeni maaş mutemedinin sık yaptığı hataları listele.\n"
        "EXTRA_HINT\n"
        "Aşağıdaki JSON şemasını kullan ve BAŞKA HİÇBİR ŞEY yazma:\n\n"
        '{\n'
        '  "hata_kartlari": [\n'
        '    {\n'
        '      "kart_no": 1,\n'
        '      "hata": "...",\n'
        '      "kök_neden": "...",\n'
        '      "tespit_yöntemi": "...",\n'
        '      "dogru_uygulama": "...",\n'
        '      "kaynak_chunk_indeksleri": [1, 3]\n'
        '    }\n'
        '  ]\n'
        '}'
    ),
    "glossary": (
        "Yukarıdaki bağlam belgelerine dayanarak maaş mutemetliği alanına özgü terim sözlüğü oluştur.\n"
        "EXTRA_HINT\n"
        "Aşağıdaki JSON şemasını kullan ve BAŞKA HİÇBİR ŞEY yazma:\n\n"
        '{\n'
        '  "terimler": [\n'
        '    {\n'
        '      "terim": "...",\n'
        '      "tanim": "...",\n'
        '      "kullanim_ornegi": "...",\n'
        '      "kaynak_chunk_indeksleri": [0]\n'
        '    }\n'
        '  ]\n'
        '}'
    ),
    "simulation": (
        "Yukarıdaki bağlam belgelerine dayanarak maaş mutemetliği için etkileşimli bir simülasyon senaryosu oluştur.\n"
        "EXTRA_HINT\n"
        "Aşağıdaki JSON şemasını kullan ve BAŞKA HİÇBİR ŞEY yazma:\n\n"
        '{\n'
        '  "senaryo_basligi": "...",\n'
        '  "durum_aciklamasi": "...",\n'
        '  "soru": "...",\n'
        '  "secenekler": [\n'
        '    {"id": "A", "metin": "...", "dogru_mu": true, "geri_bildirim": "...", "sonuc": "..."},\n'
        '    {"id": "B", "metin": "...", "dogru_mu": false, "geri_bildirim": "...", "sonuc": "..."},\n'
        '    {"id": "C", "metin": "...", "dogru_mu": false, "geri_bildirim": "...", "sonuc": "..."}\n'
        '  ],\n'
        '  "ogrenme_hedefi": "...",\n'
        '  "kaynak_chunk_indeksleri": [2, 5]\n'
        '}'
    ),
}


def _build_instruction(generator_key: str, extra_hint: str) -> str:
    template = _INSTRUCTION_TEMPLATES[generator_key]
    replacement = extra_hint if extra_hint else ""
    return template.replace("EXTRA_HINT", replacement).strip()


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
Sen bir kamu kurumu eğitim içeriği değerlendirme uzmanısın.
Aşağıdaki yapay zeka tarafından üretilen içeriği üç kriter üzerinden 0-10 arasında puanla:

1. DAYANDIRMA (grounding): İçerik kaynak chunk indekslerine dayandırılmış mı? Atıflar makul görünüyor mu?
2. TAMLAMA (completeness): Tüm zorunlu JSON alanları eksiksiz ve anlamlı biçimde doldurulmuş mu?
3. PRATIK FAYDALILIK (usefulness): İçerik yeni bir maaş mutemedi için gerçekten öğretici ve uygulanabilir mi?

ÜRETİLEN İÇERİK:
{content}

YALNIZCA şu JSON formatında yanıt ver, başka hiçbir şey yazma:
{{"grounding": <0-10>, "completeness": <0-10>, "usefulness": <0-10>, "reasoning": "<kısa Türkçe gerekçe>"}}
"""


def _score_output(client, content: dict) -> dict:
    """Score a generated content dict with the LLM judge. Returns scores dict."""
    content_str = json.dumps(content, ensure_ascii=False, indent=2)[:3000]
    prompt = _JUDGE_PROMPT.format(content=content_str)
    try:
        raw = _call_llm(client, prompt)
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
# Main optimization loop
# ---------------------------------------------------------------------------

def run_optimization(generator_key: str, progress_callback=None) -> list[dict]:
    """
    Run all prompt variants for a generator, score each with an LLM judge, and
    return results sorted by total score (best first).

    Parameters
    ----------
    generator_key : str
        One of: "process_map", "error_cards", "glossary", "simulation"
    progress_callback : callable or None
        Called as progress_callback(variant_index, variant_name, stage) where
        stage is "generating" or "scoring".

    Returns
    -------
    list[dict]
        Each item: {name, query, extra_hint, instruction, output, scores, total_score}
        Sorted best-first by total_score.
    """
    if generator_key not in VARIANTS:
        raise ValueError(f"Bilinmeyen generator: {generator_key}")

    client = _get_genai_client()
    variants = VARIANTS[generator_key]
    results = []

    for i, variant in enumerate(variants):
        name = variant["name"]
        query = variant["query"]
        extra_hint = variant["extra_hint"]
        instruction = _build_instruction(generator_key, extra_hint)

        logger.info("Varyant %d/%d: %s", i + 1, len(variants), name)

        if progress_callback:
            progress_callback(i, name, "generating")

        try:
            prompt, _ = _build_grounded_prompt(query, instruction, top_k=8)
            raw = _call_llm(client, prompt)
            output = _parse_json_with_retry(client, prompt, raw)
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
            scores = _score_output(client, output)

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
# Persistence
# ---------------------------------------------------------------------------

def save_best_prompt(generator_key: str, best_result: dict) -> None:
    """
    Persist the best variant's query and instruction to optimized_prompts.json.
    Generators read this file at runtime and use it instead of their defaults.
    """
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
    """Load all saved optimized prompts from optimized_prompts.json."""
    if not OPTIMIZED_PROMPTS_PATH.exists():
        return {}
    try:
        return json.loads(OPTIMIZED_PROMPTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
