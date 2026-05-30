"""
Phase 3 — giriş/çıkış güvenlik katmanı (unsupervised production sertleştirme).

`src/chatbot.py`'deki tek-katmanlı sistem talimatı, denetimsiz (insan onayı
olmadan) kullanım için yetersizdir. Bu modül iki BAĞIMSIZ katman ekler:

1. `screen_input()` — GİRİŞ taraması. Yalnızca YÜKSEK-KESİNLİKLİ kötüye kullanım
   sinyallerini reddeder: prompt-injection / jailbreak ve sınav-içeriği üretimi
   (bunlar bir DELF eğitim asistanı için asla meşru değildir). Puan-çıkarma ve
   grounding-bypass gibi meşru örtüşmesi olan istekler BLOKLANMAZ — bunlar sistem
   talimatı + çıkış grounding kapısı tarafından ele alınır; burada yalnızca
   denetim (audit) için etiketlenir. Bu, `chatbot.py`'nin "kırılgan, yanlış-
   pozitif üreten anahtar kelime sınıflandırıcısından kaçın" tasarım kararına
   saygı gösterir.

2. `enforce_grounding()` — ÇIKIŞ kapısı. Alıntıları getirilen kaynaklara karşı
   doğrulanamayan (ungrounded) bir yanıtı SERVİS ETMEZ; yerine net bir reddetme
   döndürür. Eski davranış (yalnızca uyarı eklemek) ungrounded içeriği yine de
   sunardı — denetimsiz, yüksek-riskli bir sınav aracı için kabul edilemez.

Her iki katman da bayrakla (config) geri alınabilir; varsayılan AÇIK.
Hepsi DETERMİNİSTİKTİR — LLM/ağ çağrısı yapmaz, MOCK_MODE'da da çalışır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.config import settings

Language = Literal["tr", "fr"]

# Kategoriler
CAT_INJECTION = "prompt_injection"
CAT_EXAM_CONTENT = "exam_content_generation"
CAT_SCORE_EXTRACTION = "score_extraction"
CAT_GROUNDING_BYPASS = "grounding_bypass"


@dataclass
class InputVerdict:
    """Giriş taraması sonucu."""

    allowed: bool
    category: str | None = None      # eşleşen kategori (bloklanmasa bile etiketlenir)
    reason: str | None = None        # kısa, loglanabilir gerekçe
    refusal_text: str | None = None  # allowed=False ise kullanıcıya gösterilecek metin


@dataclass
class OutputVerdict:
    """Çıkış grounding kapısı sonucu."""

    allowed: bool
    answer: str               # servis edilecek metin (reddedilirse reddetme metni)
    citation_report: dict
    reason: str | None = None


# ---------------------------------------------------------------------------
# Desenler — yüksek kesinlik. İngilizce de dahil (saldırılar çoğu kez İngilizce).
# ---------------------------------------------------------------------------

# Prompt-injection / jailbreak / talimat geçersiz kılma — asla meşru değil → BLOKLA.
_INJECTION_PATTERNS = [
    r"\bignore\s+(?:the\s+|all\s+|your\s+|previous\s+|prior\s+|above\s+)*"
    r"(?:instructions?|rules?|prompts?|context|guidelines?)\b",
    r"\bdisregard\s+(?:the\s+|all\s+|your\s+|previous\s+|above\s+)*"
    r"(?:instructions?|rules?|prompts?|guidelines?)\b",
    r"\bforget\s+(?:the\s+|all\s+|your\s+|previous\s+|everything\s+)*"
    r"(?:instructions?|rules?|above|you\s+were\s+told)\b",
    r"\b(?:reveal|show|print|repeat|display|expose)\s+(?:me\s+)?(?:your\s+|the\s+)*"
    r"(?:system\s+)?(?:prompt|instructions?|rules?)\b",
    r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|act\s+as|pretend\s+to\s+be|"
    r"roleplay\s+as|you\s+must\s+now\s+act)\b",
    r"\b(?:developer\s+mode|jailbreak|do\s+anything\s+now|\bDAN\b)\b",
    r"\bbypass\s+(?:your\s+|the\s+|all\s+)*(?:rules?|filters?|guard(?:rails?)?|safety|restrictions?)\b",
    # Türkçe
    r"(?:talimatlar(?:ını|ı)?|kurallar(?:ını|ı)?|yönergeler(?:ini|i)?|sistem\s+talimat\w*)\s+"
    r"(?:unut|yoksay|boş\s*ver|boşver|görmezden\s+gel|dikkate\s+alma|umursama|geçersiz\s+kıl)",
    r"(?:sistem\s+)?(?:prompt(?:unu|u)?|talimat\w*)\s*(?:n[ıi])?\s*"
    r"(?:göster|yazdır|söyle|açıkla|paylaş)",
    r"\bartık\s+(?:sen\s+)?\w+\s*(?:sin|sın|davran|rolünü\s+üstlen)",
    # Fransızca
    r"\b(?:ignore[zr]?|oublie[zr]?|n[ée]glige[zr]?)\s+(?:les\s+|tes\s+|toutes\s+les\s+|vos\s+)*"
    r"(?:instructions?|r[èe]gles?|consignes?|directives?)\b",
    r"\b(?:r[ée]v[èe]le[zr]?|montre[zr]?|affiche[zr]?)\s+(?:ton|votre|le)\s+"
    r"(?:prompt|syst[èe]me|instructions?)\b",
    r"\b(?:tu\s+es\s+maintenant|d[ée]sormais\s+tu\s+es|agis\s+comme|fais\s+semblant)\b",
]

# Sınav-içeriği / sahte aday kopyası üretimi — asla meşru değil → BLOKLA.
# İki sinyal yakınlık (proximity) ile: bir ÜRETİM FİİLİ + bir SINAV-ESERİ ismi.
# İngilizce/Fransızca fiil-önce, Türkçe fiil-sonra olduğundan İKİ sıra da denenir.
_EXAM_VERB = (
    r"(?:write|create|generate|compose|draft|produce|invent|fabricate|prepare|"
    r"[ée]cri[ts]|cr[ée]e[zr]?|g[ée]n[èe]re[zr]?|r[ée]dige[zr]?|produi[ts]|"
    r"fabrique[zr]?|invente[zr]?|pr[ée]pare[zr]?|yaz|üret|oluştur|hazırla)"
)
_EXAM_ARTIFACT = (
    r"(?:exam\s+(?:subject|topic|questions?|paper)|"
    r"sujets?(?:\s+(?:officiel|d['’]examen|de\s+delf|de\s+dalf))?|"
    r"[ée]preuves?|"
    r"candidates?\s+(?:cop(?:y|ies)|essays?|productions?|scripts?)|"
    r"(?:student|[ée]l[èe]ves?)\s+cop(?:y|ies)|"
    r"copies?\s+(?:de\s+candidat|d['’]?[ée]l[èe]ve|fictives?)|"
    r"sınav\s+(?:konu|soru)\w*|deneme\s+sınav\w*|"
    r"aday\s+(?:kopya|metn|cevab|yazı)\w*|"
    r"(?:fake|sample|mock)\s+(?:cop(?:y|ies)|exams?|subjects?))"
)
_GAP = r"[\s\S]{0,40}?"
_EXAM_CONTENT_PATTERNS = [
    rf"\b{_EXAM_VERB}\b{_GAP}\b{_EXAM_ARTIFACT}",   # fiil → eser (EN/FR)
    rf"\b{_EXAM_ARTIFACT}{_GAP}\b{_EXAM_VERB}\b",    # eser → fiil (TR, fiil-sonra)
]

# Puan-çıkarma — meşru örtüşme var (kriterleri tartışmak normal) → yalnızca ETİKETLE.
_SCORE_EXTRACTION_PATTERNS = [
    r"\b(?:give\s+me|tell\s+me|what['’]?s|just\s+say|decide)\s+(?:the\s+)?(?:final\s+)?"
    r"(?:score|grade|band|mark|note)\b",
    r"(?:nihai|kesin|son)\s+(?:puan|not|bant|skor)\w*\s*(?:söyle|ver|nedir|kaç|belirle)",
    r"(?:bant|band)\s*\d", r"\bbu\s+ka[çc]\s+(?:puan|not)\b",
    r"\b(?:donne|dis)[\s-]*(?:moi)?\s+(?:la\s+)?(?:note|bande)\s+(?:finale|exacte|d[ée]finitive)\b",
]

# Grounding-bypass — kaynaksız/genel-bilgi talebi → ETİKETLE (çıkış kapısı yakalar).
_GROUNDING_BYPASS_PATTERNS = [
    r"\b(?:ignore|forget|without|don['’]?t\s+use|skip)\s+(?:the\s+)?(?:sources?|citations?|documents?|corpus|context)\b",
    r"(?:kaynak\w*|belge\w*|alıntı\w*)\s*(?:olmadan|kullanma|boş\s*ver|boşver|gerekmez|umursama)",
    r"(?:genel\s+bilgi\w*|kendi\s+bilgin\w*)\s+(?:ile|kullan)",
    r"\b(?:sans\s+(?:citer|sources?|le\s+corpus)|ignore[zr]?\s+les\s+sources?)\b",
]

_COMPILED = {
    CAT_INJECTION: [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS],
    CAT_EXAM_CONTENT: [re.compile(p, re.IGNORECASE) for p in _EXAM_CONTENT_PATTERNS],
    CAT_SCORE_EXTRACTION: [re.compile(p, re.IGNORECASE) for p in _SCORE_EXTRACTION_PATTERNS],
    CAT_GROUNDING_BYPASS: [re.compile(p, re.IGNORECASE) for p in _GROUNDING_BYPASS_PATTERNS],
}

# Bloklanan kategoriler (yüksek kesinlik). Diğerleri yalnızca etiketlenir.
_BLOCKING_CATEGORIES = (CAT_INJECTION, CAT_EXAM_CONTENT)

_REFUSAL_TEXT: dict[str, dict[Language, str]] = {
    CAT_INJECTION: {
        "tr": "Bu isteği yerine getiremem. Sistem talimatlarımı veya güvenlik "
              "kurallarımı değiştiremem. DELF/DALF değerlendirme metodolojisi, "
              "grille'ler veya descripteur'lar hakkında yardımcı olabilirim.",
        "fr": "Je ne peux pas accéder à cette demande. Je ne peux pas modifier mes "
              "instructions ni mes règles de sécurité. Je peux vous aider sur la "
              "méthodologie d'évaluation DELF/DALF, les grilles ou les descripteurs.",
    },
    CAT_EXAM_CONTENT: {
        "tr": "Sınav konusu, sınav sorusu veya aday kopyası üretemem. Bunun yerine "
              "değerlendirme metodolojisi, grille kriterleri veya örnek üzerinden "
              "kriter uygulaması konusunda yardımcı olabilirim.",
        "fr": "Je ne peux pas générer de sujet d'examen, de question d'examen ni de "
              "copie de candidat. Je peux en revanche vous aider sur la méthodologie "
              "d'évaluation, les critères des grilles ou leur application.",
    },
}


def screen_input(message: str, language: Language = "tr") -> InputVerdict:
    """Kullanıcı mesajını kötüye kullanım için tarar.

    Bloklananlar (allowed=False): prompt-injection/jailbreak, sınav-içeriği üretimi.
    Etiketlenip geçenler (allowed=True, category dolu): puan-çıkarma, grounding-bypass
    — bunlar sistem talimatı + çıkış grounding kapısı tarafından ele alınır.
    """
    if not settings.ENABLE_INPUT_SAFETY_SCREEN:
        return InputVerdict(allowed=True)

    text = message or ""
    # Önce bloklayan kategoriler (öncelik sırası: injection > exam_content).
    for category in _BLOCKING_CATEGORIES:
        for pat in _COMPILED[category]:
            if pat.search(text):
                lang = language if language in ("tr", "fr") else "tr"
                return InputVerdict(
                    allowed=False,
                    category=category,
                    reason=f"matched {category}: /{pat.pattern[:48]}/",
                    refusal_text=_REFUSAL_TEXT[category][lang],
                )
    # Bloklamayan ama etiketlenen kategoriler (audit için).
    for category in (CAT_SCORE_EXTRACTION, CAT_GROUNDING_BYPASS):
        for pat in _COMPILED[category]:
            if pat.search(text):
                return InputVerdict(allowed=True, category=category,
                                    reason=f"flagged {category}")
    return InputVerdict(allowed=True)


# ---------------------------------------------------------------------------
# Çıkış grounding kapısı
# ---------------------------------------------------------------------------

_UNGROUNDED_REFUSAL: dict[Language, str] = {
    "tr": "Bu yanıtı güvenle veremiyorum: ürettiğim ifadeleri korpustaki "
          "kaynaklara dayandıramadım. Yanlış bilgi riskine karşı, doğrulanabilir "
          "kaynak bulunmadığında yanıt vermiyorum. Soruyu daha belirgin bir DELF/DALF "
          "konusuna (örn. belirli bir grille kriteri veya seviye) daraltabilir misiniz?",
    "fr": "Je ne peux pas fournir cette réponse en toute confiance : je n'ai pas pu "
          "ancrer mes affirmations dans les sources du corpus. Pour éviter tout risque "
          "d'information erronée, je m'abstiens lorsque aucune source vérifiable n'est "
          "disponible. Pouvez-vous préciser votre question (par ex. un critère de grille "
          "ou un niveau précis) ?",
}


def enforce_grounding(
    answer: str,
    sources: list[dict],
    citation_report: dict,
    language: Language = "tr",
) -> OutputVerdict:
    """Ungrounded yanıtı reddeder (kapı açıkken).

    `citation_report` `chatbot._citation_report()` çıktısıdır. `passed` True ise
    yanıt olduğu gibi servis edilir. Değilse:
      - ENABLE_GROUNDING_GATE açık → yanıt reddetme metniyle DEĞİŞTİRİLİR (allowed=False).
      - kapalı → eski davranış: yanıt aynen geçer (çağıran uyarı ekleyebilir).
    """
    lang = language if language in ("tr", "fr") else "tr"
    if citation_report.get("passed"):
        return OutputVerdict(allowed=True, answer=answer, citation_report=citation_report)

    if not settings.ENABLE_GROUNDING_GATE:
        return OutputVerdict(
            allowed=True, answer=answer, citation_report=citation_report,
            reason="grounding_gate_disabled",
        )

    return OutputVerdict(
        allowed=False,
        answer=_UNGROUNDED_REFUSAL[lang],
        citation_report=citation_report,
        reason="ungrounded_answer_refused",
    )
