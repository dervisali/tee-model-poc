"""
Cross-lingual sorgu genişletme — Phase 3.5.

Türkçe sınavcılar Fransızca DELF korpusuna sorgu attığında BM25 leksikal
yolunun da çalışabilmesi için sorgu Gemini Flash ile hedef korpus diline
çevrilir. Yoğun (dense) yol için çeviri gerekli değildir; `gemini-embedding-001`
multilingual olduğundan Türkçe sorgu da doğrudan Fransızca chunk'lara yakın
yerleşir.

Dil tespiti hafif heuristic: Türkçe'ye özgü karakterler (ğ, ş, ı, İ, Ğ, Ş)
veya yaygın Türkçe sonek desenleri varsa 'tr', aksi halde varsayılan 'fr'
(korpus dili). Heuristic yetersiz kalırsa kullanıcı QUERY_LANGUAGE_AUTO_DETECT
flag'ini kapatıp doğrudan tek-dilli moda dönebilir.

Çeviri çağrıları sürecin bellek-içi cache'inde tutulur (lru_cache, 1024 giriş);
süreç ömrü kadar yaşar. Kalıcı cache şu an gerekmiyor.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Literal

from src.config import settings


logger = logging.getLogger(__name__)


# Türkçe'ye özgü karakterler. ç, ö, ü Fransızca'da da bulunduğundan kullanılmaz;
# ğ, ş, ı (noktasız küçük i), İ (noktalı büyük I) güçlü TR sinyalidir.
_TURKISH_CHARS_RE = re.compile(r"[ğşıİĞŞ]")

# Yaygın Türkçe sonekler; sorgunun TR olduğunu güçlü bir biçimde işaret eder.
# Liste tüketici değildir; hedef yanlış-pozitif değil yanlış-negatif minimizasyonudur:
# kaçırılan TR sorgu en kötü ihtimalle BM25 sinyali kaybeder, yine dense ile gelir.
_TURKISH_SUFFIX_RE = re.compile(
    r"\b\w+("
    r"iyor|ıyor|uyor|üyor|"      # şimdiki zaman
    r"mıştır|miştir|muştur|müştür|"
    r"larak|lerek|"               # ulaç
    r"ları|leri|"                 # iyelik / çoğul -i halinin son ekleri
    r"ında|ünde|inde|unda|"       # bulunma hali
    r"makta|mekte|"               # ilerleme
    r"madık|medik|"
    r"acak|ecek"                  # gelecek
    r")\b",
    re.IGNORECASE,
)


def detect_query_language(text: str) -> Literal["tr", "fr"]:
    """
    Sorgunun diline ilişkin heuristic karar. Güvenli varsayılan korpus
    dilidir ('fr') — belirsiz kısa sorgular için BM25 zaten Fransızca
    indekste, bu durumda çeviri devre dışı kalır.
    """
    if not text or not text.strip():
        return settings.CORPUS_PRIMARY_LANGUAGE  # type: ignore[return-value]
    if _TURKISH_CHARS_RE.search(text):
        return "tr"
    if _TURKISH_SUFFIX_RE.search(text):
        return "tr"
    return "fr"


_TRANSLATION_PROMPT_TEMPLATE = """You are translating a search query from {source_name} to {target_name} for retrieving DELF/DALF examiner training materials (French language proficiency exam evaluation grids, descriptors, methodology guides).

PRESERVE these technical CECRL/DELF terms EXACTLY (do not translate, do not localize):
- CECRL level codes: A1, A2, B1, B2, C1, C2
- Skill codes: PE (production écrite), PO (production orale), CE (compréhension écrite), CO (compréhension orale)
- Document types: grille, descripteur, stagiaire, sujet, copie, méthodologie

Output ONLY the translated query — no explanations, no quotes, no preamble.

Query ({source_name}): {query}
Query ({target_name}):"""


@lru_cache(maxsize=1024)
def translate_query(query: str, target_language: Literal["tr", "fr"] = "fr") -> str:
    """
    Sorguyu hedef dile çevirir. Kaynak dil heuristic ile tespit edilir; sorgu
    zaten hedef dilde ise olduğu gibi döndürülür. lru_cache ile süreç-içi cache.

    MOCK_MODE açıksa LLM çağrısı yapılmaz; deterministik bir mock dize döner
    (testlerin canlı Vertex'e ulaşmaması için).
    """
    if not query or not query.strip():
        return query

    source_language = detect_query_language(query)
    if source_language == target_language:
        return query

    if settings.MOCK_MODE:
        return f"[MOCK-{target_language.upper()}] {query}"

    # Lazy import to keep MOCK_MODE / non-vertex environments importable.
    from src.llm import generate as llm_generate

    names = {"tr": "Turkish", "fr": "French"}
    prompt = _TRANSLATION_PROMPT_TEMPLATE.format(
        source_name=names[source_language],
        target_name=names[target_language],
        query=query,
    )
    try:
        raw = llm_generate(prompt=prompt, temperature=0.0).strip()
        # Modelin nadiren eklediği başta/sonda alıntı işaretlerini temizle.
        translated = raw.strip('"\'').strip()
        logger.info(
            "Query translated",
            extra={
                "event": "query_translated",
                "source_lang": source_language,
                "target_lang": target_language,
                "original_len": len(query),
                "translated_len": len(translated),
            },
        )
        return translated or query
    except Exception as exc:  # noqa: BLE001 — translation hatası retrieval'ı kırmamalı
        logger.warning("Query translation failed (%s); falling back to original.", exc)
        return query
