"""
Path-based document classifier — Phase 2.

Dosya yolu ve adından `ChunkMetadata` çıkarır. Tasarım hedefi: korpusun
~%95'i tamamen kural-bazlı (regex) olarak deterministik etiketlenmeli;
yalnızca kural setine sığmayan kenar durumlar düşük güven ile manuel
inceleme listesine düşmeli.

Sınıflandırma sırası:
  1. Authority blocklist (brouillon / draft / WhatsApp Image) →
     is_authoritative=False, doc_type=other, düşük güven.
  2. Level: yol bileşenlerinde \\b(a1|a2|b1|b2|c1)\\b veya /general/ klasörü.
  3. Skill: dosya adında PE|PO|CE|CO (kelime sınırları + alt çizgi).
  4. Doc type: anahtar sözcükler (grille / descripteur / stagiaire /
     manuel / methodologie / sujet / copie / presentation).
  5. Güven puanı: explicit eşleşmelerin birikim toplamı (0.6 baz).

Aksanlar matching'den önce NFKD ile çıkarılır; "Méthodologie" → "methodologie".
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import cast

from src.chunk_metadata import ChunkMetadata, DocType, Level, Skill


# ---------------------------------------------------------------------------
# Yardımcı: Aksan-bağımsız ASCII normalizasyon
# ---------------------------------------------------------------------------

def _ascii_lower(s: str) -> str:
    """NFKD normalize + combining marks strip + lowercase. Regex hazırlığı."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    ).lower()


# ---------------------------------------------------------------------------
# Regex desenleri
# ---------------------------------------------------------------------------

# Authority blocklist — brouillon, WhatsApp Image, draft.
_NON_AUTHORITATIVE = re.compile(r"\b(brouillon|whatsapp\s*image|draft)\b", re.IGNORECASE)

# Level — \b sınırlı a1/a2/b1/b2/c1 (yol veya dosya adı).
_LEVEL_RE = re.compile(r"\b(a1|a2|b1|b2|c1)\b", re.IGNORECASE)

# Skill — alt çizgi/boşluk/tire ile sınırlı PE/PO/CE/CO.
_SKILL_RE = re.compile(r"(?<![a-z])(pe|po|ce|co)(?![a-z])", re.IGNORECASE)

# Doc type — sıra önemli (önce daha spesifik desenler).
_DOC_TYPE_PATTERNS: list[tuple[re.Pattern[str], DocType]] = [
    (re.compile(r"grille", re.IGNORECASE), "grille"),
    (re.compile(r"stagiaire", re.IGNORECASE), "stagiaire"),
    (re.compile(r"copies?[_\s-]?atypiques?", re.IGNORECASE), "copie"),
    (re.compile(r"descripteur|echelle|niveaux", re.IGNORECASE), "descripteur"),
    (re.compile(r"methodolog|portfolio|introduction", re.IGNORECASE), "methodologie"),
    (re.compile(r"manuel|exacor|tableau|inventaire|dispositif|fei", re.IGNORECASE), "manuel"),
    (
        re.compile(
            r"sujet|veganisme|le[\s_-]*corps|candidat|"
            r"^(pe|po)[_\s-]+(a1|a2|b1|b2|c1)",
            re.IGNORECASE,
        ),
        "sujet",
    ),
]


# ---------------------------------------------------------------------------
# Çıkarıcılar
# ---------------------------------------------------------------------------

def _extract_level(rel_norm: str) -> tuple[Level, bool]:
    """
    Returns (level, was_explicit). Explicit = path/name'de açık eşleşme;
    /general/ klasörü de explicit sayılır. Her ikisi de bulunamazsa
    ('general', False).
    """
    m = _LEVEL_RE.search(rel_norm)
    if m:
        return cast(Level, m.group(1).upper()), True
    if "/general/" in rel_norm or rel_norm.startswith("general/"):
        return "general", True
    return "general", False


def _extract_skill(name_norm: str) -> tuple[Skill, bool]:
    m = _SKILL_RE.search(name_norm)
    if m:
        return cast(Skill, m.group(1).upper()), True
    return "general", False


def _extract_doc_type(name_norm: str, suffix: str) -> tuple[DocType, bool]:
    for pattern, dt in _DOC_TYPE_PATTERNS:
        if pattern.search(name_norm):
            return dt, True
    if suffix == ".pptx":
        return "presentation", True
    return "other", False


# ---------------------------------------------------------------------------
# Genel sınıflandırma
# ---------------------------------------------------------------------------

def classify_path(path: Path, *, root: Path | None = None) -> ChunkMetadata:
    """
    Dosya yolundan ChunkMetadata üretir. Hiçbir IO yapmaz.

    `root` verilirse `source_file` o köke göre rölatif olarak yazılır;
    yoksa yolun string hali kullanılır.
    """
    rel = str(path.relative_to(root)) if root else str(path)
    rel_norm = _ascii_lower(rel)
    name_norm = _ascii_lower(path.name)
    suffix = path.suffix.lower()

    # 1. Authority blocklist — brouillon/draft/WhatsApp.
    if _NON_AUTHORITATIVE.search(rel_norm):
        level, _ = _extract_level(rel_norm)
        return ChunkMetadata(
            source_file=rel,
            source_filename=path.name,
            level=level,
            skill="general",
            doc_type="other",
            language="fr",
            is_authoritative=False,
            classifier_confidence=0.3,
        )

    level, level_explicit = _extract_level(rel_norm)
    skill, skill_explicit = _extract_skill(name_norm)
    doc_type, doc_type_explicit = _extract_doc_type(name_norm, suffix)

    confidence = 0.6
    if level_explicit:
        confidence += 0.2
    if skill_explicit:
        confidence += 0.1
    if doc_type_explicit:
        confidence += 0.1
    confidence = min(confidence, 1.0)

    return ChunkMetadata(
        source_file=rel,
        source_filename=path.name,
        level=level,
        skill=skill,
        doc_type=doc_type,
        language="fr",
        is_authoritative=True,
        classifier_confidence=round(confidence, 2),
    )


def classify_corpus(
    root: Path,
    *,
    confidence_threshold: float = 0.8,
) -> tuple[list[ChunkMetadata], list[ChunkMetadata]]:
    """
    `root` altındaki tüm desteklenen dosyaları sınıflandırır.

    Döner: (high_confidence, manual_review). manual_review listesi
    confidence < threshold olan kayıtları içerir; insan gözüyle
    incelenmeli ve gerekirse classifier kuralı eklenmeli.
    """
    from src.document_loaders import SUPPORTED_EXTENSIONS

    high: list[ChunkMetadata] = []
    manual: list[ChunkMetadata] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        meta = classify_path(path, root=root)
        if meta.classifier_confidence < confidence_threshold:
            manual.append(meta)
        else:
            high.append(meta)
    return high, manual
