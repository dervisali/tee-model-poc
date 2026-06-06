"""
Metadata ön-filtreleme — vektör aramadan önce aday havuzunu daraltma.

DELF/DALF korpusunda her child chunk `level` (CECRL), `skill` (PE/PO/CE/CO),
`doc_type` ve `is_authoritative` metadata'sı taşır (bkz. src/chunk_metadata.py
ve ingestion). "B2 PO grilleri" gibi sorgularda alakasız seviye/beceri
parçalarını arama öncesinde elemek hem precision'ı artırır hem de aday
havuzunu küçülterek RRF füzyonunu netleştirir.

İki giriş noktası:
  - build_where_clause(...) : açık parametrelerden ChromaDB `where` sözlüğü.
  - detect_filters(query)   : sorgudan YALNIZCA kesin token'ları (CECRL seviye,
                              beceri kodu) tespit eder. Belirsizse {} döner —
                              asla tahmin etmez, böylece recall'ı düşürmez.

ChromaDB `where` sözleşmesi: tek alan için {alan: değer}; çok alan için
{"$and": [{alan: {"$eq": değer}}, ...]} (chromadb>=0.5 örtük AND'i kabul etmez).
BM25 yolu aynı sözlüğü Python tarafında düz eşitlikle uygular.
"""

from __future__ import annotations

import re
import unicodedata
from typing import get_args

from src.chunk_metadata import DocType, Level, Skill


# Şema literal'lerinden türetilen geçerli değer kümeleri — tek kaynak.
_VALID_LEVELS: frozenset[str] = frozenset(get_args(Level))
_VALID_SKILLS: frozenset[str] = frozenset(get_args(Skill))
_VALID_DOC_TYPES: frozenset[str] = frozenset(get_args(DocType))

# Sorgudan kesin tespit için — yalnızca tek anlamlı token'lar.
# CECRL seviye: A1/A2/B1/B2/C1/C2 (C2 şemada 'general'a düşmez; yine de yakalanır).
_LEVEL_RE = re.compile(r"\b([ABC][12])\b", re.IGNORECASE)
# Beceri kodları: PE/PO/CE/CO. Kelime sınırı + büyük harf zorunlu (Fransızca
# "po"/"ce" gibi sözcüklerle karışmasın diye yalnızca büyük harf eşleşir).
_SKILL_RE = re.compile(r"\b(PE|PO|CE|CO)\b")

_SKILL_PHRASES: dict[str, tuple[str, ...]] = {
    "PO": (
        "production orale",
        "epreuve orale",
        "epreuve de production orale",
        "oral production",
        "sozlu uretim",
        "sozlu anlatim",
        "sozlu sinav",
    ),
    "PE": (
        "production ecrite",
        "epreuve ecrite",
        "epreuve de production ecrite",
        "written production",
        "yazili uretim",
        "yazili anlatim",
    ),
    "CO": (
        "comprehension orale",
        "oral comprehension",
        "dinleme",
        "sozlu anlama",
    ),
    "CE": (
        "comprehension ecrite",
        "written comprehension",
        "okuma",
        "yazili anlama",
    ),
}


def _ascii_lower(text: str) -> str:
    """Aksanları sadeleştirip Türkçe noktasız/noktalı i farkını yumuşatır."""
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped.casefold().replace("ı", "i")


def build_where_clause(
    level: str | None = None,
    skill: str | None = None,
    doc_type: str | None = None,
    *,
    authoritative_only: bool = False,
) -> dict | None:
    """
    Açık metadata parametrelerinden ChromaDB `where` sözlüğü kurar.

    Geçersiz değerler (şema literal'lerinde olmayan) sessizce yok sayılır —
    çağıran yanlış bir filtre yüzünden hiç sonuç alamamaktansa filtresiz arar.
    Hiç geçerli koşul yoksa None döner (= filtre uygulama).
    """
    conditions: list[dict] = []
    if level and level in _VALID_LEVELS:
        conditions.append({"level": {"$eq": level}})
    if skill and skill in _VALID_SKILLS:
        conditions.append({"skill": {"$eq": skill}})
    if doc_type and doc_type in _VALID_DOC_TYPES:
        conditions.append({"doc_type": {"$eq": doc_type}})
    if authoritative_only:
        conditions.append({"is_authoritative": {"$eq": True}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def detect_filters(query: str) -> dict:
    """
    Sorgu metninden YALNIZCA kesin metadata token'larını tespit eder.

    Döner — build_where_clause'a verilebilecek {level?, skill?} sözlüğü.
    Belirsiz/tespit edilemeyen durumda {} döner (filtre uygulanmaz).
    Birden çok seviye/beceri geçiyorsa tutarsızlık var demektir → o alanı atlar.
    """
    out: dict[str, str] = {}

    levels = {m.group(1).upper() for m in _LEVEL_RE.finditer(query)}
    if len(levels) == 1:
        out["level"] = next(iter(levels))

    skills = set(_SKILL_RE.findall(query))
    normalized_query = _ascii_lower(query)
    for skill, phrases in _SKILL_PHRASES.items():
        if any(phrase in normalized_query for phrase in phrases):
            skills.add(skill)
    if len(skills) == 1:
        out["skill"] = next(iter(skills))

    return out


def merge_where(*clauses: dict | None) -> dict | None:
    """
    Birden çok `where` sözlüğünü tek bir AND koşulunda birleştirir.

    None'lar atlanır. Tek anlamlı sözlük kalırsa olduğu gibi döner; birden
    çoksa hepsini `$and` altında toplar. retrieval, legacy source_filter ile
    metadata_filter'ı tek where'e indirgemek için kullanır.
    """
    present = [c for c in clauses if c]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    # Her sözlüğü $and koşullarına aç; iç içe $and'leri düzleştir.
    conditions: list[dict] = []
    for clause in present:
        if "$and" in clause and len(clause) == 1:
            conditions.extend(clause["$and"])
        else:
            conditions.append(clause)
    return {"$and": conditions}


def matches_metadata(meta: dict, where: dict | None) -> bool:
    """
    Bir metadata sözlüğünün `where` koşulunu sağlayıp sağlamadığını Python
    tarafında değerlendirir. BM25 yolu (ChromaDB sorgusu yapmayan) bunu kullanır.

    Desteklenen biçimler: {"$and": [...]}, {alan: {"$eq": değer}}, {alan: değer}.
    Tanınmayan operatörlerde güvenli taraf: True (filtreyi sessizce gevşetir).
    """
    if not where:
        return True
    if "$and" in where and len(where) == 1:
        return all(matches_metadata(meta, cond) for cond in where["$and"])
    for field, condition in where.items():
        if isinstance(condition, dict):
            expected = condition.get("$eq")
            if expected is None:
                continue  # desteklenmeyen operatör — gevşet
            if meta.get(field) != expected:
                return False
        else:
            if meta.get(field) != condition:
                return False
    return True
