"""
Chunk metadata sözleşmesi — DELF/DALF examiner korpusu için merkezi şema.

Her loader, classifier ve ingestion bileşeni bu şemayı kullanır; ChromaDB
metadata sözlüğü bu modeli `model_dump()` ile düzleştirilerek yazılır.
Şema değişiklikleri tek noktadan yapılır; downstream kodlar yalnızca
buradan import eder.

ChromaDB sınırlaması: metadata değerleri yalnızca str/int/float/bool/None
olabilir. Pydantic `Literal` tipleri string olarak serileşir; `int | None`
chromadb>=0.5 ile desteklenir.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Type aliases — classifier ve filter modülleri de bu literal'leri import eder.
Level = Literal["A1", "A2", "B1", "B2", "C1", "general"]
Skill = Literal["PE", "PO", "CE", "CO", "general"]
DocType = Literal[
    "grille",
    "descripteur",
    "stagiaire",
    "sujet",
    "methodologie",
    "copie",
    "manuel",
    "presentation",
    "other",
]
Language = Literal["fr", "tr", "other"]


class ChunkMetadata(BaseModel):
    """Her ChromaDB child kaydının taşıdığı sabit metadata sözleşmesi."""

    source_file: str = Field(
        description="DATA_DIR'a göre rölatif dosya yolu (örn. 'b2/B2_Grille_PE.pdf').",
    )
    source_filename: str = Field(
        description="Dosya adı (basename).",
    )
    page: int | None = Field(
        default=None,
        description="PDF sayfa numarası veya PPTX slide numarası; DOCX/yalnız-imaj için None.",
    )
    level: Level = Field(
        description="CECRL seviyesi. Tanımlanamadığında 'general'.",
    )
    skill: Skill = Field(
        description="Beceri alanı: PE/PO/CE/CO. Tanımlanamadığında 'general'.",
    )
    doc_type: DocType = Field(
        description="Belge türü; sınıflandırılamadığında 'other'.",
    )
    language: Language = Field(
        default="fr",
        description="Belge dili. Korpus ağırlıklı olarak Fransızca.",
    )
    is_authoritative: bool = Field(
        default=True,
        description=(
            "Üretim retrieval'ında öncelikli kullanılır. brouillon/draft/atypique "
            "gibi referans-dışı materyaller False işaretlenir."
        ),
    )
    classifier_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Path classifier güven skoru. <0.8 manuel inceleme listesine girer.",
    )
