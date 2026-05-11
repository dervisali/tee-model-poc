"""
Merkezi yapılandırma — TEE-Model üretim RAG sistemi.

Tüm modeller, eşikler, özellik bayrakları ve servis URL'leri burada toplanır.
.env dosyasından okunur; kodun başka hiçbir yerinde sabit değer (magic number)
bulunmamalıdır. Pydantic-Settings, tipleri ve aralıkları doğrular.

Kullanım:
    from src.config import settings
    if settings.ENABLE_CONTEXTUAL_ENRICHMENT:
        ...
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """TEE-Model çalışma zamanı yapılandırması."""

    # -------------------------------------------------------------------
    # Modeller
    # -------------------------------------------------------------------
    GENERATION_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Vertex AI Gemini üretim modeli adı.",
    )
    EMBEDDING_MODEL: str = Field(
        default="gemini-embedding-001",
        description="Vertex AI text embedding modeli (varsayılan 3072 boyut).",
    )
    EMBEDDING_DIMENSION: Literal[768, 1536, 3072] = Field(
        default=3072,
        description="gemini-embedding-001 output_dimensionality değeri.",
    )
    GOOGLE_CLOUD_PROJECT: str | None = Field(
        default=None,
        description="Vertex AI projesi. Boşsa Google Gen AI SDK ortam değişkenini kullanır.",
    )
    GOOGLE_CLOUD_LOCATION: str = Field(
        default="us-central1",
        description="Vertex AI bölgesi. Cloud Run servisinizle aynı bölge önerilir.",
    )

    # -------------------------------------------------------------------
    # RAG parametreleri
    # -------------------------------------------------------------------
    CHUNK_SIZE_PARENT: int = Field(default=700, ge=200, le=2000)
    CHUNK_SIZE_PARENT_MAX: int = Field(default=800, ge=200, le=2000)
    CHUNK_SIZE_PARENT_MIN: int = Field(default=300, ge=100, le=1000)
    CHUNK_SIZE_CHILD: int = Field(default=200, ge=50, le=500)
    CHUNK_SIZE_CHILD_MIN: int = Field(default=50, ge=20, le=200)

    RETRIEVAL_TOP_K: int = Field(default=5, ge=1, le=20)
    RETRIEVAL_DISTANCE_THRESHOLD: float = Field(default=0.7, ge=0.0, le=2.0)
    RETRIEVAL_FALLBACK_THRESHOLD: float = Field(default=1.2, ge=0.0, le=2.0)

    HYBRID_ALPHA: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Yoğun (dense) ağırlığı; (1-alpha) seyrek (BM25) ağırlığıdır.",
    )
    HYBRID_RRF_K: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion sabiti (Cormack 2009).",
    )

    DEDUP_SIMILARITY_THRESHOLD: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Bu eşiği aşan yeni chunk'lar mevcutla aynı sayılır ve atlanır.",
    )

    SEMANTIC_SIM_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Anlamsal chunklamada cümleler arası kosinüs eşiği.",
    )

    CHUNKING_STRATEGY: str = Field(
        default="paragraph",
        description="'paragraph' | 'semantic' | 'fixed'. Mevcut belge davranışı = paragraph.",
    )

    CORPUS_LANGUAGE: Literal["fr", "tr"] = Field(
        default="fr",
        description=(
            "Aktif korpusun dili. BM25 tokenizer ve stop-word seçimi bu değere "
            "göre yapılır. DELF/DALF korpusu için 'fr', Türkçe mevzuat için 'tr'."
        ),
    )

    # -------------------------------------------------------------------
    # LLM çağrı parametreleri
    # -------------------------------------------------------------------
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    LLM_MAX_RETRIES: int = Field(default=3, ge=1, le=10)
    LLM_RETRY_MIN_WAIT: float = Field(default=2.0, ge=0.1)
    LLM_RETRY_MAX_WAIT: float = Field(default=10.0, ge=1.0)

    # -------------------------------------------------------------------
    # Özellik bayrakları
    # -------------------------------------------------------------------
    MOCK_MODE: bool = Field(default=False)
    ENABLE_QUERY_REWRITING: bool = Field(default=False)
    ENABLE_CONTEXTUAL_ENRICHMENT: bool = Field(default=True)
    ENABLE_CONFIDENCE_SCORING: bool = Field(default=True)
    ENABLE_HYBRID_SEARCH: bool = Field(default=True)
    ENABLE_DEDUPLICATION: bool = Field(default=True)

    # -------------------------------------------------------------------
    # Değerlendirme (RAGAS)
    # -------------------------------------------------------------------
    RAGAS_ENABLED: bool = Field(default=False)
    RAGAS_TEST_SET_PATH: Path = Field(
        default=_BASE_DIR / "evaluation" / "test_questions.json"
    )
    RAGAS_RESULTS_DIR: Path = Field(
        default=_BASE_DIR / "evaluation" / "results"
    )

    # -------------------------------------------------------------------
    # Yollar (paths)
    # -------------------------------------------------------------------
    BASE_DIR: Path = Field(default=_BASE_DIR)
    DATA_DIR: Path = Field(default=_BASE_DIR / "data")
    PROCESSED_DIR: Path = Field(default=_BASE_DIR / "processed")
    CHROMA_DIR: Path = Field(default=_BASE_DIR / "chroma_db")
    LOGS_DIR: Path = Field(default=_BASE_DIR / "logs")

    # -------------------------------------------------------------------
    # Koleksiyon adları
    # -------------------------------------------------------------------
    CHILD_COLLECTION_NAME: str = Field(default="tee_children")

    model_config = SettingsConfigDict(
        env_file=str(_BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
