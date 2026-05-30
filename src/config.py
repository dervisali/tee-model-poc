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
    INFERENCE_BACKEND: Literal["vertex", "public_genai"] = Field(
        default="vertex",
        description=(
            "Hangi Gemini arka ucunun kullanılacağı. 'vertex' Vertex AI üzerinden "
            "ADC ile, 'public_genai' ise GOOGLE_API_KEY ile genai.google.com'a çağrı "
            "yapar. Aynı google-genai SDK her ikisini destekler."
        ),
    )

    GOOGLE_API_KEY: str | None = Field(
        default=None,
        description=(
            "INFERENCE_BACKEND='public_genai' iken kullanılan public Gemini API "
            "anahtarı. Vertex modunda yok sayılır."
        ),
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

    # -------------------------------------------------------------------
    # Reranking (LLM-as-reranker — ikinci-aşama precision)
    # -------------------------------------------------------------------
    ENABLE_RERANKING: bool = Field(
        default=False,
        description=(
            "Açıkken hibrit/dense füzyon sonrası adaylar LLM yargıç ile yeniden "
            "sıralanır (precision artışı; +1 LLM çağrısı gecikme). Varsayılan kapalı "
            "— gecikme optimizasyonlarını korumak için. RAGAS ile kazanç ölçülüp açılır."
        ),
    )
    RERANK_FETCH_K: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Reranking açıkken top_k'ye kırpılmadan önce yargıca verilen aday sayısı. "
            "Over-fetch: daha geniş aday havuzu → reranker daha iyi seçim yapar."
        ),
    )
    RERANK_MODEL: str | None = Field(
        default=None,
        description="Reranker LLM modeli; None ise GENERATION_MODEL kullanılır.",
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

    CORPUS_PRIMARY_LANGUAGE: Literal["fr", "tr"] = Field(
        default="fr",
        description=(
            "Korpus içeriğinin dili. BM25 indeks tokenizer'ı ve stop-word seçimi "
            "bu değere göre yapılır. DELF/DALF için 'fr', Türkçe mevzuat için 'tr'."
        ),
    )

    QUERY_LANGUAGE_AUTO_DETECT: bool = Field(
        default=True,
        description=(
            "Açıkken sorgu dili heuristic ile tespit edilir. Korpus dilinden "
            "farklı (cross-lingual) sorgularda davranış ENABLE_CROSS_LINGUAL_BM25 "
            "ile belirlenir. Dense yol her durumda orijinal sorgu ile çalışır "
            "(multilingual embedding)."
        ),
    )

    ENABLE_CROSS_LINGUAL_BM25: bool = Field(
        default=False,
        description=(
            "Cross-lingual (örn. TR sorgu → FR korpus) durumda BM25 için sorguyu "
            "korpus diline çevirip hibrit aramaya dahil eder. Varsayılan KAPALI: "
            "retrieval değerlendirmesi (evaluation/delf_questions.json) BM25+çevirinin "
            "TR sorgularda recall/MRR'a katkısı OLMADIĞINI, buna karşın çeviri LLM "
            "çağrısının ~0.6-1.1 sn gecikme eklediğini gösterdi. Kapalıyken cross-lingual "
            "sorgular dense-only'ye düşer (hızlı, eş/üstün kalite). Aynı-dil (FR) sorgular "
            "her durumda hibrit kalır — BM25 leksikal eşleşme orada sıralamayı iyileştirir."
        ),
    )

    OUTPUT_LANGUAGE: Literal["tr", "fr"] = Field(
        default="tr",
        description=(
            "Üretici fonksiyonların (process_map, error_cards, glossary, "
            "simulation) varsayılan çıktı dili. UI sidebar toggle bu değeri "
            "geçersiz kılabilir."
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
        default=_BASE_DIR / "evaluation" / "delf_questions.json"
    )
    RAGAS_RESULTS_DIR: Path = Field(
        default=_BASE_DIR / "evaluation" / "results"
    )

    # -------------------------------------------------------------------
    # Phase 2 — Deterministic recall@k raporu (kaynak-recall)
    #
    # retrieve_context'in döndürdüğü parent kaynaklarını, eval setindeki
    # expected_sources (gerçek korpus dosya adları) ile karşılaştırarak
    # recall@k / hit@k ölçer. RAGAS'tan bağımsız; LLM-yargıç GEREKTİRMEZ.
    # Yalnızca gömme (embedding) çağrıları yapılır (MOCK_MODE'da hiç çağrı yok).
    # -------------------------------------------------------------------
    RECALL_EVAL_KS: list[int] = Field(
        default=[1, 3, 5, 10],
        description="recall@k / hit@k raporunun hesaplandığı k değerleri.",
    )
    RECALL_M3_K: int | None = Field(
        default=None,
        description=(
            "M3 (top-source recall ≥%90) hangi k'de ölçülür. None ise "
            "RETRIEVAL_TOP_K kullanılır (varsayılan 5)."
        ),
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

    # -------------------------------------------------------------------
    # Kalıcılık (persistence) — ChromaDB anlık görüntüsünü GCS'te tut
    #
    # Cloud Run dosya sistemi geçicidir; her soğuk başlangıçta chroma_db/
    # silinir. backend="gcs" iken anlık görüntü (tek tar.gz) GCS'ten indirilip
    # yerel diske açılır; ChromaDB değişmeden yerel FS üzerinde çalışır.
    # Varsayılan "local" mevcut davranışı birebir korur (GCS bağımlılığı yok).
    # -------------------------------------------------------------------
    PERSISTENCE_BACKEND: Literal["local", "gcs"] = Field(
        default="local",
        description="'local' = yalnızca yerel disk (mevcut davranış); 'gcs' = açılışta GCS'ten senkronize et.",
    )
    GCS_BUCKET: str | None = Field(
        default=None,
        description="Anlık görüntü bucket adı. backend='gcs' iken zorunlu.",
    )
    GCS_SNAPSHOT_PREFIX: str = Field(
        default="tee-corpus",
        description="Bucket içindeki nesne ön eki (klasör).",
    )
    GCS_SNAPSHOT_OBJECT: str | None = Field(
        default=None,
        description="Belirli bir anlık görüntü tarball nesne yolunu sabitler; None ise latest.json takip edilir.",
    )
    SYNC_ON_STARTUP: bool = Field(
        default=True,
        description="backend='gcs' iken açılışta yerel DB boş/eksikse GCS'ten indir. 'local' iken etkisizdir.",
    )
    REQUIRE_CORPUS_ON_STARTUP: bool = Field(
        default=False,
        description="True ise korpus hazır değilken bootstrap hata verir (üretimde revizyon sağlıksız sayılır).",
    )
    GCS_DOWNLOAD_TIMEOUT_S: int = Field(
        default=120,
        ge=10,
        le=600,
        description="GCS anlık görüntü indirme zaman aşımı (saniye).",
    )

    model_config = SettingsConfigDict(
        env_file=str(_BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
