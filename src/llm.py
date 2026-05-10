"""
Vertex AI Gemini LLM istemcisi — TEE-Model'in tek üretim noktası.

Bu modül, sistemdeki TÜM LLM çağrılarının tek geçtiği yerdir. Yeniden deneme
(retry), yapılandırılmış JSON çıktısı (response_format), zamanlama ve
loglamayı bir arada sağlar. Üretim, yargı ve enrichment çağrıları aynı API
üzerinden geçer; bu sayede Phase 3 sertleştirmesi (yapılandırılmış log,
asenkron kuyruk, yeniden deneme) tek dosyada uygulanır.

Karar gerekçesi:
- Cloud Run üzerinde ayrı bir model sunucusu yönetmeden Vertex AI Gemini
  kullanılabilir.
- Google Gen AI SDK `response_schema` desteğiyle Pydantic şemaları üzerinden
  yapılandırılmış JSON çıktısı alınır.
- `tenacity` ile sarılmış üretim çağrısı, geçici Vertex AI bağlantı hatalarına
  karşı dayanıklılık sağlar (Phase 3.4).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

# `google-genai` import isteğe bağlıdır: testler ve hafif yardımcılar bu modülü
# yüklediğinde paket yüklü olmasa bile import etmek hata vermemeli.
try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai.types import HttpOptions  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    HttpOptions = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_health_cache: dict[str, float | bool] = {"checked_at": 0.0, "ok": False}
_HEALTH_CACHE_TTL_SECONDS = 300.0


# ---------------------------------------------------------------------------
# İstemci (lazy singleton — testlerde monkey-patch edilebilir olsun diye)
# ---------------------------------------------------------------------------

_client = None  # type: ignore[var-annotated]


def get_vertex_client():
    """Ortak Vertex AI Gemini istemcisini döndürür; ilk çağrıda oluşturulur."""
    global _client
    if genai is None or HttpOptions is None:
        raise ImportError(
            "google-genai paketi yüklü değil. Lütfen 'pip install google-genai' ile kurun."
        )
    if _client is None:
        kwargs = {
            "vertexai": True,
            "location": settings.GOOGLE_CLOUD_LOCATION,
            "http_options": HttpOptions(api_version="v1"),
        }
        if settings.GOOGLE_CLOUD_PROJECT:
            kwargs["project"] = settings.GOOGLE_CLOUD_PROJECT
        _client = genai.Client(**kwargs)
    return _client


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _is_retryable_exception(exc: BaseException) -> bool:
    """Yalnızca geçici Vertex AI / ağ hatalarında yeniden dene."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        from google.api_core import exceptions as google_exceptions
    except ImportError:  # pragma: no cover
        return False
    retryable = (
        google_exceptions.ServiceUnavailable,
        google_exceptions.ResourceExhausted,
        google_exceptions.DeadlineExceeded,
        google_exceptions.InternalServerError,
        google_exceptions.TooManyRequests,
    )
    return isinstance(exc, retryable)


def _schema_to_response_schema(schema: type[BaseModel] | dict | None) -> Any:
    """Pydantic modelini Gemini `response_schema` parametresine çevirir."""
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    raise TypeError(f"Beklenmeyen şema türü: {type(schema)!r}")


# ---------------------------------------------------------------------------
# Genel üretim
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
    wait=wait_exponential(
        multiplier=1,
        min=settings.LLM_RETRY_MIN_WAIT,
        max=settings.LLM_RETRY_MAX_WAIT,
    ),
    retry=retry_if_exception(_is_retryable_exception),
    reraise=True,
)
def generate(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    response_format: type[BaseModel] | dict | None = None,
) -> str:
    """
    Tek geçişli LLM çağrısı.

    Parametreler
    -----------
    prompt : str
        Kullanıcı istemi.
    system : str | None
        Varsa sistem talimatı (grounding bloğu).
    model : str | None
        Geçersiz kılma için model adı; varsayılan settings.GENERATION_MODEL.
    temperature : float | None
        Geçersiz kılma için sıcaklık; varsayılan settings.LLM_TEMPERATURE.
    response_format : Pydantic model | dict | None
        Verilirse Gemini `response_schema` ile yapılandırılmış JSON döner.

    Döner
    -----
    str — Modelin ham metin yanıtı (response_format verildiyse JSON metni).
    """
    client = get_vertex_client()
    config: dict[str, Any] = {
        "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
    }
    if system:
        config["system_instruction"] = system

    response_schema = _schema_to_response_schema(response_format)
    if response_schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = response_schema

    started = time.perf_counter()
    response = client.models.generate_content(
        model=model or settings.GENERATION_MODEL,
        contents=prompt,
        config=config,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    text = response.text or ""
    logger.info(
        "LLM üretim tamamlandı",
        extra={
            "event": "llm_generation_complete",
            "model": model or settings.GENERATION_MODEL,
            "duration_ms": round(elapsed_ms, 1),
            "response_length": len(text),
            "structured": response_schema is not None,
        },
    )
    return text


# ---------------------------------------------------------------------------
# Sağlık kontrolü
# ---------------------------------------------------------------------------

def is_vertex_available() -> bool:
    """Vertex AI Gemini erişilebilir mi? Billable generation çağrısı yapmaz."""
    now = time.time()
    if now - float(_health_cache["checked_at"]) < _HEALTH_CACHE_TTL_SECONDS:
        return bool(_health_cache["ok"])
    try:
        client = get_vertex_client()
        next(client.models.list(config={"page_size": 1}), None)
        _health_cache.update({"checked_at": now, "ok": True})
        return True
    except Exception as exc:
        _health_cache.update({"checked_at": now, "ok": False})
        logger.warning("Vertex AI erişilemedi: %s", exc)
        return False
