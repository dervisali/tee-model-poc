"""
Ollama LLM istemcisi — TEE-Model'in tek üretim noktası.

Bu modül, sistemdeki TÜM LLM çağrılarının tek geçtiği yerdir. Yeniden deneme
(retry), yapılandırılmış JSON çıktısı (response_format), zamanlama ve
loglamayı bir arada sağlar. Üretim, yargı ve enrichment çağrıları aynı API
üzerinden geçer; bu sayede Phase 3 sertleştirmesi (yapılandırılmış log,
asenkron kuyruk, yeniden deneme) tek dosyada uygulanır.

Karar gerekçesi:
- Ollama'nın `format=<json-schema>` parametresi v0.5+ ile birlikte gelir ve
  Pydantic'in `model_json_schema()` çıktısını doğrudan kabul eder. Bu sayede
  Gemini'deki `response_schema=PydanticModel` davranışı korunur.
- `tenacity` ile sarılmış üretim çağrısı, geçici Ollama bağlantı hatalarına
  karşı dayanıklılık sağlar (Phase 3.4).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import ollama
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# İstemci (lazy singleton — testlerde monkey-patch edilebilir olsun diye)
# ---------------------------------------------------------------------------

_client: ollama.Client | None = None


def get_ollama_client() -> ollama.Client:
    """Ortak Ollama istemcisini döndürür; ilk çağrıda oluşturulur."""
    global _client
    if _client is None:
        _client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _schema_to_format(schema: type[BaseModel] | dict | None) -> dict | None:
    """Pydantic modelini Ollama `format` parametresine uygun JSON şemasına çevirir."""
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
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
    retry=retry_if_exception_type((ollama.ResponseError, ConnectionError, TimeoutError)),
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
        Verilirse Ollama `format=<schema>` ile yapılandırılmış JSON döner.

    Döner
    -----
    str — Modelin ham metin yanıtı (response_format verildiyse JSON metni).
    """
    client = get_ollama_client()
    options: dict[str, Any] = {"temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE}

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    fmt = _schema_to_format(response_format)

    started = time.perf_counter()
    response = client.chat(
        model=model or settings.GENERATION_MODEL,
        messages=messages,
        options=options,
        format=fmt,
        stream=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    text = response["message"]["content"]
    logger.info(
        "LLM üretim tamamlandı",
        extra={
            "event": "llm_generation_complete",
            "model": model or settings.GENERATION_MODEL,
            "duration_ms": round(elapsed_ms, 1),
            "response_length": len(text),
            "structured": fmt is not None,
        },
    )
    return text


# ---------------------------------------------------------------------------
# Sağlık kontrolü
# ---------------------------------------------------------------------------

def is_ollama_available() -> bool:
    """Ollama servisi erişilebilir ve hedef model yüklü mü?"""
    try:
        client = get_ollama_client()
        models = client.list().get("models", [])
        names = {m.get("model", m.get("name", "")) for m in models}
        return any(settings.GENERATION_MODEL in n for n in names)
    except Exception as exc:
        logger.warning("Ollama erişilemedi: %s", exc)
        return False
