"""
Yapılandırılmış log konfigürasyonu (Phase 3.2).

Tüm log kayıtları:
  - stdout'a okunaklı (renksiz) JSON satırları,
  - logs/tee_model_YYYY-MM-DD.jsonl dosyasına kalıcı,

biçiminde yazılır. Mevcut `logging.getLogger(__name__)` kullanan modüllerde
hiçbir değişiklik gerekmez; format ortak handler'lar üzerinden uygulanır.

Yapılandırılmış alanlar:
  - timestamp (UTC, ISO-8601)
  - level
  - logger
  - event   — log mesajının ilk argümanı veya extra="event" alanı
  - duration_ms / model / search_mode vb. — extra= ile geçen alanlar
  - exception (varsa traceback)

structlog opsiyoneldir: yüklü değilse stdlib JSON formatter ile çalışmaya
devam eder.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings


# stdlib LogRecord üzerinde özel bir alan adı kullanmıyoruz; "extra"
# anahtarları doğrudan record __dict__'e yazılır. JsonFormatter, varsayılan
# alanların DIŞINDA kalan her şeyi structured_fields olarak ekler.
_RESERVED_FIELDS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """Stdlib LogRecord'unu JSON satırına dönüştüren formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_FIELDS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


_CONFIGURED = False


def configure_logging(level: str | int = "INFO") -> None:
    """
    Tek seferlik log yapılandırması — birden fazla çağrıda no-op.

    JSON satırları stdout'a yazılır VE
    settings.LOGS_DIR/tee_model_YYYY-MM-DD.jsonl dosyasına eklenir
    (gece yarısında otomatik rotasyon).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    formatter = JsonFormatter()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)

    log_path = settings.LOGS_DIR / f"tee_model_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_path),
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Önceki handler'ları temizle (Streamlit yeniden çalıştırmada birikme önler)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(stream)
    root.addHandler(file_handler)

    # Üçüncü taraf gürültüsünü kıs
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "Loglama yapılandırıldı",
        extra={"event": "logging_configured", "log_file": str(log_path)},
    )
