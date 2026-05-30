"""
Phase 3 — denetim (audit) günlüğü.

Denetimsiz bir sınav aracında her yanıt incelenebilir / itiraz edilebilir olmalı.
Bu modül her canlı sohbet turunu append-only JSONL olarak kalıcı yazar: zaman
damgası, (hash'lenmiş) session/user kimliği, sorgu, getirilen kaynak parent_id'leri,
yanıt, citation raporu ve güvenlik kararları.

PII yönetişimi (data governance):
  - `AUDIT_LOG_REDACT_PII` açıkken sorgu ve yanıt, mevcut `src/anonymizer.py`
    ile maskelenir (e-posta, telefon, TC kimlik vb.) — aday metinleri
    ("copies atypiques") yanlışlıkla loglanırsa kişisel veri sızmaz.
  - `session_id` ham saklanmaz; SHA-256 ile kısaltılmış hash olarak yazılır.
  - Günlükler `AUDIT_LOG_DIR` altında günlük dosyalara döner
    (audit_YYYY-MM-DD.jsonl) — saklama/erişim politikası dağıtımda dosya
    sistemi izinleri + yaşam döngüsü kuralıyla uygulanır (Phase 4/5).

Yazma DAYANIKLIDIR: hata olursa sessizce yutulur (audit, sohbet turunu asla
bozmamalı) ama stderr'e loglanır.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


def _hash_id(value: str | None) -> str | None:
    if not value:
        return None
    return "sid_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


# anonymizer.py telefon/TC/IBAN/isim maskeler ama E-POSTA maskelemez; audit
# kayıtlarında (örn. yanlışlıkla loglanan aday metni) e-posta da maskelenmeli.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def _redact(text: str) -> str:
    """PII maskeleme (bayrak açıkken): anonymizer (telefon/TC/IBAN/isim) + e-posta."""
    if not text or not settings.AUDIT_LOG_REDACT_PII:
        return text
    result = text
    try:
        from src.anonymizer import anonymize_text

        result = anonymize_text(result).get("anonymized_text", result)
    except Exception:  # noqa: BLE001 — anonymizer hatası audit'i bozmamalı
        logger.debug("audit PII redaction atlandı.", exc_info=True)
    return _EMAIL_RE.sub("[EPOSTA]", result)


def _audit_path() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Path(settings.AUDIT_LOG_DIR) / f"audit_{day}.jsonl"


def build_record(
    *,
    session_id: str | None,
    user_message: str,
    answer: str,
    sources: list[dict],
    citation_report: dict | None,
    input_verdict: dict | None,
    output_verdict: dict | None,
    language: str,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Bir denetim kaydı (dict) kurar — PII maskelenmiş, kimlik hash'lenmiş."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": _hash_id(session_id),
        "language": language,
        "query": _redact(user_message or ""),
        "answer": _redact(answer or ""),
        "source_parent_ids": [s.get("parent_id") for s in (sources or []) if s.get("parent_id")],
        "source_count": len(sources or []),
        "citation_passed": (citation_report or {}).get("passed"),
        "cited_parent_ids": (citation_report or {}).get("cited_parent_ids"),
        "input_verdict": input_verdict,
        "output_verdict": output_verdict,
        **(extra or {}),
    }


def write_record(record: dict, *, audit_dir: Path | None = None) -> Path | None:
    """Bir kaydı JSONL olarak append eder. Hata olursa None döner (sohbeti bozmaz)."""
    try:
        path = (Path(audit_dir) / record_filename()) if audit_dir else _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _write_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return path
    except Exception as exc:  # noqa: BLE001 — audit asla turu bozmamalı
        logger.error("Audit yazma başarısız: %s", exc)
        return None


def record_filename() -> str:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"audit_{day}.jsonl"


def log_turn(
    *,
    session_id: str | None,
    user_message: str,
    answer: str,
    sources: list[dict],
    citation_report: dict | None = None,
    input_verdict: dict | None = None,
    output_verdict: dict | None = None,
    language: str = "tr",
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Yüksek seviye: bayrak açıksa bir sohbet turunu denetim günlüğüne yazar."""
    if not settings.ENABLE_AUDIT_LOG:
        return None
    record = build_record(
        session_id=session_id, user_message=user_message, answer=answer,
        sources=sources, citation_report=citation_report,
        input_verdict=input_verdict, output_verdict=output_verdict,
        language=language, extra=extra,
    )
    return write_record(record)
