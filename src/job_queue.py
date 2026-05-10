"""
Üretim isteklerini seri sıraya koyan tek-işçili kuyruk (Phase 3.1).

Streamlit, paralel kullanıcı tıklamalarında üretici fonksiyonları aynı anda
çağırırsa Ollama bunları sıraya alamaz; tek-modelli yerel sunucu pratiğinde
genelde kuyrukta birikme sessizce bağlantı zaman aşımına döner.

Bu modül:
  - `concurrent.futures.ThreadPoolExecutor(max_workers=1)` ile tek bir
    arka plan işçi iş parçacığı tutar,
  - `submit_job(callable, *args, **kwargs) -> job_id`
  - `get_status(job_id) -> {"status": queued|running|done|error, "result": ...}`
  - `wait_for_job(job_id, timeout)` — Streamlit spinner içinde polling.

Kalıcılık YOK: kuyruk süreç ömrü boyunca yaşar. Birden fazla Streamlit
worker'ı varsa (gunicorn vb.), kuyruk uygulamadan ayrılmalıdır; POC'de tek
süreçli streamlit run yeterli.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


logger = logging.getLogger(__name__)


_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tee-gen")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def submit_job(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """
    İşi kuyruğa al; hemen job_id döndür. İş arka planda seri olarak çalışır.
    """
    job_id = uuid.uuid4().hex[:12]

    def _wrapper() -> Any:
        with _lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = time.time()
        logger.info(
            "İş başladı",
            extra={"event": "job_running", "job_id": job_id, "fn": fn.__name__},
        )
        return fn(*args, **kwargs)

    future: Future = _executor.submit(_wrapper)
    with _lock:
        _jobs[job_id] = {
            "status": "queued",
            "future": future,
            "submitted_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "fn_name": fn.__name__,
        }

    def _on_complete(fut: Future) -> None:
        with _lock:
            entry = _jobs.get(job_id)
            if entry is None:
                return
            entry["completed_at"] = time.time()
            try:
                entry["result"] = fut.result()
                entry["status"] = "done"
            except Exception as exc:
                entry["error"] = str(exc)
                entry["status"] = "error"
                logger.exception("İş başarısız", extra={"job_id": job_id})

    future.add_done_callback(_on_complete)
    logger.info(
        "İş kuyruğa alındı",
        extra={"event": "job_queued", "job_id": job_id, "fn": fn.__name__},
    )
    return job_id


def get_status(job_id: str) -> dict[str, Any]:
    """İşin son durumunu (sözlük olarak) döner."""
    with _lock:
        entry = _jobs.get(job_id)
        if entry is None:
            return {"status": "unknown", "job_id": job_id}
        return {
            "status": entry["status"],
            "job_id": job_id,
            "fn": entry["fn_name"],
            "result": entry["result"],
            "error": entry["error"],
            "submitted_at": entry["submitted_at"],
            "started_at": entry["started_at"],
            "completed_at": entry["completed_at"],
        }


def wait_for_job(
    job_id: str,
    *,
    timeout: float | None = None,
    poll_interval: float = 0.5,
) -> dict[str, Any]:
    """
    Streamlit spinner içinden çağrılmak üzere blocking bekleme.
    timeout aşılırsa son bilinen durumu döner ("running" olabilir).
    """
    deadline = (time.time() + timeout) if timeout else None
    while True:
        status = get_status(job_id)
        if status["status"] in ("done", "error", "unknown"):
            return status
        if deadline and time.time() >= deadline:
            return status
        time.sleep(poll_interval)


def queue_size() -> int:
    """Kuyruktaki bekleyen veya çalışan iş sayısını döner."""
    with _lock:
        return sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
