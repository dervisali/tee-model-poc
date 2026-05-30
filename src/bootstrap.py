"""
Açılış bootstrap'ı — korpusu sunuma hazır hale getirir.

İki yerden çağrılır:
  1. Konteyner entrypoint'i: ``python -m src.bootstrap`` (Streamlit'ten ÖNCE).
     Senkronize eder + doğrular; ``REQUIRE_CORPUS_ON_STARTUP=true`` iken hazır
     değilse sıfırdan farklı kodla çıkar (Cloud Run revizyonu sağlıksız sayılır).
  2. ``app.py`` içinde ``@st.cache_resource`` ile (süreç başına bir kez):
     hazır değilse net bir "korpus kullanılamıyor" durumu gösterilir.

``PERSISTENCE_BACKEND="local"`` iken senkronizasyon adımı NO-OP'tur; davranış
mevcut yerel akışla birebir aynıdır.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from src.config import settings
from src.readiness_check import ReadinessResult, check_readiness

logger = logging.getLogger(__name__)


@dataclass
class BootstrapStatus:
    """Açılış senkronizasyonu + hazırlık kontrolünün sonucu."""

    backend: str
    synced: bool
    readiness: ReadinessResult
    error: str | None = None


def _reset_runtime_caches() -> None:
    """Aynı süreç içinde indirme sonrası önbellekleri sıfırla (retrieval taze veriyi görsün)."""
    try:
        from src.chroma_client import reset_chroma_client

        reset_chroma_client()
    except Exception:  # pragma: no cover - en iyi çaba
        logger.debug("reset_chroma_client atlandı.", exc_info=True)
    try:
        from src.retrieval import _load_parents

        _load_parents.cache_clear()
    except Exception:  # pragma: no cover
        logger.debug("_load_parents.cache_clear atlandı.", exc_info=True)
    # NOT: BM25Index.load ve parents tazelik-duyarlı anahtar kullanır; dosya
    # içeriği değişince otomatik yeniden yüklenir.


def _sync_from_gcs() -> bool:
    """GCS'ten gerekiyorsa indir. İndirme yapıldıysa True döner."""
    from src import persistence

    if not settings.GCS_BUCKET:
        raise RuntimeError("PERSISTENCE_BACKEND=gcs ama GCS_BUCKET ayarlı değil.")

    tar_object, remote_manifest = persistence.resolve_remote_snapshot(
        bucket=settings.GCS_BUCKET,
        prefix=settings.GCS_SNAPSHOT_PREFIX,
        pinned_object=settings.GCS_SNAPSHOT_OBJECT,
    )

    if persistence.is_already_present(settings.CHROMA_DIR, remote_manifest):
        logger.info(
            "Yerel korpus güncel (snapshot=%s); indirme atlanıyor.",
            remote_manifest.snapshot_id,
        )
        return False

    logger.info(
        "GCS'ten korpus indiriliyor (snapshot=%s, child=%d, parent=%d)...",
        remote_manifest.snapshot_id, remote_manifest.child_count, remote_manifest.parent_count,
    )
    persistence.download_and_extract(
        bucket=settings.GCS_BUCKET,
        prefix=settings.GCS_SNAPSHOT_PREFIX,
        chroma_dir=settings.CHROMA_DIR,
        pinned_object=tar_object,
        timeout_s=settings.GCS_DOWNLOAD_TIMEOUT_S,
    )
    _reset_runtime_caches()
    return True


def ensure_corpus_ready(run_smoke: bool = True) -> BootstrapStatus:
    """Korpusu sunuma hazırlar: (gerekirse) senkronize et → hazırlık kontrolü yap.

    ``run_smoke=False`` retrieval duman testini (Vertex çağrısı) atlar; UI'nin
    ucuz açılış güvencesi için uygundur.
    """
    backend = settings.PERSISTENCE_BACKEND
    synced = False
    error: str | None = None

    if backend == "gcs" and settings.SYNC_ON_STARTUP:
        try:
            synced = _sync_from_gcs()
        except Exception as exc:
            error = f"GCS senkronizasyonu başarısız: {exc}"
            logger.error(error, extra={"event": "gcs_sync_failed"})

    readiness = check_readiness(run_smoke=run_smoke)
    status = BootstrapStatus(backend=backend, synced=synced, readiness=readiness, error=error)

    if settings.REQUIRE_CORPUS_ON_STARTUP and not readiness.ok:
        detail = error or "; ".join(readiness.failures) or "bilinmeyen hata"
        raise RuntimeError(f"Korpus açılışta hazır değil (REQUIRE_CORPUS_ON_STARTUP=true): {detail}")

    return status


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    try:
        status = ensure_corpus_ready()
    except Exception as exc:
        logger.error("Bootstrap başarısız: %s", exc)
        sys.exit(1)

    logger.info(
        "Bootstrap tamam. backend=%s synced=%s ok=%s child=%d parent=%d",
        status.backend, status.synced, status.readiness.ok,
        status.readiness.child_count, status.readiness.parent_count,
    )
    if not status.readiness.ok:
        for f in status.readiness.failures:
            logger.error("FAIL: %s", f)
        # REQUIRE_CORPUS_ON_STARTUP=false ise uygulama açılır (app.py banner gösterir).
        sys.exit(1 if settings.REQUIRE_CORPUS_ON_STARTUP else 0)
    sys.exit(0)
