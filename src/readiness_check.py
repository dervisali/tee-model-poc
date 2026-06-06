"""
Açılış hazırlık kontrolü ve duman testi — DELF/DALF Sınav Görevlisi RAG.

Konteyner açılışında (Cloud Run) korpusun sağlıklı olduğunu doğrular:
  1. ChromaDB dizini var.
  2. parents.json var ve geçerli, boş olmayan bir sözlük.
  3. bm25_index.pkl var.
  4. ChromaDB koleksiyonu bağlanıyor ve boş değil.
  5. Hibrit retrieval çekirdeği derleniyor ve bir duman sorgusu çalışıyor.

Yapılandırılmış sonuç (``ReadinessResult``) döndürür; CLI, ``bootstrap`` ve
testler aynı fonksiyonu paylaşır. ``check_readiness()`` KISA DEVRE YAPMAZ — tüm
hataları toplar ki tek bir açılış logunda tüm sorunlar görünsün.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("readiness_check")

# Beklenen korpus boyutları (CLAUDE.md — bilgi amaçlı log için).
EXPECTED_CHILD_COUNT = 4950
EXPECTED_PARENT_COUNT = 1302

SMOKE_QUERY = "B2 PE"


@dataclass
class ReadinessResult:
    """Korpus hazırlık durumunun yapılandırılmış özeti."""

    ok: bool
    failures: list[str] = field(default_factory=list)
    child_count: int = 0
    parent_count: int = 0
    bm25_present: bool = False
    smoke_hits: int | None = None


def check_readiness(run_smoke: bool = True, chroma_dir: Path | None = None) -> ReadinessResult:
    """Korpusun sunuma hazır olup olmadığını doğrular (tüm hataları toplar).

    ``run_smoke=False`` ile retrieval duman testi (Vertex embedding çağrısı)
    atlanır — UI her yeniden-çalıştırmada ucuz bir varlık kontrolü için kullanır.
    ``chroma_dir`` verilirse dosya ve ChromaDB koleksiyon kontrolleri bu dizine
    göre yapılır; böylece snapshot yayınlama gibi akışlar tam olarak yayınlanacak
    veritabanını doğrular.
    """
    from src.config import settings

    failures: list[str] = []
    child_count = 0
    parent_count = 0
    bm25_present = False
    smoke_hits: int | None = None

    chroma_path = Path(chroma_dir or settings.CHROMA_DIR)
    if not chroma_path.exists() or not chroma_path.is_dir():
        failures.append(
            f"ChromaDB dizini yok veya geçersiz: '{chroma_path}'. "
            "GCS senkronizasyonu başarısız olmuş olabilir."
        )
        # Dizin yoksa diğer kontroller anlamsız — erken dön.
        return ReadinessResult(ok=False, failures=failures)

    # parents.json
    parents_json = chroma_path / "parents.json"
    if not parents_json.exists():
        failures.append(f"parents.json yok: '{parents_json}'.")
    else:
        try:
            data = json.loads(parents_json.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not data:
                failures.append("parents.json boş veya geçersiz şema (boş olmayan JSON nesnesi olmalı).")
            else:
                parent_count = len(data)
        except Exception as exc:
            failures.append(f"parents.json ayrıştırılamadı: {exc}")

    # bm25_index.pkl
    bm25_path = chroma_path / "bm25_index.pkl"
    bm25_present = bm25_path.exists()
    if not bm25_present:
        failures.append(f"bm25_index.pkl yok: '{bm25_path}'. Hibrit lexical arama çalışamaz.")

    # ChromaDB koleksiyonu
    try:
        if chroma_dir is None:
            from src.chroma_client import get_child_collection

            col = get_child_collection()
        else:
            import chromadb

            client = chromadb.PersistentClient(path=str(chroma_path))
            col = client.get_collection(settings.CHILD_COLLECTION_NAME)
        child_count = col.count()
        if child_count == 0:
            failures.append(f"Koleksiyon '{col.name}' boş. Uygulama ingest edilmiş veri gerektirir.")
    except Exception as exc:
        failures.append(f"ChromaDB koleksiyonuna bağlanılamadı: {exc}")

    # Retrieval duman testi (yalnızca veri varsa ve istenmişse)
    if run_smoke and child_count > 0 and parent_count > 0:
        try:
            from src.retrieval import retrieve_context

            hits = retrieve_context(SMOKE_QUERY, top_k=1)
            smoke_hits = len(hits)
            if not hits:
                failures.append("Retrieval duman testi boş sonuç döndürdü (DB bozuk/okunamaz olabilir).")
        except Exception as exc:
            failures.append(f"Retrieval duman testi hata verdi: {exc}")

    return ReadinessResult(
        ok=not failures,
        failures=failures,
        child_count=child_count,
        parent_count=parent_count,
        bm25_present=bm25_present,
        smoke_hits=smoke_hits,
    )


def run_readiness_checks() -> bool:
    """Geriye dönük uyumlu sarmalayıcı: yapılandırılmış kontrolü çalıştırır, loglar, bool döndürür."""
    logger.info("=== DELF/DALF RAG AÇILIŞ HAZIRLIK KONTROLLERİ BAŞLIYOR ===")
    result = check_readiness()

    if result.ok:
        logger.info(
            "PASS: Korpus hazır. child=%d (beklenen ~%d), parent=%d (beklenen ~%d), bm25=%s, duman_isabet=%s",
            result.child_count, EXPECTED_CHILD_COUNT,
            result.parent_count, EXPECTED_PARENT_COUNT,
            result.bm25_present, result.smoke_hits,
        )
        logger.info("=== TÜM AÇILIŞ HAZIRLIK KONTROLLERİ GEÇTİ. ===")
    else:
        for f in result.failures:
            logger.error("FAIL: %s", f)
        logger.error("SİSTEM HAZIR DEĞİL: %d kontrol başarısız.", len(result.failures))

    return result.ok


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    sys.exit(0 if run_readiness_checks() else 1)
