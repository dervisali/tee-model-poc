"""
Soğuk-başlangıç kabul testi (Phase 1) — OPT-IN, GERÇEK korpus gerektirir.

Kabul kriteri: uygulama, manuel re-ingest OLMADAN simüle bir soğuk başlangıçtan
sonra TAM korpusu (4.950 child / 1.302 parent) sunabilmeli.

Çalıştırma:
    RUN_COLDSTART_TEST=1 pytest tests/test_coldstart_integration.py -m requires_cold_start -v

Bu test GERÇEK ``chroma_db/``'ye DOKUNMAZ: salt-okunur anlık görüntü alır, BOŞ
bir geçici dizine geri yükler, çalışma zamanını oraya yönlendirir, doğrular ve
teardown'da gerçek yollara döner. Retrieval duman testi canlı Vertex AI gerektirir.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_cold_start

EXPECTED_CHILD = 4950
EXPECTED_PARENT = 1302


# --- bellek-içi sahte GCS (gerçek bucket/kimlik gerekmez) ---
class _FakeBlob:
    def __init__(self, store, key):
        self.store, self.key = store, key

    def exists(self):
        return self.key in self.store

    def upload_from_filename(self, path):
        self.store[self.key] = Path(path).read_bytes()

    def upload_from_string(self, data, content_type=None):
        self.store[self.key] = data.encode("utf-8") if isinstance(data, str) else data

    def download_to_filename(self, path, timeout=None):
        Path(path).write_bytes(self.store[self.key])

    def download_as_text(self):
        return self.store[self.key].decode("utf-8")


class _FakeBucket:
    def __init__(self, store):
        self.store = store

    def blob(self, key):
        return _FakeBlob(self.store, key)


class _FakeClient:
    def __init__(self):
        self.store = {}

    def bucket(self, name):
        return _FakeBucket(self.store)


def _clear_runtime_caches():
    from src.chroma_client import reset_chroma_client

    reset_chroma_client()
    try:
        from src.retrieval import _load_parents

        _load_parents.cache_clear()
    except Exception:
        pass
    try:
        from src.hybrid_search import BM25Index

        BM25Index.load.cache_clear()
    except Exception:
        pass


def test_cold_start_restores_and_serves_full_corpus(tmp_path, monkeypatch, request):
    from src.config import settings
    from src import persistence as P

    real_chroma = Path(settings.CHROMA_DIR)
    if not real_chroma.exists() or not (real_chroma / "parents.json").exists():
        pytest.skip("Gerçek korpus (chroma_db/) bulunamadı.")

    parents = json.loads((real_chroma / "parents.json").read_text(encoding="utf-8"))

    # 1. Gerçek korpusu (salt-okunur) anlık görüntüle → sahte GCS'e yayınla.
    fake = _FakeClient()
    P.publish_snapshot(
        real_chroma, bucket="b", prefix="tee-corpus",
        child_count=EXPECTED_CHILD, parent_count=len(parents), storage_client=fake,
    )

    # 2. BOŞ bir hedefe geri yükle (soğuk-başlangıç simülasyonu).
    dest = tmp_path / "restored" / "chroma_db"
    manifest = P.download_and_extract(
        bucket="b", prefix="tee-corpus", chroma_dir=dest,
        timeout_s=120, storage_client=fake,
    )
    assert manifest.parent_count == len(parents)

    # 3. Çalışma zamanını yeni dizine yönlendir (import-zamanı sabitleri dahil).
    import src.retrieval as retr
    import src.hybrid_search as hs
    import src.ingestion as ing

    monkeypatch.setattr(settings, "CHROMA_DIR", dest, raising=False)
    monkeypatch.setattr(retr, "CHROMA_DIR", dest, raising=False)
    monkeypatch.setattr(retr, "PARENTS_JSON", dest / "parents.json", raising=False)
    monkeypatch.setattr(hs, "BM25_INDEX_PATH", dest / "bm25_index.pkl", raising=False)
    monkeypatch.setattr(ing, "CHROMA_DIR", dest, raising=False)
    monkeypatch.setattr(ing, "PARENTS_JSON", dest / "parents.json", raising=False)

    _clear_runtime_caches()
    request.addfinalizer(_clear_runtime_caches)  # teardown: gerçek yollara dön

    # 4. Doğrula: tam korpus sunuluyor, manuel re-ingest YOK.
    from src.chroma_client import get_child_collection

    assert get_child_collection().count() == EXPECTED_CHILD

    from src.retrieval import _load_current_parents

    assert len(_load_current_parents()) == EXPECTED_PARENT

    # 5. Uçtan uca hazırlık (retrieval duman testi dahil — canlı Vertex gerektirir).
    from src.readiness_check import check_readiness

    result = check_readiness(run_smoke=True)
    assert result.ok, f"Hazırlık başarısız: {result.failures}"
    assert result.child_count == EXPECTED_CHILD
    assert result.parent_count == EXPECTED_PARENT
    assert result.smoke_hits and result.smoke_hits >= 1
