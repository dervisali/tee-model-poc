"""
Kalıcılık katmanı testleri (Phase 1) — GCS anlık görüntü senkronizasyonu.

Gerçek GCS veya kimlik bilgisi GEREKMEZ: ``storage_client`` enjeksiyonu ile
bellek-içi sahte bir istemci kullanılır. Çoğu test düz dosyalarla çalışır
(chromadb gerekmez); yalnızca tek bir test gerçek bir ChromaDB tohumlar.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src import persistence as P


# ---------------------------------------------------------------------------
# Bellek-içi sahte GCS istemcisi
# ---------------------------------------------------------------------------
class FakeBlob:
    def __init__(self, store: dict, key: str):
        self.store, self.key = store, key

    def exists(self) -> bool:
        return self.key in self.store

    def upload_from_filename(self, path):
        self.store[self.key] = Path(path).read_bytes()

    def upload_from_string(self, data, content_type=None):
        self.store[self.key] = data.encode("utf-8") if isinstance(data, str) else data

    def download_to_filename(self, path, timeout=None):
        Path(path).write_bytes(self.store[self.key])

    def download_as_text(self) -> str:
        return self.store[self.key].decode("utf-8")


class FakeBucket:
    def __init__(self, store: dict):
        self.store = store

    def blob(self, key: str) -> FakeBlob:
        return FakeBlob(self.store, key)


class FakeClient:
    """Tek bucket'lı bellek-içi GCS taklidi."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self.store)


# ---------------------------------------------------------------------------
# Fixture'lar
# ---------------------------------------------------------------------------
def _seed_plain_corpus(root: Path, *, n_parents: int = 2) -> Path:
    """Korpusu taklit eden düz dosyalar (chromadb gerekmez)."""
    chroma = root / "chroma_db"
    chroma.mkdir(parents=True)
    parents = {f"p{i}": {"text": f"parent {i}"} for i in range(n_parents)}
    (chroma / "parents.json").write_text(json.dumps(parents), encoding="utf-8")
    (chroma / "bm25_index.pkl").write_bytes(b"\x00fake-bm25\x00" * 10)
    (chroma / "chroma.sqlite3").write_bytes(b"SQLite format 3\x00" + b"\x01" * 500)
    uuid_dir = chroma / "abc-uuid-collection"
    uuid_dir.mkdir()
    (uuid_dir / "data_level0.bin").write_bytes(b"\x07" * 2048)
    # Anlık görüntüye dahil EDİLMEMESİ gerekenler:
    (chroma / "process_map_cache.json").write_text("{}", encoding="utf-8")
    (chroma / "parents.json.tmp").write_text("garbage", encoding="utf-8")
    return chroma


@pytest.fixture
def plain_corpus(tmp_path) -> Path:
    return _seed_plain_corpus(tmp_path / "src_corpus")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def test_manifest_json_round_trip():
    m = P.CorpusManifest(
        schema_version=1, snapshot_id="s1",
        embedding_model="gemini-embedding-001", embedding_dim=3072,
        child_count=5, parent_count=2, git_sha="abc123",
        created_at="2026-05-29T00:00:00+00:00", tarball_sha256="deadbeef",
    )
    assert P.CorpusManifest.from_json(m.to_json()) == m


def test_manifest_from_json_missing_field_raises():
    with pytest.raises(P.CorpusValidationError):
        P.CorpusManifest.from_json(json.dumps({"snapshot_id": "x"}))


# ---------------------------------------------------------------------------
# Tarball
# ---------------------------------------------------------------------------
def test_tarball_excludes_cache_and_tmp(plain_corpus, tmp_path):
    import tarfile

    dest = tmp_path / "out" / P.TARBALL_NAME
    P.create_snapshot_tarball(plain_corpus, dest)
    with tarfile.open(dest, "r:gz") as tar:
        names = {Path(n).name for n in tar.getnames()}
    assert "parents.json" in names
    assert "bm25_index.pkl" in names
    assert "data_level0.bin" in names
    assert "process_map_cache.json" not in names  # çalışma-zamanı önbelleği
    assert "parents.json.tmp" not in names  # geçici dosya


# ---------------------------------------------------------------------------
# Yükle → indir round-trip (sahte GCS)
# ---------------------------------------------------------------------------
def test_publish_download_round_trip(plain_corpus, tmp_path):
    client = FakeClient()
    P.publish_snapshot(
        plain_corpus, bucket="b", prefix="tee-corpus",
        child_count=5, parent_count=2, storage_client=client,
    )
    # latest.json + manifest.json + corpus.tar.gz yüklendi
    assert any(k.endswith("latest.json") for k in client.store)
    assert any(k.endswith(P.TARBALL_NAME) for k in client.store)

    dest = tmp_path / "restored" / "chroma_db"
    manifest = P.download_and_extract(
        bucket="b", prefix="tee-corpus", chroma_dir=dest,
        timeout_s=30, storage_client=client,
    )
    assert manifest.parent_count == 2
    restored = {p.name for p in dest.rglob("*") if p.is_file()}
    assert {"parents.json", "bm25_index.pkl", "chroma.sqlite3", "data_level0.bin"} <= restored
    assert "process_map_cache.json" not in restored  # dışlama korunur
    assert (dest / P.LOCAL_MANIFEST_NAME).exists()  # yerel manifest yazıldı
    # HNSW ikili dosyası bayt-bayt geri geldi
    assert (dest / "abc-uuid-collection" / "data_level0.bin").read_bytes() == b"\x07" * 2048


def test_is_already_present_skips_redownload(plain_corpus, tmp_path):
    client = FakeClient()
    P.publish_snapshot(plain_corpus, bucket="b", prefix="tee-corpus",
                       child_count=5, parent_count=2, storage_client=client)
    dest = tmp_path / "restored" / "chroma_db"
    P.download_and_extract(bucket="b", prefix="tee-corpus", chroma_dir=dest,
                           timeout_s=30, storage_client=client)
    _, remote_manifest = P.resolve_remote_snapshot(
        bucket="b", prefix="tee-corpus", storage_client=client
    )
    assert P.is_already_present(dest, remote_manifest) is True
    # Farklı/boş hedef: present değil
    assert P.is_already_present(tmp_path / "empty", remote_manifest) is False


# ---------------------------------------------------------------------------
# Doğrulama / bütünlük
# ---------------------------------------------------------------------------
def _tamper_manifest(client: FakeClient, **changes) -> None:
    """Yüklenen manifest nesnesini değiştirir (sha256 tarball'ın aynısı kalır)."""
    key = next(k for k in client.store if k.endswith("manifest.json") and "latest" not in k)
    m = P.CorpusManifest.from_json(client.store[key].decode("utf-8"))
    tampered = dataclasses.replace(m, **changes)
    client.store[key] = tampered.to_json().encode("utf-8")


def test_download_rejects_sha_mismatch(plain_corpus, tmp_path):
    client = FakeClient()
    P.publish_snapshot(plain_corpus, bucket="b", prefix="tee-corpus",
                       child_count=5, parent_count=2, storage_client=client)
    _tamper_manifest(client, tarball_sha256="0" * 64)
    with pytest.raises(P.CorpusValidationError, match="sha256"):
        P.download_and_extract(bucket="b", prefix="tee-corpus",
                               chroma_dir=tmp_path / "r" / "chroma_db",
                               timeout_s=30, storage_client=client)


def test_download_rejects_embedding_dim_mismatch(plain_corpus, tmp_path):
    client = FakeClient()
    P.publish_snapshot(plain_corpus, bucket="b", prefix="tee-corpus",
                       child_count=5, parent_count=2, storage_client=client)
    _tamper_manifest(client, embedding_dim=768)  # ≠ settings 3072
    with pytest.raises(P.CorpusValidationError, match="embedding_dim"):
        P.download_and_extract(bucket="b", prefix="tee-corpus",
                               chroma_dir=tmp_path / "r" / "chroma_db",
                               timeout_s=30, storage_client=client)


def test_download_rejects_parent_count_mismatch(plain_corpus, tmp_path):
    client = FakeClient()
    P.publish_snapshot(plain_corpus, bucket="b", prefix="tee-corpus",
                       child_count=5, parent_count=2, storage_client=client)
    _tamper_manifest(client, parent_count=999)
    with pytest.raises(P.CorpusValidationError, match="parent"):
        P.download_and_extract(bucket="b", prefix="tee-corpus",
                               chroma_dir=tmp_path / "r" / "chroma_db",
                               timeout_s=30, storage_client=client)


def test_resolve_missing_snapshot_raises():
    client = FakeClient()  # boş store
    with pytest.raises(P.SnapshotNotFoundError):
        P.resolve_remote_snapshot(bucket="b", prefix="tee-corpus", storage_client=client)


def test_local_backend_does_not_import_gcs_sdk():
    """persistence import edildiğinde google-cloud-storage YÜKLENMEMELİ (tembel import)."""
    import sys

    assert "google.cloud.storage" not in sys.modules


# ---------------------------------------------------------------------------
# Gerçek ChromaDB ile round-trip (count doğrulaması)
# ---------------------------------------------------------------------------
@pytest.mark.requires_chromadb_seeded
def test_round_trip_preserves_chromadb_count(tmp_path):
    import chromadb

    src_chroma = tmp_path / "src" / "chroma_db"
    src_chroma.mkdir(parents=True)

    client = chromadb.PersistentClient(path=str(src_chroma))
    col = client.get_or_create_collection(name="tee_children", metadata={"hnsw:space": "cosine"})
    n = 6
    col.add(
        ids=[f"c{i}" for i in range(n)],
        embeddings=[[float(i), float(i + 1), float(i + 2)] for i in range(n)],
        documents=[f"chunk {i}" for i in range(n)],
        metadatas=[{"parent_id": f"p{i}"} for i in range(n)],
    )
    del client  # dosya kilidini bırak

    (src_chroma / "parents.json").write_text(
        json.dumps({f"p{i}": {"text": f"parent {i}"} for i in range(n)}), encoding="utf-8"
    )
    (src_chroma / "bm25_index.pkl").write_bytes(b"\x00bm25\x00")

    fake = FakeClient()
    P.publish_snapshot(src_chroma, bucket="b", prefix="tee-corpus",
                       child_count=n, parent_count=n, storage_client=fake)

    dest = tmp_path / "restored" / "chroma_db"
    P.download_and_extract(bucket="b", prefix="tee-corpus", chroma_dir=dest,
                           timeout_s=30, storage_client=fake)

    rclient = chromadb.PersistentClient(path=str(dest))
    rcol = rclient.get_or_create_collection(name="tee_children", metadata={"hnsw:space": "cosine"})
    assert rcol.count() == n
