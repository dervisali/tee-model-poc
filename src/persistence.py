"""
Kalıcılık (persistence) — ChromaDB korpusunu GCS anlık görüntüsü olarak yönet.

Cloud Run dosya sistemi geçicidir; her soğuk başlangıçta yerel ``chroma_db/``
silinir. Bu modül, korpusu **tek bir tar.gz nesnesi** olarak GCS'te tutar ve
konteyner açılışında yerel ephemeral diske indirip açar. ChromaDB böylece tam
POSIX semantiğine sahip yerel FS üzerinde, ``PersistentClient`` koduna hiç
dokunmadan çalışır (FUSE/SQLite kilit sorunları yok).

Korpus üç **karşılıklı bağımlı** dosyadan oluşur ve birlikte sürümlenmelidir:
  - ``chroma.sqlite3`` + HNSW dizini ``<uuid>/*.bin``  (child gömme vektörleri)
  - ``parents.json``                                   (parent metinleri / PDR)
  - ``bm25_index.pkl``                                 (seyrek/lexical indeks)
``process_map_cache.json`` yeniden üretilebilir bir çalışma-zamanı önbelleğidir
ve anlık görüntüye DAHİL EDİLMEZ.

Tasarım notları:
  - ``google.cloud.storage`` yalnızca fonksiyon içinde, tembel (lazy) import
    edilir; ``backend="local"`` iken bağımlılık hiç gerekmez.
  - Tüm GCS fonksiyonları test edilebilirlik için enjekte edilebilir bir
    ``storage_client`` parametresi alır (gerçek GCS/kimlik bilgisi gerekmez).
  - İndirme **atomiktir**: geçici dizine açılır, doğrulanır, sonra
    ``os.replace`` ile yerine taşınır (mevcut ``_save_parents`` deseniyle uyumlu).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

# Anlık görüntüye dahil EDİLMEYEN dosyalar (yeniden üretilebilir / sürüm dışı).
LOCAL_MANIFEST_NAME = "manifest.json"
_EXCLUDED_NAMES = {"process_map_cache.json", LOCAL_MANIFEST_NAME}

# GCS nesne adları.
TARBALL_NAME = "corpus.tar.gz"
MANIFEST_OBJECT_NAME = "manifest.json"
LATEST_POINTER_NAME = "latest.json"

MANIFEST_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Hatalar
# ---------------------------------------------------------------------------
class PersistenceError(Exception):
    """Kalıcılık katmanı genel hatası."""


class SnapshotNotFoundError(PersistenceError):
    """Uzak (GCS) anlık görüntü bulunamadı."""


class CorpusValidationError(PersistenceError):
    """İndirilen anlık görüntü çalışma-zamanı ayarlarıyla uyumsuz (model/boyut/sayım)."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusManifest:
    """Bir korpus anlık görüntüsünü tanımlayan değişmez meta veri."""

    schema_version: int
    snapshot_id: str
    embedding_model: str
    embedding_dim: int
    child_count: int
    parent_count: int
    git_sha: str | None
    created_at: str
    tarball_sha256: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "CorpusManifest":
        data = json.loads(raw)
        try:
            return cls(
                schema_version=int(data["schema_version"]),
                snapshot_id=str(data["snapshot_id"]),
                embedding_model=str(data["embedding_model"]),
                embedding_dim=int(data["embedding_dim"]),
                child_count=int(data["child_count"]),
                parent_count=int(data["parent_count"]),
                git_sha=(str(data["git_sha"]) if data.get("git_sha") is not None else None),
                created_at=str(data["created_at"]),
                tarball_sha256=str(data["tarball_sha256"]),
            )
        except KeyError as exc:
            raise CorpusValidationError(f"Manifest alanı eksik: {exc}") from exc


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_snapshot_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_sha() -> str | None:
    """En iyi çaba: mevcut commit SHA'sı (kısa). Başarısızsa None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(settings.BASE_DIR),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:  # pragma: no cover - ortam bağımlı
        return None


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_parents(parents_json: Path) -> int:
    if not parents_json.exists():
        return 0
    data = json.loads(parents_json.read_text(encoding="utf-8"))
    return len(data) if isinstance(data, dict) else 0


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Yeniden üretilebilir / sürüm dışı dosyaları tar'dan ayıkla."""
    if tarinfo.issym() or tarinfo.islnk():
        raise PersistenceError(f"Güvensiz tar üyesi (link desteklenmez): {tarinfo.name}")
    if tarinfo.isdev():
        raise PersistenceError(f"Güvensiz tar üyesi (özel dosya desteklenmez): {tarinfo.name}")
    parts = Path(tarinfo.name).parts
    name = Path(tarinfo.name).name
    if name in _EXCLUDED_NAMES or name.endswith(".tmp"):
        return None
    # Geçici/yedek dizinleri (.chroma_staging*, *.old) dahil etme.
    for p in parts:
        if p.startswith(".chroma_staging") or p.endswith(".old"):
            return None
    return tarinfo


# ---------------------------------------------------------------------------
# Anlık görüntü oluşturma (yerel)
# ---------------------------------------------------------------------------
def create_snapshot_tarball(chroma_dir: Path, dest_tar: Path) -> Path:
    """``chroma_dir`` içeriğini (filtrelenmiş) tek bir tar.gz'e atomik yazar."""
    chroma_dir = Path(chroma_dir)
    dest_tar = Path(dest_tar)
    if not chroma_dir.is_dir():
        raise PersistenceError(f"ChromaDB dizini yok: {chroma_dir}")
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_tar.with_name(dest_tar.name + ".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(str(chroma_dir), arcname=".", filter=_tar_filter)
    tmp.replace(dest_tar)
    return dest_tar


def build_manifest(
    chroma_dir: Path,
    *,
    snapshot_id: str,
    git_sha: str | None,
    child_count: int,
    parent_count: int,
    tarball_sha256: str,
) -> CorpusManifest:
    """Anlık görüntü meta verisini, çalışma-zamanı gömme ayarlarıyla birlikte üretir."""
    return CorpusManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        embedding_model=settings.EMBEDDING_MODEL,
        embedding_dim=int(settings.EMBEDDING_DIMENSION),
        child_count=int(child_count),
        parent_count=int(parent_count),
        git_sha=git_sha,
        created_at=_now_iso(),
        tarball_sha256=tarball_sha256,
    )


def make_snapshot(
    chroma_dir: Path,
    out_dir: Path,
    *,
    child_count: int,
    parent_count: int,
    snapshot_id: str | None = None,
) -> tuple[Path, CorpusManifest, Path]:
    """``out_dir`` içine corpus.tar.gz + manifest.json üretir. (tar, manifest, manifest_path)."""
    chroma_dir = Path(chroma_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_id = snapshot_id or _new_snapshot_id()

    tar_path = create_snapshot_tarball(chroma_dir, out_dir / TARBALL_NAME)
    manifest = build_manifest(
        chroma_dir,
        snapshot_id=snapshot_id,
        git_sha=_git_sha(),
        child_count=child_count,
        parent_count=parent_count,
        tarball_sha256=_sha256_file(tar_path),
    )
    manifest_path = out_dir / MANIFEST_OBJECT_NAME
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    return tar_path, manifest, manifest_path


# ---------------------------------------------------------------------------
# GCS istemcisi (tembel + enjekte edilebilir)
# ---------------------------------------------------------------------------
def _resolve_client(storage_client=None):
    """Enjekte edilen istemciyi döndürür; yoksa ``google.cloud.storage`` ile kurar."""
    if storage_client is not None:
        return storage_client
    try:
        from google.cloud import storage  # tembel import — local backend gerektirmez
    except ImportError as exc:  # pragma: no cover - bağımlılık yoksa
        raise PersistenceError(
            "google-cloud-storage kurulu değil; PERSISTENCE_BACKEND=gcs için gereklidir."
        ) from exc
    return storage.Client()


def _join(prefix: str, *parts: str) -> str:
    segs = [prefix.strip("/")] + [p.strip("/") for p in parts]
    return "/".join(s for s in segs if s)


# ---------------------------------------------------------------------------
# Yükleme (admin / ingestion sonrası)
# ---------------------------------------------------------------------------
def upload_snapshot(
    local_tar: Path,
    manifest: CorpusManifest,
    *,
    bucket: str,
    prefix: str,
    storage_client=None,
) -> str:
    """tar + manifest'i GCS'e yükler ve EN SON ``latest.json`` işaretçisini yazar.

    Döndürür: yüklenen tarball'ın nesne yolu (``prefix/<id>/corpus.tar.gz``).
    """
    client = _resolve_client(storage_client)
    bkt = client.bucket(bucket)
    snap_dir = _join(prefix, manifest.snapshot_id)
    tar_object = _join(snap_dir, TARBALL_NAME)
    manifest_object = _join(snap_dir, MANIFEST_OBJECT_NAME)

    bkt.blob(tar_object).upload_from_filename(str(local_tar))
    bkt.blob(manifest_object).upload_from_string(
        manifest.to_json(), content_type="application/json"
    )
    # latest.json EN SON yazılır → atomik yayın / geri alma.
    pointer = json.dumps(
        {
            "snapshot_id": manifest.snapshot_id,
            "tarball": tar_object,
            "manifest": manifest_object,
            "updated_at": _now_iso(),
        },
        ensure_ascii=False,
        indent=2,
    )
    bkt.blob(_join(prefix, LATEST_POINTER_NAME)).upload_from_string(
        pointer, content_type="application/json"
    )
    logger.info(
        "Anlık görüntü yüklendi.",
        extra={"event": "snapshot_uploaded", "snapshot_id": manifest.snapshot_id,
               "tar_object": tar_object, "bucket": bucket},
    )
    return tar_object


def publish_snapshot(
    chroma_dir: Path,
    *,
    bucket: str,
    prefix: str,
    child_count: int,
    parent_count: int,
    snapshot_id: str | None = None,
    storage_client=None,
) -> str:
    """Yerel ``chroma_dir``'den anlık görüntü üretip GCS'e yayınlar (yüksek seviye)."""
    with tempfile.TemporaryDirectory(prefix="tee-snapshot-") as tmp:
        tar_path, manifest, _ = make_snapshot(
            chroma_dir, Path(tmp),
            child_count=child_count, parent_count=parent_count, snapshot_id=snapshot_id,
        )
        return upload_snapshot(
            tar_path, manifest, bucket=bucket, prefix=prefix, storage_client=storage_client
        )


# ---------------------------------------------------------------------------
# İndirme / senkronizasyon (açılışta)
# ---------------------------------------------------------------------------
def resolve_remote_snapshot(
    *,
    bucket: str,
    prefix: str,
    pinned_object: str | None = None,
    storage_client=None,
) -> tuple[str, CorpusManifest]:
    """Hangi tarball'ın indirileceğini ve manifest'ini çözer. (tar_object, manifest)."""
    client = _resolve_client(storage_client)
    bkt = client.bucket(bucket)

    if pinned_object:
        tar_object = pinned_object.strip("/")
        manifest_object = _join(str(Path(tar_object).parent), MANIFEST_OBJECT_NAME)
    else:
        pointer_blob = bkt.blob(_join(prefix, LATEST_POINTER_NAME))
        if not pointer_blob.exists():
            raise SnapshotNotFoundError(
                f"latest.json bulunamadı: gs://{bucket}/{_join(prefix, LATEST_POINTER_NAME)}"
            )
        pointer = json.loads(pointer_blob.download_as_text())
        tar_object = str(pointer["tarball"]).strip("/")
        manifest_object = str(pointer.get("manifest")
                              or _join(str(Path(tar_object).parent), MANIFEST_OBJECT_NAME)).strip("/")

    manifest_blob = bkt.blob(manifest_object)
    if not manifest_blob.exists():
        raise SnapshotNotFoundError(f"Manifest bulunamadı: gs://{bucket}/{manifest_object}")
    manifest = CorpusManifest.from_json(manifest_blob.download_as_text())
    return tar_object, manifest


def _validate_against_settings(manifest: CorpusManifest, staging: Path) -> None:
    """İndirilen anlık görüntünün çalışma-zamanı ayarlarıyla uyumunu doğrular."""
    failures: list[str] = []
    if manifest.embedding_model != settings.EMBEDDING_MODEL:
        failures.append(
            f"embedding_model uyumsuz: anlık görüntü={manifest.embedding_model} "
            f"≠ çalışma-zamanı={settings.EMBEDDING_MODEL}"
        )
    if int(manifest.embedding_dim) != int(settings.EMBEDDING_DIMENSION):
        failures.append(
            f"embedding_dim uyumsuz: anlık görüntü={manifest.embedding_dim} "
            f"≠ çalışma-zamanı={settings.EMBEDDING_DIMENSION}"
        )
    actual_parents = _count_parents(staging / "parents.json")
    if actual_parents != manifest.parent_count:
        failures.append(
            f"parent sayımı uyumsuz: parents.json={actual_parents} "
            f"≠ manifest={manifest.parent_count}"
        )
    if failures:
        raise CorpusValidationError("; ".join(failures))


def download_and_extract(
    *,
    bucket: str,
    prefix: str,
    chroma_dir: Path,
    pinned_object: str | None = None,
    timeout_s: int = 120,
    storage_client=None,
) -> CorpusManifest:
    """GCS'ten anlık görüntüyü indirir, doğrular ve ``chroma_dir`` yerine atomik koyar."""
    chroma_dir = Path(chroma_dir)
    client = _resolve_client(storage_client)
    bkt = client.bucket(bucket)

    tar_object, manifest = resolve_remote_snapshot(
        bucket=bucket, prefix=prefix, pinned_object=pinned_object, storage_client=client
    )

    parent_dir = chroma_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    staging = parent_dir / ".chroma_staging"
    if staging.exists():
        _rmtree(staging)
    staging.mkdir(parents=True)

    tar_blob = bkt.blob(tar_object)
    if not tar_blob.exists():
        raise SnapshotNotFoundError(f"Tarball bulunamadı: gs://{bucket}/{tar_object}")

    with tempfile.NamedTemporaryFile(prefix="tee-corpus-", suffix=".tar.gz", delete=False) as tf:
        local_tar = Path(tf.name)
    try:
        # timeout, google-cloud-storage'da desteklenir; sahte istemcide yoksa görmezden gel.
        try:
            tar_blob.download_to_filename(str(local_tar), timeout=timeout_s)
        except TypeError:
            tar_blob.download_to_filename(str(local_tar))

        actual_sha = _sha256_file(local_tar)
        if actual_sha != manifest.tarball_sha256:
            raise CorpusValidationError(
                f"tarball sha256 uyumsuz: indirilen={actual_sha} ≠ manifest={manifest.tarball_sha256}"
            )

        with tarfile.open(local_tar, "r:gz") as tar:
            _safe_extractall(tar, staging)

        _validate_against_settings(manifest, staging)

        # manifest.json'u açılan korpusun yanına yaz (is_already_present için).
        (staging / LOCAL_MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")

        # Atomik yer değiştirme: varsa mevcut dizini kenara al, staging'i yerine koy.
        backup: Path | None = None
        if chroma_dir.exists():
            backup = parent_dir / f"{chroma_dir.name}.old-{_new_snapshot_id()}"
            os.replace(chroma_dir, backup)
        os.replace(staging, chroma_dir)
        if backup is not None:
            _rmtree(backup)
    finally:
        local_tar.unlink(missing_ok=True)
        if staging.exists():
            _rmtree(staging)

    logger.info(
        "Anlık görüntü indirildi ve açıldı.",
        extra={"event": "snapshot_restored", "snapshot_id": manifest.snapshot_id,
               "child_count": manifest.child_count, "parent_count": manifest.parent_count},
    )
    return manifest


# ---------------------------------------------------------------------------
# Yerel durum
# ---------------------------------------------------------------------------
def read_local_manifest(chroma_dir: Path) -> CorpusManifest | None:
    path = Path(chroma_dir) / LOCAL_MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return CorpusManifest.from_json(path.read_text(encoding="utf-8"))
    except (PersistenceError, json.JSONDecodeError):
        return None


def is_already_present(chroma_dir: Path, remote_manifest: CorpusManifest) -> bool:
    """Yerel korpus zaten uzak anlık görüntüyle aynı sürümdeyse True (indirmeyi atla)."""
    local = read_local_manifest(chroma_dir)
    if local is None:
        return False
    return (
        local.snapshot_id == remote_manifest.snapshot_id
        and local.tarball_sha256 == remote_manifest.tarball_sha256
    )


# ---------------------------------------------------------------------------
# Güvenli dosya işlemleri
# ---------------------------------------------------------------------------
def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Path-traversal'a karşı korumalı extractall (tar slip önlemi)."""
    dest = Path(dest).resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise PersistenceError(f"Güvensiz tar üyesi (link desteklenmez): {member.name}")
        if member.isdev():
            raise PersistenceError(f"Güvensiz tar üyesi (özel dosya desteklenmez): {member.name}")
        target = (dest / member.name).resolve()
        try:
            target.relative_to(dest)
        except ValueError:
            raise PersistenceError(f"Güvensiz tar üyesi (path traversal): {member.name}")
    # Python 3.12+ 'data' filtresi (ek güvenlik); 3.11'de parametre yok → geri düş.
    try:
        tar.extractall(dest, filter="data")  # noqa: S202 - üyeler yukarıda doğrulandı
    except TypeError:
        tar.extractall(dest)  # noqa: S202 - üyeler yukarıda doğrulandı
