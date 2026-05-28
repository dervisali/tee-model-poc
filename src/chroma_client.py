"""
ChromaDB istemci tekili (singleton) — süreç başına tek PersistentClient.

Neden gerekli:
1. Doğruluk — Her retrieval çağrısı yeni bir `chromadb.PersistentClient` kurardı.
   Süreç haritası retrieval'ı 3 fazı PARALEL çalıştırdığından, soğuk başlangıçta
   (henüz hiç istemci kurulmamışken) birden çok thread aynı anda istemci kurmaya
   çalışıp paylaşılan Rust binding / default_tenant başlatmasında yarışır ve
   "Could not connect to tenant default_tenant" / "RustBindingsAPI ... bindings"
   hatalarıyla çöker. Tek istemciyi kilit altında bir kez kurmak bunu önler.
2. Hız — İstemci kurulumu (SQLite açma + sistem DB okuma) çağrı başına yüke
   yol açar. Tek uzun-ömürlü istemci her retrieval'da bu yükü ortadan kaldırır.

PersistentClient thread-safe sorgu içindir; tek paylaşılan istemci Chroma'nın
önerdiği kullanım biçimidir. Yeniden ingestion koleksiyonu silip yeniden
oluştursa da `get_or_create_collection` her çağrıda güncel koleksiyonu döndürür.
"""

from __future__ import annotations

import threading

import chromadb

from src.config import settings


_client: chromadb.ClientAPI | None = None
_lock = threading.Lock()


def get_chroma_client() -> chromadb.ClientAPI:
    """Süreç ömrü boyunca paylaşılan tek PersistentClient'i döndürür (lazy + kilitli)."""
    global _client
    if _client is None:
        with _lock:
            # Double-checked locking: soğuk-başlangıç paralel yarışında yalnızca
            # ilk thread istemciyi kurar; diğerleri kuruluşu bekler ve paylaşır.
            if _client is None:
                _client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    return _client


def get_child_collection() -> chromadb.Collection:
    """Aktif child (tee_children) koleksiyonunu paylaşılan istemci üzerinden döndürür."""
    return get_chroma_client().get_or_create_collection(
        name=settings.CHILD_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_chroma_client() -> None:
    """
    Paylaşılan istemciyi sıfırlar (test / yeniden ingestion sonrası temizlik).
    Bir sonraki get_chroma_client() çağrısı yeni istemci kurar.
    """
    global _client
    with _lock:
        _client = None
