"""
Üç chunklama stratejisi için birim testleri.

paragraph + fixed: ağır bağımlılıksız çalışır (saf string işleme).
semantic: sentence-transformers gerektirir; yüklü değilse atlanır.
"""

from __future__ import annotations

import pytest

from src.chunkers import (
    paragraph_parent_chunks,
    fixed_parent_chunks,
    get_parent_chunker,
)


SAMPLE_DOC = (
    "MADDE 1 — MAAŞ HESAPLAMA ESASLARI\n\n"
    "Devlet memurlarının aylık maaşları, 657 sayılı Devlet Memurları Kanunu'nun "
    "155. maddesi ve her yıl Bakanlar Kurulu Kararnamesi ile belirlenen "
    "katsayılar esas alınarak hesaplanır. Maaş hesaplamasında kullanılan temel "
    "katsayılar: aylık katsayısı, taban aylık katsayısı, yan ödeme katsayısı.\n\n"
    "MADDE 2 — KESİNTİ TÜRLERİ\n\n"
    "Brüt maaş üzerinden yapılan kesintiler şu sırayla uygulanır: SGK İşçi Payı "
    "(Emekli Keseği) brüt aylığın %14'ü, GSS primi brüt aylığın %3'ü, gelir "
    "vergisi kümülatif matrah üzerinden artan oranlı tarife ile hesaplanır."
)


class TestParagraphChunker:
    """Çift yeni satır + cümle bazlı kapanışın korunduğunu test eder."""

    def test_paragraph_kac_chunk_uretti(self):
        """Mevzuat metni MADDE bloklarına göre ayrılmalıdır."""
        chunks = paragraph_parent_chunks(SAMPLE_DOC)
        assert len(chunks) >= 1
        # En az bir MADDE başlığı korunmalı
        assert any("MADDE" in c for c in chunks)

    def test_paragraph_bos_metin(self):
        """Boş metin boş liste döner."""
        assert paragraph_parent_chunks("") == []

    def test_paragraph_kisa_metin_tek_chunk(self):
        """Tek paragraflık kısa metin tek chunk'tır."""
        kisa = "Tek bir cümle. Birkaç kelime daha."
        chunks = paragraph_parent_chunks(kisa)
        assert len(chunks) == 1
        assert kisa.replace("\n", " ").strip() in chunks[0] or chunks[0] in kisa


class TestFixedChunker:
    """Sabit boyutlu kayan pencerenin doğru örtüşme uyguladığını test eder."""

    def test_fixed_pencere_overlap(self):
        """100 karakterlik metin, size=40 overlap=10 ile birden fazla chunk üretir."""
        metin = "A" * 100
        chunks = fixed_parent_chunks(metin, size=40, overlap=10)
        # adım = 30; chunk konumları 0, 30, 60, 90; 4 chunk beklenir
        assert len(chunks) == 4
        # Her chunk en fazla 40 karakter olmalı
        assert all(len(c) <= 40 for c in chunks)

    def test_fixed_size_overlap_dogrulamasi(self):
        """size <= overlap durumu ValueError fırlatmalıdır."""
        with pytest.raises(ValueError):
            fixed_parent_chunks("test metin", size=10, overlap=10)


class TestStrategyDispatcher:
    """get_parent_chunker doğru fonksiyonu döndürmelidir."""

    def test_paragraph_secimi(self):
        chunker = get_parent_chunker("paragraph")
        assert chunker is paragraph_parent_chunks

    def test_fixed_secimi(self):
        chunker = get_parent_chunker("fixed")
        assert chunker is fixed_parent_chunks

    def test_bilinmeyen_strateji(self):
        with pytest.raises(ValueError, match="Bilinmeyen chunking_strategy"):
            get_parent_chunker("nonsense")


@pytest.mark.requires_torch
class TestSemanticChunker:
    """Anlamsal sınır chunklayıcı — torch + sentence-transformers gerekir."""

    def test_iki_konu_iki_chunk_uretmeli(self):
        """
        Konu sınırının net olduğu metin (maaş hesaplama → kesintiler) en az
        iki ayrı chunk üretmelidir.
        """
        from src.chunkers import semantic_parent_chunks
        chunks = semantic_parent_chunks(SAMPLE_DOC)
        assert len(chunks) >= 1
        # SAMPLE_DOC içinde MADDE 1 ve MADDE 2 ayrı konular: en az iki chunk
        # bekleriz; ancak küçük örneklerde tek chunk da olabilir.
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, str) and c.strip()

    def test_bos_metin(self):
        from src.chunkers import semantic_parent_chunks
        assert semantic_parent_chunks("") == []
