"""
TEE-Model içerik üreticileri — Ollama gemma3:4b üzerinde grounded üretim.

Tüm üreticiler kesin biçimde dayandırılmıştır (grounded): LLM'e yalnızca
retrieve edilen parent parçaları gönderilir; sistemin "BAĞLAM dışında bilgi
üretme" talimatı her çağrıda zorunludur.

Çıktılar Pydantic şemaları ile sınırlandırılır; Ollama'nın `format=`
yapılandırılmış JSON modu, modelin şemaya uyumlu JSON üretmesini garanti eder.
Önceki Gemini `response_schema` davranışı bire bir karşılanır.

MOCK_MODE açıkken hiçbir yerel veya uzak servise dokunulmaz; sabit fixture
çıktıları döner. Bu, hem CI ortamlarında hem de Streamlit UI testinde gerekli.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from src.config import settings
from src.llm import generate as llm_generate
from src.retrieval import build_context_text, retrieve_context


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grounding sistem talimatı
# ---------------------------------------------------------------------------

_GROUNDING_TEMPLATE = """\
Sen bir kamu kurumu eğitim içerik uzmanısın.
YALNIZCA aşağıda sağlanan bağlam belgelerini kullan.
Bağlamda yer almayan hiçbir bilgiyi üretme veya tahmin etme.
Eğer cevap bağlamda mevcut değilse, şunu yaz:
"Bu bilgi mevcut belgelerde yer almamaktadır."

BAĞLAM:
{context_text}

KAYNAK CHUNK İNDEKSLERİ: {chunk_indices}
"""


# ---------------------------------------------------------------------------
# Pydantic şemaları — yapılandırılmış çıktı için
# ---------------------------------------------------------------------------

class ProcessStep(BaseModel):
    adim_no: int
    baslik: str
    giris: str
    cikis: str
    karar_noktasi: str
    risk: str
    kontrol: str
    kaynak_chunk_indeksleri: list[str]


class ProcessMapSchema(BaseModel):
    steps: list[ProcessStep]


class ErrorCard(BaseModel):
    kart_no: int
    hata: str
    kok_neden: str
    tespit_yontemi: str
    dogru_uygulama: str
    kaynak_chunk_indeksleri: list[str]


class ErrorCardsSchema(BaseModel):
    hata_kartlari: list[ErrorCard]


class Term(BaseModel):
    terim: str
    tanim: str
    kullanim_ornegi: str
    kaynak_chunk_indeksleri: list[str]


class GlossarySchema(BaseModel):
    terimler: list[Term]


class SimulationOption(BaseModel):
    id: str
    metin: str
    dogru_mu: bool
    geri_bildirim: str
    sonuc: str


class SimulationSchema(BaseModel):
    senaryo_basligi: str
    durum_aciklamasi: str
    soru: str
    secenekler: list[SimulationOption]
    ogrenme_hedefi: str
    kaynak_chunk_indeksleri: list[str]


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

_MIN_CONTEXT_CHUNKS = 2
_OPTIMIZED_PROMPTS_PATH = settings.BASE_DIR / "optimized_prompts.json"


def _load_optimized_prompt(generator_key: str) -> dict | None:
    """Prompt Optimizer tarafından kaydedilen en iyi varyantı yükler."""
    if not _OPTIMIZED_PROMPTS_PATH.exists():
        return None
    try:
        data = json.loads(_OPTIMIZED_PROMPTS_PATH.read_text(encoding="utf-8"))
        return data.get(generator_key)
    except Exception:
        return None


def _build_grounded_prompt(
    query: str,
    extra_instruction: str,
    top_k: int = 7,
) -> tuple[str, str, list[str]]:
    """
    Sorgu için bağlam çek ve grounded prompt'u kompoze et.

    Yetersiz bağlam koruması: eşik altında çok az parça döndüyse, retrieval
    daha geniş eşikle yeniden çalıştırılır. Bu, ince bağlam durumlarını
    görünür kılar (sessizce halüsinasyona kaymaz).
    """
    chunks = retrieve_context(query, top_k=top_k)
    if len(chunks) < _MIN_CONTEXT_CHUNKS:
        logger.warning(
            "Yetersiz bağlam: '%s' sorgusu için %d chunk; eşik genişletiliyor.",
            query, len(chunks),
        )
        chunks = retrieve_context(
            query,
            top_k=top_k,
            distance_threshold=settings.RETRIEVAL_FALLBACK_THRESHOLD,
        )

    context_text, chunk_indices = build_context_text(chunks)
    system_instruction = _GROUNDING_TEMPLATE.format(
        context_text=context_text,
        chunk_indices=chunk_indices,
    )
    return system_instruction, extra_instruction, chunk_indices


# ---------------------------------------------------------------------------
# MOCK fixtures
# ---------------------------------------------------------------------------

_MOCK_PROCESS_MAP = {
    "steps": [
        {
            "adim_no": 1,
            "baslik": "Kadro Değişikliklerini Sisteme İşle",
            "giris": "Ay başı kadro bilgileri",
            "cikis": "Güncel kadro listesi",
            "karar_noktasi": "Değişiklik var mı?",
            "risk": "Eksik bildirim nedeniyle yanlış hesaplama",
            "kontrol": "Sisteme girilen veriler muhasebe birimiyle karşılaştırılır",
            "kaynak_chunk_indeksleri": ["explicit_mevzuat_p1", "explicit_mevzuat_p2"],
        },
        {
            "adim_no": 2,
            "baslik": "Göreve Başlama ve Ayrılış Bildirimlerini Tamamla",
            "giris": "Personel hareketleri listesi",
            "cikis": "Onaylı göreve başlama belgesi",
            "karar_noktasi": "Belge eksiksiz mi?",
            "risk": "Belgesiz personele maaş işlenmesi",
            "kontrol": "Özlük dosyasında belge varlığı kontrol edilir",
            "kaynak_chunk_indeksleri": ["explicit_mevzuat_p3"],
        },
        {
            "adim_no": 3,
            "baslik": "Ek Ödeme ve Kesintileri Gir",
            "giris": "İcra yazıları, tazminat kararları",
            "cikis": "Eksiksiz kesinti bordrosu",
            "karar_noktasi": "İcra 1/4 sınırını aşıyor mu?",
            "risk": "Yasal limit aşımı ve personel şikayeti",
            "kontrol": "Net aylığın 1/4 hesabı yapılır",
            "kaynak_chunk_indeksleri": ["explicit_mevzuat_p4"],
        },
    ]
}

_MOCK_ERROR_CARDS = {
    "hata_kartlari": [
        {
            "kart_no": 1,
            "hata": "Göreve başlama belgesi alınmadan maaş sisteme işlenmesi",
            "kok_neden": "Yeni mutemet, personelin fiziksel varlığını belge yerine geçerli sayıyor",
            "tespit_yontemi": "Denetimde özlük dosyasında göreve başlama belgesi bulunamaması",
            "dogru_uygulama": "Belge imzalanmadan maaş sisteme girilmez; belge süreci tamamlanana kadar beklenir",
            "kaynak_chunk_indeksleri": ["explicit_mevzuat_p3", "tacit_interview_clean_p1"],
        },
        {
            "kart_no": 2,
            "hata": "Ocak ayında kümülatif gelir vergisi matrahının sıfırlanmaması",
            "kok_neden": "Sistem otomatik sıfırlamaz; mutemet manuel adımı unutuyor",
            "tespit_yontemi": "Personel yanlış vergi diliminden vergi ödediğini fark edip şikayet ediyor",
            "dogru_uygulama": "Her yılın ilk bordrosunda kümülatif matrah sıfırlanır ve kayıt altına alınır",
            "kaynak_chunk_indeksleri": ["explicit_mevzuat_p1"],
        },
    ]
}

_MOCK_GLOSSARY = {
    "terimler": [
        {
            "terim": "Kümülatif Matrah",
            "tanim": "Yıl başından itibaren biriken gelir vergisi hesaplama tabanı; her Ocak ayında sıfırlanır.",
            "kullanim_ornegi": "Ocak bordrosunda kümülatif matrah sıfırlanmazsa vergi dilimi yanlış hesaplanır.",
            "kaynak_chunk_indeksleri": ["explicit_mevzuat_p1"],
        },
        {
            "terim": "İcra Kesintisi",
            "tanim": "Mahkeme veya icra müdürlüğü kararıyla maaştan yapılan yasal kesinti; net aylığın 1/4'ünü geçemez.",
            "kullanim_ornegi": "İcra kesintisi uygulamak için yazılı tebligat şarttır.",
            "kaynak_chunk_indeksleri": ["explicit_mevzuat_p1", "explicit_mevzuat_p4"],
        },
        {
            "terim": "Göreve Başlama Belgesi",
            "tanim": "Personelin kuruma ilk katıldığı günü resmi olarak belgeleyen, amir onaylı formdur.",
            "kullanim_ornegi": "Göreve başlama belgesi olmadan maaş sisteme işlenemez.",
            "kaynak_chunk_indeksleri": ["explicit_mevzuat_p3"],
        },
    ]
}

_MOCK_SIMULATION = {
    "senaryo_basligi": "Ay Ortasında İşe Başlayan Personelin Maaş Hesabı",
    "durum_aciklamasi": (
        "Kurum bünyesine 23 Mart tarihinde katılan Memur A'nın ilk maaşını hesaplamanız gerekiyor. "
        "Mart ayı 31 gün çekmektedir. Brüt maaşı 25.000 TL olarak belirlendi."
    ),
    "soru": "Bu personelin Mart ayı brüt maaşını nasıl hesaplarsınız?",
    "secenekler": [
        {
            "id": "A",
            "metin": "Brüt maaşı 31'e bölüp 9 ile çarparım. SGK Form 4A'yı 23 Mart tarihi ile bildiririm.",
            "dogru_mu": True,
            "geri_bildirim": "Doğru! Ay ortası işe girişte kısmi maaş hesabı bu şekilde yapılır.",
            "sonuc": "Personel doğru tutar üzerinden maaş alır.",
        },
        {
            "id": "B",
            "metin": "Tam ay maaşı öderim, zira personel resmi kadro listesinde bu ay yer alıyor.",
            "dogru_mu": False,
            "geri_bildirim": "Yanlış. Çalışılmayan günler için maaş ödenmez.",
            "sonuc": "Fazla ödeme nedeniyle iade süreci başlar.",
        },
        {
            "id": "C",
            "metin": "Bir sonraki ay toplu öderim; bu ay için herhangi bir işlem yapmam.",
            "dogru_mu": False,
            "geri_bildirim": "Yanlış. Personelin maaşı fiilen çalıştığı dönem için aynı ay ödenmelidir.",
            "sonuc": "Gecikmiş ödeme nedeniyle idari işlem başlatılabilir.",
        },
    ],
    "ogrenme_hedefi": "Ay ortası işe girişlerde kısmi maaş hesabı ve SGK Form 4A bildirim tarihini doğru uygulamak.",
    "kaynak_chunk_indeksleri": ["explicit_mevzuat_p2", "tacit_interview_clean_p3"],
}


# ---------------------------------------------------------------------------
# Tek geçişli yapılandırılmış üretim
# ---------------------------------------------------------------------------

def _structured_generate(
    *,
    query: str,
    instruction: str,
    schema: type[BaseModel],
    top_k: int,
) -> dict:
    """
    Grounded sistem talimatı + verilen şema ile tek LLM çağrısı yapar.
    JSON metnini parse edip dict olarak döndürür; başarısız olursa
    {"hata": ..., "ham_cikti": ...} döndürür.
    """
    system, user_prompt, chunk_indices = _build_grounded_prompt(query, instruction, top_k=top_k)
    logger.info("Yapılandırılmış üretim: chunk_count=%d", len(chunk_indices))
    raw = llm_generate(
        prompt=user_prompt,
        system=system,
        response_format=schema,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse hatası: %s", exc)
        return {"hata": str(exc), "ham_cikti": raw}


# ---------------------------------------------------------------------------
# Üreticiler
# ---------------------------------------------------------------------------

def generate_process_map() -> dict:
    """Maaş mutemetliği süreç haritasını grounded biçimde üretir."""
    if settings.MOCK_MODE:
        logger.info("MOCK_MODE: süreç haritası fixture döndürülüyor.")
        return _MOCK_PROCESS_MAP

    opt = _load_optimized_prompt("process_map")
    query = opt["query"] if opt else "maaş hesaplama adımları süreç akışı prosedür"
    instruction = opt["instruction"] if opt else (
        "Yukarıdaki bağlam belgelerine dayanarak maaş mutemetliği süreç haritasını oluştur. "
        "Her adım için kullandığın kaynak parent_id değerlerini kaynak_chunk_indeksleri "
        "alanına yaz. Sadece geçerli JSON döndür."
    )
    try:
        return _structured_generate(
            query=query,
            instruction=instruction,
            schema=ProcessMapSchema,
            top_k=8,
        )
    except Exception as exc:
        logger.error("Süreç haritası üretim hatası: %s", exc)
        return {"hata": str(exc), "ham_cikti": ""}


def generate_error_cards() -> dict:
    """Yaygın hatalar üzerine 'hata kartları' üretir."""
    if settings.MOCK_MODE:
        logger.info("MOCK_MODE: hata kartları fixture döndürülüyor.")
        return _MOCK_ERROR_CARDS

    opt = _load_optimized_prompt("error_cards")
    query = opt["query"] if opt else "sık yapılan hatalar yanlış uygulama kaçırılan adım"
    instruction = opt["instruction"] if opt else (
        "Yukarıdaki bağlam belgelerine dayanarak yeni maaş mutemedinin sık yaptığı hataları "
        "listele. Her kart için kullandığın kaynak parent_id değerlerini kaynak_chunk_indeksleri "
        "alanına yaz. Sadece geçerli JSON döndür."
    )
    try:
        return _structured_generate(
            query=query,
            instruction=instruction,
            schema=ErrorCardsSchema,
            top_k=8,
        )
    except Exception as exc:
        logger.error("Hata kartları üretim hatası: %s", exc)
        return {"hata": str(exc), "ham_cikti": ""}


def generate_glossary() -> dict:
    """Alan terimleri sözlüğünü üretir."""
    if settings.MOCK_MODE:
        logger.info("MOCK_MODE: terim sözlüğü fixture döndürülüyor.")
        return _MOCK_GLOSSARY

    opt = _load_optimized_prompt("glossary")
    query = opt["query"] if opt else "kuruma özgü terimler teknik kavramlar kısaltmalar"
    instruction = opt["instruction"] if opt else (
        "Yukarıdaki bağlam belgelerine dayanarak maaş mutemetliği alanına özgü terim sözlüğü oluştur. "
        "Her terim için kaynak parent_id değerlerini kaynak_chunk_indeksleri alanına yaz. "
        "Sadece geçerli JSON döndür."
    )
    try:
        return _structured_generate(
            query=query,
            instruction=instruction,
            schema=GlossarySchema,
            top_k=8,
        )
    except Exception as exc:
        logger.error("Terim sözlüğü üretim hatası: %s", exc)
        return {"hata": str(exc), "ham_cikti": ""}


def generate_simulation_scenario() -> dict:
    """Etkileşimli karar simülasyon senaryosu üretir."""
    if settings.MOCK_MODE:
        logger.info("MOCK_MODE: simülasyon fixture döndürülüyor.")
        return _MOCK_SIMULATION

    opt = _load_optimized_prompt("simulation")
    query = opt["query"] if opt else "kritik karar noktası yüksek hata riski zor durum"
    instruction = opt["instruction"] if opt else (
        "Yukarıdaki bağlam belgelerine dayanarak maaş mutemetliği için etkileşimli bir simülasyon "
        "senaryosu oluştur. Doğru cevap seçeneğinde dogru_mu = true olmalı. "
        "Kaynak parent_id değerlerini kaynak_chunk_indeksleri alanına yaz. "
        "Sadece geçerli JSON döndür."
    )
    try:
        return _structured_generate(
            query=query,
            instruction=instruction,
            schema=SimulationSchema,
            top_k=7,
        )
    except Exception as exc:
        logger.error("Simülasyon üretim hatası: %s", exc)
        return {"hata": str(exc), "ham_cikti": ""}
