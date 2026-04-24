"""
Content generation module for TEE-Model POC.

All generators are strictly grounded: the LLM is instructed to use ONLY the
retrieved context chunks and cite their indices. Includes JSON retry logic
and mock-mode fixtures for UI testing without API quota consumption.
"""

import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.retrieval import retrieve_context, build_context_text

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
GENERATION_MODEL = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Grounding system prompt (injected into every generator call)
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
# Shared helpers
# ---------------------------------------------------------------------------


def _get_genai_client() -> genai.Client:
    """Initialise and return the google-genai client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY ortam değişkeni bulunamadı. "
            "Lütfen .env dosyasını kontrol edin."
        )
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Output
# ---------------------------------------------------------------------------

class ProcessStep(BaseModel):
    adim_no: int
    baslik: str
    giris: str
    cikis: str
    karar_noktasi: str
    risk: str
    kontrol: str
    kaynak_chunk_indeksleri: list[int]

class ProcessMapSchema(BaseModel):
    steps: list[ProcessStep]

class ErrorCard(BaseModel):
    kart_no: int
    hata: str
    kök_neden: str
    tespit_yöntemi: str
    dogru_uygulama: str
    kaynak_chunk_indeksleri: list[int]

class ErrorCardsSchema(BaseModel):
    hata_kartlari: list[ErrorCard]

class Term(BaseModel):
    terim: str
    tanim: str
    kullanim_ornegi: str
    kaynak_chunk_indeksleri: list[int]

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
    kaynak_chunk_indeksleri: list[int]


_MIN_CONTEXT_CHUNKS = 2       # Warn if retrieval returns fewer than this many chunks.
_FALLBACK_THRESHOLD = 1.2     # Widened distance threshold used on retry.

_OPTIMIZED_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "optimized_prompts.json"


def _load_optimized_prompt(generator_key: str) -> dict | None:
    """
    Return saved optimized {query, instruction} for a generator, or None if not saved.

    Reads from optimized_prompts.json written by the Prompt Optimizer tab.
    """
    if not _OPTIMIZED_PROMPTS_PATH.exists():
        return None
    try:
        data = json.loads(_OPTIMIZED_PROMPTS_PATH.read_text(encoding="utf-8"))
        return data.get(generator_key)
    except Exception:
        return None


def _build_grounded_prompt(query: str, extra_instruction: str, top_k: int = 7) -> tuple[str, str, list[int]]:
    """
    Retrieve context for a query and compose a grounded prompt.

    Includes a retrieval health check: if fewer than _MIN_CONTEXT_CHUNKS pass
    the default distance threshold, the call is retried with a wider threshold
    (_FALLBACK_THRESHOLD) and a warning is logged. This surfaces thin-context
    situations before they silently become hallucinated or "not found" output.

    Parameters
    ----------
    query : str
        Semantic search query (Turkish).
    extra_instruction : str
        Task-specific instruction appended after the grounding block.
    top_k : int
        Number of context chunks to retrieve.

    Returns
    -------
    tuple[str, str, list[int]]
        - System instruction string.
        - User prompt string.
        - List of chunk indices used (for citation).
    """
    chunks = retrieve_context(query, top_k=top_k)

    if len(chunks) < _MIN_CONTEXT_CHUNKS:
        logger.warning(
            "Yetersiz bağlam: '%s' sorgusu için yalnızca %d chunk döndü "
            "(eşik: varsayılan). Daha geniş eşikle (%.1f) yeniden deneniyor.",
            query, len(chunks), _FALLBACK_THRESHOLD,
        )
        chunks = retrieve_context(query, top_k=top_k, distance_threshold=_FALLBACK_THRESHOLD)
        logger.warning(
            "Geniş eşikle %d chunk döndü. İçerik kalitesi düşük olabilir; "
            "sonuçları dikkatle inceleyin.", len(chunks),
        )

    context_text, chunk_indices = build_context_text(chunks)
    system_instruction = _GROUNDING_TEMPLATE.format(
        context_text=context_text,
        chunk_indices=chunk_indices,
    )
    user_prompt = extra_instruction
    return system_instruction, user_prompt, chunk_indices


# ---------------------------------------------------------------------------
# Mock fixtures
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
            "kaynak_chunk_indeksleri": [0, 1],
        },
        {
            "adim_no": 2,
            "baslik": "Göreve Başlama ve Ayrılış Bildirimlerini Tamamla",
            "giris": "Personel hareketleri listesi",
            "cikis": "Onaylı göreve başlama belgesi",
            "karar_noktasi": "Belge eksiksiz mi?",
            "risk": "Belgesiz personele maaş işlenmesi",
            "kontrol": "Özlük dosyasında belge varlığı kontrol edilir",
            "kaynak_chunk_indeksleri": [2],
        },
        {
            "adim_no": 3,
            "baslik": "Ek Ödeme ve Kesintileri Gir",
            "giris": "İcra yazıları, tazminat kararları",
            "cikis": "Eksiksiz kesinti bordrosu",
            "karar_noktasi": "İcra 1/4 sınırını aşıyor mu?",
            "risk": "Yasal limit aşımı ve personel şikayeti",
            "kontrol": "Net aylığın 1/4 hesabı yapılır",
            "kaynak_chunk_indeksleri": [1, 3],
        },
        {
            "adim_no": 4,
            "baslik": "Bordroyu Amir Onayına Sun",
            "giris": "Tamamlanmış bordro taslağı",
            "cikis": "Onaylı bordro",
            "karar_noktasi": "Onay alındı mı?",
            "risk": "Geç onay nedeniyle maaş gecikmesi",
            "kontrol": "Onay tarihi 10. günü aşmıyor mu?",
            "kaynak_chunk_indeksleri": [2],
        },
        {
            "adim_no": 5,
            "baslik": "Bordroyu Muhasebe Birimine İlet",
            "giris": "Onaylı bordro",
            "cikis": "Muhasebe fişi ve ödeme emri",
            "karar_noktasi": "Son teslim tarihi (9. gün 17:00) geçildi mi?",
            "risk": "Geç teslim halinde maaş gecikmesi",
            "kontrol": "Teslim saati muhasebe birimi iç akışıyla uyumlu mu?",
            "kaynak_chunk_indeksleri": [0, 4],
        },
    ]
}

_MOCK_ERROR_CARDS = {
    "hata_kartlari": [
        {
            "kart_no": 1,
            "hata": "Göreve başlama belgesi alınmadan maaş sisteme işlenmesi",
            "kök_neden": "Yeni mutemet, personelin fiziksel varlığını belge yerine geçerli sayıyor",
            "tespit_yöntemi": "Denetimde özlük dosyasında göreve başlama belgesi bulunamaması",
            "dogru_uygulama": "Belge imzalanmadan maaş sisteme girilmez; belge süreci tamamlanana kadar beklenir",
            "kaynak_chunk_indeksleri": [2, 5],
        },
        {
            "kart_no": 2,
            "hata": "Ocak ayında kümülatif gelir vergisi matrahının sıfırlanmaması",
            "kök_neden": "Sistem otomatik sıfırlamaz; mutemet manuel adımı unutuyor",
            "tespit_yöntemi": "Personel yanlış vergi diliminden vergi ödediğini fark edip şikayet ediyor",
            "dogru_uygulama": "Her yılın ilk bordrosunda kümülatif matrah sıfırlanır ve kayıt altına alınır",
            "kaynak_chunk_indeksleri": [1],
        },
        {
            "kart_no": 3,
            "hata": "İcra kesintisinde net aylığın 1/4 sınırının aşılması",
            "kök_neden": "Birden fazla icra kararı varken toplam kontrol yapılmaması",
            "tespit_yöntemi": "Personelden idari şikayet; hukuk birimi incelemesi",
            "dogru_uygulama": "Tüm icra kesintilerinin toplamı net aylığın 1/4'ünü geçmemeli; sıra ile uygulanmalı",
            "kaynak_chunk_indeksleri": [1, 3],
        },
    ]
}

_MOCK_GLOSSARY = {
    "terimler": [
        {
            "terim": "Kümülatif Matrah",
            "tanim": "Yıl başından itibaren biriken gelir vergisi hesaplama tabanı; her Ocak ayında sıfırlanır.",
            "kullanim_ornegi": "Ocak bordrosunda kümülatif matrah sıfırlanmazsa vergi dilimi yanlış hesaplanır.",
            "kaynak_chunk_indeksleri": [1],
        },
        {
            "terim": "İcra Kesintisi",
            "tanim": "Mahkeme veya icra müdürlüğü kararıyla maaştan yapılan yasal kesinti; net aylığın 1/4'ünü geçemez.",
            "kullanim_ornegi": "İcra kesintisi uygulamak için yazılı tebligat şarttır.",
            "kaynak_chunk_indeksleri": [1, 3],
        },
        {
            "terim": "Göreve Başlama Belgesi",
            "tanim": "Personelin kuruma ilk katıldığı günü resmi olarak belgeleyen, amir onaylı formdur.",
            "kullanim_ornegi": "Göreve başlama belgesi olmadan maaş sisteme işlenemez.",
            "kaynak_chunk_indeksleri": [2],
        },
        {
            "terim": "Aylık Katsayısı",
            "tanim": "Maaş göstergesi rakamının çarpıldığı ve her yıl Bakanlar Kurulu ile belirlenen katsayı.",
            "kullanim_ornegi": "Katsayı geç girilirse eksik ödeme ve fark mahsubu zorunlu olur.",
            "kaynak_chunk_indeksleri": [0],
        },
        {
            "terim": "Form-ADB-01",
            "tanim": "Aile Durum Bildirimi formu; medeni hal, çocuk sayısı ve bakmakla yükümlü kişileri gösterir.",
            "kullanim_ornegi": "Evlilik durumunda 30 gün içinde ADB-01 güncellenmesi zorunludur.",
            "kaynak_chunk_indeksleri": [2],
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
            "metin": "Brüt maaşı 31'e bölüp 9 ile çarparım (23 Mart'tan 31 Mart'a kadar 9 gün). SGK Form 4A'yı 23 Mart tarihi ile bildiririm.",
            "dogru_mu": True,
            "geri_bildirim": "Doğru! Ay ortası işe girişte kısmi maaş hesabı bu şekilde yapılır. Form 4A bildirimi gerçek işe başlama tarihi olan 23 Mart olmalıdır.",
            "sonuc": "Personel doğru tutar üzerinden maaş alır ve SGK prim günü eksiksiz bildirilir.",
        },
        {
            "id": "B",
            "metin": "Tam ay maaşı öderim, zira personel resmi kadro listesinde bu ay yer alıyor.",
            "dogru_mu": False,
            "geri_bildirim": "Yanlış. Çalışılmayan günler için maaş ödenmez; tam ay ödeme fazla ödeme sayılır ve iade sürecini başlatır.",
            "sonuc": "Fazla ödeme nedeniyle personele yazılı tebligat gönderilmek zorunda kalınır.",
        },
        {
            "id": "C",
            "metin": "Bir sonraki ay toplu öderim; bu ay için herhangi bir işlem yapmam.",
            "dogru_mu": False,
            "geri_bildirim": "Yanlış. Personelin maaşı fiilen çalıştığı dönem için aynı ay ödenmelidir. Erteleme yasal değildir.",
            "sonuc": "Gecikmiş ödeme nedeniyle idari işlem başlatılabilir ve personel mağduriyeti doğar.",
        },
    ],
    "ogrenme_hedefi": "Ay ortası işe girişlerde kısmi maaş hesabı ve SGK Form 4A bildirim tarihini doğru uygulamak.",
    "kaynak_chunk_indeksleri": [2, 4],
}

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_process_map() -> dict:
    """
    Generate a structured payroll process map grounded in the knowledge base.

    Queries for payroll procedure steps and formats the result as a JSON dict
    containing a list of step objects with source citations.

    Returns
    -------
    dict
        Keys: "steps" (list of step dicts with kaynak_chunk_indeksleri).
        On LLM failure: {"hata": ..., "ham_çıktı": ...}.
    """
    if MOCK_MODE:
        logger.info("MOCK_MODE: Süreç haritası fixture döndürülüyor.")
        return _MOCK_PROCESS_MAP

    _opt = _load_optimized_prompt("process_map")
    query = _opt["query"] if _opt else "maaş hesaplama adımları süreç akışı prosedür"
    instruction = _opt["instruction"] if _opt else """\
Yukarıdaki bağlam belgelerine dayanarak maaş mutemetliği süreç haritasını oluştur.
Aşağıdaki JSON şemasını kullan ve BAŞKA HİÇBİR ŞEY yazma:

{
  "steps": [
    {
      "adim_no": 1,
      "baslik": "...",
      "giris": "...",
      "cikis": "...",
      "karar_noktasi": "...",
      "risk": "...",
      "kontrol": "...",
      "kaynak_chunk_indeksleri": [0, 2]
    }
  ]
}

Her adım için kullandığın kaynak chunk numaralarını kaynak_chunk_indeksleri alanına yaz.
"""

    try:
        system_instruction, user_prompt, chunk_indices = _build_grounded_prompt(query, instruction, top_k=8)
        logger.info("Süreç haritası oluşturuluyor. Kullanılan chunk indeksleri: %s", chunk_indices)
        client = _get_genai_client()
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=ProcessMapSchema,
            )
        )
        return json.loads(response.text)
    except Exception as exc:
        logger.error("Süreç haritası oluşturma hatası: %s", exc)
        return {"hata": str(exc), "ham_çıktı": ""}


def generate_error_cards() -> dict:
    """
    Generate error/mistake awareness cards grounded in the knowledge base.

    Focuses on common mistakes made by new payroll officers, sourced from
    both explicit regulation and tacit interview knowledge.

    Returns
    -------
    dict
        Keys: "hata_kartlari" (list of card dicts with kaynak_chunk_indeksleri).
        On LLM failure: {"hata": ..., "ham_çıktı": ...}.
    """
    if MOCK_MODE:
        logger.info("MOCK_MODE: Hata kartları fixture döndürülüyor.")
        return _MOCK_ERROR_CARDS

    _opt = _load_optimized_prompt("error_cards")
    query = _opt["query"] if _opt else "sık yapılan hatalar yanlış uygulama kaçırılan adım"
    instruction = _opt["instruction"] if _opt else """\
Yukarıdaki bağlam belgelerine dayanarak yeni maaş mutemedinin sık yaptığı hataları listele.
Aşağıdaki JSON şemasını kullan ve BAŞKA HİÇBİR ŞEY yazma:

{
  "hata_kartlari": [
    {
      "kart_no": 1,
      "hata": "...",
      "kök_neden": "...",
      "tespit_yöntemi": "...",
      "dogru_uygulama": "...",
      "kaynak_chunk_indeksleri": [1, 3]
    }
  ]
}

Her kart için kullandığın kaynak chunk numaralarını kaynak_chunk_indeksleri alanına yaz.
"""

    try:
        system_instruction, user_prompt, chunk_indices = _build_grounded_prompt(query, instruction, top_k=8)
        logger.info("Hata kartları oluşturuluyor. Kullanılan chunk indeksleri: %s", chunk_indices)
        client = _get_genai_client()
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=ErrorCardsSchema,
            )
        )
        return json.loads(response.text)
    except Exception as exc:
        logger.error("Hata kartları oluşturma hatası: %s", exc)
        return {"hata": str(exc), "ham_çıktı": ""}


def generate_glossary() -> dict:
    """
    Generate a domain-specific glossary grounded in the knowledge base.

    Extracts technical terms, abbreviations, and institution-specific phrases
    from both regulation documents and interview transcripts.

    Returns
    -------
    dict
        Keys: "terimler" (list of term dicts with kaynak_chunk_indeksleri).
        On LLM failure: {"hata": ..., "ham_çıktı": ...}.
    """
    if MOCK_MODE:
        logger.info("MOCK_MODE: Terim sözlüğü fixture döndürülüyor.")
        return _MOCK_GLOSSARY

    _opt = _load_optimized_prompt("glossary")
    query = _opt["query"] if _opt else "kuruma özgü terimler teknik kavramlar kısaltmalar"
    instruction = _opt["instruction"] if _opt else """\
Yukarıdaki bağlam belgelerine dayanarak maaş mutemetliği alanına özgü terim sözlüğü oluştur.
Aşağıdaki JSON şemasını kullan ve BAŞKA HİÇBİR ŞEY yazma:

{
  "terimler": [
    {
      "terim": "...",
      "tanim": "...",
      "kullanim_ornegi": "...",
      "kaynak_chunk_indeksleri": [0]
    }
  ]
}

Her terim için kullandığın kaynak chunk numaralarını kaynak_chunk_indeksleri alanına yaz.
"""

    try:
        system_instruction, user_prompt, chunk_indices = _build_grounded_prompt(query, instruction, top_k=8)
        logger.info("Terim sözlüğü oluşturuluyor. Kullanılan chunk indeksleri: %s", chunk_indices)
        client = _get_genai_client()
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=GlossarySchema,
            )
        )
        return json.loads(response.text)
    except Exception as exc:
        logger.error("Terim sözlüğü oluşturma hatası: %s", exc)
        return {"hata": str(exc), "ham_çıktı": ""}


def generate_simulation_scenario() -> dict:
    """
    Generate an interactive simulation scenario grounded in the knowledge base.

    Focuses on high-risk decision points such as mid-month hires, retroactive
    raises, or icra (enforcement) limit breaches.

    Returns
    -------
    dict
        Simulation scenario with three answer options, feedback, and source citations.
        On LLM failure: {"hata": ..., "ham_çıktı": ...}.
    """
    if MOCK_MODE:
        logger.info("MOCK_MODE: Simülasyon senaryosu fixture döndürülüyor.")
        return _MOCK_SIMULATION

    _opt = _load_optimized_prompt("simulation")
    query = _opt["query"] if _opt else "kritik karar noktası yüksek hata riski zor durum"
    instruction = _opt["instruction"] if _opt else """\
Yukarıdaki bağlam belgelerine dayanarak maaş mutemetliği için etkileşimli bir simülasyon senaryosu oluştur.
Aşağıdaki JSON şemasını kullan ve BAŞKA HİÇBİR ŞEY yazma:

{
  "senaryo_basligi": "...",
  "durum_aciklamasi": "...",
  "soru": "...",
  "secenekler": [
    {
      "id": "A",
      "metin": "...",
      "dogru_mu": true,
      "geri_bildirim": "...",
      "sonuc": "..."
    },
    {
      "id": "B",
      "metin": "...",
      "dogru_mu": false,
      "geri_bildirim": "...",
      "sonuc": "..."
    },
    {
      "id": "C",
      "metin": "...",
      "dogru_mu": false,
      "geri_bildirim": "...",
      "sonuc": "..."
    }
  ],
  "ogrenme_hedefi": "...",
  "kaynak_chunk_indeksleri": [2, 5]
}

Doğru cevap seçeneğinde "dogru_mu": true olmalı. Kaynak chunk numaralarını doldur.
"""

    try:
        system_instruction, user_prompt, chunk_indices = _build_grounded_prompt(query, instruction, top_k=7)
        logger.info("Simülasyon senaryosu oluşturuluyor. Kullanılan chunk indeksleri: %s", chunk_indices)
        client = _get_genai_client()
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=SimulationSchema,
            )
        )
        return json.loads(response.text)
    except Exception as exc:
        logger.error("Simülasyon senaryosu oluşturma hatası: %s", exc)
        return {"hata": str(exc), "ham_çıktı": ""}
