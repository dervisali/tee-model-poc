"""
TEE-Model içerik üreticileri — Vertex AI Gemini üzerinde grounded üretim.

Tüm üreticiler kesin biçimde dayandırılmıştır (grounded): LLM'e yalnızca
retrieve edilen parent parçaları gönderilir; sistemin "BAĞLAM dışında bilgi
üretme" talimatı her çağrıda zorunludur.

Çıktılar Pydantic şemaları ile sınırlandırılır; Gemini `response_schema`
modu modelin şemaya uyumlu JSON üretmesini hedefler.

MOCK_MODE açıkken hiçbir cloud servisine dokunulmaz; sabit fixture
çıktıları döner. Bu, hem CI ortamlarında hem de Streamlit UI testinde gerekli.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.config import settings
from src.llm import generate as llm_generate
from src.retrieval import build_context_text, retrieve_context


OutputLanguage = Literal["tr", "fr"]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grounding sistem talimatı
# ---------------------------------------------------------------------------

_GROUNDING_TEMPLATES: dict[OutputLanguage, str] = {
    "tr": """\
Sen bir DELF/DALF sınavcı-düzelticisi eğitmenisin.
YALNIZCA aşağıda sağlanan Fransızca bağlam belgelerini kullan.
Bağlamda yer almayan hiçbir bilgiyi üretme veya tahmin etme.
Eğer cevap bağlamda mevcut değilse, şunu yaz:
"Bu bilgi mevcut belgelerde yer almamaktadır."
Çıktıyı TÜRKÇE üret; teknik DELF/CECRL terimlerini (PE, PO, CE, CO, A1-C1,
grille, descripteur, stagiaire) Fransızca formunda koru.

BAĞLAM:
{context_text}

KAYNAK CHUNK İNDEKSLERİ: {chunk_indices}
""",
    "fr": """\
Vous êtes un formateur d'examinateurs-correcteurs DELF/DALF.
N'utilisez QUE les documents de contexte fournis ci-dessous.
Ne générez ou ne devinez aucune information ne figurant pas dans le contexte.
Si la réponse n'est pas présente dans le contexte, écrivez :
« Cette information n'est pas disponible dans les documents fournis. »
Produisez la sortie en FRANÇAIS.

CONTEXTE :
{context_text}

INDICES DE CHUNK SOURCES : {chunk_indices}
""",
}


# Her üreticinin TR ve FR varyantları. Anahtarlar generator_key tarafında
# sabittir; UI'dan veya app.py'den `language` parametresi ile seçilir.
# Optimized prompts varsa (`optimized_prompts.json`) o tercih edilir; aksi
# halde bu sözlükten okunur.
_GENERATOR_PROMPTS: dict[str, dict[OutputLanguage, dict[str, str]]] = {
    "process_map": {
        "tr": {
            "query": "DELF değerlendirme süreci sınavcı-düzeltici protokolü iş akışı",
            "instruction": (
                "Yukarıdaki Fransızca DELF kaynak metinlerine dayanarak "
                "sınavcı-düzeltici değerlendirme sürecinin kapsamlı adım adım "
                "haritasını TÜRKÇE üret. "
                "EN AZ 10, EN FAZLA 15 adım üret. "
                "Adımları aşağıdaki dört fazdan birine ata (faz alanı): "
                "1-Hazırlık, 2-Bireysel Düzeltme, 3-Çift Düzeltme ve Uzlaşma, 4-Finalizasyon. "
                "Her adım için kullandığın kaynak parent_id değerlerini "
                "kaynak_chunk_indeksleri alanına yaz. Sadece geçerli JSON döndür."
            ),
        },
        "fr": {
            "query": "processus d'évaluation DELF protocole examinateur correcteur",
            "instruction": (
                "Sur la base des sources DELF françaises ci-dessus, produire la "
                "carte complète et détaillée du processus d'évaluation par les "
                "examinateurs-correcteurs en FRANÇAIS. "
                "Produire ENTRE 10 ET 15 étapes. "
                "Attribuer chaque étape à l'une des quatre phases suivantes (champ faz) : "
                "1-Préparation, 2-Correction individuelle, 3-Double correction et délibération, 4-Finalisation. "
                "Pour chaque étape, indiquer les parent_id sources dans "
                "kaynak_chunk_indeksleri. Retourner uniquement du JSON valide."
            ),
        },
    },
    "error_cards": {
        "tr": {
            "query": "sınavcının sık yaptığı değerlendirme hataları yanlış puanlama halo etkisi",
            "instruction": (
                "Yukarıdaki Fransızca DELF kaynak metinlerine dayanarak yeni "
                "sınavcı-düzelticinin DELF değerlendirmesinde sık yaptığı "
                "hataları TÜRKÇE listele. Her kart için kaynak parent_id "
                "değerlerini kaynak_chunk_indeksleri alanına yaz. Sadece "
                "geçerli JSON döndür."
            ),
        },
        "fr": {
            "query": "erreurs fréquentes correcteur évaluation DELF notation halo",
            "instruction": (
                "Sur la base des sources DELF françaises ci-dessus, lister les "
                "erreurs fréquentes des nouveaux examinateurs-correcteurs en "
                "FRANÇAIS. Pour chaque fiche, indiquer les parent_id sources "
                "dans kaynak_chunk_indeksleri. Retourner uniquement du JSON valide."
            ),
        },
    },
    "glossary": {
        "tr": {
            "query": "DELF CECRL terimleri descripteur grille kriterler bant",
            "instruction": (
                "Yukarıdaki Fransızca DELF kaynak metinlerine dayanarak "
                "sınavcılar için DELF/CECRL terim sözlüğünü TÜRKÇE oluştur. "
                "Terimler Fransızca kalabilir; tanım ve kullanım örnekleri "
                "Türkçe yazılır. Kaynak parent_id değerlerini "
                "kaynak_chunk_indeksleri alanına yaz. Sadece geçerli JSON döndür."
            ),
        },
        "fr": {
            "query": "lexique DELF CECRL descripteur grille critères bande",
            "instruction": (
                "Sur la base des sources DELF françaises ci-dessus, créer un "
                "glossaire DELF/CECRL pour examinateurs en FRANÇAIS. Indiquer "
                "les parent_id sources dans kaynak_chunk_indeksleri. Retourner "
                "uniquement du JSON valide."
            ),
        },
    },
    "simulation": {
        "tr": {
            "query": "kritik değerlendirme kararı atipik kopya ikilemli durum",
            "instruction": (
                "Yukarıdaki Fransızca DELF kaynak metinlerine dayanarak "
                "sınavcı-düzelticiler için etkileşimli bir değerlendirme karar "
                "senaryosu TÜRKÇE oluştur. Doğru cevap seçeneğinde dogru_mu = "
                "true olmalı. Kaynak parent_id değerlerini "
                "kaynak_chunk_indeksleri alanına yaz. Sadece geçerli JSON döndür."
            ),
        },
        "fr": {
            "query": "décision critique évaluation copie atypique cas difficile",
            "instruction": (
                "Sur la base des sources DELF françaises ci-dessus, créer un "
                "scénario interactif de décision pour examinateurs-correcteurs "
                "en FRANÇAIS. L'option correcte doit avoir dogru_mu = true. "
                "Indiquer les parent_id sources dans kaynak_chunk_indeksleri. "
                "Retourner uniquement du JSON valide."
            ),
        },
    },
}


# Three targeted queries — one per examination phase — used by the multi-phase
# retrieval path in generate_process_map to ensure all phases are represented.
_PROCESS_MAP_PHASE_QUERIES: dict[OutputLanguage, list[str]] = {
    "tr": [
        "sınavcı-düzeltici habilitasyon hazırlık referans manüeli oturum öncesi",
        "bireysel düzeltme notlandırma grille kriterleri adayın üretimini değerlendirme",
        "çift düzeltme skoru karşılaştırma delibération uzlaşma atipik kopya finalizasyon",
    ],
    "fr": [
        "habilitation examinateur-correcteur préparation référentiel session convocation",
        "correction individuelle notation grille critères évaluation production candidat",
        "double correction comparaison délibération copie atypique finalisation résultats",
    ],
}


# ---------------------------------------------------------------------------
# Pydantic şemaları — yapılandırılmış çıktı için
# ---------------------------------------------------------------------------

class ProcessStep(BaseModel):
    adim_no: int
    faz: str
    baslik: str
    giris: str
    cikis: str
    karar_noktasi: str
    risk: str
    kontrol: str
    kaynak_chunk_indeksleri: list[str]


class ProcessMapSchema(BaseModel):
    steps: list[ProcessStep] = Field(min_length=8)


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
    language: OutputLanguage | None = None,
) -> tuple[str, str, list[str]]:
    """
    Sorgu için bağlam çek ve grounded prompt'u kompoze et.
    Backward-compat: 3-tuple döndürür (system, user, chunk_indices).

    Confidence scoring için tam chunk listesi gerekenler
    `_build_grounded_prompt_with_chunks` kullanmalı.
    """
    system, user, indices, _ = _build_grounded_prompt_with_chunks(query, extra_instruction, top_k, language)
    return system, user, indices


def _build_grounded_prompt_with_chunks(
    query: str,
    extra_instruction: str,
    top_k: int = 7,
    language: OutputLanguage | None = None,
) -> tuple[str, str, list[str], list[dict]]:
    """
    `_build_grounded_prompt` ile aynıdır, ek olarak retrieve edilen tam chunk
    listesini de döner — Phase 2.2 confidence scoring bunu kullanır.

    Phase 1.4: ENABLE_QUERY_REWRITING true ise, retrieval çok-sorgulu
    moda geçer (orijinal + 3 varyant); sonuçlar parent_id bazında
    deduplicate edilip birleşik skora göre sıralanır.
    """
    if settings.ENABLE_QUERY_REWRITING:
        from src.query_rewriter import multi_query_retrieve
        logger.info("Multi-query retrieval aktif (ENABLE_QUERY_REWRITING=true).")
        result = multi_query_retrieve(query, top_k=top_k)
        chunks = result["merged_results"]
        if result["variants"]:
            logger.info("Sorgu varyantları: %s", result["variants"])
    else:
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
    active_lang: OutputLanguage = language or settings.OUTPUT_LANGUAGE  # type: ignore[assignment]
    template = _GROUNDING_TEMPLATES.get(active_lang, _GROUNDING_TEMPLATES["tr"])
    system_instruction = template.format(
        context_text=context_text,
        chunk_indices=chunk_indices,
    )
    return system_instruction, extra_instruction, chunk_indices, chunks


# ---------------------------------------------------------------------------
# MOCK fixtures
# ---------------------------------------------------------------------------

_MOCK_CONFIDENCE_HIGH = {
    "guven_skoru": 0.91,
    "desteklenen_iddialar": 14,
    "desteklenmeyen_iddialar": 1,
    "desteklenmeyen_liste": ["Ay başında kadro değişikliklerinin sisteme işlenme zorunluluğu mevzuat dışı."],
    "uzman_onay_tavsiyesi": "hizli_inceleme",
    "rozet_renk": "green",
    "rozet_metin": "Yüksek Güven",
    "hata": False,
}

_MOCK_CONFIDENCE_MED = {
    "guven_skoru": 0.74,
    "desteklenen_iddialar": 8,
    "desteklenmeyen_iddialar": 3,
    "desteklenmeyen_liste": [
        "Tespit yöntemi olarak 'denetim' ifadesi mevzuatta yer almamaktadır.",
        "Belirli yasal yaptırımlar belge dışı.",
    ],
    "uzman_onay_tavsiyesi": "detayli_inceleme",
    "rozet_renk": "yellow",
    "rozet_metin": "Orta Güven — İncele",
    "hata": False,
}

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

_MOCK_PROCESS_MAP_FR = {
    "steps": [
        {
            "adim_no": 1,
            "baslik": "Vérifier la conformité format de la copie candidat",
            "giris": "Production écrite/orale remise par le candidat",
            "cikis": "Copie anonymisée conforme au format DELF",
            "karar_noktasi": "La copie est-elle anonyme et le format conforme?",
            "risk": "Identification du candidat compromettant l'objectivité",
            "kontrol": "Vérifier l'absence de nom ou numéro en page de garde",
            "kaynak_chunk_indeksleri": ["manuel-exacor_par1", "B2_stagiaire_par0"],
        },
        {
            "adim_no": 2,
            "baslik": "Réaliser une première évaluation holistique",
            "giris": "Copie anonyme + grille du niveau visé",
            "cikis": "Positionnement holistique (A1/A2/B1/B2/C1)",
            "karar_noktasi": "Le candidat est-il au niveau visé?",
            "risk": "Dominance d'un seul critère biaisant le jugement global",
            "kontrol": "Croiser avec le descripteur pour vérifier la cohérence de bande",
            "kaynak_chunk_indeksleri": ["B2_Grille_PE_par0", "Descripteurs_CECRL_A1_B2_par3"],
        },
        {
            "adim_no": 3,
            "baslik": "Appliquer la notation analytique critère par critère",
            "giris": "Décision holistique + grille du niveau",
            "cikis": "Score par critère + total",
            "karar_noktasi": "Identifie-t-on une copie atypique?",
            "risk": "Double comptage entre critères ou effet de halo",
            "kontrol": "Déclencher la procédure copie atypique, demander un second avis",
            "kaynak_chunk_indeksleri": ["B2_Grille_PE_par1", "Présentation_copies_atypiques_par4"],
        },
    ]
}

_MOCK_ERROR_CARDS_FR = {
    "hata_kartlari": [
        {
            "kart_no": 1,
            "hata": "Sur-pondération de la grammaire au détriment de la cohérence discursive",
            "kok_neden": "Le correcteur applique des critères de niveau supérieur sans s'aligner sur le descripteur cible",
            "tespit_yontemi": "Décalage systématique entre note grammaire et note cohérence sur plusieurs copies",
            "dogru_uygulama": "Recadrer la notation sur les descripteurs du niveau visé; chaque critère a sa propre échelle",
            "kaynak_chunk_indeksleri": ["B2_Descripteurs_PE_par2", "manuel-exacor_par5"],
        },
        {
            "kart_no": 2,
            "hata": "Non-déclenchement de la procédure copie atypique sur une production hors profil",
            "kok_neden": "Examinateur novice tente de forcer la copie dans la grille standard",
            "tespit_yontemi": "Écart de plus de deux bandes entre critères pour une même copie",
            "dogru_uygulama": "Marquer la copie comme atypique et solliciter un second correcteur formé",
            "kaynak_chunk_indeksleri": ["Présentation_copies_atypiques_par4"],
        },
    ]
}

_MOCK_GLOSSARY_FR = {
    "terimler": [
        {
            "terim": "Grille d'évaluation",
            "tanim": "Outil normé qui décompose la production en critères pondérés; sert de référence pour la notation analytique.",
            "kullanim_ornegi": "La grille PE B2 distingue cohérence, lexique, morphosyntaxe et adéquation au sujet.",
            "kaynak_chunk_indeksleri": ["B2_Grille_PE_par0"],
        },
        {
            "terim": "Descripteur",
            "tanim": "Énoncé qui décrit ce qu'un candidat est capable de faire à un niveau CECRL donné.",
            "kullanim_ornegi": "Les descripteurs A2 PE soulignent la production de phrases simples reliées par des connecteurs basiques.",
            "kaynak_chunk_indeksleri": ["Descripteurs_CECRL_A1_B2_par1"],
        },
        {
            "terim": "Copie atypique",
            "tanim": "Production dont le profil de compétences sort des bandes habituelles de la grille et déclenche une procédure d'arbitrage.",
            "kullanim_ornegi": "Une copie atypique exige un second correcteur indépendant.",
            "kaynak_chunk_indeksleri": ["Présentation_copies_atypiques_par2"],
        },
    ]
}

_MOCK_SIMULATION_FR = {
    "senaryo_basligi": "Décision face à une copie B2 PE hors profil",
    "durum_aciklamasi": (
        "Une copie B2 PE présente une cohérence discursive de niveau C1 mais des erreurs "
        "morphosyntaxiques relevant de A2. Vous êtes en première correction. "
        "Quelle conduite adoptez-vous?"
    ),
    "soru": "Comment notez-vous cette copie?",
    "secenekler": [
        {
            "id": "A",
            "metin": "Je déclenche la procédure copie atypique et sollicite un second correcteur.",
            "dogru_mu": True,
            "geri_bildirim": "Correct. Un écart de plus de deux bandes entre critères justifie la procédure copie atypique.",
            "sonuc": "La copie reçoit une double correction et une note médiane stabilisée.",
        },
        {
            "id": "B",
            "metin": "Je note tous les critères selon la bande la plus basse (A2).",
            "dogru_mu": False,
            "geri_bildirim": "Incorrect. Chaque critère doit être noté indépendamment selon son descripteur.",
            "sonuc": "Le candidat est sous-évalué; sa réelle compétence discursive n'est pas reconnue.",
        },
        {
            "id": "C",
            "metin": "Je note tous les critères selon la bande la plus haute (C1).",
            "dogru_mu": False,
            "geri_bildirim": "Incorrect. La sur-notation des compétences faibles fausse le profil du candidat.",
            "sonuc": "Le candidat passe le niveau sans maîtriser la morphosyntaxe B2.",
        },
    ],
    "ogrenme_hedefi": "Reconnaître une copie atypique et appliquer la procédure d'arbitrage plutôt que de forcer la grille standard.",
    "kaynak_chunk_indeksleri": ["B2_Grille_PE_par1", "Présentation_copies_atypiques_par4"],
}


# Mock fixture dispatch by language.
_MOCK_FIXTURES_BY_LANG: dict[OutputLanguage, dict[str, dict]] = {
    "tr": {},  # filled lazily below after all _MOCK_*_TR are defined
    "fr": {
        "process_map": _MOCK_PROCESS_MAP_FR,
        "error_cards": _MOCK_ERROR_CARDS_FR,
        "glossary": _MOCK_GLOSSARY_FR,
        "simulation": _MOCK_SIMULATION_FR,
    },
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


# Populate the TR fixtures lazily (kept payroll content for backward compat;
# follow-up commit can shift these to DELF too).
_MOCK_FIXTURES_BY_LANG["tr"] = {
    "process_map": _MOCK_PROCESS_MAP,
    "error_cards": _MOCK_ERROR_CARDS,
    "glossary": _MOCK_GLOSSARY,
    "simulation": _MOCK_SIMULATION,
}


def _resolve_language(language: OutputLanguage | None) -> OutputLanguage:
    """Belirtilmemişse settings.OUTPUT_LANGUAGE'a düşer."""
    return language or settings.OUTPUT_LANGUAGE  # type: ignore[return-value]


def _get_prompts(generator_key: str, language: OutputLanguage) -> dict[str, str]:
    """Optimized varyant varsa onu, yoksa bilingual default'u döndürür."""
    opt = _load_optimized_prompt(f"{generator_key}_{language}") or _load_optimized_prompt(generator_key)
    if opt and "query" in opt and "instruction" in opt:
        return opt
    return _GENERATOR_PROMPTS[generator_key][language]


def _mock_fixture(generator_key: str, language: OutputLanguage) -> dict:
    return _MOCK_FIXTURES_BY_LANG[language][generator_key]


# ---------------------------------------------------------------------------
# Tek geçişli yapılandırılmış üretim
# ---------------------------------------------------------------------------

def _structured_generate(
    *,
    query: str,
    instruction: str,
    schema: type[BaseModel],
    top_k: int,
    content_type: str,
    language: OutputLanguage | None = None,
) -> dict:
    """
    Grounded sistem talimatı + verilen şema ile tek LLM çağrısı yapar.
    JSON metnini parse edip dict olarak döndürür; başarısız olursa
    {"hata": ..., "ham_cikti": ...} döndürür.

    Phase 2.2: ENABLE_CONFIDENCE_SCORING true ise, üretilen içeriğe
    `_confidence` alanı eklenir — kaynak chunk'lara karşı ikinci-geçiş
    güven skoru.
    """
    system, user_prompt, chunk_indices, chunks = _build_grounded_prompt_with_chunks(
        query, instruction, top_k=top_k, language=language,
    )
    logger.info("Yapılandırılmış üretim: chunk_count=%d", len(chunk_indices))
    raw = llm_generate(
        prompt=user_prompt,
        system=system,
        response_format=schema,
    )
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse hatası: %s", exc)
        return {"hata": str(exc), "ham_cikti": raw}

    # Phase 2.2 — güven skoru
    if settings.ENABLE_CONFIDENCE_SCORING:
        from src.confidence import score_generated_content
        try:
            score = score_generated_content(result, chunks, content_type)
            result["_confidence"] = score
        except Exception as exc:
            logger.warning("Confidence skorlaması atlandı: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Üreticiler
# ---------------------------------------------------------------------------

def generate_process_map(language: OutputLanguage | None = None) -> dict:
    """DELF değerlendirme süreci haritasını grounded biçimde üretir.

    Üç fazlı retrieval stratejisi: her faz için ayrı bir sorgu çalıştırılır
    (hazırlık / bireysel düzeltme / çift düzeltme+finalizasyon), sonuçlar
    parent_id bazında tekilleştirilip en iyi 20'si bağlam olarak sunulur.
    Bu sayede tek sorguda gözden kaçabilecek fazlar kapsama alınır.
    """
    lang = _resolve_language(language)
    if settings.MOCK_MODE:
        logger.info("MOCK_MODE: process_map fixture döndürülüyor (lang=%s).", lang)
        return {**_mock_fixture("process_map", lang), "_confidence": _MOCK_CONFIDENCE_HIGH}

    prompts = _get_prompts("process_map", lang)
    try:
        # --- multi-phase retrieval ---
        phase_queries = _PROCESS_MAP_PHASE_QUERIES[lang]
        seen: dict[str, dict] = {}
        for q in phase_queries:
            for chunk in retrieve_context(q, top_k=8):
                pid = chunk["parent_id"]
                if pid not in seen or chunk.get("score", 0) > seen[pid].get("score", 0):
                    seen[pid] = chunk
        merged = sorted(seen.values(), key=lambda c: c.get("score", 0), reverse=True)[:20]
        logger.info("Multi-phase retrieval: %d unique parent chunks", len(merged))

        context_text, chunk_indices = build_context_text(merged)
        template = _GROUNDING_TEMPLATES.get(lang, _GROUNDING_TEMPLATES["tr"])
        system = template.format(context_text=context_text, chunk_indices=chunk_indices)

        raw = llm_generate(
            prompt=prompts["instruction"],
            system=system,
            response_format=ProcessMapSchema,
        )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse hatası: %s", exc)
            return {"hata": str(exc), "ham_cikti": raw}

        if settings.ENABLE_CONFIDENCE_SCORING:
            from src.confidence import score_generated_content
            try:
                result["_confidence"] = score_generated_content(result, merged, "process_map")
            except Exception as exc:
                logger.warning("Confidence skorlaması atlandı: %s", exc)

        return result
    except Exception as exc:
        logger.error("process_map üretim hatası: %s", exc)
        return {"hata": str(exc), "ham_cikti": ""}


def generate_error_cards(language: OutputLanguage | None = None) -> dict:
    """Sınavcı hata kartlarını üretir."""
    lang = _resolve_language(language)
    if settings.MOCK_MODE:
        logger.info("MOCK_MODE: error_cards fixture döndürülüyor (lang=%s).", lang)
        return {**_mock_fixture("error_cards", lang), "_confidence": _MOCK_CONFIDENCE_MED}

    prompts = _get_prompts("error_cards", lang)
    try:
        return _structured_generate(
            query=prompts["query"],
            instruction=prompts["instruction"],
            schema=ErrorCardsSchema,
            top_k=8,
            content_type="error_cards",
            language=lang,
        )
    except Exception as exc:
        logger.error("error_cards üretim hatası: %s", exc)
        return {"hata": str(exc), "ham_cikti": ""}


def generate_glossary(language: OutputLanguage | None = None) -> dict:
    """DELF/CECRL terim sözlüğünü üretir."""
    lang = _resolve_language(language)
    if settings.MOCK_MODE:
        logger.info("MOCK_MODE: glossary fixture döndürülüyor (lang=%s).", lang)
        return {**_mock_fixture("glossary", lang), "_confidence": _MOCK_CONFIDENCE_HIGH}

    prompts = _get_prompts("glossary", lang)
    try:
        return _structured_generate(
            query=prompts["query"],
            instruction=prompts["instruction"],
            schema=GlossarySchema,
            top_k=8,
            content_type="glossary",
            language=lang,
        )
    except Exception as exc:
        logger.error("glossary üretim hatası: %s", exc)
        return {"hata": str(exc), "ham_cikti": ""}


def generate_simulation_scenario(language: OutputLanguage | None = None) -> dict:
    """Sınavcı karar simülasyon senaryosu üretir."""
    lang = _resolve_language(language)
    if settings.MOCK_MODE:
        logger.info("MOCK_MODE: simulation fixture döndürülüyor (lang=%s).", lang)
        return {**_mock_fixture("simulation", lang), "_confidence": _MOCK_CONFIDENCE_MED}

    prompts = _get_prompts("simulation", lang)
    try:
        return _structured_generate(
            query=prompts["query"],
            instruction=prompts["instruction"],
            schema=SimulationSchema,
            top_k=7,
            content_type="simulation",
            language=lang,
        )
    except Exception as exc:
        logger.error("simulation üretim hatası: %s", exc)
        return {"hata": str(exc), "ham_cikti": ""}
