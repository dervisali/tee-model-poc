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
import hashlib
import time
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
        "habilitation examinateur-correcteur préparation référentiel session convocation",
        "correction individuelle notation grille critères évaluation production candidat",
        "double correction comparaison délibération copie atypique finalisation résultats",
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
_PROCESS_MAP_CACHE_PATH = settings.CHROMA_DIR / "process_map_cache.json"


def _load_optimized_prompt(generator_key: str) -> dict | None:
    """Prompt Optimizer tarafından kaydedilen en iyi varyantı yükler."""
    if not _OPTIMIZED_PROMPTS_PATH.exists():
        return None
    try:
        data = json.loads(_OPTIMIZED_PROMPTS_PATH.read_text(encoding="utf-8"))
        return data.get(generator_key)
    except Exception:
        return None


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("JSON cache okunamadı (%s): %s", path, exc)
        return {}


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _process_map_cache_key(lang: OutputLanguage, prompts: dict[str, str]) -> str:
    corpus_path = settings.CHROMA_DIR / "parents.json"
    bm25_path = settings.CHROMA_DIR / "bm25_index.pkl"
    fingerprint = {
        "lang": lang,
        "generation_model": settings.GENERATION_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "corpus_primary_language": settings.CORPUS_PRIMARY_LANGUAGE,
        "child_collection": settings.CHILD_COLLECTION_NAME,
        "enable_hybrid_search": settings.ENABLE_HYBRID_SEARCH,
        "enable_confidence_scoring": settings.ENABLE_CONFIDENCE_SCORING,
        "hybrid_rrf_k": settings.HYBRID_RRF_K,
        "parents_hash": _hash_file(corpus_path),
        "bm25_hash": _hash_file(bm25_path),
        "phase_queries": _PROCESS_MAP_PHASE_QUERIES["fr"],
        "instruction": prompts["instruction"],
        "grounding_template": _GROUNDING_TEMPLATES[lang],
        "schema": ProcessMapSchema.model_json_schema(),
    }
    return _hash_text(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, default=str))


def _read_process_map_cache(cache_key: str) -> dict | None:
    cache = _load_json_file(_PROCESS_MAP_CACHE_PATH)
    entry = cache.get(cache_key)
    if not isinstance(entry, dict):
        return None
    result = entry.get("result")
    if not isinstance(result, dict):
        return None
    logger.info(
        "Process map cache hit",
        extra={
            "event": "process_map_cache_hit",
            "cache_key": cache_key[:12],
            "created_at": entry.get("created_at"),
        },
    )
    return result


def _write_process_map_cache(cache_key: str, result: dict) -> None:
    cache = _load_json_file(_PROCESS_MAP_CACHE_PATH)
    cache[cache_key] = {
        "created_at": int(time.time()),
        "result": result,
    }
    # Keep the cache tiny; old entries are only useful across recent prompt/corpus edits.
    if len(cache) > 12:
        cache = dict(
            sorted(
                cache.items(),
                key=lambda item: item[1].get("created_at", 0) if isinstance(item[1], dict) else 0,
                reverse=True,
            )[:12]
        )
    _write_json_file(_PROCESS_MAP_CACHE_PATH, cache)


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
    "desteklenmeyen_liste": ["Habilitasyon seans süresi sınırı mevcut dokümanlar dışında kalmaktadır."],
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
        "Delibération eşik puanı dokümanların dışında kalmaktadır.",
        "Belirli bant sınır değerleri kaynak belgelerde yer almamaktadır.",
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
            "faz": "1-Hazırlık",
            "baslik": "Habilitasyon Belgelerini ve Grille'leri Hazırla",
            "giris": "Sınav dönemi başlangıcı, ilgili düzey grille'leri ve CECRL descripteur belgeleri",
            "cikis": "Sınavcıya atanan düzey ve beceri için eksiksiz grille seti",
            "karar_noktasi": "Sınavcının ilgili düzey (A1-C1) ve beceri (PE/PO/CE/CO) için habilitasyonu var mı?",
            "risk": "Yanlış düzey grille'si kullanarak puanlama yapılması",
            "kontrol": "Grille başlığındaki düzey ve beceri kodu, sınavcının habilitation belgesiyle karşılaştırılır",
            "kaynak_chunk_indeksleri": ["manuel-exacor_par1", "B2_stagiaire_par0"],
        },
        {
            "adim_no": 2,
            "faz": "2-Bireysel Düzeltme",
            "baslik": "Bireysel Analitik Notlandırmayı Gerçekleştir",
            "giris": "Anonim adayın kopyası, ilgili düzeyin grille'si ve descripteur'ları",
            "cikis": "Her kriter için bant puanı ve toplam not",
            "karar_noktasi": "Kopya atipik mi? (Kriterler arasında iki banttan fazla fark var mı?)",
            "risk": "Halo etkisi — tek bir kriterin diğer kriterlerin puanlamasını etkilemesi",
            "kontrol": "Her kriter bağımsız olarak değerlendirilir; grille tanımlayıcılarıyla çapraz kontrol yapılır",
            "kaynak_chunk_indeksleri": ["B2_Grille_PE_par0", "Présentation_copies_atypiques_par4"],
        },
        {
            "adim_no": 3,
            "faz": "3-Çift Düzeltme ve Uzlaşma",
            "baslik": "Çift Düzeltme ve Delibération Sürecini Yönet",
            "giris": "İki sınavcının bağımsız notları ve gerekçeleri",
            "cikis": "Uzlaşılan nihai not veya arbitraj kararı",
            "karar_noktasi": "İki sınavcının notları arasındaki fark kabul edilebilir sınırda mı?",
            "risk": "Sınavcıların birbirini etkileyerek bağımsızlıklarını yitirmesi",
            "kontrol": "Delibération tutanağı tutulur; anlaşmazlık halinde üçüncü sınavcı devreye girer",
            "kaynak_chunk_indeksleri": ["manuel-exacor_par3", "DELF_B2_examinateur_par2"],
        },
    ]
}

_MOCK_ERROR_CARDS = {
    "hata_kartlari": [
        {
            "kart_no": 1,
            "hata": "Halo Etkisi — Bir Kriterin Diğer Kriterlerin Puanlamasını Etkilemesi",
            "kok_neden": "Sınavcı, adayın güçlü yanını (örn. sözcük zenginliği) fark edince diğer kriterlere de bilinçsizce yüksek puan veriyor",
            "tespit_yontemi": "Tüm kriterler için neredeyse aynı bant puanı verildiğinde veya bir kriterin beklenen profille örtüşmediği durumlarda",
            "dogru_uygulama": "Her kriter grille'deki tanımlayıcıya göre bağımsız değerlendirilir; diğer kriterlerin puanı kapsandıktan sonra bakılır",
            "kaynak_chunk_indeksleri": ["B2_Grille_PE_par1", "manuel-exacor_par5"],
        },
        {
            "kart_no": 2,
            "hata": "Atipik Kopya Prosedürünü Başlatmama",
            "kok_neden": "Yeni sınavcı, kopyanın standart grille'e uymadığını fark etmeden doğrudan puanlama yapmaya çalışıyor",
            "tespit_yontemi": "Aynı kopyadaki kriterler arasında iki banttan fazla fark (örn. B2 düzeyi sözdizimi ama A1 düzeyi söylem yapısı)",
            "dogru_uygulama": "Kopya atipik olarak işaretlenir, ikinci bir uzman sınavcıya yönlendirilir; tek başına puanlama tamamlanmaz",
            "kaynak_chunk_indeksleri": ["Présentation_copies_atypiques_par2", "manuel-exacor_par7"],
        },
    ]
}

_MOCK_GLOSSARY = {
    "terimler": [
        {
            "terim": "Grille d'évaluation",
            "tanim": "Her DELF/DALF düzeyi ve becerisi için geliştirilmiş; adayın üretimini ölçüt ve bantlara göre analitik olarak puanlayan resmi araç.",
            "kullanim_ornegi": "B2 PE grille'si söylem tutarlılığı, sözcüksel yeterlik ve biçimbilimsel-sözdizimsel doğruluk olmak üzere dört kriter içerir.",
            "kaynak_chunk_indeksleri": ["B2_Grille_PE_par0"],
        },
        {
            "terim": "Descripteur",
            "tanim": "Bir adayın belirli bir CECRL düzeyinde neler yapabileceğini betimleyen ölçütsel ifade; grille bantlarının referans çerçevesini oluşturur.",
            "kullanim_ornegi": "A2 PE descripteur'u, adayın temel bağlaçlar kullanan basit cümleler üretebileceğini belirtir.",
            "kaynak_chunk_indeksleri": ["Descripteurs_CECRL_A1_B2_par1"],
        },
        {
            "terim": "Copie atypique",
            "tanim": "Yeterlik profili grille'nin olağan bantlarından çıkan, arbitraj prosedürü gerektiren aday üretimi.",
            "kullanim_ornegi": "Kriterler arasında iki banttan fazla fark olduğunda kopya atipik sayılır ve ikinci bir sınavcıya yönlendirilir.",
            "kaynak_chunk_indeksleri": ["Présentation_copies_atypiques_par2"],
        },
    ]
}

_MOCK_PROCESS_MAP_FR = {
    "steps": [
        {
            "adim_no": 1,
            "faz": "1-Préparation",
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
            "faz": "2-Correction individuelle",
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
            "faz": "2-Correction individuelle",
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
    "senaryo_basligi": "B2 PE Değerlendirmesinde Halo Etkisini Tanıma",
    "durum_aciklamasi": (
        "Bir B2 PE kopyasını değerlendiriyorsunuz. Adayın sözcük zenginliği ve sözdizimi beklentilerin üzerinde. "
        "Ancak söylem tutarlılığını incelerken paragraflar arasındaki geçişlerin zayıf olduğunu fark ediyorsunuz. "
        "Sözcüksel yeterliliğe 3/3, sözdizimsel doğruluğa 2/3 verdikten sonra söylem tutarlılığı kriteri için ne yapacaksınız?"
    ),
    "soru": "Söylem tutarlılığı kriteri için hangi puanı verirsiniz?",
    "secenekler": [
        {
            "id": "A",
            "metin": "Söylem tutarlılığını diğer kriterlerden bağımsız olarak grille tanımlayıcısına göre değerlendiririm; zayıf bağlantılar nedeniyle 1/3 veririm.",
            "dogru_mu": True,
            "geri_bildirim": "Doğru! Her kriter grille tanımlayıcısına göre bağımsız değerlendirilir. Diğer kriterlerin puanı bu kararı etkilememelidir.",
            "sonuc": "Aday profili doğru yansıtılır; söylem zayıflığı raporlanmış olur.",
        },
        {
            "id": "B",
            "metin": "Diğer kriterler yüksek olduğundan söylem tutarlılığına da 3/3 veririm; genel izlenim güçlü.",
            "dogru_mu": False,
            "geri_bildirim": "Yanlış. Bu halo etkisidir. Güçlü sözcük kullanımı söylem tutarlılığı kriterini örtmemeli.",
            "sonuc": "Aday gerçek profil yerine yapay olarak yüksek puan alır; değerlendirme güvenilirliği düşer.",
        },
        {
            "id": "C",
            "metin": "Kopya atipik olarak işaretlerim ve ikinci sınavcıya gönderirim.",
            "dogru_mu": False,
            "geri_bildirim": "Yanlış. Atipik prosedür kriterler arasında iki banttan fazla fark olduğunda devreye girer; bu senaryoda durum bu değil.",
            "sonuc": "Gereksiz arbitraj süreci başlatılır; süreç uzar.",
        },
    ],
    "ogrenme_hedefi": "Halo etkisini tanımak ve her DELF kriterini grille tanımlayıcısına göre bağımsız değerlendirmek.",
    "kaynak_chunk_indeksleri": ["B2_Grille_PE_par1", "manuel-exacor_par5"],
}


# Populate the TR fixtures lazily — all DELF/DALF examiner-correction domain.
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

def score_process_map_confidence(
    process_map: dict,
    source_chunks: list[dict],
    cache_key: str | None = None,
) -> dict:
    """Score an already-generated process map and persist the scored cache entry."""
    result = dict(process_map)
    result.pop("_confidence_status", None)
    result.pop("_confidence_error", None)
    try:
        from src.confidence import score_generated_content

        confidence_started = time.perf_counter()
        result["_confidence"] = score_generated_content(result, source_chunks, "process_map")
        logger.info(
            "Process map confidence scoring tamamlandı",
            extra={
                "event": "process_map_confidence_complete",
                "duration_ms": round((time.perf_counter() - confidence_started) * 1000, 1),
            },
        )
    except Exception as exc:
        logger.warning("Confidence skorlaması atlandı: %s", exc)
        result["_confidence_status"] = "error"
        result["_confidence_error"] = str(exc)

    if cache_key:
        _write_process_map_cache(cache_key, result)
        logger.info(
            "Process map cache yazıldı",
            extra={"event": "process_map_cache_write", "cache_key": cache_key[:12]},
        )
    return result


def generate_process_map(
    language: OutputLanguage | None = None,
    *,
    defer_confidence: bool = False,
) -> dict:
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
    cache_key = _process_map_cache_key(lang, prompts)
    cached = _read_process_map_cache(cache_key)
    if cached is not None:
        return cached

    try:
        # --- multi-phase retrieval ---
        started = time.perf_counter()
        phase_queries = _PROCESS_MAP_PHASE_QUERIES[lang]
        seen: dict[str, dict] = {}
        for q in phase_queries:
            phase_started = time.perf_counter()
            phase_count = 0
            for chunk in retrieve_context(q, top_k=8):
                phase_count += 1
                pid = chunk["parent_id"]
                if pid not in seen or chunk.get("score", 0) > seen[pid].get("score", 0):
                    seen[pid] = chunk
            logger.info(
                "Process map phase retrieval tamamlandı",
                extra={
                    "event": "process_map_phase_retrieval_complete",
                    "query": q,
                    "results": phase_count,
                    "duration_ms": round((time.perf_counter() - phase_started) * 1000, 1),
                },
            )
        merged = sorted(seen.values(), key=lambda c: c.get("score", 0), reverse=True)[:20]
        logger.info(
            "Multi-phase retrieval tamamlandı",
            extra={
                "event": "process_map_multi_phase_retrieval_complete",
                "unique_parent_chunks": len(merged),
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

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
            if defer_confidence:
                result["_confidence_status"] = "pending"
                result["_confidence_source_chunks"] = merged
                result["_process_map_cache_key"] = cache_key
                logger.info(
                    "Process map confidence scoring ertelendi",
                    extra={"event": "process_map_confidence_deferred", "cache_key": cache_key[:12]},
                )
                return result

            result = score_process_map_confidence(result, merged, cache_key)
            return result

        _write_process_map_cache(cache_key, result)
        logger.info(
            "Process map cache yazıldı",
            extra={"event": "process_map_cache_write", "cache_key": cache_key[:12]},
        )
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
