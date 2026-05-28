"""
TEE-Model — Asistan (Tab 7) konuşma motoru.

DELF/DALF sınavcı-düzelticileri için serbest formlu, korpusa dayandırılmış
(grounded) sohbet asistanı. Her olgusal yanıt korpus kaynaklarını alıntılar;
nihai puanlama kararı vermez; sınav konusu / aday metni üretmez.

Tasarım kararları — mevcut üretici (generators.py) deseninden ayrılan noktalar
ve gerekçeleri:

1. `response_format` (Pydantic şema) KULLANILMAZ. Süreç haritası / hata kartı
   gibi üreticiler yapılandırılmış JSON üretir; sohbet ise serbest metindir.
   Alıntılar metnin içine [Kaynak: dosya — parent_id] biçiminde gömülür.

2. Modül DURUMSUZDUR (stateless). Konuşma geçmişi st.session_state'te app.py
   tarafından tutulur; `chat()` her çağrıda geçmişi parametre olarak alır.
   Bu, Streamlit'in rerun modeliyle uyumludur ve test edilebilirliği artırır.

3. Hafıza = kayan pencere (sliding window). Tam geçmiş session_state'te
   saklanır, ancak prompt'a yalnızca son `HISTORY_WINDOW` mesaj eklenir —
   bu, bağlam kayması (context drift) ve token şişmesini önler.

4. Güvenlik korumaları tek katmanlı: sistem talimatı. Kırılgan ve yanlış-pozitif
   üreten anahtar kelime sınıflandırıcısı yerine, net ve örnekli sistem kuralları
   tercih edildi (Gemini düşük sıcaklıkta bu kurallara güvenilir biçimde uyar).

5. İş kuyruğu (job_queue) atlanır. Sohbet etkileşimlidir; tek kısa LLM çağrısı
   yapar ve anında yanıt beklenir. Kuyruk, sohbeti diğer sekmelerin uzun
   üretim işleriyle seri hale getirir — bu, sohbet için kötü bir UX olur.

6. Yanıt AKIŞLA (streaming) verilir: `chat_stream()` + Streamlit
   `st.write_stream()`. DELF yanıtları uzun olabildiğinden token token
   gösterim algılanan gecikmeyi büyük ölçüde düşürür. Akışsız `chat()`
   sürümü testler ve programatik kullanım için korunur.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from src.config import settings
from src.llm import generate as llm_generate, generate_stream as llm_generate_stream
from src.retrieval import build_context_text, retrieve_context

logger = logging.getLogger(__name__)

OutputLanguage = Literal["tr", "fr"]

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

MAX_TURNS = 10              # Oturum başına en fazla kullanıcı turu (bağlam kayması koruması)
HISTORY_WINDOW = 10         # Prompt'a dahil edilen son mesaj sayısı (= son 5 tam tur)
CHAT_TOP_K = 6              # Sohbet turu başına çekilen parent chunk sayısı
CHAT_TEMPERATURE = 0.3      # Üreticilerden (0.1) biraz yüksek — daha doğal sohbet
_SNIPPET_LEN = 320          # Kaynak panelinde gösterilen alıntı uzunluğu
_ASSISTANT_CTX_LEN = 180    # Retrieval zenginleştirme için asistan yanıtından alınan max karakter
# Alıntı etiketlerini retrieval sorgusundan ayıklar. Modelin hem talimatlı
# ([Kaynak: ...]) hem de gözlemlenen numaralı ([Kaynak 2: ...]) biçimini
# yakalar — iki nokta üst üste opsiyoneldir.
_CITATION_RE = re.compile(r"\[Kaynak[^\]]*\]")
# Citation ayırıcı — Gemini em-dash'i bazen en-dash veya düz tireye
# normalize ediyor. parent_id'lerin kendisi tire içerebildiğinden (örn.
# `manuel-exacor_par5`) ayırıcının iki yanında boşluk şart.
_PARENT_ID_SEP_RE = re.compile(r"\s+[—–-]\s+")
_VAGUE_FOLLOWUP_RE = re.compile(
    r"^\s*(biraz\s+daha\s+açıkla|daha\s+açıkla|açıklar\s+mısın|"
    r"örnek\s+ver|peki\s+.+|devam\s+et|bunu\s+aç|"
    r"explique\s+un\s+peu\s+plus|plus\s+de\s+détails|donne\s+un\s+exemple|"
    r"et\s+pour\s+.+|continue|précise)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_ROLE_LABELS: dict[OutputLanguage, dict[str, str]] = {
    "tr": {"user": "Kullanıcı", "assistant": "Asistan"},
    "fr": {"user": "Utilisateur", "assistant": "Assistant"},
}

_NO_CONTEXT_MARKER: dict[OutputLanguage, str] = {
    "tr": "(Bu soruyla ilgili korpus belgesi bulunamadı.)",
    "fr": "(Aucun document du corpus pertinent pour cette question.)",
}

_NO_CONTEXT_ANSWER: dict[OutputLanguage, str] = {
    "tr": (
        "Bu konuda elimdeki belgelerde yeterli bilgi yok. Korpustaki DELF/DALF "
        "grille, descripteur veya metodoloji belgelerine dayanmayan bir yanıt "
        "üretmem doğru olmaz."
    ),
    "fr": (
        "Je ne dispose pas d'informations suffisantes sur ce point dans mes "
        "documents. Je ne peux pas produire une réponse qui ne soit pas ancrée "
        "dans les grilles, descripteurs ou documents méthodologiques DELF/DALF "
        "du corpus."
    ),
}

_CITATION_WARNING: dict[OutputLanguage, str] = {
    "tr": (
        "\n\n_Not: Bu yanıttaki kaynak atıfları getirilen korpus kaynaklarıyla "
        "tam olarak doğrulanamadı; lütfen kaynak panelini kontrol edin._"
    ),
    "fr": (
        "\n\n_Note : les citations de cette réponse n'ont pas pu être validées "
        "entièrement avec les sources récupérées ; veuillez vérifier le panneau "
        "des sources._"
    ),
}

# ---------------------------------------------------------------------------
# Sistem talimatı — güvenlik korumalarının uygulandığı tek katman
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATES: dict[OutputLanguage, str] = {
    "tr": """\
Sen "TEE-Model Asistanı"sın — Türkiye'deki DELF/DALF sınavcı-düzelticileri için
bir eğitim asistanısın. Sınavcıların değerlendirme protokolü, grille'ler,
descripteur'lar ve metodoloji hakkındaki sorularını yanıtlarsın.

KURALLAR — istisnasız uygulanır:

1. DAYANDIRMA: Yalnızca aşağıdaki BAĞLAM belgelerindeki bilgileri kullan.
   Bağlamda yer almayan hiçbir olguyu üretme veya tahmin etme.

2. ALINTILAMA: Her olgusal ifadenin sonunda kaynağını
   [Kaynak: dosya_adı — parent_id] biçiminde belirt. Bağlamdaki
   [Kaynak N: dosya — parent_id] etiketlerinden dosya adını ve parent_id'yi al.
   Selamlama, teşekkür gibi olgusal olmayan ifadeler alıntı gerektirmez.

3. BİLGİ YOKSA: Soruyla ilgili bağlam belgesi yoksa açıkça şunu söyle:
   "Bu konuda elimdeki belgelerde yeterli bilgi yok." Asla uydurma yapma.

4. PUAN KARARI VERME: Hiçbir koşulda nihai puan/bant kararı verme
   ("Bu Bant 2'dir" deme). Kriterleri açıklayabilir, bir üretimin
   özelliklerini analiz edebilir, karar ağacı sunabilirsin — ama nihai karar
   sınavcıya aittir. Kullanıcı kesin bir puan dayatırsa kibarca hatırlat:
   "Nihai puanlama kararı yalnızca sınavcının sorumluluğundadır. Size grille
   kriterlerini ve dikkat edilmesi gereken noktaları açıklayabilirim."

5. SINAV İÇERİĞİ ÜRETME: Sınav konusu, sınav sorusu veya örnek aday metni/
   kopyası üretme taleplerini reddet. Bunun yerine değerlendirme metodolojisi
   konusunda yardımcı olabileceğini belirt.

6. DİL: Yanıtını TÜRKÇE ver. DELF/CECRL teknik terimlerini (grille, descripteur,
   PE, PO, CE, CO, A1-C1, copie atypique, délibération) Fransızca biçiminde koru.

YAPABİLECEKLERİN: grille/descripteur kriter açıklama; borderline durumlar için
adım adım karar ağacı; istek üzerine anlık mini hata kartı; istek üzerine mini
quiz (soru üret, sonraki turda cevabı kontrol et, puanı takip et); atipik kopya
rehberliği; sınav fazı navigasyonu; iki dilli terim arama; iki düzeltici
arasındaki délibération için konuşma metni; oturum özeti.

Önceki konuşmayı dikkate alarak tutarlı, çok turlu yanıtlar ver.

BAĞLAM:
{context_text}
""",
    "fr": """\
Vous êtes « l'Assistant TEE-Model » — un assistant de formation pour les
examinateurs-correcteurs DELF/DALF en Turquie. Vous répondez aux questions sur
le protocole d'évaluation, les grilles, les descripteurs et la méthodologie.

RÈGLES — appliquées sans exception :

1. ANCRAGE : N'utilisez QUE les informations des documents de CONTEXTE
   ci-dessous. Ne générez ni ne devinez aucun fait absent du contexte.

2. CITATION : Terminez chaque affirmation factuelle par sa source au format
   [Kaynak: nom_fichier — parent_id]. Reprenez le nom de fichier et le
   parent_id des étiquettes [Kaynak N: fichier — parent_id] du contexte.
   Les formules non factuelles (salutations, remerciements) n'exigent pas
   de citation.

3. AUCUNE INFORMATION : Si aucun document de contexte ne concerne la question,
   dites clairement : « Je ne dispose pas d'informations suffisantes sur ce
   point dans mes documents. » N'inventez jamais.

4. NE DÉCIDEZ JAMAIS D'UNE NOTE : Ne rendez en aucun cas une décision finale
   de note/bande (« C'est la Bande 2 »). Vous pouvez expliquer les critères,
   analyser les caractéristiques d'une production, proposer un arbre de
   décision — mais la décision finale appartient à l'examinateur. Si
   l'utilisateur insiste pour une note définitive, rappelez poliment :
   « La décision finale de notation relève de la seule responsabilité de
   l'examinateur. Je peux vous expliquer les critères de la grille et les
   points de vigilance. »

5. AUCUN CONTENU D'EXAMEN : Refusez les demandes de génération de sujets
   d'examen, de questions d'examen ou de textes/copies de candidats fictifs.
   Proposez plutôt votre aide sur la méthodologie d'évaluation.

6. LANGUE : Répondez en FRANÇAIS.

CE QUE VOUS POUVEZ FAIRE : expliquer les critères grille/descripteur ; produire
un arbre de décision pas à pas pour les cas limites ; générer une mini-fiche
d'erreur à la demande ; lancer un mini-quiz à la demande (générer des questions,
vérifier les réponses au tour suivant, suivre le score) ; guider le traitement
des copies atypiques ; naviguer dans les phases de l'examen ; rechercher des
termes bilingues ; rédiger un script verbal pour la délibération entre deux
correcteurs ; résumer la session.

Tenez compte de la conversation précédente pour des réponses cohérentes et
multi-tours.

CONTEXTE :
{context_text}
""",
}

# ---------------------------------------------------------------------------
# MOCK fixtures — MOCK_MODE açıkken LLM/ChromaDB çağrısı yapılmaz
# ---------------------------------------------------------------------------

_MOCK_RESPONSES: dict[OutputLanguage, dict] = {
    "tr": {
        "answer": (
            "**Halo etkisi (effet de halo)**, sınavcının adayın güçlü bir "
            "yanını fark edip diğer kriterlere de bilinçsizce yüksek puan "
            "vermesidir [Kaynak: manuel-exacor.pdf — manuel-exacor_par5]. "
            "Örneğin zengin bir sözcük dağarcığı görüldüğünde söylem tutarlılığı "
            "kriterine de hak edilmemiş yüksek puan verilebilir.\n\n"
            "Önlemek için her kriteri grille tanımlayıcısına göre **bağımsız** "
            "değerlendirin [Kaynak: B2_Grille_PE.pdf — B2_Grille_PE_par1].\n\n"
            "_(MOCK MODU: bu yanıt sabit bir örnektir — gerçek API çağrısı "
            "yapılmadı.)_"
        ),
        "sources": [
            {
                "filename": "manuel-exacor.pdf",
                "parent_id": "manuel-exacor_par5",
                "snippet": "L'effet de halo conduit le correcteur à généraliser une "
                           "impression positive à l'ensemble des critères...",
                "score": 0.88,
            },
            {
                "filename": "B2_Grille_PE.pdf",
                "parent_id": "B2_Grille_PE_par1",
                "snippet": "Chaque critère est évalué indépendamment selon son "
                           "descripteur propre...",
                "score": 0.81,
            },
        ],
    },
    "fr": {
        "answer": (
            "**L'effet de halo** désigne la tendance du correcteur à généraliser "
            "une impression positive sur un point fort à l'ensemble des critères "
            "[Kaynak: manuel-exacor.pdf — manuel-exacor_par5]. Par exemple, un "
            "lexique riche peut entraîner une note imméritée sur la cohérence "
            "discursive.\n\n"
            "Pour l'éviter, évaluez chaque critère **indépendamment** selon son "
            "descripteur [Kaynak: B2_Grille_PE.pdf — B2_Grille_PE_par1].\n\n"
            "_(MODE MOCK : réponse fixe — aucun appel API réel.)_"
        ),
        "sources": [
            {
                "filename": "manuel-exacor.pdf",
                "parent_id": "manuel-exacor_par5",
                "snippet": "L'effet de halo conduit le correcteur à généraliser une "
                           "impression positive à l'ensemble des critères...",
                "score": 0.88,
            },
            {
                "filename": "B2_Grille_PE.pdf",
                "parent_id": "B2_Grille_PE_par1",
                "snippet": "Chaque critère est évalué indépendamment selon son "
                           "descripteur propre...",
                "score": 0.81,
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _resolve_language(language: OutputLanguage | None) -> OutputLanguage:
    """Belirtilmemişse settings.OUTPUT_LANGUAGE'a düşer."""
    if language in ("tr", "fr"):
        return language  # type: ignore[return-value]
    return settings.OUTPUT_LANGUAGE  # type: ignore[return-value]


def _build_retrieval_query(user_message: str, history: list[dict]) -> str:
    """
    Retrieval sorgusunu kurar.

    Üç katmanlı zenginleştirme:
    1. Son kullanıcı mesajı — konuyu sabitler ("halo etkisi" gibi).
    2. Son asistan yanıtının alıntısız ilk _ASSISTANT_CTX_LEN karakteri —
       retrieval'ı devam eden konuya yönlendirir; "biraz daha açıkla" gibi
       vague takip sorularında kritik bağlamı korur.
    3. Geçerli kullanıcı mesajı.

    Alıntı etiketleri ([Kaynak: ...]) ChromaDB sorgusunu bozacağından temizlenir.
    """
    prev_user = ""
    prev_assistant = ""
    for msg in reversed(history):
        role = msg.get("role")
        if role == "assistant" and not prev_assistant:
            cleaned = _CITATION_RE.sub("", msg.get("content", "")).strip()
            prev_assistant = cleaned[:_ASSISTANT_CTX_LEN]
        elif role == "user" and not prev_user:
            prev_user = msg.get("content", "")
        if prev_user and prev_assistant:
            break

    current = user_message.strip()
    if _VAGUE_FOLLOWUP_RE.match(current) and (prev_user or prev_assistant):
        current = f"takip sorusu / question de suivi: {current}"

    parts = [p for p in (prev_user, prev_assistant, current) if p.strip()]
    return " ".join(parts).strip()


def _format_history(history: list[dict], language: OutputLanguage) -> str:
    """Son HISTORY_WINDOW mesajı prompt metnine çevirir (kayan pencere)."""
    labels = _ROLE_LABELS[language]
    recent = history[-HISTORY_WINDOW:]
    lines = []
    for msg in recent:
        role = labels.get(msg.get("role", "user"), msg.get("role", ""))
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n\n".join(lines)


def _extract_cited_parent_ids(answer: str) -> list[str]:
    """Yanıttaki [Kaynak: dosya — parent_id] etiketlerinden parent_id çıkarır."""
    cited: list[str] = []
    for citation in _CITATION_RE.findall(answer):
        inner = citation.strip("[]")
        parts = _PARENT_ID_SEP_RE.split(inner)
        if len(parts) < 2:
            continue
        parent_id = parts[-1].strip()
        if parent_id and parent_id not in cited:
            cited.append(parent_id)
    return cited


def _citation_report(answer: str, sources: list[dict]) -> dict:
    """
    Modelin ürettiği atıfları retrieved parent_id listesine karşı doğrular.

    `passed` yalnızca atıf varsa ve tüm atıflar retrieved kaynaklardan geliyorsa
    true olur. Selamlama gibi kaynak gerektirmeyen yanıtlar için chatbot LLM'e
    gitmeden önce no-context fallback kullandığımızdan, kaynaklı yanıtların en az
    bir doğrulanabilir atıf taşımasını bekliyoruz.
    """
    cited = _extract_cited_parent_ids(answer)
    available = {
        str(source.get("parent_id", "")).strip()
        for source in sources
        if source.get("parent_id")
    }
    invalid = [parent_id for parent_id in cited if parent_id not in available]
    return {
        "passed": bool(cited) and not invalid,
        "cited_parent_ids": cited,
        "invalid_parent_ids": invalid,
        "cited_count": len(cited),
        "retrieved_count": len(sources),
    }


def validate_citations(answer: str, sources: list[dict]) -> dict:
    """Public helper for UI/tests: validate answer citations against sources."""
    return _citation_report(answer, sources)


def _attach_citation_report(sources: list[dict], report: dict) -> list[dict]:
    """UI'nin kaynak panelinde gösterebilmesi için raporu her source'a ekler."""
    return [{**source, "_citation_report": report} for source in sources]


def _guard_answer(answer: str, sources: list[dict], lang: OutputLanguage) -> tuple[str, list[dict], dict]:
    """Yanıtı citation validation'dan geçirir ve gerekirse görünür uyarı ekler."""
    report = _citation_report(answer, sources)
    guarded = answer.strip()
    if not report["passed"]:
        logger.warning(
            "Sohbet citation validation başarısız",
            extra={
                "event": "chat_citation_validation_failed",
                "cited_parent_ids": report["cited_parent_ids"],
                "invalid_parent_ids": report["invalid_parent_ids"],
                "retrieved_count": report["retrieved_count"],
            },
        )
        guarded += _CITATION_WARNING[lang]
    return guarded, _attach_citation_report(sources, report), report


def _retrieve(query: str) -> tuple[str, list[dict]]:
    """
    Sorgu için korpus bağlamı çeker.

    Döner
    -----
    tuple[str, list[dict]]
        - context_text: LLM sistem talimatına gömülecek etiketli bağlam.
        - sources: UI kaynak paneli için sadeleştirilmiş chunk listesi.

    Hiç sonuç yoksa veya retrieval başarısız olursa ("", []) döner —
    sistem talimatı bu durumda "bilgi yok" yanıtını zorunlu kılar.
    """
    try:
        chunks = retrieve_context(query, top_k=CHAT_TOP_K)
    except (ValueError, RuntimeError) as exc:
        logger.info("Sohbet retrieval boş/başarısız: %s", exc)
        return "", []

    if not chunks:
        return "", []

    context_text, _ = build_context_text(chunks)
    sources = [
        {
            "filename": c.get("filename", "bilinmiyor"),
            "parent_id": c.get("parent_id", ""),
            "snippet": (c.get("parent_text", "") or "")[:_SNIPPET_LEN],
            "score": round(float(c.get("score", 0.0) or 0.0), 4),
        }
        for c in chunks
    ]
    return context_text, sources


# ---------------------------------------------------------------------------
# Genel API
# ---------------------------------------------------------------------------

def _prepare(
    user_message: str,
    history: list[dict],
    lang: OutputLanguage,
) -> tuple[str, str, list[dict]]:
    """
    Retrieval + prompt kurulumu — `chat()` ve `chat_stream()` ortak adımı.

    Döner
    -----
    tuple[str, str, list[dict]]
        (system_prompt, user_prompt, sources)
    """
    retrieval_query = _build_retrieval_query(user_message, history)
    context_text, sources = _retrieve(retrieval_query)

    system_prompt = _SYSTEM_TEMPLATES[lang].format(
        context_text=context_text or _NO_CONTEXT_MARKER[lang],
    )

    labels = _ROLE_LABELS[lang]
    history_text = _format_history(history, lang)
    if history_text:
        user_prompt = (
            f"{history_text}\n\n"
            f"{labels['user']}: {user_message}\n\n"
            f"{labels['assistant']}:"
        )
    else:
        user_prompt = f"{labels['user']}: {user_message}\n\n{labels['assistant']}:"
    return system_prompt, user_prompt, sources


def _error_text(lang: OutputLanguage) -> str:
    """UI'ya gösterilecek dostane hata mesajı (akış modunda satır içi görünür)."""
    return (
        "⚠️ Yanıt üretilirken bir hata oluştu. Lütfen tekrar deneyin."
        if lang == "tr"
        else "⚠️ Une erreur s'est produite lors de la génération. Veuillez réessayer."
    )


def chat(
    user_message: str,
    history: list[dict] | None = None,
    language: OutputLanguage | None = None,
) -> dict:
    """
    Tek bir sohbet turunu işler — akışsız sürüm.

    Yapılandırılmış dict döndürür; testler ve programatik kullanım içindir.
    Streamlit UI, algılanan gecikmeyi düşürmek için `chat_stream()` kullanır.

    Parametreler
    -----------
    user_message : str
        Kullanıcının yeni mesajı.
    history : list[dict] | None
        Önceki turlar — [{"role": "user"|"assistant", "content": str}, ...].
        Geçerli mesajı İÇERMEZ.
    language : "tr" | "fr" | None
        Yanıt dili; None ise settings.OUTPUT_LANGUAGE.

    Döner
    -----
    dict — {"answer": str, "sources": list[dict], "error": str | None}
    """
    lang = _resolve_language(language)
    history = history or []

    if settings.MOCK_MODE:
        logger.info("MOCK_MODE: sohbet fixture döndürülüyor (lang=%s).", lang)
        fixture = _MOCK_RESPONSES[lang]
        report = _citation_report(fixture["answer"], fixture["sources"])
        return {
            "answer": fixture["answer"],
            "sources": _attach_citation_report(list(fixture["sources"]), report),
            "error": None,
            "citation_report": report,
        }

    try:
        system_prompt, user_prompt, sources = _prepare(user_message, history, lang)
        if not sources:
            return {
                "answer": _NO_CONTEXT_ANSWER[lang],
                "sources": [],
                "error": None,
                "citation_report": _citation_report(_NO_CONTEXT_ANSWER[lang], []),
            }
        answer = llm_generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=CHAT_TEMPERATURE,
        )
        guarded_answer, checked_sources, report = _guard_answer(answer, sources, lang)
        return {
            "answer": guarded_answer,
            "sources": checked_sources,
            "error": None,
            "citation_report": report,
        }

    except Exception as exc:  # noqa: BLE001 — UI'ya zarif hata döndürülür
        logger.error("Sohbet turu hatası: %s", exc)
        return {"answer": _error_text(lang), "sources": [], "error": str(exc)}


def _mock_stream(text: str):
    """MOCK fixture'ı kelime kelime yield ederek akış UI'sini canlandırır."""
    for token in re.findall(r"\S+\s*", text):
        yield token


def _llm_stream(user_prompt: str, system_prompt: str, lang: OutputLanguage):
    """
    LLM akışını sarar. Akış sırasında bir hata olursa istisnayı yutar ve
    dostane bir hata mesajı yield eder — akış modelinde hatalar satır içinde
    gösterilir (kısmen yazılmış metin geri alınamaz)."""
    try:
        for piece in llm_generate_stream(
            prompt=user_prompt,
            system=system_prompt,
            temperature=CHAT_TEMPERATURE,
        ):
            yield piece
    except Exception as exc:  # noqa: BLE001
        logger.error("Sohbet akış hatası: %s", exc)
        yield "\n\n" + _error_text(lang)


def chat_stream(
    user_message: str,
    history: list[dict] | None = None,
    language: OutputLanguage | None = None,
) -> tuple[object, list[dict]]:
    """
    `chat()` ile aynı tur işleme — ancak yanıtı akış (streaming) olarak verir.

    Streamlit `st.write_stream()` ile kullanılır. Retrieval senkrondur, bu
    yüzden `sources` çağrı döndüğünde hemen kesinleşir; yalnızca LLM üretimi
    akışla gelir. Bu, kaynak panelinin doğru render edilmesini sağlarken
    yanıtın token token görünmesine izin verir.

    Döner
    -----
    tuple[generator, list[dict]]
        - generator: yanıt metnini parça parça yield eder; tam tüketildiğinde
          birleşmiş yanıt elde edilir (st.write_stream bunu döndürür).
        - sources: UI kaynak paneli için chunk listesi.

    Hata durumunda generator dostane bir hata mesajı yield eder.
    """
    lang = _resolve_language(language)
    history = history or []

    if settings.MOCK_MODE:
        logger.info("MOCK_MODE: sohbet akış fixture'ı döndürülüyor (lang=%s).", lang)
        fixture = _MOCK_RESPONSES[lang]
        report = _citation_report(fixture["answer"], fixture["sources"])
        return _mock_stream(fixture["answer"]), _attach_citation_report(list(fixture["sources"]), report)

    try:
        system_prompt, user_prompt, sources = _prepare(user_message, history, lang)
    except Exception as exc:  # noqa: BLE001 — retrieval/hazırlık hatası
        logger.error("Sohbet akış hazırlık hatası: %s", exc)

        def _err_gen():
            yield _error_text(lang)

        return _err_gen(), []

    if not sources:
        return _mock_stream(_NO_CONTEXT_ANSWER[lang]), []

    def _validated_stream():
        pieces: list[str] = []
        for piece in _llm_stream(user_prompt, system_prompt, lang):
            pieces.append(piece)
            yield piece
        report = _citation_report("".join(pieces), sources)
        if not report["passed"]:
            logger.warning(
                "Sohbet akış citation validation başarısız",
                extra={
                    "event": "chat_stream_citation_validation_failed",
                    "cited_parent_ids": report["cited_parent_ids"],
                    "invalid_parent_ids": report["invalid_parent_ids"],
                    "retrieved_count": report["retrieved_count"],
                },
            )
            yield _CITATION_WARNING[lang]

    return _validated_stream(), sources
