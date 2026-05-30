"""
TEE-Model — Streamlit Uygulaması

DELF/DALF sınav düzeltici eğitimi için yapay zeka destekli içerik üretimi.
Veri yükleme, süreç haritası, hata kartları, terim sözlüğü, simülasyon
ve uzman onay panelini tek arayüzde sunar.
"""

import os
import sys
import logging

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# Ensure src/ is importable when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.config import settings  # noqa: E402
from src.logging_config import configure_logging  # noqa: E402
from src.job_queue import submit_job, wait_for_job, queue_size, get_status  # noqa: E402
from src.chatbot import chat_stream, MAX_TURNS, validate_citations  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)

MOCK_MODE = settings.MOCK_MODE


def _run_via_queue(label: str, fn, *args, **kwargs):
    """Üretim çağrılarını paylaşılan kuyruktan geçirir, Streamlit spinner ile."""
    job_id = submit_job(fn, *args, **kwargs)
    queued_count = queue_size()
    waiting_label = (
        f"{label} ({job_id}) — kuyrukta {queued_count} iş..."
        if queued_count > 1
        else f"{label} ({job_id})..."
    )
    with st.spinner(waiting_label):
        status = wait_for_job(job_id, timeout=600)
    if status["status"] == "done":
        return status["result"], None
    if status["status"] == "error":
        return None, status["error"]
    return None, f"İş zaman aşımına uğradı (status={status['status']})"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TEE-Model — DELF/DALF Eğitim Sistemi",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Korpus kalıcılık güvencesi (Phase 1) — açılışta net durum, gizemli hata yok
#
# backend="gcs" iken korpus süreç başına bir kez GCS'ten indirilir; korpus
# hazır değilse tek bir net banner gösterilir (her sekmede derinlerde patlayan
# ValueError yerine). backend="local" ve MOCK_MODE'da davranış değişmez.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Korpus hazırlanıyor (GCS senkronizasyonu)...")
def _sync_corpus_once():
    """GCS backend'inde korpusu süreç başına bir kez indirir; hata mesajını döndürür."""
    if (
        settings.PERSISTENCE_BACKEND == "gcs"
        and settings.SYNC_ON_STARTUP
        and not settings.MOCK_MODE
    ):
        from src.bootstrap import ensure_corpus_ready

        try:
            return ensure_corpus_ready(run_smoke=False).error
        except Exception as exc:  # noqa: BLE001 - banner ile göster, traceback ile çökme
            logger.error("Açılış GCS senkronizasyonu başarısız: %s", exc)
            return str(exc)
    return None


def _render_corpus_guard():
    """Korpus hazır değilse net bir banner gösterir; üretimde uygulamayı durdurur."""
    if settings.MOCK_MODE:
        return
    sync_error = _sync_corpus_once()
    from src.readiness_check import check_readiness

    ready = check_readiness(run_smoke=False)  # ucuz: dosya + koleksiyon sayımı (Vertex yok)
    if ready.ok:
        return
    st.error("⚠️ Korpus şu anda kullanılamıyor — sorgulara dayanıklı yanıt verilemeyebilir.")
    with st.expander("Korpus durumu (ayrıntılar)"):
        if sync_error:
            st.write(f"GCS senkronizasyonu: {sync_error}")
        for failure in ready.failures:
            st.write(f"- {failure}")
        st.caption(
            "Üretimde bu durum bir dağıtım hatasıdır. Yerelde Tab 0'dan "
            "'Veritabanını Yenile' ile ingest edebilirsiniz."
        )
    if settings.REQUIRE_CORPUS_ON_STARTUP:
        st.stop()


_render_corpus_guard()

# ---------------------------------------------------------------------------
# Custom CSS — red left border for error cards, general polish
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .hata-karti {
        border-left: 4px solid #e74c3c;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        background: #fff5f5;
        border-radius: 4px;
    }
    .onay-banner {
        font-size: 0.85rem;
        color: #555;
    }
    .chunk-card {
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 5px solid #ccc;
        background: #fafafa;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .chunk-card.mevzuat {
        border-left-color: #2980b9;
        background: #eaf4fb;
    }
    .chunk-card.tacit {
        border-left-color: #27ae60;
        background: #eafaf1;
    }
    .chunk-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-mevzuat { background: #2980b9; color: white; }
    .badge-tacit   { background: #27ae60; color: white; }
    .chunk-meta    { color: #888; font-size: 0.78rem; margin-bottom: 0.4rem; }
    .chunk-text    { color: #333; }
    .confidence-banner {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0 1rem 0;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .confidence-banner.green  { background: #d4edda; border-left: 5px solid #27ae60; color: #155724; }
    .confidence-banner.yellow { background: #fff3cd; border-left: 5px solid #f1c40f; color: #856404; }
    .confidence-banner.red    { background: #f8d7da; border-left: 5px solid #e74c3c; color: #721c24; }
    .confidence-banner.pending { background: #eef2f7; border-left: 5px solid #7f8c8d; color: #2c3e50; }
    .confidence-meta { font-size: 0.82rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_confidence_banner(content: dict) -> None:
    """Üretilen içerikteki _confidence rozetini Tab 4'te render eder."""
    if content.get("_confidence_status") == "pending":
        st.markdown(
            """<div class="confidence-banner pending">
            <b>🛡️ Güven Skoru Hesaplanıyor</b> &nbsp;·&nbsp;
            İçerik görüntülenebilir; kaynak destek kontrolü arka planda devam ediyor.
            <span class="confidence-meta"> &nbsp;|&nbsp; Uzman onayı öncesi rozetin tamamlanmasını bekleyin.</span>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    if content.get("_confidence_status") == "error":
        st.markdown(
            f"""<div class="confidence-banner yellow">
            <b>🛡️ Güven Skoru Tamamlanamadı</b> &nbsp;·&nbsp;
            Manuel inceleme önerilir.
            <span class="confidence-meta"> &nbsp;|&nbsp; Hata: <b>{content.get('_confidence_error', 'bilinmiyor')}</b></span>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    score = content.get("_confidence") if isinstance(content, dict) else None
    if not score or score.get("hata"):
        return
    color = score.get("rozet_renk", "yellow")
    label = score.get("rozet_metin", "Güven Skoru")
    skor = score.get("guven_skoru", 0.0)
    sup = score.get("desteklenen_iddialar", 0)
    nosup = score.get("desteklenmeyen_iddialar", 0)
    advice_map = {
        "hizli_inceleme": "Hızlı İnceleme",
        "detayli_inceleme": "Detaylı İnceleme",
        "reddet": "Reddet",
    }
    advice = advice_map.get(score.get("uzman_onay_tavsiyesi"), "İncele")
    unsupported = score.get("desteklenmeyen_liste") or []
    extra = ""
    if unsupported:
        items = "".join(f"<li>{u}</li>" for u in unsupported[:5])
        source_hint = ""
        if color in ("yellow", "red"):
            source_hint = (
                "<p style='margin-top:0.4rem;font-size:0.82rem;'>"
                "💡 Desteklenmeyen iddiaların kaynak belgelerini doğrulamak için "
                "<b>Veritabanı Gezgini</b> sekmesini kullanın.</p>"
            )
        extra = (
            f"<details><summary>Desteklenmeyen iddialar ({len(unsupported)})</summary>"
            f"<ul>{items}</ul>{source_hint}</details>"
        )
    st.markdown(
        f"""<div class="confidence-banner {color}">
        <b>🛡️ {label}</b> &nbsp;·&nbsp;
        Güven skoru: <b>{skor:.2f}</b> &nbsp;·&nbsp;
        Desteklenen: <b>{sup}</b> · Desteklenmeyen: <b>{nosup}</b>
        <span class="confidence-meta"> &nbsp;|&nbsp; Uzman tavsiyesi: <b>{advice}</b></span>
        {extra}
        </div>""",
        unsafe_allow_html=True,
    )


def _strip_process_map_confidence_payload(result: dict) -> tuple[dict, list[dict], str | None]:
    """Remove bulky private confidence payload before storing in session state."""
    clean = dict(result)
    chunks = clean.pop("_confidence_source_chunks", [])
    cache_key = clean.pop("_process_map_cache_key", None)
    return clean, chunks, cache_key


def _start_process_map_confidence_job(result: dict) -> dict:
    """Queue deferred confidence scoring for a process map result."""
    clean, chunks, cache_key = _strip_process_map_confidence_payload(result)
    if clean.get("_confidence_status") != "pending" or not chunks:
        return clean

    from src.generators import score_process_map_confidence

    job_id = submit_job(score_process_map_confidence, clean, chunks, cache_key)
    clean["_confidence_job_id"] = job_id
    st.session_state["process_map_confidence_job_id"] = job_id
    logger.info(
        "Process map confidence job kuyruğa alındı",
        extra={"event": "process_map_confidence_job_queued", "job_id": job_id},
    )
    return clean


def _poll_process_map_confidence_job() -> None:
    """Patch process-map confidence into session state when the background job finishes."""
    job_id = st.session_state.get("process_map_confidence_job_id")
    pm = st.session_state.get("process_map")
    if not job_id or not isinstance(pm, dict):
        return

    status = get_status(job_id)
    if status["status"] == "done" and isinstance(status.get("result"), dict):
        scored = dict(status["result"])
        scored.pop("_confidence_job_id", None)
        st.session_state["process_map"] = scored
        st.session_state["process_map_confidence_job_id"] = None
        st.rerun()
    if status["status"] == "error":
        pm["_confidence_status"] = "error"
        pm["_confidence_error"] = status.get("error") or "bilinmeyen hata"
        st.session_state["process_map_confidence_job_id"] = None
        st.rerun()


def _strip_error_cards_confidence_payload(result: dict) -> tuple[dict, list[dict]]:
    """Remove bulky private confidence payload before storing in session state."""
    clean = dict(result)
    chunks = clean.pop("_confidence_source_chunks", [])
    clean.pop("_confidence_content_type", None)
    return clean, chunks


def _start_error_cards_confidence_job(result: dict) -> dict:
    """Queue deferred confidence scoring for an error_cards result."""
    clean, chunks = _strip_error_cards_confidence_payload(result)
    if clean.get("_confidence_status") != "pending" or not chunks:
        return clean

    from src.generators import score_error_cards_confidence

    job_id = submit_job(score_error_cards_confidence, clean, chunks)
    clean["_confidence_job_id"] = job_id
    st.session_state["error_cards_confidence_job_id"] = job_id
    logger.info(
        "Error cards confidence job kuyruğa alındı",
        extra={"event": "error_cards_confidence_job_queued", "job_id": job_id},
    )
    return clean


def _poll_error_cards_confidence_job() -> None:
    """Patch error_cards confidence into session state when the background job finishes."""
    job_id = st.session_state.get("error_cards_confidence_job_id")
    ec = st.session_state.get("error_cards")
    if not job_id or not isinstance(ec, dict):
        return

    status = get_status(job_id)
    if status["status"] == "done" and isinstance(status.get("result"), dict):
        scored = dict(status["result"])
        scored.pop("_confidence_job_id", None)
        st.session_state["error_cards"] = scored
        st.session_state["error_cards_confidence_job_id"] = None
        st.rerun()
    if status["status"] == "error":
        ec["_confidence_status"] = "error"
        ec["_confidence_error"] = status.get("error") or "bilinmeyen hata"
        st.session_state["error_cards_confidence_job_id"] = None
        st.rerun()


def _schedule_confidence_autorefresh() -> None:
    """Ask the browser to refresh while deferred confidence scoring is pending."""
    components.html(
        """
        <script>
        setTimeout(() => window.parent.location.reload(), 3000);
        </script>
        """,
        height=0,
    )

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "ingestion_summary": None,
    "mask_log": None,
    "process_map": None,
    "process_map_confidence_job_id": None,
    "error_cards": None,
    "error_cards_confidence_job_id": None,
    "glossary": None,
    "simulation": None,
    "sim_selected": None,
    "sim_answered": False,
    # Approval state: {content_type: {item_index: "onaylandi" | "reddedildi" | None}}
    "approvals": {
        "process_map": {},
        "error_cards": {},
        "glossary": {},
        "simulation": {},
    },
    # Prompt optimizer
    "optimizer_results": None,
    "optimizer_generator": None,
    # Phase 8 — generated output language (TR by default for Turkish examiners).
    "output_language": settings.OUTPUT_LANGUAGE,
    # Tab 7 — chatbot conversation state.
    # chat_history: [{"role": "user"|"assistant", "content": str, "sources": list}]
    "chat_history": [],
    "chat_turns": 0,
    "chat_inject": None,   # set by example-prompt buttons; consumed once on next rerun
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

# Sidebar — output language toggle (Phase 8).
with st.sidebar:
    st.markdown("### Çıktı Dili / Langue de sortie")
    lang_choice = st.radio(
        "Üretici dili",
        options=["tr", "fr"],
        format_func=lambda v: "Türkçe (TR)" if v == "tr" else "Français (FR)",
        index=0 if st.session_state.get("output_language", "tr") == "tr" else 1,
        key="output_language_radio",
        help="Süreç haritası / Hata kartları / Terim sözlüğü / Simülasyon çıktısının dili. Retrieval kaynak korpusu (FR) her durumda aynıdır.",
    )
    st.session_state["output_language"] = lang_choice

st.title("🏛️ TEE-Model — Tacit-Explicit Entegre Eğitim Sistemi")
st.caption(
    "DELF/DALF Sınav Düzeltici | "
    + ("⚠️ MOCK MODU AKTİF — cloud servis çağrısı yapılmıyor" if MOCK_MODE else f"🟢 Vertex AI Gemini: {settings.GENERATION_MODEL}")
)

tabs = st.tabs(
    [
        "📂 Veri Yükleme ve İşleme",
        "🗺️ Süreç Haritası ve Hata Kartları",
        "🎮 İnteraktif Simülasyon",
        "✅ Uzman Onay Paneli",
        "🔬 Prompt Optimizer",
        "🗄️ Veritabanı Gezgini",
        "💬 Asistan",
    ]
)

# ===========================================================================
# TAB 1 — Veri Yükleme ve İşleme
# ===========================================================================

with tabs[0]:
    st.subheader("Veri Yükleme ve İşleme")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            "Aşağıdaki butona tıklayarak belgeleri anonimleştirin, parçalayın ve "
            "vektör veritabanına yükleyin."
        )

    with col2:
        refresh_btn = st.button(
            "🔄 Veritabanını Yenile",
            help="Mevcut koleksiyonu siler ve tüm belgeleri yeniden yükler.",
            use_container_width=True,
        )

    if refresh_btn:
        if MOCK_MODE:
            st.session_state["ingestion_summary"] = {
                "total_chunks": 18,
                "explicit_chunks": 10,
                "tacit_chunks": 8,
                "collection_size": 18,
            }
            st.session_state["mask_log"] = [
                {"original": "Ahmet Yılmaz", "replaced_with": "[İSİM]"},
                {"original": "Selin Çelik", "replaced_with": "[İSİM]"},
                {"original": "Fahri Şahin", "replaced_with": "[İSİM]"},
                {"original": "Zeynep Hanım", "replaced_with": "[İSİM]"},
                {"original": "12345678901", "replaced_with": "[TC-KİMLİK]"},
                {"original": "TR330006100519786457841326", "replaced_with": "[IBAN]"},
                {"original": "05321234567", "replaced_with": "[TELEFON]"},
            ]
            st.success("MOCK MODU: Veriler başarıyla yüklendi.")
        else:
            from src.anonymizer import anonymize_text
            from src.ingestion import clear_and_reingest
            from pathlib import Path

            with st.spinner("Belgeler işleniyor ve vektör veritabanına yükleniyor..."):
                try:
                    # Run anonymizer first to capture mask log
                    base_dir = Path(__file__).resolve().parent
                    raw_path = base_dir / "data" / "tacit_interview_raw.txt"
                    if raw_path.exists():
                        raw_text = raw_path.read_text(encoding="utf-8")
                        anon_result = anonymize_text(raw_text)
                        st.session_state["mask_log"] = anon_result["mask_log"]
                        # Write anonymized file
                        clean_path = base_dir / "processed" / "tacit_interview_clean.txt"
                        clean_path.parent.mkdir(parents=True, exist_ok=True)
                        clean_path.write_text(anon_result["anonymized_text"], encoding="utf-8")

                    summary = clear_and_reingest()
                    st.session_state["ingestion_summary"] = summary
                    st.success(
                        f"✅ Yükleme tamamlandı — Toplam: {summary['total_chunks']} chunk | "
                        f"Mevzuat: {summary['explicit_chunks']} | "
                        f"Tacit: {summary['tacit_chunks']} | "
                        f"DB boyutu: {summary['collection_size']}"
                    )
                except Exception as exc:
                    logger.error("Yükleme hatası: %s", exc)
                    st.error(f"Yükleme hatası: {exc}")

    # Show current status
    summary = st.session_state.get("ingestion_summary")
    if summary:
        st.markdown("---")
        st.markdown("### Mevcut Veritabanı Durumu")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Toplam Chunk", summary.get("total_chunks", 0))
        m2.metric("Mevzuat Chunk", summary.get("explicit_chunks", 0))
        m3.metric("Tacit Chunk", summary.get("tacit_chunks", 0))
        m4.metric("DB Boyutu", summary.get("collection_size", 0))
    else:
        # Try to read existing status without triggering ingestion
        if not MOCK_MODE:
            try:
                from src.ingestion import get_ingestion_status
                status = get_ingestion_status()
                if status["collection_size"] > 0:
                    st.session_state["ingestion_summary"] = status
                    st.info(
                        f"Mevcut veritabanı bulundu: {status['collection_size']} chunk. "
                        "Yenilemek için butona tıklayın."
                    )
            except Exception:
                pass

        if not st.session_state.get("ingestion_summary"):
            st.info("ℹ️ Veritabanı henüz yüklenmedi. Başlamak için 'Veritabanını Yenile' butonuna tıklayın.")

    # Anonymization log
    mask_log = st.session_state.get("mask_log")
    if mask_log:
        with st.expander(f"🔒 Anonimleştirme Kaydı ({len(mask_log)} kayıt maskelendi)"):
            df_mask = pd.DataFrame(mask_log)
            df_mask.columns = ["Orijinal Değer", "Maskelendi"]
            st.dataframe(df_mask, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2 — Süreç Haritası ve Hata Kartları
# ===========================================================================

with tabs[1]:
    st.subheader("Süreç Haritası, Hata Kartları ve Terim Sözlüğü")
    _poll_process_map_confidence_job()
    _poll_error_cards_confidence_job()

    # --- Process Map ---
    st.markdown("#### 🗺️ Süreç Haritası")
    if st.button(
        "Süreç Haritası Oluştur", key="gen_process_map",
        help="DELF/DALF sınavcı-düzeltici değerlendirme sürecinin adım adım haritasını oluşturur. "
             "Hazırlık, Bireysel Düzeltme, Çift Düzeltme ve Finalizasyon fazlarını kapsar.",
    ):
        from src.generators import generate_process_map

        result, error = _run_via_queue(
            "Süreç haritası oluşturuluyor", generate_process_map,
            language=st.session_state.get("output_language", "tr"),
            defer_confidence=True,
        )
        if error:
            st.error(f"Hata: {error}")
        else:
            st.session_state["process_map"] = _start_process_map_confidence_job(result)
            st.session_state["approvals"]["process_map"] = {}

    pm = st.session_state.get("process_map")
    if pm:
        if "hata" in pm:
            st.error(f"İçerik üretilemedi: {pm.get('hata')}")
            if pm.get("ham_çıktı"):
                with st.expander("Ham Çıktı"):
                    st.text(pm["ham_çıktı"])
        else:
            _render_confidence_banner(pm)
            steps = pm.get("steps", [])
            if steps:
                df_pm = pd.DataFrame(steps)
                st.dataframe(df_pm, use_container_width=True, hide_index=True)
            else:
                st.warning("Adım bulunamadı.")
            if pm.get("_confidence_status") == "pending":
                _schedule_confidence_autorefresh()

    st.markdown("---")

    # --- Error Cards ---
    st.markdown("#### ⚠️ Hata Kartları")
    if st.button(
        "Hata Kartlarını Oluştur", key="gen_error_cards",
        help="Yeni DELF/DALF sınavcılarının sık yaptığı değerlendirme hatalarını kartlar halinde listeler. "
             "Her kart: hata, kök neden, tespit yöntemi ve doğru uygulama içerir.",
    ):
        from src.generators import generate_error_cards

        result, error = _run_via_queue(
            "Hata kartları oluşturuluyor", generate_error_cards,
            language=st.session_state.get("output_language", "tr"),
            defer_confidence=True,
        )
        if error:
            st.error(f"Hata: {error}")
        else:
            st.session_state["error_cards"] = _start_error_cards_confidence_job(result)
            st.session_state["approvals"]["error_cards"] = {}

    ec = st.session_state.get("error_cards")
    if ec:
        if "hata" in ec:
            st.error(f"İçerik üretilemedi: {ec.get('hata')}")
        else:
            _render_confidence_banner(ec)
            cards = ec.get("hata_kartlari", [])
            for card in cards:
                with st.expander(f"Kart {card.get('kart_no', '?')}: {card.get('hata', '')}"):
                    st.markdown(
                        f"""<div class="hata-karti">
                        <b>Hata:</b> {card.get('hata', '')}<br>
                        <b>Kök Neden:</b> {card.get('kok_neden', '')}<br>
                        <b>Tespit Yöntemi:</b> {card.get('tespit_yontemi', '')}<br>
                        <b>Doğru Uygulama:</b> {card.get('dogru_uygulama', '')}<br>
                        <small>📌 Kaynak Chunk İndeksleri: {card.get('kaynak_chunk_indeksleri', [])}</small>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            if ec.get("_confidence_status") == "pending":
                _schedule_confidence_autorefresh()

    st.markdown("---")

    # --- Glossary ---
    st.markdown("#### 📖 Terim Sözlüğü")
    if st.button(
        "Terim Sözlüğü Oluştur", key="gen_glossary",
        help="DELF/DALF sınavcıları için CECRL terim sözlüğü oluşturur. "
             "Grille, descripteur, copie atypique gibi teknik terimleri tanım ve kullanım örnekleriyle açıklar.",
    ):
        from src.generators import generate_glossary

        result, error = _run_via_queue(
            "Terim sözlüğü oluşturuluyor", generate_glossary,
            language=st.session_state.get("output_language", "tr"),
        )
        if error:
            st.error(f"Hata: {error}")
        else:
            st.session_state["glossary"] = result
            st.session_state["approvals"]["glossary"] = {}

    gl = st.session_state.get("glossary")
    if gl:
        if "hata" in gl:
            st.error(f"İçerik üretilemedi: {gl.get('hata')}")
        else:
            terms = gl.get("terimler", [])
            if terms:
                df_gl = pd.DataFrame(terms)
                search_term = st.text_input("🔍 Terim ara:", key="glossary_search", placeholder="Örn: matrah")
                if search_term:
                    mask = df_gl.apply(
                        lambda row: search_term.lower() in str(row).lower(), axis=1
                    )
                    df_gl = df_gl[mask]
                st.dataframe(df_gl, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 3 — İnteraktif Simülasyon
# ===========================================================================

with tabs[2]:
    st.subheader("İnteraktif Simülasyon Senaryosu")

    if st.button(
        "🎲 Yeni Senaryo Oluştur", key="gen_simulation",
        help="Gerçekçi bir DELF/DALF değerlendirme karar senaryosu üretir. "
             "Atipik kopya, puan uyuşmazlığı veya bant sınır kararları gibi kritik durumları test eder.",
    ):
        from src.generators import generate_simulation_scenario

        result, error = _run_via_queue(
            "Senaryo oluşturuluyor", generate_simulation_scenario,
            language=st.session_state.get("output_language", "tr"),
        )
        if error:
            st.error(f"Hata: {error}")
        else:
            st.session_state["simulation"] = result
            st.session_state["sim_selected"] = None
            st.session_state["sim_answered"] = False
            st.session_state["approvals"]["simulation"] = {}

    sim = st.session_state.get("simulation")
    if sim:
        if "hata" in sim:
            st.error(f"Senaryo üretilemedi: {sim.get('hata')}")
        else:
            st.markdown(f"### {sim.get('senaryo_basligi', '')}")
            st.info(sim.get("durum_aciklamasi", ""))
            st.markdown(f"**❓ Soru:** {sim.get('soru', '')}")
            st.markdown("---")

            options = sim.get("secenekler", [])
            option_texts = [f"{opt['id']}) {opt['metin']}" for opt in options]

            answered = st.session_state.get("sim_answered", False)

            selected_label = st.radio(
                "Seçeneğinizi seçin:",
                options=option_texts,
                key="sim_radio",
                disabled=answered,
                index=None,
            )

            if selected_label and not answered:
                if st.button("✔️ Cevabı Onayla", key="confirm_answer"):
                    selected_id = selected_label.split(")")[0].strip()
                    st.session_state["sim_selected"] = selected_id
                    st.session_state["sim_answered"] = True

            if answered:
                selected_id = st.session_state.get("sim_selected")
                for opt in options:
                    if opt["id"] == selected_id:
                        if opt.get("dogru_mu"):
                            st.success(f"✅ **Doğru!** {opt.get('geri_bildirim', '')}")
                            st.markdown(f"**Sonuç:** {opt.get('sonuc', '')}")
                        else:
                            st.error(f"❌ **Yanlış.** {opt.get('geri_bildirim', '')}")
                            st.markdown(f"**Sonuç:** {opt.get('sonuc', '')}")

                        # Show correct answer
                        for correct_opt in options:
                            if correct_opt.get("dogru_mu") and correct_opt["id"] != selected_id:
                                st.info(
                                    f"💡 Doğru cevap: **{correct_opt['id']}**) {correct_opt['metin']}"
                                )
                        break

                st.markdown(f"🎯 **Öğrenme Hedefi:** {sim.get('ogrenme_hedefi', '')}")

                with st.expander("📎 Kaynak Referansları"):
                    st.write(
                        "Bu senaryo aşağıdaki chunk indekslerindeki bilgilere dayanmaktadır:"
                    )
                    st.json(sim.get("kaynak_chunk_indeksleri", []))

# ===========================================================================
# TAB 4 — Uzman Onay Paneli
# ===========================================================================

with tabs[3]:
    st.subheader("Uzman Onay Paneli")

    st.warning(
        "⚠️ Onaylanmamış içerikler eğitime aktarılmaz. Lütfen tüm içerikleri inceleyin."
    )

    from src.retrieval import lookup_parent_context

    approvals = st.session_state["approvals"]
    total_items = 0
    approved_count = 0
    rejected_count = 0

    # ---- helper to render approval controls ----
    def render_approval_section(
        section_key: str,
        section_title: str,
        items: list,
        label_fn,
        detail_fn,
        citations_fn=None,
    ) -> tuple[int, int, int]:
        """
        Render an approval section with approve/reject buttons per item.

        Returns (total, approved, rejected) counts for this section so the
        caller can accumulate them — avoids nonlocal, which requires an
        enclosing *function* scope (not a module-level with-block).

        Parameters
        ----------
        section_key : str
            Key in st.session_state["approvals"].
        section_title : str
            Display heading.
        items : list
            List of content dicts to review.
        label_fn : callable
            Returns display label string for an item dict.
        detail_fn : callable
            Returns detail string for an item dict.
        """
        section_total = 0
        section_approved = 0
        section_rejected = 0

        st.markdown(f"#### {section_title}")
        if not items:
            st.caption("Bu bölüm için henüz içerik üretilmedi.")
            return 0, 0, 0

        for idx, item in enumerate(items):
            section_total += 1
            current_state = approvals[section_key].get(idx)
            label = label_fn(item)
            detail = detail_fn(item)

            with st.expander(f"{label} — {'✅ Onaylandı' if current_state == 'onaylandi' else '❌ Reddedildi' if current_state == 'reddedildi' else '⏳ İnceleniyor'}"):
                st.markdown(detail)

                # Show source chunk details if citations are available
                if citations_fn is not None:
                    citation_ids = citations_fn(item)
                    if citation_ids:
                        with st.expander("📎 Kaynak Chunk Detayları", expanded=False):
                            for pid in citation_ids:
                                ctx = lookup_parent_context(str(pid))
                                if ctx:
                                    src_label = "📘 Mevzuat" if ctx["source"] == "explicit" else "🧠 Tacit"
                                    st.markdown(
                                        f"**{src_label}** — `{pid}` | {ctx['filename']}"
                                    )
                                    st.markdown("**Üst Bağlam — LLM'ye gönderilen:**")
                                    st.info(ctx["parent_text"])
                                    if ctx["children"]:
                                        st.markdown("**Eşleşen Alt Parçalar — vektör aramasında bulunan:**")
                                        for child in ctx["children"]:
                                            st.success(
                                                f"Alt Parça #{child['child_index']}: {child['text']}"
                                            )
                                    st.markdown("---")
                                else:
                                    st.caption(f"`{pid}` — chunk bulunamadı (veritabanı yenilenmesi gerekebilir)")

                col_a, col_r, col_c = st.columns([1, 1, 4])
                with col_a:
                    if st.button("✓ Onayla", key=f"approve_{section_key}_{idx}"):
                        approvals[section_key][idx] = "onaylandi"
                        st.rerun()
                with col_r:
                    if st.button("✗ Reddet", key=f"reject_{section_key}_{idx}"):
                        approvals[section_key][idx] = "reddedildi"
                        st.rerun()

            if current_state == "onaylandi":
                section_approved += 1
            elif current_state == "reddedildi":
                section_rejected += 1

        return section_total, section_approved, section_rejected

    # --- Process Map ---
    pm = st.session_state.get("process_map")
    pm_items = pm.get("steps", []) if pm and "hata" not in pm else []
    if pm and "hata" not in pm:
        _render_confidence_banner(pm)
    t, a, r = render_approval_section(
        "process_map",
        "🗺️ Süreç Haritası Adımları",
        pm_items,
        label_fn=lambda s: f"Adım {s.get('adim_no', '?')}: {s.get('baslik', '')}",
        detail_fn=lambda s: (
            f"**Giriş:** {s.get('giris', '')}\n\n"
            f"**Çıkış:** {s.get('cikis', '')}\n\n"
            f"**Karar Noktası:** {s.get('karar_noktasi', '')}\n\n"
            f"**Risk:** {s.get('risk', '')}\n\n"
            f"**Kontrol:** {s.get('kontrol', '')}\n\n"
            f"📌 Kaynak: {s.get('kaynak_chunk_indeksleri', [])}"
        ),
        citations_fn=lambda s: s.get("kaynak_chunk_indeksleri", []),
    )
    total_items += t; approved_count += a; rejected_count += r

    st.markdown("---")

    # --- Error Cards ---
    ec = st.session_state.get("error_cards")
    ec_items = ec.get("hata_kartlari", []) if ec and "hata" not in ec else []
    if ec and "hata" not in ec:
        _render_confidence_banner(ec)
    t, a, r = render_approval_section(
        "error_cards",
        "⚠️ Hata Kartları",
        ec_items,
        label_fn=lambda c: f"Kart {c.get('kart_no', '?')}: {c.get('hata', '')[:60]}",
        detail_fn=lambda c: (
            f"**Hata:** {c.get('hata', '')}\n\n"
            f"**Kök Neden:** {c.get('kok_neden', '')}\n\n"
            f"**Tespit Yöntemi:** {c.get('tespit_yontemi', '')}\n\n"
            f"**Doğru Uygulama:** {c.get('dogru_uygulama', '')}\n\n"
            f"📌 Kaynak: {c.get('kaynak_chunk_indeksleri', [])}"
        ),
        citations_fn=lambda c: c.get("kaynak_chunk_indeksleri", []),
    )
    total_items += t; approved_count += a; rejected_count += r

    st.markdown("---")

    # --- Glossary ---
    gl = st.session_state.get("glossary")
    gl_items = gl.get("terimler", []) if gl and "hata" not in gl else []
    if gl and "hata" not in gl:
        _render_confidence_banner(gl)
    t, a, r = render_approval_section(
        "glossary",
        "📖 Terim Sözlüğü",
        gl_items,
        label_fn=lambda t_: f"Terim: {t_.get('terim', '')}",
        detail_fn=lambda t_: (
            f"**Tanım:** {t_.get('tanim', '')}\n\n"
            f"**Kullanım Örneği:** {t_.get('kullanim_ornegi', '')}\n\n"
            f"📌 Kaynak: {t_.get('kaynak_chunk_indeksleri', [])}"
        ),
        citations_fn=lambda t_: t_.get("kaynak_chunk_indeksleri", []),
    )
    total_items += t; approved_count += a; rejected_count += r

    st.markdown("---")

    # --- Simulation ---
    sim = st.session_state.get("simulation")
    sim_items = [sim] if sim and "hata" not in sim else []
    if sim and "hata" not in sim:
        _render_confidence_banner(sim)
    t, a, r = render_approval_section(
        "simulation",
        "🎮 Simülasyon Senaryosu",
        sim_items,
        label_fn=lambda s: f"Senaryo: {s.get('senaryo_basligi', '')}",
        detail_fn=lambda s: (
            f"**Durum:** {s.get('durum_aciklamasi', '')}\n\n"
            f"**Soru:** {s.get('soru', '')}\n\n"
            f"**Öğrenme Hedefi:** {s.get('ogrenme_hedefi', '')}\n\n"
            f"📌 Kaynak: {s.get('kaynak_chunk_indeksleri', [])}"
        ),
        citations_fn=lambda s: s.get("kaynak_chunk_indeksleri", []),
    )
    total_items += t; approved_count += a; rejected_count += r

    st.markdown("---")

    # Summary
    st.markdown("### Onay Özeti")
    pending = total_items - approved_count - rejected_count
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam İçerik", total_items)
    c2.metric("✅ Onaylanan", approved_count)
    c3.metric("❌ Reddedilen", rejected_count)
    c4.metric("⏳ Bekleyen", pending)

    if total_items > 0 and pending == 0:
        st.success("🎉 Tüm içerikler incelendi!")
    elif total_items == 0:
        st.info("Henüz onaylanacak içerik yok. Lütfen diğer sekmelerde içerik üretin.")

# ===========================================================================
# TAB 5 — Prompt Optimizer
# ===========================================================================

with tabs[4]:
    st.subheader("🔬 Prompt Optimizer")
    st.markdown(
        "Her üretici için **3 farklı sorgu + talimat varyantı** çalıştırır, "
        "Gemini'yi hakem olarak kullanarak her birini **dayandırma, tamlama ve pratik fayda** "
        "kriterlerinde puanlar ve en iyisini kaydeder. "
        "Kaydedilen prompt, ilgili üreticinin sonraki çalıştırmasında otomatik olarak kullanılır."
    )
    st.info(
        "💡 **autoresearch ilhamı:** Karpathy'nin autoresearch projesindeki döngüden ilham alınmıştır — "
        "train.py'yi değiştir → 5 dk eğit → val_bpb ölç → en iyiyi sakla. "
        "Burada: sorgu+talimatı değiştir → içerik üret → LLM hakemiyle puanla → en iyiyi sakla."
    )

    if MOCK_MODE:
        st.warning("⚠️ MOCK MODU aktif — Optimizer gerçek API çağrısı yapar. Lütfen MOCK_MODE=false yapın.")

    st.markdown("---")

    # --- Current saved prompts ---
    from src.prompt_optimizer import load_optimized_prompts, GENERATOR_LABELS

    saved = load_optimized_prompts()
    if saved:
        st.markdown("#### Mevcut Kaydedilmiş Promptlar")
        for key, val in saved.items():
            label = GENERATOR_LABELS.get(key, key)
            scores = val.get("scores", {})
            total = scores.get("total", "—")
            with st.expander(f"**{label}** — Varyant: {val.get('name', '?')} | Toplam puan: {total}/30"):
                st.markdown(f"**Sorgu:** `{val.get('query', '')}`")
                st.markdown(f"**Dayandırma:** {scores.get('grounding', '—')} | "
                            f"**Tamlama:** {scores.get('completeness', '—')} | "
                            f"**Pratik Fayda:** {scores.get('usefulness', '—')}")
                if scores.get("reasoning"):
                    st.caption(f"Hakem gerekçesi: {scores['reasoning']}")
        st.markdown("---")

    # --- Run optimizer ---
    st.markdown("#### Yeni Optimizasyon Çalıştır")

    generator_options = {v: k for k, v in GENERATOR_LABELS.items()}
    selected_label = st.selectbox(
        "Optimize edilecek üretici:",
        options=list(GENERATOR_LABELS.values()),
        key="optimizer_select",
    )
    selected_key = generator_options[selected_label]

    col_run, col_save = st.columns([1, 1])

    with col_run:
        run_btn = st.button(
            "▶ Optimize Et",
            key="run_optimizer",
            use_container_width=True,
            help="3 varyant çalıştırır ve LLM hakemiyle puanlar (6 API çağrısı)",
        )

    if run_btn:
        from src.prompt_optimizer import run_optimization, VARIANTS

        progress_placeholder = st.empty()
        variant_count = len(VARIANTS[selected_key])

        def _on_progress(idx, name, stage):
            stage_label = "üretiliyor" if stage == "generating" else "puanlanıyor"
            progress_placeholder.info(
                f"⏳ Varyant {idx + 1}/{variant_count}: **{name}** — {stage_label}..."
            )

        with st.spinner("Optimizasyon çalışıyor..."):
            try:
                results = run_optimization(selected_key, progress_callback=_on_progress)
                st.session_state["optimizer_results"] = results
                st.session_state["optimizer_generator"] = selected_key
                progress_placeholder.empty()
                st.success(f"✅ Optimizasyon tamamlandı! En iyi varyant: **{results[0]['name']}** ({results[0]['total_score']}/30)")
            except Exception as exc:
                progress_placeholder.empty()
                st.error(f"Optimizasyon hatası: {exc}")

    # --- Show results ---
    results = st.session_state.get("optimizer_results")
    opt_gen = st.session_state.get("optimizer_generator")

    if results and opt_gen == selected_key:
        st.markdown("#### Sonuçlar")

        rows = []
        for rank, r in enumerate(results, start=1):
            rows.append({
                "Sıra": f"{'🥇' if rank == 1 else '🥈' if rank == 2 else '🥉'} {rank}",
                "Varyant": r["name"],
                "Sorgu": r["query"],
                "Dayandırma": r["scores"].get("grounding", 0),
                "Tamlama": r["scores"].get("completeness", 0),
                "Pratik Fayda": r["scores"].get("usefulness", 0),
                "Toplam /30": r["total_score"],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Reasoning for each
        for rank, r in enumerate(results, start=1):
            reasoning = r["scores"].get("reasoning", "")
            if reasoning:
                st.caption(f"{'🥇' if rank == 1 else '🥈' if rank == 2 else '🥉'} **{r['name']}** — {reasoning}")

        st.markdown("---")

        # Save best
        with col_save:
            save_btn = st.button(
                f"💾 En İyiyi Kaydet ({results[0]['name']})",
                key="save_best_prompt",
                use_container_width=True,
                disabled=(opt_gen != selected_key),
            )

        if save_btn and results:
            from src.prompt_optimizer import save_best_prompt
            save_best_prompt(selected_key, results[0])
            st.success(
                f"✅ **{GENERATOR_LABELS.get(selected_key, selected_key)}** için "
                f"**{results[0]['name']}** promptu kaydedildi. "
                "Bir sonraki içerik üretiminde otomatik kullanılacak."
            )
            st.rerun()

# ===========================================================================
# TAB 6 — Veritabanı Gezgini
# ===========================================================================

with tabs[5]:
    st.subheader("🗄️ Veritabanı Gezgini")
    st.markdown(
        "Vektör veritabanında saklanan tüm bilgi parçacıklarını (chunk) burada görebilirsiniz. "
        "Her kart, yapay zekanın öğrendiği bir bilgi birimidir."
    )

    import chromadb as _chromadb
    from pathlib import Path as _Path

    _chroma_dir = _Path(__file__).resolve().parent / "chroma_db"

    @st.cache_data(show_spinner=False, ttl=30)
    def _load_all_chunks():
        try:
            client = _chromadb.PersistentClient(path=str(_chroma_dir))
            col = client.get_or_create_collection(
                name="tee_children",
                metadata={"hnsw:space": "cosine"},
            )
            if col.count() == 0:
                return []
            data = col.get(include=["documents", "metadatas"])
            chunks = []
            for doc, meta in zip(data["documents"], data["metadatas"]):
                original = meta.get("original_text", doc)
                # DELF corpus uses source_filename + doc_type; fall back to old keys
                filename = meta.get("source_filename", meta.get("filename", "bilinmiyor"))
                doc_type = meta.get("doc_type", meta.get("source", "bilinmiyor"))
                chunks.append({
                    "text": original,
                    "embedded_text": doc,
                    "enriched": bool(meta.get("enriched", False)),
                    "doc_type": doc_type,
                    "filename": filename,
                    "level": meta.get("level", ""),
                    "skill": meta.get("skill", ""),
                    "chunk_index": meta.get("child_index", 0),
                    "parent_id": meta.get("parent_id", ""),
                    "char_count": meta.get("char_count", len(original)),
                })
            chunks.sort(key=lambda c: (c["doc_type"], c["filename"], c["chunk_index"]))
            return chunks
        except Exception as exc:
            return []

    all_chunks = _load_all_chunks()

    if not all_chunks:
        st.info("ℹ️ Veritabanı boş. Önce 'Veri Yükleme' sekmesinden veritabanını yükleyin.")
    else:
        from collections import Counter as _Counter
        type_counts = _Counter(c["doc_type"] for c in all_chunks)
        unique_files = len(set(c["filename"] for c in all_chunks))

        # --- Top metrics ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Toplam Bilgi Parçası", len(all_chunks))
        m2.metric("📄 Kaynak Doküman", unique_files)
        m3.metric("🏷️ Doküman Tipi", len(type_counts))

        st.markdown("---")

        # --- Filters ---
        doc_types_sorted = sorted(type_counts.keys())
        levels_sorted = sorted(set(c["level"] for c in all_chunks if c["level"]))
        skills_sorted = sorted(set(c["skill"] for c in all_chunks if c["skill"]))

        fcol1, fcol2, fcol3, fcol4 = st.columns([1.5, 1, 1, 2])
        with fcol1:
            type_opts = ["Tümü"] + doc_types_sorted
            doc_type_filter = st.selectbox(
                "Doküman tipi:",
                options=type_opts,
                key="db_doctype_filter",
            )
        with fcol2:
            level_filter = st.selectbox(
                "Seviye:",
                options=["Tümü"] + levels_sorted,
                key="db_level_filter",
            )
        with fcol3:
            skill_filter = st.selectbox(
                "Beceri:",
                options=["Tümü"] + skills_sorted,
                key="db_skill_filter",
            )
        with fcol4:
            search_query = st.text_input(
                "🔍 Metin içinde ara:",
                placeholder="Örn: grille, descripteur, niveau...",
                key="db_search",
            )

        visible = all_chunks
        if doc_type_filter != "Tümü":
            visible = [c for c in visible if c["doc_type"] == doc_type_filter]
        if level_filter != "Tümü":
            visible = [c for c in visible if c["level"] == level_filter]
        if skill_filter != "Tümü":
            visible = [c for c in visible if c["skill"] == skill_filter]
        if search_query.strip():
            q = search_query.strip().lower()
            visible = [c for c in visible if q in c["text"].lower()]

        st.caption(f"{len(visible)} parça gösteriliyor")
        st.markdown("---")

        # doc_type → CSS class mapping
        _OFFICIAL_TYPES = {"grille", "descripteur", "manuel", "methodologie", "explicit"}
        _TACIT_TYPES    = {"tacit", "stagiaire", "brouillon"}

        # --- Render chunk cards ---
        for chunk in visible:
            dt = chunk["doc_type"]
            if dt in _OFFICIAL_TYPES:
                css_cls   = "mevzuat"
                badge_cls = "badge-mevzuat"
            elif dt in _TACIT_TYPES:
                css_cls   = "tacit"
                badge_cls = "badge-tacit"
            else:
                css_cls   = "mevzuat"
                badge_cls = "badge-mevzuat"

            level_tag = f" · {chunk['level']}" if chunk.get("level") else ""
            skill_tag = f" · {chunk['skill']}" if chunk.get("skill") else ""
            badge_lbl = dt.upper() if dt != "bilinmiyor" else "BILINMIYOR"

            enriched_badge = (
                "<span class='chunk-badge' style='background:#8e44ad;color:white;'>📑 Zenginleştirildi</span>"
                if chunk.get("enriched") else ""
            )
            with st.expander(
                f"[{badge_lbl}]{level_tag}{skill_tag}  ·  {chunk['filename']}  ·  "
                f"Alt Parça #{chunk['chunk_index']}  ·  {chunk['char_count']} karakter"
            ):
                st.markdown(
                    f"""<div class="chunk-card {css_cls}">
                    <div class="chunk-meta">
                        <span class="chunk-badge {badge_cls}">{badge_lbl}</span>
                        {enriched_badge}
                        Dosya: <b>{chunk['filename']}</b> &nbsp;·&nbsp;
                        Seviye: <b>{chunk.get('level') or '—'}</b> &nbsp;·&nbsp;
                        Beceri: <b>{chunk.get('skill') or '—'}</b> &nbsp;·&nbsp;
                        Alt parça: <b>#{chunk['chunk_index']}</b> &nbsp;·&nbsp;
                        Üst parça: <b>{chunk['parent_id']}</b> &nbsp;·&nbsp;
                        {chunk['char_count']} karakter
                    </div>
                    <div class="chunk-text">{chunk['text'].replace(chr(10), '<br>')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if chunk.get("enriched"):
                    with st.expander("🔬 Gömme için kullanılan zenginleştirilmiş metin"):
                        st.code(chunk["embedded_text"], language=None)

# ===========================================================================
# TAB 7 — Asistan (sohbet)
# ===========================================================================

with tabs[6]:
    _lang = st.session_state.get("output_language", "tr")
    _is_tr = _lang == "tr"

    st.subheader("💬 Asistan" if _is_tr else "💬 Assistant")
    st.markdown(
        (
            "DELF/DALF değerlendirme protokolü, grille'ler, descripteur'lar ve "
            "metodoloji hakkında serbestçe soru sorun. Her olgusal yanıt korpus "
            "kaynaklarıyla **dayandırılır ve alıntılanır**. Asistan nihai puan "
            "kararı vermez ve sınav içeriği üretmez."
        )
        if _is_tr
        else (
            "Posez librement vos questions sur le protocole d'évaluation DELF/DALF, "
            "les grilles, les descripteurs et la méthodologie. Chaque réponse "
            "factuelle est **ancrée et citée** depuis le corpus. L'assistant ne "
            "décide pas de la note finale et ne génère pas de contenu d'examen."
        )
    )

    if MOCK_MODE:
        st.warning(
            "⚠️ MOCK MODU — sabit örnek yanıt döner, API çağrısı yapılmaz."
            if _is_tr
            else "⚠️ MODE MOCK — réponse fixe, aucun appel API."
        )

    # --- Toolbar: turn counter + new conversation ---
    _turns = st.session_state.get("chat_turns", 0)
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        st.caption(
            f"Tur: {_turns}/{MAX_TURNS}" if _is_tr else f"Tour : {_turns}/{MAX_TURNS}"
        )
    with tcol2:
        if st.button(
            "🔄 Yeni Konuşma" if _is_tr else "🔄 Nouvelle conversation",
            key="chat_reset",
            use_container_width=True,
            help=(
                "Konuşma geçmişini temizler ve tur sayacını sıfırlar."
                if _is_tr
                else "Efface l'historique et réinitialise le compteur de tours."
            ),
        ):
            st.session_state["chat_history"] = []
            st.session_state["chat_turns"] = 0
            st.rerun()

    # --- Example prompts (shown only on an empty conversation) ---
    if not st.session_state["chat_history"]:
        _examples_tr = [
            "B2 PE söylem tutarlılığı kriterinde bant 2 ne anlama gelir?",
            "İki düzeltici 2 banttan fazla farklı puan verdiğinde ne yapmalıyım?",
            "DELF düzeltmesinde halo etkisi nedir?",
            "Beni B2 PE üzerine kısa bir quizle sına.",
        ]
        _examples_fr = [
            "Que signifie la bande 2 pour le critère de cohérence en B2 PE ?",
            "Que faire si deux correcteurs divergent de plus de 2 bandes ?",
            "Qu'est-ce que l'effet de halo dans la correction DELF ?",
            "Lance-moi un mini-quiz sur le B2 PE.",
        ]
        st.markdown("**Örnek sorular:**" if _is_tr else "**Exemples de questions :**")
        for _i, _ex in enumerate(_examples_tr if _is_tr else _examples_fr):
            if st.button(_ex, key=f"chat_example_{_i}", use_container_width=True):
                st.session_state["chat_inject"] = _ex
                st.rerun()

    # Consume an injected message from example buttons (cleared immediately so it
    # doesn't fire again on the next rerun after the conversation is appended).
    _inject: str | None = st.session_state.get("chat_inject")
    if _inject:
        st.session_state["chat_inject"] = None

    # --- Render conversation history ---
    for _msg in st.session_state["chat_history"]:
        with st.chat_message(_msg["role"]):
            st.markdown(_msg["content"])
            _srcs = _msg.get("sources") or []
            if _srcs:
                with st.expander(
                    (f"📎 Kaynaklar ({len(_srcs)})" if _is_tr else f"📎 Sources ({len(_srcs)})")
                ):
                    _citation_report = _srcs[0].get("_citation_report") or {}
                    if _citation_report:
                        _passed = bool(_citation_report.get("passed"))
                        _cited = int(_citation_report.get("cited_count", 0) or 0)
                        _retrieved = int(_citation_report.get("retrieved_count", len(_srcs)) or 0)
                        _invalid = _citation_report.get("invalid_parent_ids") or []
                        st.caption(
                            (
                                f"✅ Atıf kontrolü geçti · {_cited}/{_retrieved} kaynak kullanıldı"
                                if _passed
                                else f"⚠️ Atıf kontrolü uyarısı · {_cited}/{_retrieved} kaynak kullanıldı"
                            )
                            if _is_tr
                            else (
                                f"✅ Citations validées · {_cited}/{_retrieved} sources utilisées"
                                if _passed
                                else f"⚠️ Avertissement citations · {_cited}/{_retrieved} sources utilisées"
                            )
                        )
                        if _invalid:
                            st.caption(
                                ("Doğrulanamayan parent_id: " if _is_tr else "parent_id non validé : ")
                                + ", ".join(_invalid)
                            )
                    for _s in _srcs:
                        st.markdown(
                            f"**{_s.get('filename', '?')}** — `{_s.get('parent_id', '')}` "
                            f"· skor: {_s.get('score', 0)}"
                        )
                        if _s.get("snippet"):
                            st.caption(_s["snippet"])

    # --- Turn limit gate ---
    _limit_reached = _turns >= MAX_TURNS
    if _limit_reached:
        st.warning(
            "Bu konuşma tur sınırına ulaştı. Bağlam kaymasını önlemek için "
            "lütfen **🔄 Yeni Konuşma** ile yeni bir konuşma başlatın."
            if _is_tr
            else "Cette conversation a atteint la limite de tours. Démarrez une "
            "**🔄 Nouvelle conversation** pour éviter la dérive de contexte."
        )

    # --- Chat input ---
    _user_msg = st.chat_input(
        ("Sorunuzu yazın..." if _is_tr else "Saisissez votre question..."),
        key="chat_input",
        disabled=_limit_reached,
    )

    _active_msg = (_user_msg or "").strip() or (_inject or "").strip()

    if _active_msg:
        history_before = list(st.session_state["chat_history"])
        st.session_state["chat_history"].append(
            {"role": "user", "content": _active_msg, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(_active_msg)

        with st.chat_message("assistant"):
            # Retrieval senkrondur (spinner); LLM yanıtı akışla render edilir.
            with st.spinner("Kaynaklar aranıyor..." if _is_tr else "Recherche des sources..."):
                _stream, _sources = chat_stream(
                    _active_msg, history=history_before, language=_lang
                )
            _answer = st.write_stream(_stream)

        _citation_report = validate_citations(_answer, _sources) if _sources else {}
        if _citation_report:
            _sources = [{**_s, "_citation_report": _citation_report} for _s in _sources]

        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": _answer,
                "sources": _sources,
            }
        )
        st.session_state["chat_turns"] = _turns + 1
        st.rerun()
