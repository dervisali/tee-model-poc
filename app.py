"""
TEE-Model POC — Streamlit Uygulaması

Maaş mutemedi onboarding eğitimi için yapay zeka destekli içerik üretimi.
Veri yükleme, süreç haritası, hata kartları, terim sözlüğü, simülasyon
ve uzman onay panelini tek arayüzde sunar.
"""

import os
import sys
import logging

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Ensure src/ is importable when run from project root
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TEE-Model POC — Maaş Mutemedi Eğitim Sistemi",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "ingestion_summary": None,
    "mask_log": None,
    "process_map": None,
    "error_cards": None,
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
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🏛️ TEE-Model — Tacit-Explicit Entegre Eğitim Sistemi")
st.caption(
    "Maaş Mutemedi Onboarding POC | "
    + ("⚠️ MOCK MODU AKTİF — API çağrısı yapılmıyor" if MOCK_MODE else "🟢 Canlı API Modu")
)

tabs = st.tabs(
    [
        "📂 Veri Yükleme ve İşleme",
        "🗺️ Süreç Haritası ve Hata Kartları",
        "🎮 İnteraktif Simülasyon",
        "✅ Uzman Onay Paneli",
        "🔬 Prompt Optimizer",
        "🗄️ Veritabanı Gezgini",
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

    # --- Process Map ---
    st.markdown("#### 🗺️ Süreç Haritası")
    if st.button("Süreç Haritası Oluştur", key="gen_process_map"):
        from src.generators import generate_process_map

        with st.spinner("Süreç haritası oluşturuluyor..."):
            try:
                result = generate_process_map()
                st.session_state["process_map"] = result
                # Reset approvals for this content
                st.session_state["approvals"]["process_map"] = {}
            except Exception as exc:
                st.error(f"Hata: {exc}")

    pm = st.session_state.get("process_map")
    if pm:
        if "hata" in pm:
            st.error(f"İçerik üretilemedi: {pm.get('hata')}")
            if pm.get("ham_çıktı"):
                with st.expander("Ham Çıktı"):
                    st.text(pm["ham_çıktı"])
        else:
            steps = pm.get("steps", [])
            if steps:
                df_pm = pd.DataFrame(steps)
                st.dataframe(df_pm, use_container_width=True, hide_index=True)
            else:
                st.warning("Adım bulunamadı.")

    st.markdown("---")

    # --- Error Cards ---
    st.markdown("#### ⚠️ Hata Kartları")
    if st.button("Hata Kartlarını Oluştur", key="gen_error_cards"):
        from src.generators import generate_error_cards

        with st.spinner("Hata kartları oluşturuluyor..."):
            try:
                result = generate_error_cards()
                st.session_state["error_cards"] = result
                st.session_state["approvals"]["error_cards"] = {}
            except Exception as exc:
                st.error(f"Hata: {exc}")

    ec = st.session_state.get("error_cards")
    if ec:
        if "hata" in ec:
            st.error(f"İçerik üretilemedi: {ec.get('hata')}")
        else:
            cards = ec.get("hata_kartlari", [])
            for card in cards:
                with st.expander(f"Kart {card.get('kart_no', '?')}: {card.get('hata', '')}"):
                    st.markdown(
                        f"""<div class="hata-karti">
                        <b>Hata:</b> {card.get('hata', '')}<br>
                        <b>Kök Neden:</b> {card.get('kök_neden', '')}<br>
                        <b>Tespit Yöntemi:</b> {card.get('tespit_yöntemi', '')}<br>
                        <b>Doğru Uygulama:</b> {card.get('dogru_uygulama', '')}<br>
                        <small>📌 Kaynak Chunk İndeksleri: {card.get('kaynak_chunk_indeksleri', [])}</small>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # --- Glossary ---
    st.markdown("#### 📖 Terim Sözlüğü")
    if st.button("Terim Sözlüğü Oluştur", key="gen_glossary"):
        from src.generators import generate_glossary

        with st.spinner("Terim sözlüğü oluşturuluyor..."):
            try:
                result = generate_glossary()
                st.session_state["glossary"] = result
                st.session_state["approvals"]["glossary"] = {}
            except Exception as exc:
                st.error(f"Hata: {exc}")

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

    if st.button("🎲 Yeni Senaryo Oluştur", key="gen_simulation"):
        from src.generators import generate_simulation_scenario

        with st.spinner("Senaryo oluşturuluyor..."):
            try:
                result = generate_simulation_scenario()
                st.session_state["simulation"] = result
                st.session_state["sim_selected"] = None
                st.session_state["sim_answered"] = False
                st.session_state["approvals"]["simulation"] = {}
            except Exception as exc:
                st.error(f"Hata: {exc}")

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
            f"📌 Chunk indeksleri: {s.get('kaynak_chunk_indeksleri', [])}"
        ),
    )
    total_items += t; approved_count += a; rejected_count += r

    st.markdown("---")

    # --- Error Cards ---
    ec = st.session_state.get("error_cards")
    ec_items = ec.get("hata_kartlari", []) if ec and "hata" not in ec else []
    t, a, r = render_approval_section(
        "error_cards",
        "⚠️ Hata Kartları",
        ec_items,
        label_fn=lambda c: f"Kart {c.get('kart_no', '?')}: {c.get('hata', '')[:60]}",
        detail_fn=lambda c: (
            f"**Hata:** {c.get('hata', '')}\n\n"
            f"**Kök Neden:** {c.get('kök_neden', '')}\n\n"
            f"**Tespit Yöntemi:** {c.get('tespit_yöntemi', '')}\n\n"
            f"**Doğru Uygulama:** {c.get('dogru_uygulama', '')}\n\n"
            f"📌 Chunk indeksleri: {c.get('kaynak_chunk_indeksleri', [])}"
        ),
    )
    total_items += t; approved_count += a; rejected_count += r

    st.markdown("---")

    # --- Glossary ---
    gl = st.session_state.get("glossary")
    gl_items = gl.get("terimler", []) if gl and "hata" not in gl else []
    t, a, r = render_approval_section(
        "glossary",
        "📖 Terim Sözlüğü",
        gl_items,
        label_fn=lambda t_: f"Terim: {t_.get('terim', '')}",
        detail_fn=lambda t_: (
            f"**Tanım:** {t_.get('tanim', '')}\n\n"
            f"**Kullanım Örneği:** {t_.get('kullanim_ornegi', '')}\n\n"
            f"📌 Chunk indeksleri: {t_.get('kaynak_chunk_indeksleri', [])}"
        ),
    )
    total_items += t; approved_count += a; rejected_count += r

    st.markdown("---")

    # --- Simulation ---
    sim = st.session_state.get("simulation")
    sim_items = [sim] if sim and "hata" not in sim else []
    t, a, r = render_approval_section(
        "simulation",
        "🎮 Simülasyon Senaryosu",
        sim_items,
        label_fn=lambda s: f"Senaryo: {s.get('senaryo_basligi', '')}",
        detail_fn=lambda s: (
            f"**Durum:** {s.get('durum_aciklamasi', '')}\n\n"
            f"**Soru:** {s.get('soru', '')}\n\n"
            f"**Öğrenme Hedefi:** {s.get('ogrenme_hedefi', '')}\n\n"
            f"📌 Chunk indeksleri: {s.get('kaynak_chunk_indeksleri', [])}"
        ),
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
                name="tee_knowledge_base",
                metadata={"hnsw:space": "cosine"},
            )
            if col.count() == 0:
                return []
            data = col.get(include=["documents", "metadatas"])
            chunks = []
            for doc, meta in zip(data["documents"], data["metadatas"]):
                chunks.append({
                    "text": doc,
                    "source": meta.get("source", "bilinmiyor"),
                    "filename": meta.get("filename", "bilinmiyor"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "char_count": meta.get("char_count", len(doc)),
                })
            chunks.sort(key=lambda c: (c["source"], c["chunk_index"]))
            return chunks
        except Exception as exc:
            return []

    all_chunks = _load_all_chunks()

    if not all_chunks:
        st.info("ℹ️ Veritabanı boş. Önce 'Veri Yükleme' sekmesinden veritabanını yükleyin.")
    else:
        mevzuat_chunks = [c for c in all_chunks if c["source"] == "explicit"]
        tacit_chunks   = [c for c in all_chunks if c["source"] == "tacit"]

        # --- Top metrics ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Toplam Bilgi Parçası", len(all_chunks))
        m2.metric("📘 Mevzuat (Resmi Kural)", len(mevzuat_chunks))
        m3.metric("🧠 Tacit (Deneyim)", len(tacit_chunks))

        st.markdown("---")

        # --- Filters and search ---
        fcol1, fcol2 = st.columns([1, 2])
        with fcol1:
            source_filter = st.radio(
                "Kaynak filtresi:",
                options=["Tümü", "📘 Mevzuat", "🧠 Tacit"],
                horizontal=True,
                key="db_source_filter",
            )
        with fcol2:
            search_query = st.text_input(
                "🔍 Metin içinde ara:",
                placeholder="Örn: maaş, vergi, ödeme...",
                key="db_search",
            )

        if source_filter == "📘 Mevzuat":
            visible = mevzuat_chunks
        elif source_filter == "🧠 Tacit":
            visible = tacit_chunks
        else:
            visible = all_chunks

        if search_query.strip():
            q = search_query.strip().lower()
            visible = [c for c in visible if q in c["text"].lower()]

        st.caption(f"{len(visible)} parça gösteriliyor")
        st.markdown("---")

        # --- Render chunk cards ---
        for chunk in visible:
            src = chunk["source"]
            css_cls  = "mevzuat" if src == "explicit" else "tacit"
            badge_cls = "badge-mevzuat" if src == "explicit" else "badge-tacit"
            badge_lbl = "📘 Mevzuat" if src == "explicit" else "🧠 Tacit Deneyim"
            preview = chunk["text"][:280].replace("\n", " ")
            if len(chunk["text"]) > 280:
                preview += "…"

            with st.expander(
                f"{badge_lbl}  ·  Parça #{chunk['chunk_index']}  ·  {chunk['char_count']} karakter"
            ):
                st.markdown(
                    f"""<div class="chunk-card {css_cls}">
                    <div class="chunk-meta">
                        <span class="chunk-badge {badge_cls}">{badge_lbl}</span>
                        Dosya: <b>{chunk['filename']}</b> &nbsp;·&nbsp;
                        Parça indeksi: <b>{chunk['chunk_index']}</b> &nbsp;·&nbsp;
                        {chunk['char_count']} karakter
                    </div>
                    <div class="chunk-text">{chunk['text'].replace(chr(10), '<br>')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
