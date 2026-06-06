from pathlib import Path


APP_PY = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_branding_is_delf_dalf_facing():
    app_source = APP_PY.read_text(encoding="utf-8")

    assert 'page_title="DELF/DALF Sınavcı Asistanı"' in app_source
    assert 'st.title("🏛️ DELF/DALF Sınavcı-Düzeltici Asistanı")' in app_source
    assert "TEE-Model — DELF/DALF Eğitim Sistemi" not in app_source
    assert "Tacit-Explicit Entegre Eğitim Sistemi" not in app_source
