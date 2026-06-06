import types

import pytest

from src import document_loaders as loaders


class _FakePdfPage:
    def extract_text(self):
        return ""

    def to_image(self, resolution):
        return types.SimpleNamespace(original=object())


class _FakePdf:
    pages = [_FakePdfPage()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_load_pdf_continues_when_tesseract_error_decode_fails(monkeypatch, tmp_path):
    fake_pdfplumber = types.SimpleNamespace(open=lambda _: _FakePdf())
    fake_pytesseract = types.SimpleNamespace(
        TesseractNotFoundError=RuntimeError,
        TesseractError=RuntimeError,
        image_to_string=lambda *_, **__: (_ for _ in ()).throw(
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "pdfplumber", fake_pdfplumber)
    monkeypatch.setitem(__import__("sys").modules, "pytesseract", fake_pytesseract)

    pages = loaders.load_pdf(tmp_path / "scan.pdf")

    assert pages == [loaders.PageContent(text="", page=1, total_pages=1)]


def test_load_image_ocr_falls_back_when_pytesseract_stderr_decode_fails(monkeypatch, tmp_path):
    """pytesseract'ın get_errors() katı UTF-8 stderr decode'u çökerse OCR
    bozulmamalı: ikili doğrudan çağrılır ve sonucu döner (regresyon: önceden
    UnicodeDecodeError tüm sayfayı boş bırakıyordu)."""
    class FakeImage:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_pil = types.SimpleNamespace(Image=types.SimpleNamespace(open=lambda _: FakeImage()))
    fake_pytesseract = types.SimpleNamespace(
        TesseractNotFoundError=RuntimeError,
        TesseractError=RuntimeError,
        image_to_string=lambda *_, **__: (_ for _ in ()).throw(
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "PIL", fake_pil)
    monkeypatch.setitem(__import__("sys").modules, "PIL.Image", fake_pil.Image)
    monkeypatch.setitem(__import__("sys").modules, "pytesseract", fake_pytesseract)
    monkeypatch.setattr(loaders, "_tesseract_via_subprocess", lambda *_, **__: "Notre cerveau a-t-il un sexe ?")

    pages = loaders.load_image_ocr(tmp_path / "scan.jpeg")

    assert pages == [loaders.PageContent(text="Notre cerveau a-t-il un sexe ?", page=None, total_pages=1)]


def test_tesseract_via_subprocess_ocrs_source_file_and_decodes_leniently(monkeypatch, tmp_path):
    """source_path verilince orijinal dosya OCR'lanır (geçici kopya yok) ve
    stdout, UTF-8 olmayan baytlar olsa bile hoşgörüyle çözülür."""
    captured = {}

    def fake_run(cmd, capture_output):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="Café".encode("utf-8") + b"\xff", stderr=b"\xff noise")

    monkeypatch.setattr("subprocess.run", fake_run)
    src = tmp_path / "img.jpeg"
    src.write_bytes(b"not-a-real-image")

    out = loaders._tesseract_via_subprocess(object(), lang="fra", source_path=src)

    assert "Café" in out                  # valid text preserved, no crash on 0xff
    assert str(src) in captured["cmd"]     # ran on the original file, not a temp copy
    assert "-l" in captured["cmd"] and "fra" in captured["cmd"]
