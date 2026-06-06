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


def test_load_image_ocr_wraps_tesseract_unicode_error(monkeypatch, tmp_path):
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

    with pytest.raises(loaders.LoaderError, match="OCR başarısız: UnicodeDecodeError"):
        loaders.load_image_ocr(tmp_path / "scan.jpeg")
