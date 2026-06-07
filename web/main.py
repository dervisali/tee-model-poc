"""
DELF/DALF Sınavcı-Düzeltici Asistanı — FastAPI + HTMX + Tailwind front-end.

Server-rendered UI that reuses the Python backend in `src/` directly (no API
layer, one deployable service). Run:

    MOCK_MODE=true python3 -m uvicorn web.main:app --reload --port 8000
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from markupsafe import Markup

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fastapi import FastAPI, Form, Request  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402

from src.config import settings  # noqa: E402
from src.chatbot import chat_stream  # noqa: E402

app = FastAPI(title="DELF/DALF Sınavcı Asistanı")
TEMPLATES = Jinja2Templates(directory=str(REPO / "web" / "templates"))
app.mount("/static", StaticFiles(directory=str(REPO / "web" / "static")), name="static")

EXAMPLES = [
    "Çift düzeltmede görüş ayrılığı nasıl çözülür?",
    "Halo etkisi (effet de halo) nedir?",
    "DELF B2 sözlü üretiminde hangi kriterler değerlendirilir?",
]

# slug -> (heading, subheading) for the not-yet-rebuilt tabs
STUBS = {
    "surec": ("Süreç Haritası & Hata Kartları", "Değerlendirme sürecinin haritası ve sık hatalar"),
    "simulasyon": ("İnteraktif Simülasyon", "Gerçekçi değerlendirme karar senaryoları"),
    "onay": ("Uzman Onay Paneli", "Üretilen içeriğin uzman onayı"),
    "optimizer": ("Prompt Optimizer", "A/B prompt varyantları ve LLM-yargıç skorları"),
    "veritabani": ("Veritabanı Gezgini", "Korpus chunk'larını arama ve inceleme"),
    "veri": ("Veri Yükleme & İşleme", "Belge yükleme, parçalama, gömme"),
}


_CITE = re.compile(r"\[Kaynak:\s*([^\]]+)\]")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_ITAL = re.compile(r"(?<!\w)_(.+?)_(?!\w)", re.S)


def render_answer(text: str) -> Markup:
    """Minimal, safe markdown→HTML for assistant answers (no markdown dep).

    Escapes first (the text is LLM-generated), then renders bold/italic,
    turns inline [Kaynak: file — id] markers into chips, and splits paragraphs.
    """
    s = html.escape(text.strip())
    s = _CITE.sub(lambda m: f'<span class="cite-chip">{m.group(1).strip()}</span>', s)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    s = _ITAL.sub(r"<em>\1</em>", s)
    paras = [p.strip().replace("\n", "<br>") for p in re.split(r"\n\s*\n", s) if p.strip()]
    return Markup("".join(f"<p>{p}</p>" for p in paras) or "<p></p>")


def base_ctx(request: Request, active: str) -> dict:
    return {
        "request": request,
        "active": active,
        "mock_mode": settings.MOCK_MODE,
        "model": settings.GENERATION_MODEL,
    }


@app.get("/", response_class=HTMLResponse)
def home() -> RedirectResponse:
    return RedirectResponse("/asistan")


@app.get("/asistan", response_class=HTMLResponse)
def asistan(request: Request) -> HTMLResponse:
    ctx = base_ctx(request, "asistan")
    ctx["examples"] = EXAMPLES
    return TEMPLATES.TemplateResponse(request, "asistan.html", ctx)


@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, message: str = Form(...), language: str = Form("tr")) -> HTMLResponse:
    text = message.strip()
    gen, sources = chat_stream(text, history=[], language=language if language in ("tr", "fr") else "tr")
    answer = "".join(gen)
    ctx = {
        "request": request,
        "message": text,
        "answer": render_answer(answer),
        "sources": sources,
        "refused": not sources,
    }
    return TEMPLATES.TemplateResponse(request, "partials/chat_exchange.html", ctx)


@app.get("/{slug}", response_class=HTMLResponse)
def stub(request: Request, slug: str) -> HTMLResponse:
    if slug not in STUBS:
        return HTMLResponse("Sayfa bulunamadı", status_code=404)
    heading, subheading = STUBS[slug]
    ctx = base_ctx(request, slug)
    ctx.update({"heading": heading, "subheading": subheading})
    return TEMPLATES.TemplateResponse(request, "stub.html", ctx)
