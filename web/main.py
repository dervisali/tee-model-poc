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

# slug -> (heading, subheading) for the not-yet-rebuilt tabs (internal/dev tools)
STUBS = {
    "optimizer": ("Prompt Optimizer", "A/B prompt varyantları ve LLM-yargıç skorları"),
    "veritabani": ("Veritabanı Gezgini", "Korpus chunk'larını arama ve inceleme"),
    "veri": ("Veri Yükleme & İşleme", "Belge yükleme, parçalama, gömme"),
}


def _lang(language: str) -> str:
    return language if language in ("tr", "fr") else "tr"


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


def _generate(request: Request, partial: str, fn, **kwargs) -> HTMLResponse:
    try:
        data = fn(**kwargs)
        return TEMPLATES.TemplateResponse(request, partial, {"request": request, "d": data})
    except Exception as exc:  # noqa: BLE001 — surface generation failures in the UI
        return TEMPLATES.TemplateResponse(
            request, "partials/gen_error.html",
            {"request": request, "error": f"{type(exc).__name__}: {exc}"},
        )


# --- Süreç Haritası & Hata Kartları ---------------------------------------
@app.get("/surec", response_class=HTMLResponse)
def surec(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(request, "surec.html", base_ctx(request, "surec"))


@app.post("/surec/process-map", response_class=HTMLResponse)
def surec_process_map(request: Request, language: str = Form("tr")) -> HTMLResponse:
    from src.generators import generate_process_map
    return _generate(request, "partials/process_map.html", generate_process_map, language=_lang(language))


@app.post("/surec/error-cards", response_class=HTMLResponse)
def surec_error_cards(request: Request, language: str = Form("tr")) -> HTMLResponse:
    from src.generators import generate_error_cards
    return _generate(request, "partials/error_cards.html", generate_error_cards, language=_lang(language))


@app.post("/surec/glossary", response_class=HTMLResponse)
def surec_glossary(request: Request, language: str = Form("tr")) -> HTMLResponse:
    from src.generators import generate_glossary
    return _generate(request, "partials/glossary.html", generate_glossary, language=_lang(language))


# --- İnteraktif Simülasyon -------------------------------------------------
@app.get("/simulasyon", response_class=HTMLResponse)
def simulasyon(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(request, "simulasyon.html", base_ctx(request, "simulasyon"))


@app.post("/simulasyon", response_class=HTMLResponse)
def simulasyon_generate(request: Request, language: str = Form("tr")) -> HTMLResponse:
    from src.generators import generate_simulation_scenario
    return _generate(request, "partials/simulation.html", generate_simulation_scenario, language=_lang(language))


# --- Uzman Onay Paneli (light governance review) --------------------------
@app.get("/onay", response_class=HTMLResponse)
def onay(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(request, "onay.html", base_ctx(request, "onay"))


@app.post("/onay/review", response_class=HTMLResponse)
def onay_review(request: Request, language: str = Form("tr")) -> HTMLResponse:
    from src.generators import generate_error_cards
    return _generate(request, "partials/approval_queue.html", generate_error_cards, language=_lang(language))


@app.get("/{slug}", response_class=HTMLResponse)
def stub(request: Request, slug: str) -> HTMLResponse:
    if slug not in STUBS:
        return HTMLResponse("Sayfa bulunamadı", status_code=404)
    heading, subheading = STUBS[slug]
    ctx = base_ctx(request, slug)
    ctx.update({"heading": heading, "subheading": subheading})
    return TEMPLATES.TemplateResponse(request, "stub.html", ctx)
