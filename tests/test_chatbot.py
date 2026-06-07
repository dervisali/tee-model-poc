"""Focused tests for Tab 7 chatbot guardrails."""

from __future__ import annotations

import json
from pathlib import Path


def test_mock_chat_returns_validated_sources():
    from src.chatbot import chat

    out = chat("DELF düzeltmesinde halo etkisi nedir?", language="tr")

    assert out["error"] is None
    assert out["citation_report"]["passed"] is True
    assert out["citation_report"]["cited_count"] == 2
    assert out["sources"][0]["_citation_report"]["passed"] is True


def test_validate_citations_rejects_unknown_parent_id():
    from src.chatbot import validate_citations

    sources = [{"parent_id": "known_parent"}]
    answer = "Bilgi [Kaynak: doc.pdf — unknown_parent]."

    report = validate_citations(answer, sources)

    assert report["passed"] is False
    assert report["invalid_parent_ids"] == ["unknown_parent"]


def test_validate_citations_passes_with_one_valid_despite_invalid():
    """Bir geçerli atıf varsa, tek bir geçersiz/uydurma id yanıtı düşürmemeli;
    geçersiz id raporlanır ama grounding kapısı reddetmez (d03 regresyonu)."""
    from src.chatbot import validate_citations

    sources = [{"parent_id": "manuel-exacor_par5"}]
    answer = (
        "İlk nokta [Kaynak: manuel-exacor.pdf — manuel-exacor_par5]. "
        "İkinci nokta [Kaynak: dalf.pdf — dalf-c1_par24_bogus]."
    )

    report = validate_citations(answer, sources)

    assert report["passed"] is True
    assert report["valid_parent_ids"] == ["manuel-exacor_par5"]
    assert report["invalid_parent_ids"] == ["dalf-c1_par24_bogus"]


def test_validate_citations_accepts_dash_variants_and_keeps_internal_hyphens():
    """Gemini, em-dash'i en-dash veya düz tireye normalize edebilir; parent_id
    içindeki tireler (örn. manuel-exacor_par5) ayırıcı sayılmamalı."""
    from src.chatbot import validate_citations

    sources = [{"parent_id": "manuel-exacor_par5"}]

    for sep in ("—", "–", "-"):
        answer = f"Halo etkisi [Kaynak: manuel-exacor.pdf {sep} manuel-exacor_par5]."
        report = validate_citations(answer, sources)
        assert report["passed"] is True, f"sep={sep!r} should be accepted"
        assert report["cited_parent_ids"] == ["manuel-exacor_par5"]
        assert report["invalid_parent_ids"] == []


def test_retrieval_query_strips_citations_and_marks_vague_followup():
    from src.chatbot import _build_retrieval_query

    history = [
        {"role": "user", "content": "Halo etkisi nedir?"},
        {
            "role": "assistant",
            "content": "Halo etkisi kriterleri etkiler [Kaynak: manuel.pdf — p1].",
        },
    ]

    query = _build_retrieval_query("biraz daha açıkla", history)

    assert "Halo etkisi nedir?" in query
    assert "[Kaynak:" not in query
    assert "p1" not in query
    assert "takip sorusu" in query


def test_no_context_fallback_skips_llm(monkeypatch):
    import src.chatbot as chatbot

    monkeypatch.setattr(chatbot.settings, "MOCK_MODE", False)
    monkeypatch.setattr(chatbot, "_retrieve", lambda query: ("", []))

    def fail_generate(*args, **kwargs):
        raise AssertionError("LLM should not be called when retrieval has no sources")

    monkeypatch.setattr(chatbot, "llm_generate", fail_generate)

    out = chatbot.chat("Korpusta olmayan bir prosedür nedir?", language="tr")

    assert out["error"] is None
    assert out["sources"] == []
    assert "yeterli bilgi yok" in out["answer"]
    assert "citation_report" in out
    assert out["citation_report"]["retrieved_count"] == 0
    assert out["citation_report"]["cited_count"] == 0


def test_non_streaming_chat_appends_warning_for_bad_citation(monkeypatch):
    """Legacy path: grounding kapısı KAPALIYKEN ungrounded yanıt uyarıyla geçer.

    Phase 3 varsayılanı (ENABLE_GROUNDING_GATE=True) artık reddeder
    (bkz. test_non_streaming_chat_refuses_ungrounded_by_default); bu test eski
    'uyarı ekle' davranışının bayrak kapalıyken korunduğunu doğrular.
    """
    import src.chatbot as chatbot

    source = {
        "filename": "manuel.pdf",
        "parent_id": "known_parent",
        "snippet": "snippet",
        "score": 0.9,
    }

    monkeypatch.setattr(chatbot.settings, "MOCK_MODE", False)
    monkeypatch.setattr(chatbot.settings, "ENABLE_GROUNDING_GATE", False)
    monkeypatch.setattr(chatbot, "_retrieve", lambda query: ("context", [source]))
    monkeypatch.setattr(
        chatbot,
        "llm_generate",
        lambda **kwargs: "Yanıt [Kaynak: manuel.pdf — made_up_parent].",
    )

    out = chatbot.chat("Halo etkisi nedir?", language="tr")

    assert out["citation_report"]["passed"] is False
    assert out["citation_report"]["invalid_parent_ids"] == ["made_up_parent"]
    assert "atıf" in out["answer"].lower()
    assert out["sources"][0]["_citation_report"]["passed"] is False


def test_non_streaming_chat_refuses_ungrounded_by_default(monkeypatch):
    """Phase 3 varsayılanı: grounding kapısı AÇIKKEN ungrounded yanıt REDDEDİLİR."""
    import src.chatbot as chatbot

    source = {"filename": "manuel.pdf", "parent_id": "known_parent",
              "snippet": "snippet", "score": 0.9}

    monkeypatch.setattr(chatbot.settings, "MOCK_MODE", False)
    monkeypatch.setattr(chatbot.settings, "ENABLE_GROUNDING_GATE", True)
    monkeypatch.setattr(chatbot, "_retrieve", lambda query: ("context", [source]))
    monkeypatch.setattr(
        chatbot, "llm_generate",
        lambda **kwargs: "Yanıt [Kaynak: manuel.pdf — made_up_parent].",
    )

    out = chatbot.chat("Halo etkisi nedir?", language="tr")

    assert out["citation_report"]["passed"] is False
    assert out["safety"]["grounding_allowed"] is False
    assert "güvenle veremiyorum" in out["answer"]          # reddetme metni
    assert "made_up_parent" not in out["answer"]            # uydurma içerik servis edilmedi


def test_streaming_no_context_fallback_skips_llm(monkeypatch):
    import src.chatbot as chatbot

    monkeypatch.setattr(chatbot.settings, "MOCK_MODE", False)
    monkeypatch.setattr(chatbot, "_retrieve", lambda query: ("", []))

    def fail_stream(*args, **kwargs):
        raise AssertionError("Streaming LLM should not be called without sources")

    monkeypatch.setattr(chatbot, "llm_generate_stream", fail_stream)

    stream, sources = chatbot.chat_stream("Bilinmeyen konu", language="fr")

    assert sources == []
    assert "Je ne dispose pas d'informations suffisantes" in "".join(stream)


def test_safety_probe_set_is_present_and_covers_core_categories():
    probe_path = Path("evaluation/chatbot_safety_probes.json")
    probes = json.loads(probe_path.read_text(encoding="utf-8"))

    categories = {probe["category"] for probe in probes}

    assert {
        "score_decision",
        "exam_content_generation",
        "grounding_bypass",
        "no_context",
        "language_lock",
    }.issubset(categories)
    assert all(probe["prompt"].strip() for probe in probes)
