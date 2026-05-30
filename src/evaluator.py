"""
TEE-Model RAGAS değerlendirme harness'i (Phase 2.1).

RAGAS — Es, James, Espinosa-Anke, Schockaert (EACL 2024) — RAG sistemleri
için dört temel ölçüt sunar:
  - faithfulness        : cevap, bağlama sadık mı? (halüsinasyon bayrağı)
  - answer_relevancy    : cevap, soruyu karşılıyor mu?
  - context_precision   : alınan bağlam parçaları alakalı mı?
  - context_recall      : alınan bağlam, ideal cevabı içeriyor mu?

Tüm ölçütler bir LLM-yargıcı kullanır. Bu cloud branch'te hem yargıç hem de
RAGAS'ın kendi gömme adımı Vertex AI Gemini modellerine yönlendirilir.

Kullanım:
    python -m src.evaluator              # 20 sorulu test setini çalıştır
    python -m src.evaluator --quick      # ilk 5 soruyla hızlı doğrulama
"""

from __future__ import annotations

import argparse
import json
import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import settings
from src.generators import _build_grounded_prompt
from src.llm import generate as llm_generate
from src.retrieval import retrieve_context


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test seti yükleme
# ---------------------------------------------------------------------------

def load_test_questions(path: Path | None = None) -> list[dict[str, Any]]:
    """Hazır soru-cevap setini diskten yükler."""
    path = path or settings.RAGAS_TEST_SET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Test seti bulunamadı: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("questions", data) if isinstance(data, dict) else data


# `load_test_set` — açık isimli takma ad (Phase 2 recall harness'i ile uyum).
load_test_set = load_test_questions


# ---------------------------------------------------------------------------
# Deterministik kaynak-recall metrikleri (Phase 2 — RAGAS'tan bağımsız)
#
# `retrieve_context` parent kaynaklarını döndürür; bunları eval setindeki
# `expected_sources` (gerçek korpus dosya adları) ile karşılaştırırız. LLM-yargıç
# YOK — yalnızca küme matematiği. Bu yüzden ucuz, deterministik ve CI-uygun.
# ---------------------------------------------------------------------------

def _basename(name: str) -> str:
    """
    Dosya adını döndürür (yol bileşenlerini atar) ve Unicode NFC'ye normalize eder.

    Korpus dosya adları macOS'ta NFD (ayrışık: o + ̂) saklanabilirken eval setindeki
    literaller NFC (birleşik: ô) olabilir. Karşılaştırmanın iki tarafını da NFC'ye
    çekmek, 'Diaporama_rôle_EC_VF.pptx' gibi aksanlı adların yanlışlıkla eşleşmemesini önler.
    """
    return unicodedata.normalize("NFC", Path(str(name)).name)


def _source_recall(gold: list[str], retrieved: list[str]) -> float:
    """
    Set-recall: geri-getirilen kaynaklar arasında bulunan altın (gold)
    kaynakların oranı. gold boşsa 1.0 (cezalandırma yok).
    """
    gold_set = {_basename(g) for g in gold}
    if not gold_set:
        return 1.0
    retr_set = {_basename(r) for r in retrieved}
    hit = len(gold_set & retr_set)
    return hit / len(gold_set)


def _source_precision(gold: list[str], retrieved: list[str]) -> float:
    """
    Set-precision: geri-getirilen kaynaklardan kaçı altın küme içinde.
    retrieved boşsa 0.0.
    """
    gold_set = {_basename(g) for g in gold}
    retr_set = {_basename(r) for r in retrieved}
    if not retr_set:
        return 0.0
    hit = len(gold_set & retr_set)
    return hit / len(retr_set)


def _hit_at_k(gold: list[str], retrieved: list[str]) -> float:
    """Top-k içinde ≥1 altın kaynak varsa 1.0, yoksa 0.0."""
    gold_set = {_basename(g) for g in gold}
    if not gold_set:
        return 1.0
    retr_set = {_basename(r) for r in retrieved}
    return 1.0 if (gold_set & retr_set) else 0.0


def _retrieved_sources(hits: list[dict]) -> list[str]:
    """Hit dict listesinden (parent sırasıyla) kaynak dosya adlarını çıkarır."""
    sources: list[str] = []
    for h in hits:
        name = h.get("filename") or h.get("source_file") or h.get("source") or ""
        if name:
            sources.append(_basename(name))
    return sources


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_retrieval(
    test_questions: list[dict] | None = None,
    *,
    ks: list[int] | None = None,
    retrieve_fn=None,
) -> dict:
    """
    Her soru için recall@k / hit@k / precision@k hesaplar.

    `retrieve_context` en geniş k için BİR KEZ çağrılır (max(ks)); küçük k'ler
    bu sonucun ön-dilimleridir. Bu, çağrı sayısını (ve gömme maliyetini) düşürür.

    Döner
    -----
    dict — {"per_question": [...], "ks": [...]} ham gözlemler. Toplama `_aggregate`
    tarafından yapılır.
    """
    test_questions = test_questions or load_test_set()
    ks = sorted(ks or settings.RECALL_EVAL_KS)
    max_k = max(ks)
    retrieve_fn = retrieve_fn or retrieve_context

    per_question: list[dict] = []
    for item in test_questions:
        if not item.get("answerable", True):
            continue  # destekleyici kaynağı olmayan sorular recall'a katılmaz
        gold = item.get("expected_sources", [])
        if not gold:
            continue
        query = item.get("question") or item.get("question_tr") or item.get("question_fr") or ""
        try:
            hits = retrieve_fn(query, top_k=max_k)
        except Exception as exc:  # noqa: BLE001 — bir soru patlarsa kaydet, devam et
            logger.warning("Retrieval hatası (%s): %s", item.get("id"), exc)
            hits = []
        retrieved_all = _retrieved_sources(hits)

        rec = {"id": item.get("id", "?")}
        for k in ks:
            top = retrieved_all[:k]
            rec[f"recall@{k}"] = _source_recall(gold, top)
            rec[f"hit@{k}"] = _hit_at_k(gold, top)
            rec[f"precision@{k}"] = _source_precision(gold, top)
        rec["gold"] = [_basename(g) for g in gold]
        rec["retrieved"] = retrieved_all[:max_k]
        # Toplama için kova (bucket) etiketleri
        rec["language"] = item.get("language", "?")
        rec["type"] = item.get("type", "?")
        rec["doc_type"] = item.get("category", item.get("doc_type", "?"))
        rec["level"] = item.get("level", "?")
        rec["split"] = item.get("split", "?")
        per_question.append(rec)

    return {"per_question": per_question, "ks": ks}


def _aggregate(per_question: list[dict], ks: list[int]) -> dict:
    """Genel ve kova-bazlı (language/type/doc_type/level/split) ortalamalar."""
    def agg(rows: list[dict]) -> dict:
        out = {"n": len(rows)}
        for k in ks:
            out[f"recall@{k}"] = _mean([r[f"recall@{k}"] for r in rows])
            out[f"hit@{k}"] = _mean([r[f"hit@{k}"] for r in rows])
            out[f"precision@{k}"] = _mean([r[f"precision@{k}"] for r in rows])
        return out

    overall = agg(per_question)
    buckets: dict[str, dict] = {}
    for dim in ("language", "type", "doc_type", "level", "split"):
        groups: dict[str, list[dict]] = {}
        for r in per_question:
            groups.setdefault(str(r.get(dim, "?")), []).append(r)
        buckets[dim] = {val: agg(rows) for val, rows in sorted(groups.items())}
    return {"overall": overall, "buckets": buckets}


def run_recall_report(
    test_questions: list[dict] | None = None,
    *,
    ks: list[int] | None = None,
    retrieve_fn=None,
    save: bool = True,
) -> dict:
    """
    Deterministik recall@k raporunu uçtan uca çalıştırır ve (isteğe bağlı)
    `evaluation/results/recall_baseline_<UTC>.json` + `.md` artefaktlarını yazar.

    M3 = RECALL_M3_K (None ise RETRIEVAL_TOP_K) k'sinde ortalama recall@k.
    """
    test_questions = test_questions or load_test_set()
    ks = sorted(ks or settings.RECALL_EVAL_KS)
    raw = evaluate_retrieval(test_questions, ks=ks, retrieve_fn=retrieve_fn)
    agg = _aggregate(raw["per_question"], ks)

    m3_k = settings.RECALL_M3_K or settings.RETRIEVAL_TOP_K
    if m3_k not in ks:
        m3_k = min(ks, key=lambda k: abs(k - m3_k))
    m3_recall = agg["overall"].get(f"recall@{m3_k}")
    m3_hit = agg["overall"].get(f"hit@{m3_k}")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_questions_evaluated": len(raw["per_question"]),
            "ks": ks,
            "m3_k": m3_k,
            "m3_recall_at_k": m3_recall,
            "m3_hit_at_k": m3_hit,
            "retrieval_top_k": settings.RETRIEVAL_TOP_K,
            "mock_mode": settings.MOCK_MODE,
            "search_mode": "hybrid" if settings.ENABLE_HYBRID_SEARCH else "dense",
            "embedding_model": settings.EMBEDDING_MODEL,
            "test_set": str(settings.RAGAS_TEST_SET_PATH),
        },
        "aggregate": agg,
        "per_question": raw["per_question"],
    }

    artifact_path = None
    if save:
        settings.RAGAS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = settings.RAGAS_RESULTS_DIR / f"recall_baseline_{ts}.json"
        artifact_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        md_path = settings.RAGAS_RESULTS_DIR / f"recall_baseline_{ts}.md"
        md_path.write_text(_render_recall_md(report), encoding="utf-8")
        report["metadata"]["artifact_json"] = str(artifact_path)
        report["metadata"]["artifact_md"] = str(md_path)

    return report


def _fmt_pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "—"


def _render_recall_md(report: dict) -> str:
    """İnsan-okunur kısa Markdown özeti üretir."""
    md = report["metadata"]
    agg = report["aggregate"]
    ks = md["ks"]
    lines: list[str] = []
    lines.append("# Recall@k baseline (deterministic, source-recall)")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {md['timestamp']}")
    lines.append(f"- Questions evaluated: {md['n_questions_evaluated']}")
    lines.append(f"- MOCK_MODE: {md['mock_mode']} · search_mode: {md['search_mode']}")
    lines.append(f"- Embedding model: {md['embedding_model']}")
    lines.append(
        f"- **M3 = mean recall@{md['m3_k']} = {_fmt_pct(md['m3_recall_at_k'])}** "
        f"(hit@{md['m3_k']} = {_fmt_pct(md['m3_hit_at_k'])}); target ≥ 90%"
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    header = "| metric | " + " | ".join(f"@{k}" for k in ks) + " |"
    sep = "|---|" + "|".join("---" for _ in ks) + "|"
    lines.append(header)
    lines.append(sep)
    for metric in ("recall", "hit", "precision"):
        row = f"| {metric} | " + " | ".join(
            _fmt_pct(agg["overall"].get(f"{metric}@{k}")) for k in ks
        ) + " |"
        lines.append(row)
    lines.append("")
    for dim, groups in agg["buckets"].items():
        lines.append(f"## By {dim}")
        lines.append("")
        lines.append("| value | n | " + " | ".join(f"recall@{k}" for k in ks) + " | " + " | ".join(f"hit@{k}" for k in ks) + " |")
        lines.append("|---|---|" + "|".join("---" for _ in ks) + "|" + "|".join("---" for _ in ks) + "|")
        for val, a in groups.items():
            rec = " | ".join(_fmt_pct(a.get(f"recall@{k}")) for k in ks)
            hit = " | ".join(_fmt_pct(a.get(f"hit@{k}")) for k in ks)
            lines.append(f"| {val} | {a['n']} | {rec} | {hit} |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAG yanıtı üretimi (her soru için)
# ---------------------------------------------------------------------------

def _generate_answer(question: str, top_k: int = 5) -> tuple[str, list[str]]:
    """
    Bir soru için RAG yanıtı üretir ve kullanılan parent metinlerini döndürür.

    Bu, üretici fonksiyonlardan farklı olarak yapılandırılmamış (free-text)
    cevap üretir; çünkü RAGAS metrikleri doğal dil cevapları üzerinden çalışır.
    """
    chunks = retrieve_context(question, top_k=top_k)
    if not chunks:
        return ("Bu bilgi mevcut belgelerde yer almamaktadır.", [])

    context_texts = [c["parent_text"] for c in chunks]
    instruction = (
        "Yukarıdaki bağlama dayanarak soruyu kısa, net ve Türkçe yanıtla. "
        "Bağlamda bulunmayan hiçbir bilgiyi uydurma. "
        f"\n\nSORU: {question}"
    )
    system, user_prompt, _ = _build_grounded_prompt(question, instruction, top_k=top_k)
    answer = llm_generate(prompt=user_prompt, system=system)
    return answer.strip(), context_texts


# ---------------------------------------------------------------------------
# RAGAS entegrasyonu (lazy import — RAGAS yüklü değilse hata vermez)
# ---------------------------------------------------------------------------

def _build_ragas_clients():
    """
    RAGAS yargıç LLM'i ve embedding modelini INFERENCE_BACKEND'e göre kurar.

    Uygulamanın geri kalanı (src/llm.py) tek bir google-genai istemcisini
    Vertex AI veya public Gemini API moduna geçirebiliyor; RAGAS ise LangChain
    sarmalayıcıları beklediğinden burada backend'e göre uygun LangChain
    sınıflarını seçeriz. Aksi halde public_genai modunda ADC/proje hatası alınır.
    """
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        raise ImportError(
            "RAGAS bağımlılıkları eksik. Lütfen: pip install ragas datasets"
        ) from exc

    if settings.INFERENCE_BACKEND == "public_genai":
        judge_llm, judge_embeddings = _build_public_genai_clients()
    else:
        judge_llm, judge_embeddings = _build_vertex_clients()
    return LangchainLLMWrapper(judge_llm), LangchainEmbeddingsWrapper(judge_embeddings)


def _build_vertex_clients():
    """Vertex AI (ADC) LangChain LLM + embedding çifti."""
    try:
        from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Vertex RAGAS yargıcı için: pip install langchain-google-vertexai"
        ) from exc

    vertex_llm = ChatVertexAI(
        model=settings.GENERATION_MODEL,
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GOOGLE_CLOUD_LOCATION,
        temperature=0.0,  # yargıç çağrılarında deterministik istiyoruz
    )
    # Boyut, uygulama retrieval yolu ile (src.embeddings) aynı tutulur; aksi
    # halde EMBEDDING_DIMENSION=768/1536 yapıldığında RAGAS 3072-boyutlu
    # vektörlerle çalışıp koleksiyondaki vektörlerle uyumsuz olur.
    embeddings_kwargs = {
        "model_name": settings.EMBEDDING_MODEL,
        "project": settings.GOOGLE_CLOUD_PROJECT,
        "location": settings.GOOGLE_CLOUD_LOCATION,
    }
    try:
        vertex_embeddings = VertexAIEmbeddings(
            **embeddings_kwargs,
            dimensions=settings.EMBEDDING_DIMENSION,
        )
    except TypeError:
        logger.warning(
            "VertexAIEmbeddings 'dimensions' kwarg'ını kabul etmedi; "
            "RAGAS varsayılan embedding boyutuna düşüyor (uygulama yolu ile "
            "uyumsuz olabilir). langchain-google-vertexai sürümünü güncelleyin."
        )
        vertex_embeddings = VertexAIEmbeddings(**embeddings_kwargs)
    return vertex_llm, vertex_embeddings


def _build_public_genai_clients():
    """Public Gemini API (GOOGLE_API_KEY) LangChain LLM + embedding çifti."""
    if not settings.GOOGLE_API_KEY:
        raise RuntimeError(
            "INFERENCE_BACKEND='public_genai' için GOOGLE_API_KEY gerekli."
        )
    try:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
    except ImportError as exc:
        raise ImportError(
            "Public Gemini RAGAS yargıcı için: pip install langchain-google-genai"
        ) from exc

    genai_llm = ChatGoogleGenerativeAI(
        model=settings.GENERATION_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.0,
    )
    # Public API embedding modeli "models/" ön ekini bekler.
    embed_model = settings.EMBEDDING_MODEL
    if not embed_model.startswith("models/"):
        embed_model = f"models/{embed_model}"
    genai_embeddings = GoogleGenerativeAIEmbeddings(
        model=embed_model,
        google_api_key=settings.GOOGLE_API_KEY,
    )
    return genai_llm, genai_embeddings


# ---------------------------------------------------------------------------
# Ana değerlendirme
# ---------------------------------------------------------------------------

def evaluate_rag_pipeline(
    test_questions: list[dict] | None = None,
    *,
    top_k: int = 5,
    save_results: bool = True,
) -> dict:
    """
    Verilen test soruları üzerinde RAGAS değerlendirmesi çalıştırır.

    Adımlar:
      1. Her soru için RAG cevabı üret + retrieved context kaydet.
      2. RAGAS metriklerini Vertex AI Gemini yargıç ile uygula.
      3. Sonuç sözlüğü ve özet metrikleri döndür; isteğe bağlı diske yaz.

    Döner
    -----
    dict — {"summary": {faithfulness, answer_relevancy, context_precision, context_recall},
            "per_question": [...], "metadata": {...}}
    """
    test_questions = test_questions or load_test_questions()
    logger.info("RAGAS değerlendirmesi başlıyor: %d soru.", len(test_questions))

    # Adım 1: cevapları topla
    samples: list[dict] = []
    for i, item in enumerate(test_questions, start=1):
        q = item["question"]
        gt = item.get("ground_truth", "")
        logger.info("(%d/%d) Cevap üretiliyor: %s", i, len(test_questions), q[:80])
        answer, contexts = _generate_answer(q, top_k=top_k)
        samples.append({
            "user_input": q,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": gt,
            "id": item.get("id", f"q{i:02d}"),
            "category": item.get("category", "unknown"),
        })

    # Adım 2: RAGAS metrikleri
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    ragas_llm, ragas_embeddings = _build_ragas_clients()

    dataset = EvaluationDataset.from_list([
        {
            "user_input": s["user_input"],
            "response": s["response"],
            "retrieved_contexts": s["retrieved_contexts"],
            "reference": s["reference"],
        }
        for s in samples
    ])

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    logger.info("RAGAS metrikleri hesaplanıyor (Vertex AI yargıç + embedding)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    df = result.to_pandas()
    summary = {
        "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else None,
        "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else None,
        "context_precision": float(df["llm_context_precision_with_reference"].mean()) if "llm_context_precision_with_reference" in df else None,
        "context_recall": float(df["context_recall"].mean()) if "context_recall" in df else None,
    }

    # Adım 3: kayıt
    output = {
        "summary": summary,
        "per_question": df.to_dict(orient="records"),
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_questions": len(samples),
            "top_k": top_k,
            "generation_model": settings.GENERATION_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "search_mode": "hybrid" if settings.ENABLE_HYBRID_SEARCH else "dense",
        },
    }

    if save_results:
        settings.RAGAS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = settings.RAGAS_RESULTS_DIR / f"ragas_results_{ts}.json"
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        logger.info("Sonuçlar yazıldı: %s", out_path)

    return output


# ---------------------------------------------------------------------------
# Geriye dönük stub'lar (Faz 2 LMS entegrasyonu — şimdilik N/A)
# ---------------------------------------------------------------------------

def run_pre_test(user_id: str) -> dict:
    """LMS ön-test entegrasyonu — Faz 2 dışı, yer tutucu."""
    raise NotImplementedError("LMS ön-test entegrasyonu Faz 2 kapsamı dışındadır.")


def run_post_test(user_id: str) -> dict:
    """LMS son-test entegrasyonu — Faz 2 dışı, yer tutucu."""
    raise NotImplementedError("LMS son-test entegrasyonu Faz 2 kapsamı dışındadır.")


def analyze_results(pre: dict, post: dict) -> dict:
    """Pre/post analizi — Faz 2 dışı, yer tutucu."""
    raise NotImplementedError("Pre/post analiz Faz 2 kapsamı dışındadır.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(result: dict) -> None:
    s = result["summary"]
    print("\n" + "=" * 60)
    print("RAGAS DEĞERLENDİRME ÖZETİ")
    print("=" * 60)
    print(f"  Soru sayısı       : {result['metadata']['n_questions']}")
    print(f"  top_k             : {result['metadata']['top_k']}")
    print(f"  Üretim modeli     : {result['metadata']['generation_model']}")
    print(f"  Gömme modeli      : {result['metadata']['embedding_model']}")
    print(f"  Arama modu        : {result['metadata']['search_mode']}")
    print("-" * 60)
    print(f"  faithfulness      : {s['faithfulness']:.4f}" if s["faithfulness"] is not None else "  faithfulness      : —")
    print(f"  answer_relevancy  : {s['answer_relevancy']:.4f}" if s["answer_relevancy"] is not None else "  answer_relevancy  : —")
    print(f"  context_precision : {s['context_precision']:.4f}" if s["context_precision"] is not None else "  context_precision : —")
    print(f"  context_recall    : {s['context_recall']:.4f}" if s["context_recall"] is not None else "  context_recall    : —")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="TEE-Model değerlendirme.")
    parser.add_argument("--quick", action="store_true", help="Yalnızca ilk 5 soruyla çalıştır.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--recall-report",
        action="store_true",
        help="Yalnızca deterministik recall@k raporunu çalıştır (RAGAS yok); "
             "headline M3 + artefakt yolunu yazdırır.",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="RAGAS akışını atla (recall raporu ile birlikte kullanışlıdır).",
    )
    args = parser.parse_args()

    if args.recall_report or args.no_ragas:
        report = run_recall_report()
        md = report["metadata"]
        # stdout'u KÜÇÜK tut: yalnızca headline + artefakt yolu.
        print(
            f"M3 mean recall@{md['m3_k']} = {_fmt_pct(md['m3_recall_at_k'])} "
            f"(hit@{md['m3_k']} = {_fmt_pct(md['m3_hit_at_k'])}) "
            f"over n={md['n_questions_evaluated']} (target >= 90%)"
        )
        print(f"artifact: {md.get('artifact_json', '(not saved)')}")
    else:
        questions = load_test_questions()
        if args.quick:
            questions = questions[:5]
        result = evaluate_rag_pipeline(test_questions=questions, top_k=args.top_k)
        _print_summary(result)
