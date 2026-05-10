"""
TEE-Model RAGAS değerlendirme harness'i (Phase 2.1).

RAGAS — Es, James, Espinosa-Anke, Schockaert (EACL 2024) — RAG sistemleri
için dört temel ölçüt sunar:
  - faithfulness        : cevap, bağlama sadık mı? (halüsinasyon bayrağı)
  - answer_relevancy    : cevap, soruyu karşılıyor mu?
  - context_precision   : alınan bağlam parçaları alakalı mı?
  - context_recall      : alınan bağlam, ideal cevabı içeriyor mu?

Tüm ölçütler bir LLM-yargıcı kullanır. Burada hem yargıç hem de RAGAS'ın
kendi gömme adımı yerel Ollama + multilingual-e5-large yığınına yönlendirilir;
sistem hiçbir dış API'ye dokunmadan değerlendirilir.

Kullanım:
    python -m src.evaluator              # 20 sorulu test setini çalıştır
    python -m src.evaluator --quick      # ilk 5 soruyla hızlı doğrulama
"""

from __future__ import annotations

import argparse
import json
import logging
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
    """Vertex AI Gemini LLM ve embedding modelini RAGAS sarmalayıcılarına yerleştirir."""
    try:
        from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        raise ImportError(
            "RAGAS bağımlılıkları eksik. Lütfen: "
            "pip install ragas langchain-google-vertexai"
        ) from exc

    vertex_llm = ChatVertexAI(
        model=settings.GENERATION_MODEL,
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GOOGLE_CLOUD_LOCATION,
        temperature=0.0,  # yargıç çağrılarında deterministik istiyoruz
    )
    vertex_embeddings = VertexAIEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GOOGLE_CLOUD_LOCATION,
    )
    return LangchainLLMWrapper(vertex_llm), LangchainEmbeddingsWrapper(vertex_embeddings)


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
      2. RAGAS metriklerini Ollama yargıç ile uygula.
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

    logger.info("RAGAS metrikleri hesaplanıyor (LLM yargıç + e5 gömme)...")
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
    parser = argparse.ArgumentParser(description="TEE-Model RAGAS değerlendirme.")
    parser.add_argument("--quick", action="store_true", help="Yalnızca ilk 5 soruyla çalıştır.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = load_test_questions()
    if args.quick:
        questions = questions[:5]

    result = evaluate_rag_pipeline(test_questions=questions, top_k=args.top_k)
    _print_summary(result)
