"""
LLM tabanlı yeniden sıralama (reranker) — ikinci-aşama precision.

Hibrit (BM25 + dense) füzyon, geniş bir aday havuzu döndürür ama sıralaması
yalnızca sözcük/vektör benzerliğine dayanır. Bir LLM yargıç, sorgu ile her
adayın gerçek alaka düzeyini değerlendirerek bu havuzu yeniden sıralar ve en
alakalı top_n parçayı öne çıkarır.

Tasarım kararı (kullanıcı onaylı):
- Yerel cross-encoder (torch) YOK — Cloud Run'da ağır ML runtime taşımama
  ilkesine sadık kalınır (bkz. src/embeddings.py).
- Mevcut Gemini arka ucu (src/llm.generate) tek bir toplu çağrıyla tüm
  adayları puanlar. Çok dilli (TR sorgu, FR korpus) doğal olarak desteklenir.
- settings.ENABLE_RERANKING ile bayrak arkasında; varsayılan KAPALI (gecikme
  ekler). Açıldığında retrieval RERANK_FETCH_K aday çekip top_k'ye kırpar.

Dayanıklılık: herhangi bir hatada (LLM, parse, şema) sessizce passthrough —
reranker asla retrieval'ı bozmaz; en kötü ihtimalle füzyon sırası korunur.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from src.config import settings


logger = logging.getLogger(__name__)


# Aday parçanın yargıca verilen metin parçacığı uzunluğu. Tam parent metni
# token bütçesini şişirir; baş kısım alaka için genelde yeterlidir.
_SNIPPET_CHARS = 600


class _RerankScore(BaseModel):
    """Tek bir aday için yargıç puanı."""

    index: int = Field(description="Adayın 1-tabanlı listedeki sıra numarası.")
    score: float = Field(description="Sorguya alaka düzeyi, 0 (alakasız) - 10 (birebir).")


class RerankScores(BaseModel):
    """Yargıcın tüm adaylar için döndürdüğü puan listesi (yapılandırılmış JSON)."""

    scores: list[_RerankScore]


_SYSTEM = (
    "Sen bir bilgi getirme (retrieval) yeniden sıralama uzmanısın. Sana bir "
    "kullanıcı sorgusu ve numaralanmış aday belge parçaları verilir. Her parçanın "
    "sorguyu yanıtlamaya ne kadar yardımcı olduğunu 0-10 arası puanla: 10 = "
    "doğrudan ve birebir alakalı, 0 = tamamen alakasız. Belgeler Fransızca, sorgu "
    "Türkçe veya Fransızca olabilir; dil farkı puanı etkilememeli, yalnızca anlam "
    "alaka düzeyini değerlendir. Her aday için tam olarak bir puan döndür."
)


def _snippet(chunk: dict) -> str:
    """Adayın yargıca gösterilecek metnini seçer (parent öncelikli, kısaltılmış)."""
    text = chunk.get("parent_text") or chunk.get("text") or chunk.get("child_text") or ""
    text = text.strip().replace("\n", " ")
    return text[:_SNIPPET_CHARS]


def _build_prompt(query: str, candidates: list[dict]) -> str:
    lines = [f"SORGU: {query}", "", "ADAYLAR:"]
    for i, chunk in enumerate(candidates, start=1):
        lines.append(f"[{i}] {_snippet(chunk)}")
    lines.append("")
    lines.append(
        "Her aday için {index, score} döndür. index 1-tabanlıdır ve yukarıdaki "
        "numaralarla eşleşmelidir."
    )
    return "\n".join(lines)


def rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    Adayları LLM yargıç ile yeniden sıralar ve en alakalı top_n parçayı döndürür.

    Passthrough koşulları (LLM çağrısı YAPILMAZ):
      - settings.ENABLE_RERANKING kapalı,
      - aday sayısı <= top_n (sıralamanın bir önemi yok),
      - aday listesi boş.

    Hata halinde füzyon sırası korunarak candidates[:top_n] döner.
    """
    if not settings.ENABLE_RERANKING or not candidates or len(candidates) <= top_n:
        return candidates[:top_n]

    from src.llm import generate

    started = time.perf_counter()
    prompt = _build_prompt(query, candidates)
    try:
        raw = generate(
            prompt=prompt,
            system=_SYSTEM,
            model=settings.RERANK_MODEL or settings.GENERATION_MODEL,
            temperature=0.0,
            response_format=RerankScores,
            # Alaka puanlaması akıl yürütme gerektirmez; thinking kapalı = düşük gecikme.
            thinking_budget=0,
        )
        parsed = RerankScores.model_validate_json(raw)
    except Exception as exc:
        logger.warning(
            "Reranking atlandı (passthrough): %s",
            exc,
            extra={"event": "rerank_failed", "candidates": len(candidates)},
        )
        return candidates[:top_n]

    # 1-tabanlı index → skor. Geçersiz/eksik index'ler yok sayılır;
    # puanlanmayan adaylar -1 ile en sona düşer ama elenmez (güvenli taraf).
    score_by_index: dict[int, float] = {}
    for s in parsed.scores:
        if 1 <= s.index <= len(candidates):
            score_by_index[s.index - 1] = s.score

    ranked = sorted(
        range(len(candidates)),
        key=lambda i: score_by_index.get(i, -1.0),
        reverse=True,
    )
    reordered = [candidates[i] for i in ranked[:top_n]]

    logger.info(
        "Reranking tamamlandı",
        extra={
            "event": "rerank_complete",
            "candidates": len(candidates),
            "returned": len(reordered),
            "scored": len(score_by_index),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return reordered
