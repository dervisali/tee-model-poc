"""
Retrieval kalite karşılaştırması — dense vs sparse vs hybrid.

Türkçe mevzuat metnine özgü birebir terim eşleşmelerinin (örn.
"ek gösterge katsayısı", "net aylığın 1/4'ü") hibrit BM25 + dense füzyonu
sayesinde dense-only retrieval'a göre nasıl iyileştiğini gösterir.

Kullanım:
    # Önce ingestion + BM25 inşası: python -m src.ingestion
    python -m scripts.compare_retrieval

Çıktı: her sorgu için top-3 sonuçların ÜÇ retrieval modu altındaki
karşılaştırmalı tablosu.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import retrieve_context  # noqa: E402


# Mission Phase 1.3 spec'i: birebir Türkçe regülatör terim sorguları.
QUERIES: list[str] = [
    "ek gösterge katsayısı",
    "net aylığın 1/4'ü icra kesintisi",
    "kümülatif matrah Ocak ayı sıfırlama",
    "Form-ADB-01 aile durumu",
    "göreve başlama belgesi",
    "yıllık katsayı güncelleme",
]


def _format_hit(i: int, hit: dict) -> str:
    src = hit.get("source", "?")
    pid = hit.get("parent_id", "?")
    score = hit.get("score")
    dist = hit.get("distance")
    bm = hit.get("bm25_score")
    preview = hit.get("child_text", "")[:100].replace("\n", " ")
    score_str = f"score={score}" if score is not None else f"dist={dist}"
    bm_str = f" bm25={round(bm, 3)}" if bm is not None else ""
    return f"  [{i}] {src} {pid} | {score_str}{bm_str}\n      {preview}…"


def compare(query: str, top_k: int = 3) -> None:
    print(f"\n{'=' * 78}")
    print(f"SORGU: {query}")
    print("=" * 78)
    for mode in ("dense", "sparse", "hybrid"):
        print(f"\n[{mode.upper()}]")
        try:
            hits = retrieve_context(query, top_k=top_k, search_mode=mode)
            if not hits:
                print("  (sonuç yok)")
                continue
            for i, hit in enumerate(hits, start=1):
                print(_format_hit(i, hit))
        except Exception as exc:
            print(f"  HATA: {exc}")


def main() -> None:
    print("TEE-Model retrieval karşılaştırması — dense vs sparse vs hybrid (top-3)")
    for q in QUERIES:
        compare(q, top_k=3)


if __name__ == "__main__":
    main()
