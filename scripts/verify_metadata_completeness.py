"""
scripts/verify_metadata_completeness.py — Phase 2 M2 gate.

Loader → classifier → chunker boru hattını canlı Vertex/ChromaDB çağrıları
yapmadan exercise eder ve `ChunkMetadata` sözleşmesinin %100 doldurulduğunu
doğrular. Ingestion'ın embed/upsert öncesi hazırladığı metadata dict ile
özdeş bir audit yapar.

Çıkış kodları:
  0 — M2 met (her chunk tam metadata taşıyor)
  2 — fail (eksik veya geçersiz alan bulundu)

Kullanım:
  python -m scripts.verify_metadata_completeness new_docs/
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.chunk_metadata import ChunkMetadata  # noqa: E402
from src.chunkers import get_parent_chunker  # noqa: E402
from src.config import settings  # noqa: E402
from src.document_classifier import classify_path  # noqa: E402
from src.document_loaders import iter_documents  # noqa: E402
from src.ingestion import _sanitize_id, _split_into_child_chunks  # noqa: E402


REQUIRED_KEYS = {
    # ChunkMetadata alanları
    "source_file", "source_filename", "level", "skill", "doc_type",
    "language", "is_authoritative", "classifier_confidence", "page",
    # Ingestion ek alanları
    "child_index", "parent_id", "char_count", "chunking_strategy",
    "original_text", "enriched",
}

# None'a izin verilen alanlar
NULLABLE_KEYS = {"page"}


def main() -> int:
    parser = argparse.ArgumentParser(description="DELF korpus metadata bütünlük denetimi")
    parser.add_argument("root", type=Path, help="Korpus kökü (örn. new_docs/)")
    parser.add_argument(
        "--strategy", default=None,
        help=f"Chunking stratejisi (default: settings.CHUNKING_STRATEGY = {settings.CHUNKING_STRATEGY})",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"ERROR: dizin yok: {args.root}", file=sys.stderr)
        return 2

    strategy = args.strategy or settings.CHUNKING_STRATEGY
    parent_chunker = get_parent_chunker(strategy)

    total_chunks = 0
    invalid_chunks: list[tuple[str, str]] = []
    by_doc_type: Counter[str] = Counter()
    by_level: Counter[str] = Counter()
    by_skill: Counter[str] = Counter()
    documents_processed = 0
    documents_skipped = 0

    for load_result in iter_documents(args.root):
        if load_result.error:
            documents_skipped += 1
            continue
        non_empty = [pc for pc in load_result.pages if pc.text.strip()]
        if not non_empty:
            documents_skipped += 1
            continue

        base_meta = classify_path(load_result.path, root=args.root)
        documents_processed += 1
        safe_stem = _sanitize_id(load_result.path.stem)

        parent_idx = 0
        for pc in non_empty:
            for parent_text in parent_chunker(pc.text):
                parent_id = f"{safe_stem}_par{parent_idx}"
                for c_idx, child_text in enumerate(_split_into_child_chunks(parent_text)):
                    meta_dict = base_meta.model_dump()
                    meta_dict["page"] = pc.page
                    meta_dict.update({
                        "child_index": c_idx,
                        "parent_id": parent_id,
                        "char_count": len(child_text),
                        "chunking_strategy": strategy,
                        "original_text": child_text,
                        "enriched": False,
                    })

                    total_chunks += 1

                    missing = REQUIRED_KEYS - set(meta_dict.keys())
                    if missing:
                        invalid_chunks.append((parent_id, f"missing: {sorted(missing)}"))
                        continue
                    nulls = [
                        k for k, v in meta_dict.items()
                        if v is None and k not in NULLABLE_KEYS
                    ]
                    if nulls:
                        invalid_chunks.append((parent_id, f"null in: {sorted(nulls)}"))
                        continue

                    # Pydantic validation — Literal'lerin geçerliliği vs.
                    try:
                        ChunkMetadata.model_validate({
                            k: meta_dict[k] for k in ChunkMetadata.model_fields
                        })
                    except Exception as exc:
                        invalid_chunks.append((parent_id, f"validation: {exc}"))
                        continue

                    by_doc_type[meta_dict["doc_type"]] += 1
                    by_level[meta_dict["level"]] += 1
                    by_skill[meta_dict["skill"]] += 1

                parent_idx += 1

    print(f"\nBelge: {documents_processed} işlendi, {documents_skipped} atlandı")
    print(f"Toplam chunk: {total_chunks}")
    print(f"Geçersiz chunk: {len(invalid_chunks)}")

    if invalid_chunks:
        print("\nGeçersiz örnekler (ilk 10):")
        for pid, reason in invalid_chunks[:10]:
            print(f"  {pid}: {reason}")

    print(f"\nby_doc_type: {dict(by_doc_type.most_common())}")
    print(f"by_level:    {dict(by_level.most_common())}")
    print(f"by_skill:    {dict(by_skill.most_common())}")

    completeness = (total_chunks - len(invalid_chunks)) / total_chunks if total_chunks else 0.0
    if not invalid_chunks:
        print(f"\n[OK] M2 met: {total_chunks}/{total_chunks} chunks fully tagged")
        return 0
    print(f"\n[FAIL] M2: {completeness:.1%} ({total_chunks - len(invalid_chunks)}/{total_chunks})")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
