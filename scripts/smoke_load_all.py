"""
scripts/smoke_load_all.py — Phase 1 M1 coverage gate.

`new_docs/` altındaki tüm desteklenen dosyaları document_loaders üzerinden
ingest eder ve dosya başına extraction istatistiklerini raporlar.

Çıkış kodları:
  0 — M1 met (>=98% non-empty)
  1 — iteration band (90-98%): triage gerekiyor
  2 — fail (<90%): mimari sorun

Kullanım:
  python -m scripts.smoke_load_all new_docs/
  python -m scripts.smoke_load_all new_docs/ --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.document_loaders import iter_documents, IMAGE_EXTENSIONS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="DELF korpus extraction coverage smoke")
    parser.add_argument("root", type=Path, help="Belge kök dizini (örn. new_docs/)")
    parser.add_argument("--json", type=Path, default=None, help="JSON çıktı yolu (opsiyonel)")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"ERROR: dizin yok: {args.root}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    total = 0
    nonempty = 0
    needs_ocr_pdfs: list[str] = []
    failures: list[tuple[str, str]] = []
    ocr_unavailable: list[str] = []

    for result in iter_documents(args.root):
        total += 1
        rel = str(result.path.relative_to(args.root))
        ext = result.path.suffix.lower()

        if result.error is not None:
            status = "ERROR"
            n_pages = 0
            n_chars = 0
            note = result.error
            failures.append((rel, result.error))
            if ext in IMAGE_EXTENSIONS and "tesseract" in result.error.lower():
                ocr_unavailable.append(rel)
        else:
            n_pages = len(result.pages)
            n_chars = sum(len(p.text) for p in result.pages)
            if n_chars == 0:
                status = "EMPTY"
                note = "no extractable text"
                if ext == ".pdf":
                    needs_ocr_pdfs.append(rel)
            else:
                status = "OK"
                note = ""
                nonempty += 1

        rows.append({
            "path": rel,
            "ext": ext,
            "status": status,
            "pages": n_pages,
            "chars": n_chars,
            "note": note,
        })

    coverage = (nonempty / total) if total else 0.0

    # stdout report
    print(f"\n{'STATUS':<8} {'EXT':<6} {'PAGES':<6} {'CHARS':<8} PATH")
    print("-" * 100)
    for r in rows:
        print(f"{r['status']:<8} {r['ext']:<6} {r['pages']:<6} {r['chars']:<8} {r['path']}")

    print(f"\nProcessed: {total} files; non-empty extractions: {nonempty}")
    print(f"Coverage: {coverage:.1%}")

    if needs_ocr_pdfs:
        print(f"\nPDFs with 0 chars (OCR needed): {len(needs_ocr_pdfs)}")
        for p in needs_ocr_pdfs:
            print(f"  - {p}")
    if ocr_unavailable:
        print(f"\nImages with OCR unavailable: {len(ocr_unavailable)}")
        for p in ocr_unavailable:
            print(f"  - {p}")
    if failures:
        print(f"\nFailures: {len(failures)}")
        for p, err in failures:
            print(f"  - {p}: {err}")

    if args.json is not None:
        payload = {
            "total": total,
            "nonempty": nonempty,
            "coverage": coverage,
            "needs_ocr_pdfs": needs_ocr_pdfs,
            "ocr_unavailable": ocr_unavailable,
            "failures": [{"path": p, "error": e} for p, e in failures],
            "rows": rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON written to {args.json}")

    threshold = 0.98
    if coverage >= threshold:
        print(f"\n[OK] M1 met: {coverage:.1%} >= {threshold:.0%}")
        return 0
    if coverage >= 0.90:
        print(f"\n[WARN] M1 in iteration band: {coverage:.1%}")
        return 1
    print(f"\n[FAIL] M1: {coverage:.1%} < 90%")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
