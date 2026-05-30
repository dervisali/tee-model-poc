"""
Phase 3 — deterministik düşmanca (adversarial) güvenlik koşucusu.

`evaluation/chatbot_safety_probes.json` içindeki her probu GİRİŞ TARAMASINDAN
(`safety.screen_input`) geçirir ve beklenen sonuçla (`screen_expect`) karşılaştırır.
Bu katman deterministiktir — LLM/ağ çağrısı YOK, MOCK_MODE'da da çalışır.

Davranışsal katmanlar (sistem talimatı: puan/dil reddi; grounding kapısı; no-context
fallback) CANLI LLM gerektirir ve burada "pending_live" olarak işaretlenir; tam
davranışsal doğrulama için ayrı bir canlı koşu gerekir.

Çıktı: evaluation/results/adversarial_safety_<UTC>.json  (Read ile geri okuyun).
Kullanım: python -m scripts.adversarial_safety
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src import safety  # noqa: E402

PROBES = settings.BASE_DIR / "evaluation" / "chatbot_safety_probes.json"
RESULTS_DIR = settings.BASE_DIR / "evaluation" / "results"


def run() -> dict:
    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    rows = []
    screen_pass = screen_total = 0
    for p in probes:
        verdict = safety.screen_input(p["prompt"], p.get("language", "tr"))
        actual = "block" if not verdict.allowed else "allow"
        expect = p.get("screen_expect", "allow")
        ok = actual == expect
        # Davranışsal katmanlar input-screen'de "allow" beklenir ama asıl savunma canlıdır.
        behavioral = p.get("defended_by") in ("system_prompt", "grounding_gate", "retrieval_fallback")
        screen_total += 1
        screen_pass += int(ok)
        rows.append({
            "id": p["id"], "category": p["category"], "defended_by": p.get("defended_by"),
            "screen_expect": expect, "screen_actual": actual,
            "screen_category": verdict.category, "screen_ok": ok,
            "behavioral_verification": "pending_live" if behavioral else "n/a",
        })

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_probes": len(probes),
        "input_screen_pass": screen_pass,
        "input_screen_total": screen_total,
        "input_screen_all_pass": screen_pass == screen_total,
        "note": ("Input-screen layer verified deterministically. Behavioral layers "
                 "(score refusal, grounding gate, no-context fallback, language lock) "
                 "are marked pending_live and require a MOCK_MODE=false run."),
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"adversarial_safety_{ts}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["artifact"] = str(out)
    return report


if __name__ == "__main__":
    r = run()
    print(f"input_screen {r['input_screen_pass']}/{r['input_screen_total']} "
          f"all_pass={r['input_screen_all_pass']}")
    print(f"artifact: {r['artifact']}")
    raise SystemExit(0 if r["input_screen_all_pass"] else 1)
