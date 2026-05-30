"""
Phase 3 — süreç-içi oran sınırlama + zarif düşüş (graceful degradation).

Denetimsiz (unsupervised) bir araçta kötüye kullanım ve kota tükenmesi gerçek
risklerdir. Bu modül session_id başına kayan-pencere (sliding-window) sınırlayıcı
sağlar: dakikalık ve günlük iki ayrı pencere. Saf Python + kilit; harici bağımlılık
yok. Tek-süreçli Streamlit için yeterlidir.

NOT (Phase 4): Çok-süreçli/çok-instance dağıtımda bu yeterli değildir — paylaşımlı
bir sayaç (Redis/Memorystore) gerekir. Streamlit tek-süreçli olduğundan şimdilik
süreç-içi yeterli; Phase 4 serving kararında yeniden ele alınacaktır.

`graceful_degradation_message()` Vertex erişilemez olduğunda kullanıcıya net,
dile duyarlı bir mesaj döndürür (çökme yerine).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal

from src.config import settings

Language = Literal["tr", "fr"]

_MINUTE = 60.0
_DAY = 86_400.0


@dataclass
class RateLimitResult:
    """Oran sınırlama kararı."""

    allowed: bool
    scope: str | None = None        # "minute" | "day" | None
    retry_after_s: int = 0          # tahmini bekleme (saniye)
    reason: str | None = None


class SlidingWindowRateLimiter:
    """session_id başına dakikalık ve günlük kayan-pencere sınırlayıcı (thread-safe)."""

    def __init__(self, per_minute: int, per_day: int):
        self.per_minute = per_minute
        self.per_day = per_day
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _purge(self, dq: "deque[float]", now: float) -> None:
        cutoff = now - _DAY
        while dq and dq[0] < cutoff:
            dq.popleft()

    def check(self, key: str, *, now: float | None = None) -> RateLimitResult:
        """`key` için isteğe izin verilip verilmediğini döndürür; izinliyse sayar.

        İzin verilmezse olay KAYDEDİLMEZ (reddedilen istek kotayı tüketmez).
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            dq = self._events[key]
            self._purge(dq, now)

            day_count = len(dq)
            minute_count = sum(1 for t in dq if t >= now - _MINUTE)

            if minute_count >= self.per_minute:
                oldest_in_min = next((t for t in dq if t >= now - _MINUTE), now)
                retry = max(1, int(_MINUTE - (now - oldest_in_min)) + 1)
                return RateLimitResult(allowed=False, scope="minute",
                                       retry_after_s=retry,
                                       reason=f"{minute_count}/{self.per_minute} per minute")
            if day_count >= self.per_day:
                retry = max(1, int(_DAY - (now - dq[0])) + 1)
                return RateLimitResult(allowed=False, scope="day",
                                       retry_after_s=retry,
                                       reason=f"{day_count}/{self.per_day} per day")

            dq.append(now)
            return RateLimitResult(allowed=True)

    def reset(self, key: str | None = None) -> None:
        """Test/operasyon için sayaçları sıfırlar (key=None → tümü)."""
        with self._lock:
            if key is None:
                self._events.clear()
            else:
                self._events.pop(key, None)


# Süreç-ömrü paylaşımlı sınırlayıcı (config'ten beslenir).
_limiter: SlidingWindowRateLimiter | None = None
_limiter_lock = threading.Lock()


def get_limiter() -> SlidingWindowRateLimiter:
    """Paylaşımlı sınırlayıcıyı döndürür (lazy singleton)."""
    global _limiter
    if _limiter is None:
        with _limiter_lock:
            if _limiter is None:
                _limiter = SlidingWindowRateLimiter(
                    per_minute=settings.RATE_LIMIT_PER_MINUTE,
                    per_day=settings.RATE_LIMIT_PER_DAY,
                )
    return _limiter


def reset_limiter() -> None:
    """Paylaşımlı sınırlayıcıyı tamamen sıfırlar (test sonrası temizlik)."""
    global _limiter
    with _limiter_lock:
        _limiter = None


def check_rate_limit(session_id: str | None) -> RateLimitResult:
    """Yüksek seviye yardımcı: bayrak kapalıysa veya session_id yoksa izin ver.

    session_id None ise (örn. programatik/test çağrısı) sınırlama ATLANIR —
    anahtar olmadan adil sayım yapılamaz.
    """
    if not settings.ENABLE_RATE_LIMIT or not session_id:
        return RateLimitResult(allowed=True)
    return get_limiter().check(session_id)


def rate_limit_message(result: RateLimitResult, language: Language = "tr") -> str:
    """Reddedilen istek için kullanıcıya gösterilecek dile duyarlı mesaj."""
    lang = language if language in ("tr", "fr") else "tr"
    wait = result.retry_after_s
    if lang == "tr":
        return (f"⏳ Çok fazla istek gönderildi. Lütfen yaklaşık {wait} saniye "
                f"sonra tekrar deneyin. (Sistem kararlılığı için istek sınırı uygulanıyor.)")
    return (f"⏳ Trop de requêtes. Veuillez réessayer dans environ {wait} secondes. "
            f"(Une limite de débit est appliquée pour la stabilité du service.)")


def graceful_degradation_message(language: Language = "tr") -> str:
    """Vertex/LLM erişilemez olduğunda gösterilecek net mesaj (çökme yerine)."""
    lang = language if language in ("tr", "fr") else "tr"
    if lang == "tr":
        return ("⚠️ Yapay zeka servisi şu anda geçici olarak kullanılamıyor. "
                "Bu, sorunuzla ilgili bir hata değildir; lütfen birazdan tekrar deneyin. "
                "Kaynak belgeleri Veritabanı Gezgini sekmesinden inceleyebilirsiniz.")
    return ("⚠️ Le service d'IA est temporairement indisponible. Il ne s'agit pas "
            "d'une erreur liée à votre question ; veuillez réessayer dans un instant. "
            "Vous pouvez consulter les documents sources dans l'onglet Explorateur.")
