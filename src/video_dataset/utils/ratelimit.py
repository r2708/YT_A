"""Thread-safe token-bucket rate limiter for API-backed providers.

One bucket per (provider, model) is shared across the process, so the vision analyzer, the causal
inferencer and the paraphraser calling the same API all draw from the same quota.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from video_dataset.utils.logging import get_logger

log = get_logger("ratelimit")


class RateLimiter:
    """Token bucket: `rate_per_minute` tokens refill continuously, up to `burst` stored tokens."""

    def __init__(self, rate_per_minute: float, burst: int | None = None, clock: Callable[[], float] = time.monotonic, sleeper: Callable[[float], None] = time.sleep):
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be > 0")
        self.rate = float(rate_per_minute) / 60.0  # tokens per second
        self.capacity = float(burst if burst is not None else max(1, int(rate_per_minute)))
        self.tokens = self.capacity
        self._clock = clock
        self._sleep = sleeper
        self._last = clock()
        self._lock = threading.Lock()
        self.waits = 0
        self.total_wait_seconds = 0.0

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until `tokens` are available; returns the seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    if waited:
                        self.waits += 1
                        self.total_wait_seconds += waited
                    return waited
                delay = (tokens - self.tokens) / self.rate
            self._sleep(delay)
            waited += delay


_REGISTRY: dict[str, RateLimiter] = {}
_REGISTRY_LOCK = threading.Lock()


def get_rate_limiter(key: str, rate_per_minute: float) -> RateLimiter | None:
    """Process-wide limiter for `key` (e.g. "anthropic:claude-opus-5"). None when the rate is 0/unset."""
    if not rate_per_minute or rate_per_minute <= 0:
        return None
    with _REGISTRY_LOCK:
        limiter = _REGISTRY.get(key)
        if limiter is None or abs(limiter.rate * 60.0 - rate_per_minute) > 1e-9:
            limiter = RateLimiter(rate_per_minute)
            _REGISTRY[key] = limiter
        return limiter


class RateLimitedLLMClient:
    """Wraps any LLMClient so every `complete()` call first takes a token from the shared bucket."""

    def __init__(self, inner: Any, limiter: RateLimiter):
        self.inner = inner
        self.limiter = limiter
        self.name = getattr(inner, "name", "llm")
        self.model = getattr(inner, "model", "")
        self.supports_images = bool(getattr(inner, "supports_images", False))

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        images: list[Path] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Any:
        waited = self.limiter.acquire()
        if waited >= 1.0:
            log.info("rate limit: waited %.1fs before calling %s/%s", waited, self.name, self.model)
        return self.inner.complete(
            prompt, system=system, images=images, json_schema=json_schema, max_tokens=max_tokens, temperature=temperature
        )

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    def __getattr__(self, item: str) -> Any:  # delegate anything else (e.g. `client`) to the wrapped client
        return getattr(self.inner, item)
