"""Retry helper with exponential backoff for transient failures (network, API, OOM)."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable
from typing import TypeVar

from video_dataset.utils.logging import get_logger

T = TypeVar("T")
log = get_logger("retry")


def retry_call(
    fn: Callable[[], T],
    *,
    retries: int = 2,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: Iterable[type[BaseException]] = (Exception,),
    on_retry: Callable[[BaseException, int], None] | None = None,
    label: str = "operation",
) -> T:
    retry_types = tuple(retry_on)
    attempt = 0
    while True:
        try:
            return fn()
        except retry_types as exc:  # type: ignore[misc]
            attempt += 1
            if attempt > retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            log.warning("%s failed (%s: %s); retry %d/%d in %.1fs", label, type(exc).__name__, str(exc)[:200], attempt, retries, delay)
            if on_retry:
                on_retry(exc, attempt)
            time.sleep(delay)
