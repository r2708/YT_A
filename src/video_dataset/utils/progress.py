"""Time-throttled progress lines for long-running loops.

Every heavy stage (download, scene scan, frame extraction, ASR, OCR, vision) iterates over hundreds or
thousands of items and used to print nothing until it finished. `ProgressLog` emits one INFO line at
most every `every_seconds`, with the fraction done, elapsed time and an ETA, so the terminal always
shows which process is running and how far along it is - without flooding the log files.
"""

from __future__ import annotations

import logging
import time
from typing import Any


def fmt_seconds(seconds: float | None) -> str:
    """0:07:03 / 07:03 style, '--:--' when unknown."""
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "--:--"
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class ProgressLog:
    """Throttled `label 120/2387 (5%) elapsed 01:20 eta 25:10 <extra>` lines.

    `total` may be a count or a duration in seconds (`unit`); `update(done, extra=...)` logs when at
    least `every_seconds` passed since the last line (the first update after `min_items` is logged
    immediately, so a slow first item still shows life). `finish()` logs the closing line once.
    """

    def __init__(
        self,
        log: logging.Logger | logging.LoggerAdapter,
        label: str,
        total: float | None,
        *,
        unit: str = "",
        every_seconds: float = 10.0,
        min_items: float = 1,
        level: int = logging.INFO,
    ):
        self.log = log
        self.label = label
        self.total = float(total) if total else None
        self.unit = unit
        self.every = float(every_seconds)
        self.min_items = float(min_items)
        self.level = level
        self.t0 = time.time()
        self.last_log = 0.0
        self.done = 0.0
        self.finished = False

    def _fmt_count(self, value: float) -> str:
        if self.unit == "s":
            return fmt_seconds(value)
        return f"{int(value)}"

    def _line(self, extra: str) -> str:
        elapsed = time.time() - self.t0
        parts = [self.label]
        if self.total:
            frac = min(1.0, self.done / self.total) if self.total else 0.0
            eta = (elapsed / frac - elapsed) if frac > 0 else None
            parts.append(f"{self._fmt_count(self.done)}/{self._fmt_count(self.total)}{(' ' + self.unit) if self.unit and self.unit != 's' else ''} ({frac * 100:.0f}%)")
            parts.append(f"elapsed {fmt_seconds(elapsed)} eta {fmt_seconds(eta)}")
        else:
            parts.append(f"{self._fmt_count(self.done)}{(' ' + self.unit) if self.unit and self.unit != 's' else ''}")
            parts.append(f"elapsed {fmt_seconds(elapsed)}")
        if extra:
            parts.append(extra)
        return " ".join(parts)

    def update(self, done: float, extra: str = "", force: bool = False) -> None:
        self.done = float(done)
        now = time.time()
        if not force:
            if self.done < self.min_items:
                return
            if now - self.last_log < self.every:
                return
        self.last_log = now
        self.log.log(self.level, self._line(extra))

    def step(self, n: float = 1, extra: str = "") -> None:
        self.update(self.done + n, extra)

    def finish(self, extra: str = "") -> None:
        if self.finished:
            return
        self.finished = True
        if self.total and self.done < self.total:
            self.done = self.total
        if time.time() - self.t0 >= self.every or self.last_log:
            self.log.log(self.level, self._line(extra or "done"))

    def __enter__(self) -> ProgressLog:
        return self

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is None:
            self.finish()
