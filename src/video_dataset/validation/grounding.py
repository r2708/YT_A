"""Is the answer supported by the evidence? Lexical grounding between answer text and evidence text."""

from __future__ import annotations

import re

from video_dataset.utils.text import containment, content_words

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def answer_grounding_score(answer: str, evidence_texts: list[str], evidence_window: tuple[float, float] | None = None) -> float:
    """Fraction of the answer's content words that appear in the evidence texts.
    Numbers in the answer must fall inside the evidence time window (with 1s slack) to count."""
    words = content_words(answer)
    pool: set[str] = set()
    for t in evidence_texts:
        pool |= content_words(t)
    numbers = [float(n) for n in _NUMBER.findall(answer)]
    num_ok = 1.0
    if numbers and evidence_window is not None:
        lo, hi = evidence_window
        inside = [n for n in numbers if lo - 1.0 <= n <= hi + 1.0 or n <= 100]  # small counts/durations are allowed
        num_ok = len(inside) / len(numbers)
    if not words:
        return round(num_ok, 3)
    return round(0.85 * containment(words, pool) + 0.15 * num_ok, 3)


def numeric_grounding_score(answer: str, window: tuple[float, float], duration: float | None = None, slack: float = 1.0) -> float:
    """Grounding for answers that are numeric derivations of the evidence window (durations, timestamps).

    Every number in the answer must be explainable by the window: a timestamp inside it, its length,
    or a minutes/seconds rendering of either. Returns the fraction of numbers that are explainable.
    """
    lo, hi = window
    numbers = [float(n) for n in _NUMBER.findall(answer)]
    if not numbers:
        return 0.0
    length = max(0.0, hi - lo)
    plausible: list[float] = [lo, hi, length, round(length), round(length / 5.0) * 5, length / 60.0, round(length / 60.0, 1), round(length / 60.0)]
    plausible += [lo / 60.0, hi / 60.0, lo // 60, hi // 60, lo % 60, hi % 60, round(lo), round(hi)]
    if duration is not None:
        plausible.append(duration)
    ok = 0
    for n in numbers:
        if lo - slack <= n <= hi + slack or any(abs(n - p) <= max(slack, 0.05 * max(p, 1.0)) for p in plausible):
            ok += 1
    return round(ok / len(numbers), 3)
