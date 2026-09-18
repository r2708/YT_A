"""Text normalization, similarity and human-friendly time phrasing."""

from __future__ import annotations

import re
from collections.abc import Iterable

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "is", "are", "was", "were", "be", "been",
    "with", "by", "for", "from", "as", "it", "its", "this", "that", "these", "those", "there", "here", "then",
    "into", "onto", "over", "under", "up", "down", "out", "about", "after", "before", "during", "while",
    "what", "which", "who", "when", "where", "how", "does", "do", "did", "happens", "happen", "happening",
    "video", "seconds", "second", "approximately", "around", "first", "next", "immediately",
    "camera", "shot", "scene", "s",
}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s%']", re.UNICODE)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [t for t in normalize_text(text).split(" ") if t]


def content_words(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1 and not t.isdigit()}


def word_shingles(text: str, k: int = 3) -> set[str]:
    toks = tokenize(text)
    if len(toks) <= k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment(needle: Iterable[str], haystack: Iterable[str]) -> float:
    """Fraction of `needle` items present in `haystack` (1.0 when needle is empty)."""
    sn, sh = set(needle), set(haystack)
    if not sn:
        return 1.0
    return len(sn & sh) / len(sn)


def format_seconds(t: float, decimals: int = 1) -> str:
    return f"{t:.{decimals}f} seconds"


def format_timestamp_clock(t: float) -> str:
    t = max(0.0, t)
    m, s = divmod(int(round(t)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def approx_duration_phrase(seconds: float) -> str:
    """Round a duration to a granularity that matches its magnitude ("about 6 seconds")."""
    if seconds < 1.0:
        return "less than a second"
    if seconds < 10:
        n = max(1, round(seconds))
        return f"about {n:d} second{'s' if n != 1 else ''}"
    if seconds < 60:
        return f"about {int(round(seconds / 5.0) * 5):d} seconds"
    minutes = seconds / 60.0
    if minutes < 10:
        return f"about {minutes:.1f} minutes"
    return f"about {round(minutes):d} minutes"


def approx_timestamp_phrase(t: float) -> str:
    if t < 60:
        return f"around {t:.0f} seconds" if t >= 10 else f"around {t:.1f} seconds"
    return f"around {format_timestamp_clock(t)} ({t:.0f} seconds)"


def sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def lower_first(text: str) -> str:
    text = text.strip().rstrip(".")
    return text[0].lower() + text[1:] if text else text
