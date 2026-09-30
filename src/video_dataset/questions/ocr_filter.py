"""Decide which on-screen text events are fit to anchor a question.

OCR misreads watermarks and stylised titles differently in every frame ("SCENIC RELAXATION" comes out
as "SCEMGRELA", "XATON", "LAAATION", ...). The OCR stage already drops the exact and one-edit spellings
of a persistent overlay, but garbled variants survive as short tracks and would each become a
"what happens before on-screen text X is visible?" question. Two cheap checks remove them:

* :func:`looks_like_text` rejects strings whose tokens do not have the shape of words (no vowels,
  long consonant runs, tripled letters, stray lowercase letters inside capitals, symbol soup).
* :func:`recurring_text_events` counts, for every OCR text in a video, how many other OCR texts
  resemble it (substring or shared character bigrams) and flags those with more look-alikes than
  allowed; a text that keeps coming back is a watermark, caption template or timecode, not a
  temporal anchor. The count is per text, not transitive, so two captions that happen to share a
  word are not pulled into each other's family.
"""

from __future__ import annotations

import re

from video_dataset.schemas.events import Event, EventType

_TOKEN = re.compile(r"[A-Za-z]+|\d+")
_QUOTED = re.compile(r'"([^"]*)"')
_VOWELS = frozenset("aeiouy")
_CONSONANT_RUN = re.compile(r"[b-df-hj-np-tv-xz]{5,}")
_VOWEL_RUN = re.compile(r"[aeiou]{4,}")
_TRIPLE = re.compile(r"(.)\1\1")
_LOWER_INSIDE_CAPS = re.compile(r"[A-Z][a-z][A-Z]")

MIN_FAMILY_KEY = 3  # shorter keys ("re", "on") match everything and are never used for grouping
MIN_BIGRAM_KEY = 5  # the bigram rule needs a few characters to be meaningful
BIGRAM_CONTAINMENT = 0.6


def ocr_text(e: Event) -> str:
    """The recognised string behind a TEXT_ON_SCREEN event."""
    text = str(e.attributes.get("text") or "").strip() if e.attributes else ""
    if not text and e.entities:
        text = str(e.entities[0]).strip()
    if not text:
        m = _QUOTED.search(e.event or "")
        text = m.group(1).strip() if m else (e.event or "").strip()
    return text


def word_shaped(token: str) -> bool:
    """Does an alphabetic token of three or more letters look like a word (in any Latin-script language)?"""
    low = token.lower()
    if not any(ch in _VOWELS for ch in low):
        return False
    if _CONSONANT_RUN.search(low.replace("y", "i")) or _VOWEL_RUN.search(low) or _TRIPLE.search(low):
        return False
    upper = sum(ch.isupper() for ch in token)
    lower = sum(ch.islower() for ch in token)
    if upper >= 4 and 1 <= lower <= 2 and _LOWER_INSIDE_CAPS.search(token):
        return False  # "SCENICRnIAX": a lowercase glyph OCR dropped into a capitalised word
    return True


def looks_like_text(text: str, min_chars: int = 4) -> bool:
    """Reject OCR output that is not readable text: too short, symbol soup or tokens without word shape.

    Digit-only strings (scores, timecodes, prices) and strings made of one- or two-letter tokens
    ("OK", "4K TV") cannot be judged and are accepted; the recurrence check and the per-video cap
    still apply to them.
    """
    raw = text.strip()
    if not raw:
        return False
    compact = raw.replace(" ", "")
    alnum = sum(ch.isalnum() for ch in compact)
    if alnum < max(1, int(min_chars)) or alnum / max(1, len(compact)) < 0.6:
        return False
    tokens = _TOKEN.findall(raw)
    if not tokens:
        return False
    judged = [t for t in tokens if t.isalpha() and len(t) >= 3]
    if not judged:
        return True
    good = sum(word_shaped(t) for t in judged)
    return good / len(judged) >= 0.5 if len(judged) > 1 else good == 1


def family_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _bigrams(key: str) -> set[str]:
    return {key[i : i + 2] for i in range(len(key) - 1)}


def _related(a: str, b: str, ba: set[str], bb: set[str]) -> bool:
    """Two OCR spellings of (probably) the same sign: one contains the other, or most of the
    shorter one's character bigrams occur in the longer one."""
    short, long_, bs = (a, b, ba) if len(a) <= len(b) else (b, a, bb)
    if len(short) < MIN_FAMILY_KEY:
        return False
    if short in long_:
        return True
    if len(short) < MIN_BIGRAM_KEY or not bs:
        return False
    other = bb if short is a else ba
    return len(bs & other) / len(bs) >= BIGRAM_CONTAINMENT


def recurring_text_events(events: list[Event], max_repeats: int) -> set[str]:
    """Event ids of TEXT_ON_SCREEN events that are a *recurring* text or look like one.

    A text is recurring when it resembles more than ``max_repeats`` other on-screen texts of the same
    video (counting itself, it occurs more than ``max_repeats`` times); every text that resembles a
    recurring one is dropped too (garbled reads that only match one or two spellings of the
    watermark). Resemblance is not chained further, so captions that merely share a word with each
    other are left alone. ``max_repeats`` <= 0 disables the check."""
    if max_repeats <= 0:
        return set()
    texts = [(e.event_id, family_key(ocr_text(e))) for e in events if e.event_type == EventType.TEXT_ON_SCREEN]
    if len(texts) <= max_repeats:
        return set()
    keys = [k for _, k in texts]
    grams = [_bigrams(k) for k in keys]
    neighbours: list[set[int]] = [set() for _ in keys]
    for i in range(len(keys)):
        if len(keys[i]) < MIN_FAMILY_KEY:
            continue
        for j in range(i + 1, len(keys)):
            if len(keys[j]) >= MIN_FAMILY_KEY and _related(keys[i], keys[j], grams[i], grams[j]):
                neighbours[i].add(j)
                neighbours[j].add(i)
    recurring = {i for i in range(len(keys)) if len(neighbours[i]) + 1 > max_repeats}
    dropped = recurring | {j for i in recurring for j in neighbours[i]}
    return {texts[i][0] for i in dropped}


def unusable_text_events(events: list[Event], *, shape_filter: bool = True, min_chars: int = 4, max_repeats: int = 3) -> set[str]:
    """Ids of TEXT_ON_SCREEN events that must not anchor questions (garbled or recurring text)."""
    bad: set[str] = set()
    if shape_filter:
        bad.update(e.event_id for e in events if e.event_type == EventType.TEXT_ON_SCREEN and not looks_like_text(ocr_text(e), min_chars))
    bad.update(recurring_text_events(events, max_repeats))
    return bad


def is_text_anchored(events: list[Event]) -> bool:
    return any(e.event_type == EventType.TEXT_ON_SCREEN for e in events)
