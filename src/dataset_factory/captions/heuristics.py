"""Deterministic caption checks. They can prove a caption is *malformed*; they can never prove it
is *true*, so the best status they award is ``passed`` with ``caption_review_method: rules`` -
never "visually verified"."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dataset_factory.config import CaptionRules

REVIEW_METHOD = "rules"
SCORE_BASIS = "share of deterministic text checks passed (length, placeholders, template residue, language, duplication); says nothing about visual truth"


@dataclass
class CaptionCheck:
    status: str  # passed | failed | needs_review
    score: float
    reasons: list[str] = field(default_factory=list)
    motion_described: bool = False


def _non_ascii_fraction(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > 127) / len(letters)


def check_caption(text: str | None, rules: CaptionRules, *, duplicate_count: int = 0, actions: int = 0) -> CaptionCheck:
    checks = 0
    failed: list[str] = []
    review: list[str] = []
    if not text or not text.strip():
        return CaptionCheck("failed", 0.0, ["missing"])
    t = text.strip()
    checks += 1
    if len(t) < rules.min_chars:
        failed.append(f"too_short:{len(t)}<{rules.min_chars}")
    checks += 1
    if len(t) > rules.max_chars:
        failed.append(f"too_long:{len(t)}>{rules.max_chars}")
    checks += 1
    for pat in rules.placeholder_patterns:
        if re.search(pat, t, flags=re.I):
            failed.append(f"placeholder:{pat}")
            break
    checks += 1
    for pat in rules.template_residue_patterns:
        if re.search(pat, t, flags=re.I):
            review.append(f"template_residue:{pat}")
            break
    checks += 1
    words = re.findall(r"[A-Za-z']+", t)
    if len(words) < 3:
        failed.append("too_few_words")
    elif len(set(w.lower() for w in words)) / len(words) < 0.3:
        review.append("repetitive_wording")
    if rules.language:
        checks += 1
        if rules.language.lower() == "en" and _non_ascii_fraction(t) > 0.3:
            failed.append("language_not_en")
    checks += 1
    if duplicate_count > rules.max_duplicate_captions:
        review.append(f"shared_by_{duplicate_count}_records")
    lowered = t.lower()
    motion = actions > 0 or any(term in lowered for term in rules.motion_terms)
    if not motion:
        review.append("no_motion_described")
    status = "failed" if failed else ("needs_review" if review else "passed")
    score = round(1.0 - len(failed) / max(1, checks), 3)
    return CaptionCheck(status, score, failed + review, motion)
