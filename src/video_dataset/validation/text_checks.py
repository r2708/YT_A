"""Text-structure checks on captions and questions (spaCy where available, rules otherwise).

This is the text-consistency layer the pipeline was missing: it normalises text, finds claims
inside one text that cannot both hold (a static camera that also pans), and validates the form of
generated questions and the type of their answers. It never decides whether something *happened*
in the video; that stays with the measurements and verifiers.

spaCy is optional: when the configured model loads, verb presence is checked with the parser;
otherwise a small rule set runs. The checks are language-aware in the only way that is honest
without a model per language: English rules apply to English text (the pipeline's generated text is
English); other languages are passed through with ``not_run``.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from video_dataset.schemas.quality import CheckState
from video_dataset.schemas.vision import MOVEMENT_TAXONOMY, CameraMovement, CameraMovementFamily
from video_dataset.validation.consistency import text_movements

_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_NO_SPACE_AFTER = re.compile(r"([,;:])(?=[^\s\d])")
_MULTI_PUNCT = re.compile(r"([.!?]){2,}")
_QUESTION_START = re.compile(r"^(what|which|when|where|who|how|does|do|is|are|at|in|after|before|earlier|approximately|describe)\b", re.I)
_TIME_WORDS = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:s|sec|seconds?|min|minutes?)|\d+:\d\d|around \d|about \d|starts at|at \d)", re.I)
_ORDER_WORDS = re.compile(r"^(first\b|before\b|after\b)", re.I)
_COUNT_WORDS = re.compile(r"^\d+\s+(shots?|times?)", re.I)
_CHANGE_WORDS = re.compile(r"\bchanges? (from|to)\b|\bgoes from\b|\bbecomes\b", re.I)
_VERB_HINT = re.compile(r"\b(is|are|was|were|has|have|moves?|pans?|tilts?|zooms?|tracks?|enters?|leaves?|appears?|changes?|says?|stays?|occurs?|happens?|begins?|starts?|ends?|comes?|goes?|runs?|walks?|drives?|sits?|stands?|holds?|shows?|visible|audible|lasts?|follows?|grows?|shrinks?|orbit|cranes?|dollies|dolly)\b", re.I)

ANSWER_TYPES = {
    "timestamp": "description",
    "before_after": "description",
    "temporal_ordering": "order",
    "duration": "duration",
    "event_localization": "time",
    "state_change": "change",
    "multi_event": "order",
    "long_range": "description",
}


def answer_type_for(qa_type: str, template_id: str | None = None) -> str:
    """The answer type a question of this QA type asks for. Some templates of a type ask for
    something else (a count, or the event *between* two others), so the template wins."""
    t = template_id or ""
    if t.startswith("count"):
        return "count"
    if t.startswith(("spacy", "between")):
        return "description"
    return ANSWER_TYPES.get(str(qa_type), "description")


def normalize_caption(text: str | None) -> str | None:
    """Unicode NFKC, collapsed whitespace, no space before punctuation, single terminal punctuation.
    The raw text is kept by the caller; this is the comparison / export form."""
    if text is None:
        return None
    t = unicodedata.normalize("NFKC", str(text)).replace(" ", " ")
    t = _WS.sub(" ", t).strip()
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = _NO_SPACE_AFTER.sub(r"\1 ", t)
    t = _MULTI_PUNCT.sub(r"\1", t)
    return t


def text_contradictions(texts: list[str]) -> list[str]:
    """Camera-movement claims inside the given texts that cannot all hold: opposite directions of
    one family, or ``static`` next to any movement. Returns human-readable descriptions."""
    found: list[tuple[CameraMovement, str]] = []
    for t in texts:
        if not t:
            continue
        for lab in text_movements(t):
            found.append((lab, t[:60]))
    labels = {lab for lab, _ in found}
    out: list[str] = []
    fams: dict[CameraMovementFamily, set[str | None]] = {}
    for lab in labels:
        fam, direction = MOVEMENT_TAXONOMY[lab]
        fams.setdefault(fam, set()).add(direction)
    for fam, dirs in fams.items():
        if fam == CameraMovementFamily.STATIC:
            continue
        real = {d for d in dirs if d}
        if len(real) >= 2:
            out.append(f"text claims {fam.value} in two directions ({', '.join(sorted(real))})")
    if CameraMovementFamily.STATIC in fams and any(f not in (CameraMovementFamily.STATIC, CameraMovementFamily.UNKNOWN) for f in fams):
        moving = sorted(f.value for f in fams if f not in (CameraMovementFamily.STATIC, CameraMovementFamily.UNKNOWN))
        out.append(f"text says the camera is static and also that it moves ({', '.join(moving)})")
    return out


class TextChecker:
    """Rule-based question / caption checks with an optional spaCy pipeline for verb detection."""

    def __init__(self, nlp: Any | None = None, language: str = "en"):
        self.nlp = nlp
        self.language = language
        self.name = f"spacy:{nlp.meta.get('name')}" if nlp is not None and hasattr(nlp, "meta") else "rules"

    @classmethod
    def load(cls, model: str | None) -> TextChecker:
        if not model:
            return cls()
        try:
            import spacy

            return cls(spacy.load(model, disable=["ner", "lemmatizer"]))
        except Exception:
            return cls()

    def _has_verb(self, text: str) -> bool | None:
        if self.nlp is not None:
            try:
                doc = self.nlp(text[:400])
                return any(tok.pos_ in ("VERB", "AUX") for tok in doc)
            except Exception:
                return None
        return bool(_VERB_HINT.search(text))

    def question_form(self, question: str) -> tuple[str, list[str]]:
        """(state, problems): a question must be one well-formed interrogative sentence."""
        q = normalize_caption(question) or ""
        problems: list[str] = []
        if not q:
            return CheckState.FAILED.value, ["empty question"]
        if not q.endswith("?"):
            problems.append("question does not end with '?'")
        if q.count("?") > 1:
            problems.append("more than one question mark")
        if not _QUESTION_START.match(q):
            problems.append("question does not start with an interrogative")
        if len(q.split()) < 4:
            problems.append("question too short")
        if len(q) > 400:
            problems.append("question too long")
        if "  " in question or " ?" in question:
            problems.append("whitespace artefacts")
        verb = self._has_verb(q)
        if verb is False:
            problems.append("question has no verb")
        return (CheckState.FAILED.value if problems else CheckState.PASSED.value), problems

    def answer_matches_type(self, answer: str, answer_type: str) -> tuple[str, list[str]]:
        a = normalize_caption(answer) or ""
        problems: list[str] = []
        if not a:
            return CheckState.FAILED.value, ["empty answer"]
        if answer_type == "duration" and not re.search(r"\b(second|minute|less than)", a, re.I):
            problems.append("duration answer names no duration")
        elif answer_type == "time" and not _TIME_WORDS.search(a):
            problems.append("time answer names no timestamp")
        elif answer_type == "order" and not _ORDER_WORDS.match(a):
            problems.append("order answer does not state an order")
        elif answer_type == "count" and not _COUNT_WORDS.match(a):
            problems.append("count answer does not start with a number")
        elif answer_type == "change" and not _CHANGE_WORDS.search(a):
            problems.append("change answer names no change")
        elif answer_type == "description" and len(a.split()) < 2:
            problems.append("description answer too short")
        return (CheckState.FAILED.value if problems else CheckState.PASSED.value), problems

    def caption_consistent(self, texts: list[str]) -> tuple[str, list[str]]:
        issues = text_contradictions(texts)
        return (CheckState.FAILED.value if issues else CheckState.PASSED.value), issues
