"""Quality scores. Components are computed only from real signals; otherwise they stay None."""

from __future__ import annotations

import re

from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.events import Event
from video_dataset.schemas.qa import QARecord
from video_dataset.schemas.quality import QualityScore
from video_dataset.schemas.vision import SceneAnalysis, VerificationVerdict

_GENERIC = {"cinematic", "beautiful", "stunning", "amazing", "nice", "great", "interesting", "various", "some", "thing", "things", "stuff"}
_PLACEHOLDER = {"unknown", "unmeasured", "n/a", "null", "none"}
_CONCRETE = re.compile(
    r"\b(red|blue|green|yellow|white|black|grey|gray|orange|brown|purple|pink|left|right|center|centre|foreground|background|top|bottom|"
    r"wide|close|medium|low|high|overhead|aerial|bright|dark|dim|warm|cool|soft|hard|shadow|sunlight|window|street|road|room|table|"
    r"car|person|people|man|woman|child|dog|tree|building|sky|water|hand|door|wall|floor|camera|pan|tilt|zoom|static|handheld|"
    r"\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


def description_quality_score(text: str | None) -> float | None:
    """Heuristic specificity score in [0,1]: length, concreteness, absence of generic/placeholder words."""
    if not text:
        return None
    words = re.findall(r"[a-zA-Z0-9#]+", text.lower())
    if not words:
        return 0.0
    n = len(words)
    length_score = min(1.0, n / 25.0)
    concrete = len(_CONCRETE.findall(text))
    concrete_score = min(1.0, concrete / max(4.0, n / 6.0))
    generic_penalty = sum(1 for w in words if w in _GENERIC) / n
    placeholder_penalty = sum(1 for w in words if w in _PLACEHOLDER) / n
    score = 0.4 * length_score + 0.6 * concrete_score - 2.0 * generic_penalty - 2.0 * placeholder_penalty
    return round(max(0.0, min(1.0, score)), 3)


def _verifier_score(v) -> float | None:  # type: ignore[no-untyped-def]
    if v is None or v.verdict == VerificationVerdict.UNKNOWN:
        return None
    return v.score


def scene_quality(a: SceneAnalysis) -> QualityScore:
    grounding = _verifier_score(a.verification)
    notes = []
    if grounding is None and a.confidence is not None and a.confidence_source == ConfidenceSource.MODEL_SELF_REPORT:
        notes.append("grounding unavailable: only model self-report confidence")
    if a.provider == "heuristic":
        notes.append("measurement-derived description (no generative model)")
    return QualityScore.from_components(
        grounding=grounding,
        temporal_accuracy=None,  # scene boundaries come from the detector; accuracy of the description over time is not measured
        description_quality=description_quality_score(a.summary),
        notes=notes,
    )


def event_quality(e: Event) -> QualityScore:
    grounding = _verifier_score(e.verification)
    if grounding is None and e.confidence is not None and e.confidence_source in (ConfidenceSource.MEASUREMENT, ConfidenceSource.OCR_SCORE, ConfidenceSource.ASR_LOGPROB, ConfidenceSource.DETECTOR_SCORE):
        grounding = e.confidence  # direct signal measurements are grounding by construction
    temporal = e.confidence if e.boundary_precision in ("measured", "asr", "frame") and e.confidence is not None else None
    notes = []
    if e.boundary_precision == "scene":
        notes.append("boundaries inherited from the shot; exact action timing inside the shot unmeasured")
    return QualityScore.from_components(grounding=grounding, temporal_accuracy=temporal, description_quality=description_quality_score(e.event), notes=notes)


def relation_quality(conf: float | None, event_qualities: list[QualityScore]) -> QualityScore:
    temporals = [q.temporal_accuracy for q in event_qualities if q.temporal_accuracy is not None]
    return QualityScore.from_components(
        grounding=conf,
        temporal_accuracy=round(min(temporals), 3) if len(temporals) == len(event_qualities) and temporals else None,
        description_quality=None,
    )


def qa_quality(qa: QARecord, grounding: float | None, event_qualities: list[QualityScore]) -> QualityScore:
    temporals = [q.temporal_accuracy for q in event_qualities if q.temporal_accuracy is not None]
    temporal = round(min(temporals), 3) if temporals and len(temporals) == len(event_qualities) else None
    notes = []
    if qa.confidence is None:
        notes.append("evidence includes unscored annotations")
    return QualityScore.from_components(
        grounding=grounding,
        temporal_accuracy=temporal,
        description_quality=description_quality_score(qa.answer),
        notes=notes,
    )
