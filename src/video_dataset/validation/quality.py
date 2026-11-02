"""Quality scores. Components are computed only from real signals; otherwise they stay None."""

from __future__ import annotations

import re
from typing import Any

from video_dataset.config import QualityConfig
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.events import Event
from video_dataset.schemas.qa import QARecord
from video_dataset.schemas.quality import QualityScore
from video_dataset.schemas.vision import Measurements, SceneAnalysis, VerificationVerdict

VIDEO_LEVEL_FLAGS = ("low_resolution", "low_bitrate", "low_fps", "extreme_aspect_ratio", "duration_mismatch")


def frozen_shot(scan: Any, duration: float, cfg: QualityConfig) -> bool:
    """A shot of at least ``frozen_min_duration`` seconds whose scan never registers a content change
    above ``frozen_max_change`` and no measurable motion: a still image or a stalled stream."""
    if scan is None or not getattr(scan, "samples", None) or duration < float(cfg.frozen_min_duration):
        return False
    samples = list(scan.samples)[1:] or list(scan.samples)
    if len(samples) < 3:
        return False
    max_change = max(float(s.change or 0.0) for s in samples)
    mags = [float(s.flow_mag) for s in samples if s.flow_mag is not None]
    still = (max(mags) < 0.05) if mags else True
    return max_change <= float(cfg.frozen_max_change) and still


def scene_quality_flags(m: Measurements | None, video_flags: list[str] | None, cfg: QualityConfig, scan: Any = None, duration: float | None = None) -> list[str]:
    """Measured visual-quality flags of one shot: its own frame statistics (+ its scan: frozen) plus
    the source video's flags (resolution / bitrate / frame rate / aspect). Order is stable so
    records compare cleanly. Each flag maps to one named threshold in ``quality`` config
    (``policy_version`` records which)."""
    flags: list[str] = []
    if m is not None:
        if float(cfg.min_sharpness) > 0 and m.sharpness is not None and m.sharpness < float(cfg.min_sharpness):
            flags.append("blurry")
        if m.dark_fraction is not None and m.dark_fraction > float(cfg.max_dark_fraction):
            flags.append("underexposed")
        if m.bright_fraction is not None and m.bright_fraction > float(cfg.max_bright_fraction):
            flags.append("overexposed")
        if float(cfg.max_blockiness) > 0 and m.blockiness is not None and m.blockiness > float(cfg.max_blockiness):
            flags.append("compression_artifacts")
    if scan is not None and duration is not None and frozen_shot(scan, duration, cfg):
        flags.append("frozen")
    for f in video_flags or []:
        if f in VIDEO_LEVEL_FLAGS and f != "duration_mismatch" and f not in flags:
            flags.append(f)
    return flags

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


_INSTRUMENT_SOURCES = {
    ConfidenceSource.MEASUREMENT: "instrument:measurement",
    ConfidenceSource.OCR_SCORE: "instrument:ocr_score",
    ConfidenceSource.ASR_LOGPROB: "instrument:asr_logprob",
    ConfidenceSource.DETECTOR_SCORE: "instrument:detector_score",
}


def scene_quality(a: SceneAnalysis) -> QualityScore:
    grounding = _verifier_score(a.verification)
    method = "verifier" if grounding is not None else None
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
        grounding_method=method,
    )


def event_quality(e: Event) -> QualityScore:
    grounding = _verifier_score(e.verification)
    method = "verifier" if grounding is not None else None
    if grounding is None and e.confidence is not None and e.confidence_source in _INSTRUMENT_SOURCES:
        # the event text quotes / names what the instrument measured (OCR text, ASR utterance, detected
        # object, flow label): its recognition score is the grounding evidence, and the method says so
        grounding = e.confidence
        method = _INSTRUMENT_SOURCES[ConfidenceSource(e.confidence_source)]
    temporal = e.confidence if e.boundary_precision in ("measured", "asr", "frame") and e.confidence is not None else None
    notes = []
    if e.boundary_precision == "scene":
        notes.append("boundaries inherited from the shot; exact action timing inside the shot unmeasured")
    return QualityScore.from_components(grounding=grounding, temporal_accuracy=temporal, description_quality=description_quality_score(e.event), notes=notes, grounding_method=method)


def relation_quality(conf: float | None, event_qualities: list[QualityScore]) -> QualityScore:
    """A relation's confidence is interval algebra over its events: it says nothing about whether
    the event texts are grounded, so ``grounding`` is the minimum of the events' grounding (or None)."""
    temporals = [q.temporal_accuracy for q in event_qualities if q.temporal_accuracy is not None]
    groundings = [q.grounding for q in event_qualities if q.grounding is not None]
    grounded = bool(groundings) and len(groundings) == len(event_qualities)
    return QualityScore.from_components(
        grounding=round(min(groundings), 3) if grounded else None,
        temporal_accuracy=round(min(temporals), 3) if len(temporals) == len(event_qualities) and temporals else None,
        description_quality=None,
        grounding_method="derived_min" if grounded else None,
    )


def qa_quality(qa: QARecord, grounding: float | None, event_qualities: list[QualityScore], grounding_method: str = "lexical_overlap_v1") -> QualityScore:
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
        grounding_method=grounding_method if grounding is not None else None,
    )
