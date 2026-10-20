"""Transcript views for a time window (clip / scene) in both absolute and window-relative time.

The pipeline keeps one canonical transcript per video in source-video coordinates. A record that
covers ``[start, end)`` gets the segments overlapping that window, clipped to it: with word
timestamps the text is rebuilt from the words inside the window (``word_clipped``); without them
the whole overlapping segment is kept and flagged ``partial`` when it extends past an edge
(``segment_overlap``). Both absolute (``start`` / ``end``) and window-relative (``rel_start`` /
``rel_end``) times are stored under distinct names so they are never mixed.

Speaker identity is never inferred: ``speaker_visible`` stays ``None`` (unknown) because nothing
in the pipeline synchronises lips with audio.
"""

from __future__ import annotations

from typing import Any

from video_dataset.schemas.transcript import Transcript

SCOPES = ("word_clipped", "segment_overlap", "none", "not_run")


def transcript_scope(transcript: Transcript | None) -> str:
    if transcript is None:
        return "not_run"
    if getattr(transcript, "status", "transcribed") in ("not_run", "no_audio", "failed"):
        return "not_run"
    return "none"


def segments_in_window(transcript: Transcript | None, start: float, end: float, pad: float = 0.0) -> tuple[list[dict[str, Any]], str]:
    """(segment views, scope) for ``[start, end)`` (+ ``pad`` seconds of context on both sides)."""
    scope = transcript_scope(transcript)
    if scope == "not_run" or transcript is None:
        return [], scope
    lo, hi = start - pad, end + pad
    out: list[dict[str, Any]] = []
    all_words = True
    for seg in transcript.segments:
        if seg.end <= lo or seg.start >= hi:
            continue
        words = [w for w in seg.words if w.end > lo and w.start < hi] if seg.words else []
        if seg.words:
            text = " ".join(w.word.strip() for w in words).strip()
            s_abs = min(w.start for w in words) if words else max(seg.start, lo)
            e_abs = max(w.end for w in words) if words else min(seg.end, hi)
            partial = len(words) != len(seg.words)
            if not text:
                continue
            clipped = True
        else:
            all_words = False
            text = seg.text.strip()
            s_abs, e_abs = max(seg.start, lo), min(seg.end, hi)
            partial = seg.start < lo - 1e-3 or seg.end > hi + 1e-3
            clipped = False
        out.append({
            "segment_id": seg.segment_id,
            "start": round(s_abs, 3), "end": round(e_abs, 3),  # absolute source-video time
            "rel_start": round(max(0.0, s_abs - start), 3), "rel_end": round(max(0.0, e_abs - start), 3),  # relative to the window start
            "text": text,
            "confidence": seg.confidence,
            "confidence_source": str(seg.confidence_source),
            "partial": bool(partial),  # the segment extends beyond the window (text not word-clipped) or words were dropped
            "word_clipped": bool(clipped),
            "language": transcript.language,
            "speaker_visible": None,  # unknown: no audio-visual speaker attribution is performed
        })
    if not out:
        return [], "none"
    return out, ("word_clipped" if all_words else "segment_overlap")


def window_text(segments: list[dict[str, Any]]) -> str | None:
    text = " ".join(str(s["text"]) for s in segments if s.get("text")).strip()
    return text or None
