"""Per-video modality coverage: which optional stages really ran.

A missing transcript is not "no speech" and an empty OCR file is not "no text". The coverage
block records, per modality, ``ran`` / ``not_run`` / ``failed`` / ``disabled`` / ``no_audio`` from
the stage checkpoints and the artifacts, and is copied onto every exported record of the video so
subsets can be selected by what was measured, not by what happens to be empty.
"""

from __future__ import annotations

from typing import Any

from video_dataset.stages import Stage, StageStatus

MODALITIES = ("transcription", "ocr", "audio_events", "vision", "tracking", "verifier", "clip_alignment", "text_checks", "clip_boundaries")


def _stage_state(ctx: Any, stage: Stage) -> str:
    rec = ctx.db.get_stage(ctx.video_id, stage)
    if rec is None:
        return "not_run"
    if rec.status == StageStatus.DONE:
        return "ran"
    if rec.status == StageStatus.SKIPPED:
        return "disabled"
    if rec.status == StageStatus.FAILED:
        return "failed"
    return "not_run"


def stage_coverage(ctx: Any, transcript: Any, ocr: Any, validated: Any) -> dict[str, str]:
    cfg = ctx.config
    cov: dict[str, str] = {}
    # transcription: the artifact knows why it is empty
    t_state = _stage_state(ctx, Stage.TRANSCRIPTION)
    if transcript is not None and getattr(transcript, "status", None) in ("no_audio", "not_run", "failed"):
        cov["transcription"] = {"no_audio": "no_audio", "not_run": "disabled", "failed": "failed"}[transcript.status]
    elif transcript is not None and getattr(transcript, "provider", "none") == "none":
        cov["transcription"] = "disabled" if not cfg.transcription.enabled or cfg.transcription.provider == "none" else "no_audio"
    else:
        cov["transcription"] = t_state
    o_state = _stage_state(ctx, Stage.OCR)
    cov["ocr"] = "disabled" if (o_state == "disabled" or not cfg.ocr.enabled or cfg.ocr.provider == "none") else ("ran" if ocr is not None and o_state == "ran" else o_state)
    a_state = _stage_state(ctx, Stage.AUDIO)
    try:
        has_audio = bool(ctx.media_info.has_audio)
    except FileNotFoundError:
        has_audio = True
    cov["audio_events"] = "no_audio" if not has_audio else ("disabled" if not cfg.audio_events.enabled or cfg.audio_events.provider == "none" else a_state)
    v_state = _stage_state(ctx, Stage.VISION_ANALYSIS)
    cov["vision"] = f"{v_state}:{cfg.vision.provider}" if v_state == "ran" else v_state
    cov["tracking"] = "ran" if (v_state == "ran" and cfg.vision.provider == "cv_models" and getattr(cfg.vision, "tracking", False)) else "disabled"
    summary = getattr(validated, "summary", {}) or {}
    cov["verifier"] = "disabled" if (cfg.validation.verifier or "none") == "none" else f"ran:{summary.get('verifier') or cfg.validation.verifier}"
    align = str(summary.get("caption_alignment") or "")
    cov["clip_alignment"] = "ran" if align == "ran" else ("disabled" if align == "disabled" else "not_run")
    cov["text_checks"] = "ran" if summary.get("text_checker") not in (None, "disabled") else "disabled"
    mode = str(getattr(cfg.frame_sampling, "verify_clip_boundaries", "copy") or "copy").lower()
    cov["clip_boundaries"] = "disabled" if mode in ("none", "off", "false", "0") else f"ran:{mode}"
    return cov


def coverage_rates(manifests: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """{modality: {state: videos}} over per-video manifests."""
    out: dict[str, dict[str, int]] = {m: {} for m in MODALITIES}
    for m in manifests:
        cov = m.get("coverage") if isinstance(m, dict) else None
        if not isinstance(cov, dict):
            continue
        for key, state in cov.items():
            st = str(state).split(":")[0]
            out.setdefault(key, {})[st] = out.setdefault(key, {}).get(st, 0) + 1
    return out
