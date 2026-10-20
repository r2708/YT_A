"""Claim-level evidence for a scene annotation.

A caption is only as good as the facts it states. This module decomposes one ``SceneAnalysis``
into atomic claims (subject, action, environment, camera, lighting, temporal progression), records
for each one *where it came from* and *what checked it*, and lists the frames an auditor can open
to see the evidence. No model is consulted here: the statuses come from the measurement / detector
provenance the pipeline already stores and from the deterministic cross-checks of
``validation.consistency``.

Statuses: ``supported`` (a measurement or detector produced the value and nothing contradicts it),
``contradicted`` (a consistency rule found a conflicting measurement), ``not_checked`` (the value
was written by a generative model and no verifier judged it), ``unknown`` (the field is unknown).
"""

from __future__ import annotations

from typing import Any

from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.vision import CameraMovement, SceneAnalysis, Setting

MEASURED_SOURCES = {ConfidenceSource.MEASUREMENT.value, ConfidenceSource.DETECTOR_SCORE.value, ConfidenceSource.VERIFIER.value}


def _frames(a: SceneAnalysis, limit: int = 4) -> list[str]:
    ids = list(a.frame_ids)
    if len(ids) <= limit:
        return ids
    step = (len(ids) - 1) / (limit - 1)
    return [ids[int(round(i * step))] for i in range(limit)]


def _claim(field: str, value: Any, source: str, status: str, evidence: list[str], detail: str | None = None, score: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"field": field, "value": value, "source": source, "status": status, "evidence_frame_ids": evidence}
    if detail:
        out["detail"] = detail
    if score is not None:
        out["score"] = round(float(score), 3)
    return out


def scene_claims(a: SceneAnalysis, contradictions: list[str] | None = None, generative: bool | None = None) -> list[dict[str, Any]]:
    """Atomic claims of one scene annotation with provenance and verification status."""
    generative = (a.provider not in ("heuristic", "cv_models", None)) if generative is None else generative
    contradictions = contradictions or []
    frames = _frames(a)
    m = a.measurements
    claims: list[dict[str, Any]] = []

    def contradicted(*words: str) -> str | None:
        for c in contradictions:
            low = c.lower()
            if all(w in low for w in words):
                return c
        return None

    # ---- camera movement: measured by optical flow or stated by a model -----------------------
    cam = a.camera
    if cam.movement != CameraMovement.UNKNOWN:
        src = str(cam.confidence_source or ConfidenceSource.UNAVAILABLE)
        conflict = contradicted("camera")
        if conflict:
            status = "contradicted"
        elif src in MEASURED_SOURCES:
            status = "supported"
        else:
            status = "not_checked"
        detail = conflict or (f"optical flow: {m.camera_motion_label} ({m.camera_motion_consistency:.2f} of samples)" if m and m.camera_motion_label and m.camera_motion_consistency is not None else None)
        claims.append(_claim("camera.movement", str(cam.movement), "optical_flow" if src == ConfidenceSource.MEASUREMENT.value else src, status, frames, detail, cam.confidence))
    else:
        claims.append(_claim("camera.movement", "unknown", "none", "unknown", frames))
    if cam.shot_type and str(cam.shot_type) != "unknown":
        src = "clip_zero_shot" if "clip" in a.enrichments else ("model" if generative else "unknown")
        claims.append(_claim("camera.shot_type", str(cam.shot_type), src, "supported" if src == "clip_zero_shot" else ("not_checked" if generative else "unknown"), frames))
    # ---- subjects: detector boxes --------------------------------------------------------------
    for o in a.objects[:8]:
        src = str(o.confidence_source or ConfidenceSource.UNAVAILABLE)
        measured = src == ConfidenceSource.DETECTOR_SCORE.value
        detail = f"seen in {o.frame_fraction:.0%} of analysed frames" if o.frame_fraction is not None else None
        claims.append(_claim("subject", o.name, "detector" if measured else ("model" if generative else src), "supported" if measured else "not_checked", frames, detail, o.confidence))
    if a.people and a.people.count:
        src = "detector" if any(o.name.lower() == "person" for o in a.objects) else ("model" if generative else "unknown")
        claims.append(_claim("people.count", a.people.count, src, "supported" if src == "detector" else "not_checked", frames))
    # ---- environment: CLIP zero-shot (cv_models) or model text -----------------------------------
    env = a.environment
    clip = a.enrichments.get("clip") if isinstance(a.enrichments, dict) else None
    for field, value in (("environment.setting", env.setting if env.setting != Setting.UNKNOWN else None), ("environment.time_of_day", env.time_of_day), ("environment.location", env.location), ("environment.weather", env.weather)):
        if not value or str(value).lower() == "unknown":
            continue
        if clip and field.split(".")[1] in str(clip):
            claims.append(_claim(field, str(value), "clip_zero_shot", "supported", frames))
        else:
            claims.append(_claim(field, str(value), "model" if generative else "unknown", "not_checked" if generative else "unknown", frames))
    # ---- lighting: pixel statistics -----------------------------------------------------------
    if m is not None and m.lighting_level:
        conflict = contradicted("lighting")
        claims.append(_claim("lighting", m.lighting_level, "pixel_statistics", "contradicted" if conflict else "supported", frames, conflict or (f"mean brightness {m.brightness_mean:.0f}/255" if m.brightness_mean is not None else None)))
    elif env.lighting:
        claims.append(_claim("lighting", env.lighting, "model" if generative else "unknown", "not_checked", frames))
    # ---- actions: measured subject motion vs model text ----------------------------------------
    sm = a.subject_motion
    for act in a.actions[:6]:
        low = act.lower()
        if sm is not None and sm.label != "static" and sm.subject.lower() in low and ("moves" in low or "tracks" in low or "approach" in low or "shrink" in low or "stays" in low):
            claims.append(_claim("action", act, "tracker" if sm.method == "tracker" else "detector_boxes", "supported", frames, f"relative velocity {sm.relative_velocity}", sm.confidence))
        elif low.startswith("camera "):
            continue  # covered by camera.movement
        else:
            claims.append(_claim("action", act, "model" if generative else "unknown", "not_checked" if generative else "unknown", frames))
    # ---- temporal progression: brightness trend / change peak are measured --------------------
    if a.temporal_progression:
        measured_prog = m is not None and (m.brightness_trend in ("brightening", "darkening") or "largest visual change" in a.temporal_progression)
        claims.append(_claim("temporal_progression", a.temporal_progression[:160], "scan_statistics" if measured_prog and not generative else ("model" if generative else "unknown"), "supported" if measured_prog and not generative else ("not_checked" if generative else "unknown"), frames))
    return claims


def claim_summary(claims: list[dict[str, Any]]) -> dict[str, int]:
    out = {"supported": 0, "contradicted": 0, "not_checked": 0, "unknown": 0}
    for c in claims:
        out[str(c.get("status"))] = out.get(str(c.get("status")), 0) + 1
    return out
