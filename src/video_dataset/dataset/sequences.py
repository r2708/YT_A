"""Longer coherent motion sequences, assembled from consecutive scene records of one video.

Two shots are joined when
* the second one is an artificial `split` of the first (over-long shots are split for analysis, so
  rejoining them restores the real continuous take), or
* the boundary is a cut (or a fade / dissolve when allowed) and, where both shots know them, the
  setting (indoor / outdoor), the time of day and at least one detected object class agree.

A sequence is emitted when it spans at least two shots and ``min_duration`` seconds. Everything is
computed on exported record dicts, so re-annotation of old shards produces sequences too.
"""

from __future__ import annotations

from typing import Any

from video_dataset.config import SequenceConfig
from video_dataset.schemas.dataset import SequenceRecord
from video_dataset.schemas.quality import ValidationStatus
from video_dataset.utils.ids import record_id

_STATUS_ORDER = {ValidationStatus.ACCEPTED.value: 0, ValidationStatus.REVIEW.value: 1, ValidationStatus.REJECTED.value: 2, ValidationStatus.DUPLICATE.value: 3}


def _env(rec: dict[str, Any], key: str) -> str | None:
    env = rec.get("environment") if isinstance(rec.get("environment"), dict) else {}
    v = env.get(key)
    return str(v).strip().lower() if v and str(v).strip().lower() != "unknown" else None


def _objects(rec: dict[str, Any]) -> set[str]:
    return {str(o).strip().lower() for o in (rec.get("objects") or []) if o}


def _compatible(prev: dict[str, Any], cur: dict[str, Any], cfg: SequenceConfig) -> list[str]:
    """Reasons the two consecutive shots belong together, or [] when they do not."""
    transition = str(cur.get("transition_in") or "cut")
    if transition == "split":
        return ["rejoined_split"]
    if transition in ("fade", "dissolve") and not cfg.allow_fades:
        return []
    if transition == "start":
        return []
    reasons: list[str] = []
    if cfg.require_same_setting:
        for key in ("setting", "time_of_day"):
            a, b = _env(prev, key), _env(cur, key)
            if a and b and a != b:
                return []
            if a and b:
                reasons.append(f"same_{key}")
    if cfg.require_shared_subject:
        a, b = _objects(prev), _objects(cur)
        if a and b:
            if not (a & b):
                return []
            reasons.append("shared_subject")
    return reasons or ["consecutive_cut"]


def _camera_label(rec: dict[str, Any]) -> str:
    cam = rec.get("camera") if isinstance(rec.get("camera"), dict) else {}
    fam = cam.get("camera_movement") or "unknown"
    d = cam.get("movement_direction")
    return f"{fam}_{d}" if d else str(fam)


def _worst_status(shots: list[dict[str, Any]]) -> str:
    statuses = [((s.get("validation") or {}).get("status")) or ValidationStatus.ACCEPTED.value for s in shots]
    return max(statuses, key=lambda st: _STATUS_ORDER.get(str(st), 1))


def _min_quality(shots: list[dict[str, Any]]) -> dict[str, Any] | None:
    overalls = [((s.get("quality") or {}).get("overall")) for s in shots]
    if any(o is None for o in overalls):
        return None
    comps = sorted({c for s in shots for c in ((s.get("quality") or {}).get("components_used") or [])})
    return {"grounding": None, "temporal_accuracy": None, "description_quality": None, "overall": round(min(float(o) for o in overalls), 4), "components_used": comps, "notes": ["minimum over the joined shots"]}


def sequence_prompt(shots: list[dict[str, Any]], subjects: list[str], setting: str | None, time_of_day: str | None, duration: float) -> tuple[str | None, list[str]]:
    """(prompt, per-shot prompts): a framing sentence followed by the shots' own generation prompts."""
    shot_prompts = [str(s.get("generation_prompt")) for s in shots if s.get("generation_prompt")]
    if not shot_prompts:
        return None, []
    head = f"A continuous {duration:.0f}-second sequence of {len(shots)} shots"
    if subjects:
        head += " featuring " + ", ".join(subjects[:3])
    where = []
    if setting in ("indoor", "outdoor"):
        where.append("indoors" if setting == "indoor" else "outdoors")
    if time_of_day:
        where.append(f"at {time_of_day}" if time_of_day not in ("day", "daytime", "night", "nighttime") else ("in daylight" if time_of_day.startswith("day") else "at night"))
    if where:
        head += " " + " ".join(where)
    text = head + ". " + " ".join(f"Shot {i + 1}: {p}" for i, p in enumerate(shot_prompts))
    return text, shot_prompts


def _emit(video_id: str, shots: list[dict[str, Any]], reasons: list[str], clip_ids: dict[str, str], n: int) -> dict[str, Any]:
    start, end = float(shots[0]["start_time"]), float(shots[-1]["end_time"])
    duration = round(end - start, 3)
    subjects = sorted(set.intersection(*[_objects(s) for s in shots])) if all(_objects(s) for s in shots) else []
    settings = {_env(s, "setting") for s in shots} - {None}
    tods = {_env(s, "time_of_day") for s in shots} - {None}
    setting = settings.pop() if len(settings) == 1 else None
    tod = tods.pop() if len(tods) == 1 else None
    prompt, shot_prompts = sequence_prompt(shots, subjects, setting, tod, duration)
    confs = [s.get("confidence") for s in shots]
    conf = round(min(float(c) for c in confs), 3) if confs and all(c is not None for c in confs) else None
    status = _worst_status(shots)
    seq_id = f"seq_{n:03d}"
    rec = {
        "record_id": record_id("sequence", video_id, seq_id),
        "record_type": "sequence",
        "video_id": video_id,
        "sequence_id": seq_id,
        "scene_ids": [str(s["scene_id"]) for s in shots],
        "clip_ids": [clip_ids[str(s["scene_id"])] for s in shots if str(s["scene_id"]) in clip_ids],
        "start_time": start,
        "end_time": end,
        "duration": duration,
        "n_shots": len(shots),
        "coherence": sorted(set(reasons)),
        "subjects": subjects,
        "setting": setting,
        "time_of_day": tod,
        "camera_movements": [_camera_label(s) for s in shots],
        "summary": " ".join(f"Shot {i + 1} ({float(s['start_time']):.1f}-{float(s['end_time']):.1f}s): {s.get('summary') or ''}".strip() for i, s in enumerate(shots)),
        "generation_prompt": prompt,
        "shot_prompts": shot_prompts,
        "prompt_variants": [],
        "provider": shots[0].get("provider"),
        "confidence": conf,
        "confidence_source": "derived_min" if conf is not None else "unavailable",
        "quality": _min_quality(shots),
        "validation": {"status": status, "issues": [] if status == ValidationStatus.ACCEPTED.value else ["contains shots not accepted"], "checks": {"all_shots_accepted": status == ValidationStatus.ACCEPTED.value}},
    }
    return SequenceRecord.model_validate(rec).model_dump(mode="json")  # exactly the exported schema


def build_sequences(records: dict[str, list[dict[str, Any]]], cfg: SequenceConfig) -> list[dict[str, Any]]:
    """Sequence record dicts for every video in ``records["scenes"]``."""
    if not cfg.enabled:
        return []
    by_video: dict[str, list[dict[str, Any]]] = {}
    for rec in records.get("scenes", []):
        if rec.get("start_time") is None or rec.get("end_time") is None:
            continue
        by_video.setdefault(str(rec.get("video_id")), []).append(rec)
    clip_by_scene: dict[tuple[str, str], str] = {}
    for rec in sorted(records.get("clips", []), key=lambda r: (len(str(r.get("clip_id") or "")), str(r.get("clip_id") or "")), reverse=True):
        if rec.get("scene_id") and rec.get("clip_id"):
            clip_by_scene[(str(rec.get("video_id")), str(rec["scene_id"]))] = str(rec["clip_id"])
    out: list[dict[str, Any]] = []
    for vid, scenes in sorted(by_video.items()):
        scenes.sort(key=lambda r: (float(r["start_time"]), str(r.get("scene_id"))))
        clips = {sid: cid for (v, sid), cid in clip_by_scene.items() if v == vid}
        run: list[dict[str, Any]] = []
        reasons: list[str] = []
        n = 0

        def flush() -> None:
            nonlocal run, reasons, n
            if len(run) >= 2 and float(run[-1]["end_time"]) - float(run[0]["start_time"]) >= float(cfg.min_duration):
                n += 1
                out.append(_emit(vid, run, reasons, clips, n))
            run, reasons = [], []

        for rec in scenes:
            if not run:
                run = [rec]
                continue
            prev = run[-1]
            why = _compatible(prev, rec, cfg)
            contiguous = abs(float(rec["start_time"]) - float(prev["end_time"])) <= 0.5
            too_long = float(rec["end_time"]) - float(run[0]["start_time"]) > float(cfg.max_duration) or len(run) >= int(cfg.max_shots)
            if why and contiguous and not too_long:
                run.append(rec)
                reasons.extend(why)
            else:
                flush()
                run = [rec]
        flush()
    return out
