"""Per-task eligibility policy for the task-specific exports (issue doc P3-04, P3-10, §4, §6).

The release holds every record type side by side; a consumer training for one purpose has to know
which of them are fit for it. This module says so declaratively: ``TASKS`` lists, per task, the
record types it draws from and the requirements a record must meet, ``eligible`` applies them to
one record and returns machine-readable reason codes when it does not qualify, and
``select_task_records`` applies them to a whole export and reports the counts per reason.

Nothing here invents a label: every requirement is a test on fields the pipeline already wrote
(``validation.status``, ``subsets``, ``camera.camera_movement``, ``generation_prompt``, evidence
ids) or on facts the orchestrator supplies (whether a clip file is on disk, a video's rights).

Reason codes (stable vocabulary):

* ``status:<status>``                       validation status is not one the task accepts
* ``not_in_subset:<name>``                  the record is not in a required named subset
* ``media:missing``                         the task needs the media file and it does not resolve
* ``rights:<status>``                       the task needs release rights and the video has none
* ``camera:unknown`` / ``camera:ambiguous`` the task needs a resolved camera movement
* ``evidence:unresolved``                   evidence ids do not exist in this export for the video
* ``prompt:missing``                        the task needs ``generation_prompt``
* ``generation_target:<value>``             the record is not a generation target the task wants
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from video_dataset.dataset.curation import STRICT_SUBSET, split_manifest
from video_dataset.schemas.quality import ValidationStatus

TASK_POLICY_VERSION = "tasks-2026.11"
CINEMATIC_SUBSET = "cinematic"
QA_TYPES = ("temporal_qa", "long_video_qa")
_UNKNOWN_CAMERA = (None, "", "unknown")

MediaCheck = Callable[[dict[str, Any]], bool]


def sub_dict(rec: dict[str, Any], key: str) -> dict[str, Any]:
    """``rec[key]`` when it is a dict, else ``{}`` (nested blocks may be null on exported records)."""
    v = rec.get(key)
    return v if isinstance(v, dict) else {}

# Declarative policy. Every key below is read by ``eligible``; ``description`` and ``contains`` /
# ``does_not_contain`` feed ``policy_document``.
TASKS: dict[str, dict[str, Any]] = {
    "understanding": {
        "record_types": ["frames", "clips", "scenes", "events", "video_descriptions"],
        "statuses": [ValidationStatus.ACCEPTED.value],
        "subsets": [],
        "require_media": False,
        "require_rights": False,
        "require_evidence": False,
        "require_camera": False,
        "require_prompt": False,
        "generation_targets": [],
        "description": "Video understanding: captions, structured shot descriptions and event lists for frames, clips, scenes and whole videos.",
        "contains": [
            "measured captions / summaries and the structured fields they were built from (objects, actions, environment, camera)",
            "event records with timestamps and the scene / frame ids they rest on",
            "whole-video descriptions assembled from the per-scene fields",
        ],
        "does_not_contain": [
            "frames or clips as files: records carry relative media paths only; media follows the per-video rights",
            "records that are in review, rejected or marked duplicate",
        ],
    },
    "temporal_reasoning": {
        "record_types": ["temporal_qa", "long_video_qa", "events", "sequences"],
        "statuses": [ValidationStatus.ACCEPTED.value],
        "subsets": [],
        "require_media": False,
        "require_rights": False,
        "require_evidence": True,
        "require_camera": False,
        "require_prompt": False,
        "generation_targets": [],
        "description": "Temporal reasoning: timestamp, ordering, duration and long-range questions with evidence windows, plus the events and sequences they refer to.",
        "contains": [
            "question / answer pairs whose evidence (event, scene and frame ids) resolves inside this export for the same video",
            "rule-built hard negatives attached to the questions",
            "event records and multi-shot sequences with their time spans",
        ],
        "does_not_contain": [
            "questions whose evidence points at records that are not part of this export",
            "media files",
        ],
    },
    "cinematic": {
        "record_types": ["scenes", "clips", "frames"],
        "statuses": [ValidationStatus.ACCEPTED.value],
        "subsets": [CINEMATIC_SUBSET],
        "require_media": False,
        "require_rights": False,
        "require_evidence": False,
        "require_camera": True,
        "require_prompt": False,
        "generation_targets": [],
        "description": "Cinematic shots: scene-level records of the cinematic subset whose camera movement is resolved to one family of the standard taxonomy.",
        "contains": [
            "scene, clip and frame records of the `cinematic` subset (tier, duration, aesthetic and sharpness criteria of the export config)",
            "a standardized camera block: camera_movement family, movement_direction, movement_speed, shot_size",
        ],
        "does_not_contain": [
            "shots whose camera movement is unknown or ambiguous",
            "media files",
        ],
    },
    "generation_ready": {
        "record_types": ["clips", "sequences", "scenes"],
        "statuses": [ValidationStatus.ACCEPTED.value],
        "subsets": [STRICT_SUBSET],
        "require_media": True,
        "require_rights": True,
        "require_evidence": False,
        "require_camera": False,
        "require_prompt": True,
        "generation_targets": ["single_shot", "multi_shot"],
        "description": "Generation-ready pairs: strict-subset clips, sequences and scenes with a template generation prompt, a clip file that is on disk and a source video whose rights allow redistribution.",
        "contains": [
            "single-shot clips and multi-shot sequences with `generation_prompt` (+ `prompt_variants`) built from measured fields only",
            "the prompt -> media pairing in `pairs.jsonl` with the clip path (or the sequence's clip ids), timing, camera and subject-motion fields",
        ],
        "does_not_contain": [
            "any record whose clip file is not on disk: metadata-only records are NOT a substitute for media and are excluded with reason `media:missing`",
            "any video whose rights_status is outside the release rights: excluded with reason `rights:<status>`",
            "records outside the `strict` subset or without a generation prompt",
        ],
    },
}


def task_names() -> list[str]:
    return list(TASKS)


def _task(task: str) -> dict[str, Any]:
    try:
        return TASKS[task]
    except KeyError as exc:
        raise KeyError(f"unknown task {task!r}; known: {', '.join(TASKS)}") from exc


def index_ids(records: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, set[str]]]:
    """``{video_id: {"events": ids, "scenes": ids, "frames": ids, "clips": ids}}`` over the export,
    the reference set against which evidence ids are resolved."""
    out: dict[str, dict[str, set[str]]] = {}
    for kind, key in (("events", "event_id"), ("scenes", "scene_id"), ("frames", "frame_id"), ("clips", "clip_id")):
        for rec in records.get(kind, []):
            vid = str(rec.get("video_id") or "")
            if vid and rec.get(key):
                out.setdefault(vid, {"events": set(), "scenes": set(), "frames": set(), "clips": set()})[kind].add(str(rec[key]))
    return out


def _evidence_refs(kind: str, rec: dict[str, Any]) -> dict[str, list[str]]:
    """Which exported records this record claims as evidence, by reference type."""
    if kind in QA_TYPES:
        ev = sub_dict(rec, "evidence")
        return {"events": list(ev.get("event_ids") or []), "scenes": list(ev.get("scene_ids") or []), "frames": list(ev.get("frame_ids") or [])}
    if kind == "events":
        return {"scenes": list(rec.get("scene_ids") or []), "frames": list(rec.get("frame_ids") or []), "clips": list(rec.get("clip_ids") or [])}
    if kind == "sequences":
        return {"scenes": list(rec.get("scene_ids") or []), "clips": list(rec.get("clip_ids") or [])}
    return {}


def evidence_resolves(kind: str, rec: dict[str, Any], ids_by_video: dict[str, dict[str, set[str]]]) -> bool:
    refs = _evidence_refs(kind, rec)
    if not any(refs.values()):
        return kind not in QA_TYPES  # a QA record without any evidence ids has nothing to resolve to
    known = ids_by_video.get(str(rec.get("video_id") or ""), {})
    return all(str(i) in known.get(ref_kind, set()) for ref_kind, ids in refs.items() for i in ids)


def eligible(
    task: str,
    kind: str,
    rec: dict[str, Any],
    *,
    media_available: MediaCheck | None = None,
    rights: dict[str, str] | None = None,
    release_rights: Iterable[str] = (),
    ids_by_video: dict[str, dict[str, set[str]]] | None = None,
) -> tuple[bool, list[str]]:
    """Whether ``rec`` (of export type ``kind``, e.g. ``"clips"``) qualifies for ``task``, and the
    reason codes when it does not. ``media_available`` answers "is this record's media file on
    disk"; ``rights`` maps video id -> rights_status; ``ids_by_video`` is ``index_ids(records)``."""
    spec = _task(task)
    why: list[str] = []
    if kind not in spec["record_types"]:
        return False, [f"record_type:{kind}"]
    status = sub_dict(rec, "validation").get("status")
    if str(status or "missing") not in spec["statuses"]:
        why.append(f"status:{status or 'missing'}")
    subsets = {str(s) for s in (rec.get("subsets") or [])}
    for name in spec["subsets"]:
        if name not in subsets:
            why.append(f"not_in_subset:{name}")
    if spec["generation_targets"]:
        target = rec.get("generation_target")
        if target not in spec["generation_targets"]:
            why.append(f"generation_target:{target or 'missing'}")
    if spec["require_prompt"] and not str(rec.get("generation_prompt") or "").strip():
        why.append("prompt:missing")
    if spec["require_camera"]:
        cam = sub_dict(rec, "camera")
        if cam.get("camera_movement") in _UNKNOWN_CAMERA:
            why.append("camera:unknown")
        elif cam.get("movement_ambiguous") or cam.get("ambiguous"):
            why.append("camera:ambiguous")
    if spec["require_evidence"] and not evidence_resolves(kind, rec, ids_by_video or {}):
        why.append("evidence:unresolved")
    if spec["require_rights"]:
        status_r = str((rights or {}).get(str(rec.get("video_id") or ""), "unknown") or "unknown")
        if status_r not in {str(r) for r in release_rights}:
            why.append(f"rights:{status_r}")
    if spec["require_media"] and (media_available is None or not media_available(rec)):
        why.append("media:missing")
    return not why, why


def select_task_records(
    task: str,
    records: dict[str, list[dict[str, Any]]],
    *,
    media_available: MediaCheck | None = None,
    rights: dict[str, str] | None = None,
    release_rights: Iterable[str] = (),
    ids_by_video: dict[str, dict[str, set[str]]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """The eligible records of ``task`` per record type, plus a ``filters`` report (requirements,
    counts per type, exclusions by reason and by status, videos per split)."""
    spec = _task(task)
    release = sorted({str(r) for r in release_rights})
    index = ids_by_video if ids_by_video is not None else index_ids(records)
    selected: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}
    by_reason: dict[str, dict[str, int]] = {}
    by_status: dict[str, dict[str, int]] = {}
    for kind in spec["record_types"]:
        rows = records.get(kind, [])
        kept: list[dict[str, Any]] = []
        reasons: dict[str, int] = {}
        statuses: dict[str, int] = {}
        for rec in rows:
            ok, why = eligible(task, kind, rec, media_available=media_available, rights=rights, release_rights=release, ids_by_video=index)
            if ok:
                kept.append(rec)
                continue
            for code in why:
                reasons[code] = reasons.get(code, 0) + 1
            st = str(sub_dict(rec, "validation").get("status") or "missing")
            statuses[st] = statuses.get(st, 0) + 1
        selected[kind] = kept
        counts[kind] = {"input": len(rows), "eligible": len(kept), "excluded": len(rows) - len(kept)}
        by_reason[kind] = dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
        by_status[kind] = dict(sorted(statuses.items()))
    filters = {
        "task": task,
        "policy_version": TASK_POLICY_VERSION,
        "record_types": list(spec["record_types"]),
        "requirements": {
            "statuses": list(spec["statuses"]),
            "subsets": list(spec["subsets"]),
            "require_media": bool(spec["require_media"]),
            "require_rights": bool(spec["require_rights"]),
            "release_rights": release if spec["require_rights"] else [],
            "require_evidence": bool(spec["require_evidence"]),
            "require_camera": bool(spec["require_camera"]),
            "require_prompt": bool(spec["require_prompt"]),
            "generation_targets": list(spec["generation_targets"]),
        },
        "counts": counts,
        "excluded_by_reason": by_reason,
        "excluded_by_status": by_status,
        "splits": {split: len(vids) for split, vids in split_manifest(selected).items()},
    }
    return selected, filters


def policy_document(task: str) -> str:
    """Markdown description of the task policy: record types, required status / checks, media and
    rights requirements, what the records contain and what they do not."""
    spec = _task(task)
    lines = [f"# Task export: `{task}`", "", f"Policy version `{TASK_POLICY_VERSION}`.", "", spec["description"], ""]
    lines += ["## Eligible record types", ""] + [f"- `{t}`" for t in spec["record_types"]] + [""]
    lines += ["## Requirements", ""]
    lines.append(f"- validation status in: {', '.join(f'`{s}`' for s in spec['statuses'])} (the status the acceptance policy assigned; every required check of the record type passed, see `validation.checks` and `validation.policy_version`)")
    if spec["subsets"]:
        lines.append(f"- member of subset(s): {', '.join(f'`{s}`' for s in spec['subsets'])}")
    if spec["generation_targets"]:
        lines.append(f"- `generation_target` in: {', '.join(f'`{g}`' for g in spec['generation_targets'])}")
    if spec["require_prompt"]:
        lines.append("- `generation_prompt` present (assembled from the structured fields, no free-form claims)")
    if spec["require_camera"]:
        lines.append("- `camera.camera_movement` resolved to a taxonomy family (not `unknown`) and not flagged ambiguous")
    if spec["require_evidence"]:
        lines.append("- every evidence id (events, scenes, frames) resolves to a record of the same video inside this export")
    lines.append(f"- media required: {'yes - the clip file must be on disk (checked against the media manifest)' if spec['require_media'] else 'no - records are metadata; media paths are relative and may not be present'}")
    lines.append(f"- rights required: {'yes - the source video rights_status must be in the release rights' if spec['require_rights'] else 'no - the source video keeps its own licence (see provenance.jsonl)'}")
    lines += ["", "## What the records contain", ""] + [f"- {c}" for c in spec["contains"]]
    lines += ["", "## What they do not contain", ""] + [f"- {c}" for c in spec["does_not_contain"]]
    if task == "generation_ready":
        lines += [
            "",
            "## Media policy",
            "",
            "A generation-ready pair is a prompt *and* the media it describes. A record whose clip file is not on disk "
            "is metadata only: it is not a substitute for the media and is excluded here (reason `media:missing`), "
            "whatever its validation status. When no media is on disk this export is empty; `filters.json` says why.",
        ]
    lines += ["", "Exclusions are counted per reason code in `filters.json`.", ""]
    return "\n".join(lines)
