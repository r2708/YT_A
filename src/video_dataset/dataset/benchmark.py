"""A frozen held-out benchmark built from the export (issue doc P3-05).

Items are the records of one split (``test`` by default) that are ``accepted`` and, unless
``require_human`` is off, carry a human label (``validation.semantic_verification == "human"`` or
the ``human:accepted`` reason code). Four tasks are derived from them without inventing anything:
captioning (reference = the record's own prompt / summary), temporal QA (question, answer,
evidence, hard negatives), camera motion (the standardized camera family / direction / shot size)
and cinematic attributes (camera + lighting level + setting of the cinematic subset).

``manifest.json`` carries ``frozen_hash``: a sha256 over the sorted item ids, so two builds can be
compared and a changed item set is visible. With ``require_human`` and no human labels in the
split the benchmark is empty and the manifest says so; it never falls back to unreviewed items.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from video_dataset.dataset.tasks import CINEMATIC_SUBSET, sub_dict
from video_dataset.schemas.common import SCHEMA_VERSION, utc_now_iso
from video_dataset.schemas.quality import ValidationStatus
from video_dataset.utils.ids import stable_hash
from video_dataset.utils.io import write_json_atomic, write_jsonl
from video_dataset.validation.policy import HUMAN_ACCEPT

BENCHMARK_DIR = "benchmark"
BENCHMARK_VERSION = "bench-2026.11"
BENCHMARK_TASKS = ("captioning", "temporal_qa", "camera_motion", "cinematic")
BOUNDARY_TOLERANCE_SECONDS = 0.5
_UNKNOWN = (None, "", "unknown")

METRICS: dict[str, dict[str, Any]] = {
    "captioning": {
        "inputs": "the clip file (`clip_path`) or the scene's frames (`frame_ids`)",
        "metrics": {
            "BLEU-4": "corpus BLEU with 4-grams of the system caption against `reference`",
            "CIDEr": "consensus TF-IDF n-gram similarity of the system caption against `reference`",
            "claim_precision": "share of the system caption's atomic claims (subject, camera movement, setting, action) that match a `claims` entry of the item; claims the item does not carry are neither right nor wrong",
        },
    },
    "temporal_qa": {
        "inputs": "`question` (+ `options` for ordering questions) and the source video / its records",
        "metrics": {
            "exact_match": "normalized string equality of the predicted answer with `answer` (lower-case, stripped punctuation)",
            "evidence_iou": f"temporal IoU between the predicted window and `evidence` [start_time, end_time]; a hit is IoU >= 0.5 (boundary tolerance {BOUNDARY_TOLERANCE_SECONDS} s)",
            "numeric_tolerance": f"for time / duration answers: |predicted - answer| <= {BOUNDARY_TOLERANCE_SECONDS} s",
        },
    },
    "camera_motion": {
        "inputs": "the clip file or the scene's frames",
        "metrics": {
            "macro_f1_family": "macro F1 over the camera movement families (`camera_movement`)",
            "accuracy_family": "share of items whose predicted family equals `camera_movement`",
            "accuracy_direction": "share of items with a direction whose predicted direction equals `movement_direction`",
        },
    },
    "cinematic": {
        "inputs": "the scene's frames / clip",
        "metrics": {
            "accuracy_camera_movement": "per-attribute accuracy against `camera.camera_movement`",
            "accuracy_shot_size": "per-attribute accuracy against `camera.shot_size`",
            "accuracy_lighting_level": "per-attribute accuracy against `lighting_level` (dark | dim | normal | bright)",
            "accuracy_setting": "per-attribute accuracy against `setting` (indoor | outdoor | mixed)",
        },
    },
}

GUIDELINES = f"""# Benchmark annotation guidelines ({BENCHMARK_VERSION})

## What a human reviewer accepts

A record is accepted when every claim in its text is visible in the clip or frames it refers to:
the named objects are present, the described action happens inside the window, the camera movement
matches the taxonomy family below, and the timestamps bound the content (first and last frame
located in the source within {BOUNDARY_TOLERANCE_SECONDS} s). A record with one wrong claim is
rejected, not edited: the benchmark holds only records whose text was accepted as written.
Reviewers label records in the review queue; the pipeline writes `validation.semantic_verification
= "human"` and the reason code `{HUMAN_ACCEPT}`.

## Camera taxonomy families

`static`, `pan` (left | right), `tilt` (up | down), `dolly` (in | out), `tracking` (left | right |
forward | backward), `orbit` (clockwise | counterclockwise), `crane` (up | down), `zoom` (in | out),
`handheld`, `fpv`, `drone`, `complex`. `dolly` and `zoom` look alike in optical flow; a reviewer
labels `dolly` only when parallax shows the camera body moving. A shot with more than one movement
is `complex`; a shot whose movement cannot be told is left `unknown` and is not a benchmark item.

## Boundaries

Evidence windows and clip boundaries are compared with a tolerance of {BOUNDARY_TOLERANCE_SECONDS} s.
Temporal QA answers that are times or durations are correct within the same tolerance.

## Cinematic attributes

`lighting_level` is the measured level (dark | dim | normal | bright) of the shot's frames;
`setting` is indoor | outdoor | mixed. Both are compared exactly.
"""


def is_human_verified(rec: dict[str, Any]) -> bool:
    v = sub_dict(rec, "validation")
    return v.get("semantic_verification") == "human" or HUMAN_ACCEPT in (v.get("reason_codes") or [])


def is_held_out(rec: dict[str, Any], split: str, require_human: bool) -> bool:
    v = sub_dict(rec, "validation")
    if str(rec.get("split") or "") != split or v.get("status") != ValidationStatus.ACCEPTED.value:
        return False
    return is_human_verified(rec) if require_human else True


def _item_id(task: str, record_id: Any) -> str:
    return stable_hash(task, record_id, length=16)


def _camera(rec: dict[str, Any]) -> dict[str, Any]:
    cam = sub_dict(rec, "camera")
    return {"camera_movement": cam.get("camera_movement"), "movement_direction": cam.get("movement_direction"), "shot_size": cam.get("shot_size")}


def _base(task: str, rec: dict[str, Any]) -> dict[str, Any]:
    return {"item_id": _item_id(task, rec.get("record_id")), "record_id": rec.get("record_id"), "record_type": rec.get("record_type"), "video_id": rec.get("video_id"), "split": rec.get("split")}


def _captioning(records: dict[str, list[dict[str, Any]]], held: Any) -> list[dict[str, Any]]:
    items = []
    for kind in ("clips", "scenes"):
        for rec in records.get(kind, []):
            if not held(rec):
                continue
            ref = rec.get("generation_prompt") or rec.get("summary") or rec.get("description")
            if not ref:
                continue
            items.append({
                **_base("captioning", rec),
                "clip_path": rec.get("clip_path") if kind == "clips" else None,
                "frame_ids": list(rec.get("frame_ids") or []),
                "reference": ref,
                "reference_source": "generation_prompt" if rec.get("generation_prompt") else ("summary" if rec.get("summary") else "description"),
                "claims": [c for c in (rec.get("claims") or []) if isinstance(c, dict)],
                "start_time": rec.get("start_time"),
                "end_time": rec.get("end_time"),
            })
    return items


def _temporal_qa(records: dict[str, list[dict[str, Any]]], held: Any) -> list[dict[str, Any]]:
    items = []
    for kind in ("temporal_qa", "long_video_qa"):
        for rec in records.get(kind, []):
            if not held(rec):
                continue
            items.append({
                **_base("temporal_qa", rec),
                "record_id": rec.get("record_id") or rec.get("question_id"),
                "item_id": _item_id("temporal_qa", rec.get("record_id") or rec.get("question_id")),
                "qa_type": rec.get("type"),
                "answer_type": rec.get("answer_type"),
                "difficulty": rec.get("difficulty"),
                "question": rec.get("question"),
                "options": list(rec.get("options") or []),
                "answer": rec.get("answer"),
                "evidence": rec.get("evidence"),
                "hard_negatives": [str(h.get("text")) for h in (rec.get("hard_negatives") or []) if isinstance(h, dict) and h.get("text")],
            })
    return items


def _camera_motion(records: dict[str, list[dict[str, Any]]], held: Any) -> tuple[list[dict[str, Any]], int]:
    items, skipped = [], 0
    for kind in ("scenes", "clips"):
        for rec in records.get(kind, []):
            if not held(rec):
                continue
            cam = _camera(rec)
            if cam["camera_movement"] in _UNKNOWN:
                skipped += 1
                continue
            items.append({**_base("camera_motion", rec), "clip_path": rec.get("clip_path") if kind == "clips" else None, "frame_ids": list(rec.get("frame_ids") or []), "start_time": rec.get("start_time"), "end_time": rec.get("end_time"), **cam})
    return items, skipped


def _cinematic(records: dict[str, list[dict[str, Any]]], held: Any) -> list[dict[str, Any]]:
    items = []
    for rec in records.get("scenes", []):
        if not held(rec) or CINEMATIC_SUBSET not in (rec.get("subsets") or []):
            continue
        env = sub_dict(rec, "environment")
        m = sub_dict(rec, "measurements")
        setting = env.get("setting")
        items.append({**_base("cinematic", rec), "frame_ids": list(rec.get("frame_ids") or []), "clip_id": rec.get("clip_id"), "start_time": rec.get("start_time"), "end_time": rec.get("end_time"), "camera": _camera(rec), "lighting_level": m.get("lighting_level"), "setting": None if setting in _UNKNOWN else setting})
    return items


def frozen_hash(items: Iterable[dict[str, Any]]) -> str:
    ids = sorted(str(i["item_id"]) for i in items)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def build_benchmark(
    records: dict[str, list[dict[str, Any]]],
    final_dir: Path,
    *,
    split: str = "test",
    require_human: bool = True,
    tasks: Iterable[str] = BENCHMARK_TASKS,
) -> dict[str, Any]:
    """Write ``final_dir/benchmark/`` (``<task>.jsonl``, ``metrics.json``, ``guidelines.md``,
    ``manifest.json``) and return the manifest (counts per task, frozen hash, note)."""
    out_dir = Path(final_dir) / BENCHMARK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [t for t in tasks if t in BENCHMARK_TASKS]

    def held(rec: dict[str, Any]) -> bool:
        return is_held_out(rec, split, require_human)

    per_task: dict[str, list[dict[str, Any]]] = {}
    skipped_unknown = 0
    for task in wanted:
        if task == "captioning":
            per_task[task] = _captioning(records, held)
        elif task == "temporal_qa":
            per_task[task] = _temporal_qa(records, held)
        elif task == "camera_motion":
            per_task[task], skipped_unknown = _camera_motion(records, held)
        elif task == "cinematic":
            per_task[task] = _cinematic(records, held)
    files = []
    for task, items in per_task.items():
        write_jsonl(out_dir / f"{task}.jsonl", items)
        files.append(f"{task}.jsonl")
    write_json_atomic(out_dir / "metrics.json", {"version": BENCHMARK_VERSION, "boundary_tolerance_seconds": BOUNDARY_TOLERANCE_SECONDS, "tasks": {t: METRICS[t] for t in per_task}})
    (out_dir / "guidelines.md").write_text(GUIDELINES, encoding="utf-8")
    all_items = [i for items in per_task.values() for i in items]
    n_split = sum(1 for rows in records.values() for r in rows if str(r.get("split") or "") == split)
    n_human = sum(1 for rows in records.values() for r in rows if str(r.get("split") or "") == split and is_human_verified(r))
    note = None
    if not all_items:
        if require_human and n_human == 0:
            note = f"empty: no human-verified accepted records in split '{split}' ({n_split} records in the split); the benchmark requires human labels and does not fall back to unreviewed records"
        else:
            note = f"empty: no accepted records in split '{split}' qualify for the requested tasks"
    manifest = {
        "version": BENCHMARK_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "split": split,
        "require_human": require_human,
        "counts": {t: len(items) for t, items in per_task.items()},
        "total": len(all_items),
        "records_in_split": n_split,
        "human_verified_in_split": n_human,
        "camera_motion_skipped_unknown": skipped_unknown,
        "videos": sorted({str(i.get("video_id")) for i in all_items}),
        "frozen_hash": frozen_hash(all_items),
        "files": sorted(files + ["metrics.json", "guidelines.md", "manifest.json"]),
        "note": note,
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest
