"""Task-specific export directories under ``final/exports/<task>/`` (issue doc §6 A-D, P3-06, P3-10).

Each task directory holds the eligible records per type (JSONL + typed Parquet), ``filters.json``
(the policy that selected them and every exclusion counted by reason), ``README.md`` (the policy
document plus a counts table), ``manifest.json`` (counts, versions, file list) and, for
``generation_ready``, ``pairs.jsonl``: one prompt -> media row per eligible clip or sequence.

An empty task is still written: when no media is on disk ``generation_ready`` has zero rows and its
``filters.json`` / ``README.md`` say ``media:missing`` for every candidate. That is the honest result,
not an error.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from video_dataset.dataset.arrow import write_typed_parquet
from video_dataset.dataset.export import RECORD_MODELS
from video_dataset.dataset.tasks import (
    TASK_POLICY_VERSION,
    TASKS,
    MediaCheck,
    policy_document,
    select_task_records,
    sub_dict,
)
from video_dataset.schemas.common import SCHEMA_VERSION, utc_now_iso
from video_dataset.utils.io import write_json_atomic, write_jsonl
from video_dataset.utils.logging import get_logger

log = get_logger("dataset.task_exports")

EXPORTS_DIR = "exports"
PAIRS_FILE = "pairs.jsonl"
FILTERS_FILE = "filters.json"
README_FILE = "README.md"
MANIFEST_FILE = "manifest.json"
_CAMERA_KEYS = ("camera_movement", "movement_direction", "movement_speed", "shot_size")


def _release_rights(config: Any, release_rights: Iterable[str] | None) -> list[str]:
    if release_rights is not None:
        return sorted({str(r) for r in release_rights})
    lic = getattr(config, "licensing", None)
    return sorted({str(r) for r in (getattr(lic, "release_rights", None) or [])})


def _camera(rec: dict[str, Any]) -> dict[str, Any] | None:
    cam = rec.get("camera")
    if not isinstance(cam, dict):
        return None
    return {k: cam.get(k) for k in _CAMERA_KEYS}


def _subject_motion(rec: dict[str, Any]) -> dict[str, Any] | None:
    sm = rec.get("subject_motion")
    if not isinstance(sm, dict):
        return None
    return {"label": sm.get("label"), "direction": sm.get("direction"), "camera_relation": sm.get("camera_relation")}


def pair_row(kind: str, rec: dict[str, Any]) -> dict[str, Any]:
    """One generation pair (prompt -> media) for an eligible clip (``kind == "clips"``) or sequence."""
    v = sub_dict(rec, "validation")
    is_seq = kind == "sequences"
    start, end = rec.get("start_time"), rec.get("end_time")
    duration = rec.get("duration")
    if duration is None and start is not None and end is not None:
        duration = round(float(end) - float(start), 3)
    return {
        "record_id": rec.get("record_id"),
        "record_type": rec.get("record_type") or ("sequence" if is_seq else "clip"),
        "video_id": rec.get("video_id"),
        "prompt": rec.get("generation_prompt"),
        "prompt_variants": list(rec.get("prompt_variants") or []),
        "target": list(rec.get("clip_ids") or []) if is_seq else rec.get("clip_path"),
        "start_time": start,
        "end_time": end,
        "duration": duration,
        "fps": rec.get("fps"),
        "width": rec.get("width"),
        "height": rec.get("height"),
        "camera": _camera(rec),
        "shot_camera_movements": list(rec.get("camera_movements") or []) if is_seq else None,
        "subject_motion": _subject_motion(rec),
        "continuity": rec.get("continuity"),
        "split": rec.get("split"),
        "prompt_verification": "human" if v.get("semantic_verification") == "human" else "template",
        "task": "text_to_video",
    }


def _readme(task: str, filters: dict[str, Any]) -> str:
    counts = filters["counts"]
    lines = [policy_document(task), "## Counts", "", "| record type | input | eligible | excluded |", "|---|---:|---:|---:|"]
    for kind, c in counts.items():
        lines.append(f"| `{kind}` | {c['input']} | {c['eligible']} | {c['excluded']} |")
    total = sum(c["eligible"] for c in counts.values())
    lines += ["", f"Eligible records: **{total}**. Videos per split: " + ", ".join(f"{k} {v}" for k, v in filters["splits"].items()) + ".", ""]
    reasons = {kind: r for kind, r in filters["excluded_by_reason"].items() if r}
    if reasons:
        lines += ["## Exclusions by reason", "", "| record type | reason | records |", "|---|---|---:|"]
        for kind, r in reasons.items():
            for code, n in r.items():
                lines.append(f"| `{kind}` | `{code}` | {n} |")
        lines.append("")
    if total == 0:
        lines += ["This export is empty: no record met every requirement above. The reasons are counted in `filters.json`.", ""]
    lines.append(f"Generated {filters.get('generated_at', '')} with policy `{TASK_POLICY_VERSION}`, schema `{SCHEMA_VERSION}`.")
    return "\n".join(lines) + "\n"


def write_task_exports(
    records: dict[str, list[dict[str, Any]]],
    final_dir: Path,
    config: Any = None,
    *,
    media_available: MediaCheck | None = None,
    rights: dict[str, str] | None = None,
    release_rights: Iterable[str] | None = None,
    write_parquet: bool = True,
    tasks: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Write ``final_dir/exports/<task>/`` for every task (or the given ones). ``media_available``
    is built by the orchestrator from the media manifest; ``rights`` maps video id -> rights_status;
    ``release_rights`` defaults to ``config.licensing.release_rights``. Returns per task
    ``{"counts": {type: {input, eligible, excluded}}, "files": [relative paths], "eligible_total": n}``."""
    final_dir = Path(final_dir)
    release = _release_rights(config, release_rights)
    out: dict[str, dict[str, Any]] = {}
    for task in tasks if tasks is not None else list(TASKS):
        task_dir = final_dir / EXPORTS_DIR / task
        task_dir.mkdir(parents=True, exist_ok=True)
        selected, filters = select_task_records(task, records, media_available=media_available, rights=rights, release_rights=release)
        filters["generated_at"] = utc_now_iso()
        files: list[str] = []
        for kind, rows in selected.items():
            write_jsonl(task_dir / f"{kind}.jsonl", rows)
            files.append(f"{kind}.jsonl")
            if write_parquet:
                write_typed_parquet(RECORD_MODELS[kind], rows, task_dir / f"{kind}.parquet")
                files.append(f"{kind}.parquet")
        if task == "generation_ready":
            pairs = [pair_row(kind, r) for kind in ("clips", "sequences") for r in selected.get(kind, [])]
            write_jsonl(task_dir / PAIRS_FILE, pairs)
            files.append(PAIRS_FILE)
            filters["pairs"] = len(pairs)
        total = sum(c["eligible"] for c in filters["counts"].values())
        write_json_atomic(task_dir / FILTERS_FILE, filters)
        (task_dir / README_FILE).write_text(_readme(task, filters), encoding="utf-8")
        files += [FILTERS_FILE, README_FILE]
        manifest = {
            "task": task,
            "policy_version": TASK_POLICY_VERSION,
            "schema_version": SCHEMA_VERSION,
            "generated_at": filters["generated_at"],
            "counts": filters["counts"],
            "eligible_total": total,
            "splits": filters["splits"],
            "files": sorted(files + [MANIFEST_FILE]),
        }
        write_json_atomic(task_dir / MANIFEST_FILE, manifest)
        out[task] = {"counts": filters["counts"], "files": manifest["files"], "eligible_total": total}
        log.info("task export %s: %d eligible records -> %s", task, total, task_dir)
    return out
