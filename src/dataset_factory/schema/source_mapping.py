"""Source row -> canonical record, through configurable aliases. Fields that are absent stay
null; fields with no canonical home are preserved in ``source_payload``."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from dataset_factory.config import FactoryConfig
from dataset_factory.schema.canonical import ActionSpan, CanonicalRecord

_JSON_STRING_FIELDS = ("camera", "validation", "quality", "coverage", "claims", "transcript_segments", "subject_motion", "actions", "prompt_variants", "quality_flags", "frame_ids", "tier_reasons", "subsets", "hard_negatives")


def _maybe_json(value: Any) -> Any:
    """Typed Parquet stores free-form dicts / lists as JSON strings; JSONL keeps them native."""
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for k in _JSON_STRING_FIELDS:
        if k in out:
            out[k] = _maybe_json(out[k])
    return out


def source_record_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def first_present(row: dict[str, Any], names: list[str]) -> tuple[str | None, Any]:
    for n in names:
        if n in row and row[n] not in (None, "", [], {}):
            return n, row[n]
    return None, None


def _actions(value: Any) -> list[ActionSpan]:
    out: list[ActionSpan] = []
    for a in value or []:
        if isinstance(a, str):
            out.append(ActionSpan(name=a))
        elif isinstance(a, dict):
            name = a.get("name") or a.get("action") or a.get("description")
            if not name:
                continue
            out.append(ActionSpan(name=str(name), start_time_seconds=a.get("start_time_seconds", a.get("start_time")), end_time_seconds=a.get("end_time_seconds", a.get("end_time")), source=a.get("source")))
    return out


def map_source_row(row: dict[str, Any], cfg: FactoryConfig, *, source_dataset: str, source_config: str, video_meta: dict[str, Any] | None = None) -> CanonicalRecord:
    """``video_meta`` is the producer's per-video manifest entry (url, licence, rights)."""
    row = normalise_row(row)
    aliases = cfg.mapping.aliases
    used: set[str] = set()

    def pick(field: str) -> Any:
        name, value = first_present(row, aliases.get(field, [field]))
        if name:
            used.add(name)
        return value

    src_id = pick("source_record_id") or row.get("record_id")
    vid = pick("original_video_id")
    canon = pick("canonical_video_id") or vid
    start = pick("start_time_seconds")
    end = pick("end_time_seconds")
    duration = None
    if start is not None and end is not None:
        duration = round(float(end) - float(start), 3)
    camera = pick("camera")
    validation = pick("validation")
    meta = video_meta or {}
    used.update({"record_type", "schema_version"})
    rec = CanonicalRecord(
        record_id=f"{source_config}:{src_id}",
        source_dataset=source_dataset,
        source_config=source_config,
        source_split=pick("source_split"),
        source_record_id=str(src_id) if src_id is not None else None,
        source_record_hash=source_record_hash(row),
        source_url=meta.get("url"),
        canonical_video_id=str(canon) if canon else None,
        original_video_id=str(vid) if vid else None,
        video_path=pick("video_path"),
        start_time_seconds=float(start) if start is not None else None,
        end_time_seconds=float(end) if end is not None else None,
        duration_seconds=duration,
        width=int(pick("width")) if pick("width") is not None else None,
        height=int(pick("height")) if pick("height") is not None else None,
        fps=float(pick("fps")) if pick("fps") is not None else None,
        caption_original=pick("caption_original"),
        generation_prompt_original=pick("generation_prompt_original"),
        prompt_variants=[str(v) for v in (pick("prompt_variants") or [])],
        actions=_actions(pick("actions")),
        subject_motion=pick("subject_motion") if isinstance(pick("subject_motion"), dict) else None,
        camera_motion=camera if isinstance(camera, dict) else None,
        shot_type=(camera or {}).get("shot_size") or (camera or {}).get("shot_type") if isinstance(camera, dict) else None,
        quality_flags=[str(f) for f in (pick("quality_flags") or [])],
        source_validation=validation if isinstance(validation, dict) else None,
        source_tier=pick("tier"),
        source_confidence=float(pick("confidence")) if pick("confidence") is not None else None,
        source_license=meta.get("license") or meta.get("license_status"),
        source_payload={k: v for k, v in row.items() if k not in used},
    )
    return rec
