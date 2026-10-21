"""Measured summary of the manifest: statuses, eligibility, rights, duplicates, splits, coverage.
Rule-based checks, provider judgements and human reviews are reported in separate columns."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dataset_factory.schema.canonical import CanonicalRecord


def _bump(d: dict[str, int], key: Any) -> None:
    k = str(key if key is not None else "null")
    d[k] = d.get(k, 0) + 1


def summarize(records: Iterable[CanonicalRecord]) -> dict[str, Any]:
    s: dict[str, Any] = {"records": 0, "media_validation": {}, "caption_quality": {}, "caption_review_method": {}, "visual_quality": {}, "start_frame": {}, "rights": {}, "validation": {}, "duplicates": {}, "splits": {}, "t2v_eligible": 0, "i2v_eligible": 0, "both_eligible": 0, "neither": 0, "t2v_reasons": {}, "i2v_reasons": {}, "coverage": {"camera_movement": {}, "shot_type": {}, "duration_band": {}, "aspect_ratio": {}, "source_video": 0}, "assets_present": 0}
    videos: set[str] = set()
    for r in records:
        s["records"] += 1
        _bump(s["media_validation"], r.media_validation_status)
        _bump(s["caption_quality"], r.caption_quality_status)
        _bump(s["caption_review_method"], r.caption_review_method)
        _bump(s["visual_quality"], r.visual_quality_status)
        _bump(s["start_frame"], r.start_frame_status)
        _bump(s["rights"], r.source_rights_status)
        _bump(s["validation"], r.validation_status)
        _bump(s["duplicates"], r.duplicate_kind or "none")
        _bump(s["splits"], r.split)
        if r.video_sha256:
            s["assets_present"] += 1
        if r.t2v_eligible:
            s["t2v_eligible"] += 1
        if r.i2v_eligible:
            s["i2v_eligible"] += 1
        if r.t2v_eligible and r.i2v_eligible:
            s["both_eligible"] += 1
        if not r.t2v_eligible and not r.i2v_eligible:
            s["neither"] += 1
        for why in r.t2v_reasons:
            _bump(s["t2v_reasons"], why.split(":")[0] + ":" + why.split(":")[1] if why.count(":") >= 1 else why)
        for why in r.i2v_reasons:
            _bump(s["i2v_reasons"], why.split(":")[0] + ":" + why.split(":")[1] if why.count(":") >= 1 else why)
        cam = r.camera_motion or {}
        _bump(s["coverage"]["camera_movement"], cam.get("camera_movement") or cam.get("movement"))
        _bump(s["coverage"]["shot_type"], r.shot_type)
        d = r.duration_seconds
        _bump(s["coverage"]["duration_band"], None if d is None else ("<3s" if d < 3 else "3-10s" if d < 10 else "10-30s" if d <= 30 else ">30s"))
        if r.width and r.height:
            _bump(s["coverage"]["aspect_ratio"], f"{r.width / r.height:.2f}")
        if r.canonical_video_id:
            videos.add(r.canonical_video_id)
    s["coverage"]["source_video"] = len(videos)
    for key in ("t2v_reasons", "i2v_reasons"):
        s[key] = dict(sorted(s[key].items(), key=lambda kv: -kv[1])[:20])
    return s


def write_summary(summary: dict[str, Any], out_dir: Path, extra: dict[str, Any] | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {**summary, **(extra or {})}
    js = out_dir / "summary.json"
    js.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    md = out_dir / "summary.md"
    lines = ["# Dataset factory summary", "", "All counts are measured on the current manifest. `not_checked` means the check could not run (asset unavailable), never that it passed.", ""]
    lines.append(f"- records: {summary['records']} from {summary['coverage']['source_video']} source videos; assets present: {summary['assets_present']}")
    lines.append(f"- T2V eligible: {summary['t2v_eligible']}; I2V eligible: {summary['i2v_eligible']}; both: {summary['both_eligible']}; neither: {summary['neither']}")
    for key in ("media_validation", "caption_quality", "caption_review_method", "visual_quality", "start_frame", "rights", "validation", "duplicates", "splits"):
        lines.append(f"- {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(summary[key].items())))
    lines += ["", "## Why records are not eligible", ""]
    for key in ("t2v_reasons", "i2v_reasons"):
        lines.append(f"- {key}: " + (", ".join(f"{k}={v}" for k, v in summary[key].items()) or "none"))
    lines += ["", "## Coverage", ""]
    for k, v in summary["coverage"].items():
        if isinstance(v, dict):
            lines.append(f"- {k}: " + ", ".join(f"{a}={b}" for a, b in sorted(v.items(), key=lambda kv: -kv[1])[:12]))
    if extra:
        lines += ["", "## Run", ""] + [f"- {k}: {v}" for k, v in extra.items()]
    md.write_text("\n".join(lines) + "\n")
    return js, md
