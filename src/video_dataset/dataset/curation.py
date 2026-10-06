"""Curation of exported records: split assignment, quality tiers and the cinematic subset.

Everything here works on plain record dicts (the JSON form of the record schemas) so the same
code annotates fresh per-video exports and records merged back from earlier final files.
"""

from __future__ import annotations

import hashlib
from typing import Any

from video_dataset.config import CinematicSubsetConfig, ExportConfig, SplitConfig, TierConfig
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.curation import Split, Tier
from video_dataset.schemas.quality import ValidationStatus

SCENE_LEVEL = ("scenes", "clips", "frames")  # records that describe exactly one shot
_TIER_ORDER = {Tier.GOLD: 3, Tier.SILVER: 2, Tier.BRONZE: 1}
_TRUSTED_CONFIDENCE = {
    ConfidenceSource.VERIFIER,
    ConfidenceSource.MEASUREMENT,
    ConfidenceSource.ASR_LOGPROB,
    ConfidenceSource.OCR_SCORE,
    ConfidenceSource.DETECTOR_SCORE,
    ConfidenceSource.INTERVAL_ALGEBRA,
    ConfidenceSource.DERIVED_MIN,
}


# ------------------------------------------------------------------------------------ split
def split_for_video(video_id: str, cfg: SplitConfig) -> Split:
    """Deterministic bucket from sha1(salt | video_id); the fractions carve the unit interval."""
    h = hashlib.sha1(f"{cfg.salt}|{video_id}".encode()).hexdigest()
    u = int(h[:12], 16) / float(16**12)
    test = max(0.0, float(cfg.test_fraction))
    val = max(0.0, float(cfg.validation_fraction))
    if u < test:
        return Split.TEST
    if u < test + val:
        return Split.VALIDATION
    return Split.TRAIN


# ------------------------------------------------------------------------------------ tiers
def tier_for_record(rec: dict[str, Any], cfg: TierConfig) -> tuple[Tier | None, list[str]]:
    """(tier, reasons). Reasons explain why the record did not reach the tier above."""
    validation = rec.get("validation") or {}
    status = validation.get("status")
    if status in (ValidationStatus.REJECTED.value, ValidationStatus.DUPLICATE.value):
        return None, [f"status_{status}"]
    quality = rec.get("quality") or {}
    overall = quality.get("overall")
    components = set(quality.get("components_used") or [])
    conf = rec.get("confidence")
    conf_src = str(rec.get("confidence_source") or ConfidenceSource.UNAVAILABLE.value)
    reasons: list[str] = []

    accepted = status == ValidationStatus.ACCEPTED.value
    if not accepted:
        reasons.append(f"status_{status}" if status else "status_missing")
    if overall is None:
        reasons.append("quality_unscored")
    if accepted and overall is not None and overall >= float(cfg.gold_min_overall):
        gold_block: list[str] = []
        if cfg.gold_requires_grounding and "grounding" not in components:
            gold_block.append("no_grounding_component")
        if conf is None:
            gold_block.append("confidence_unavailable")
        elif conf < float(cfg.gold_min_confidence):
            gold_block.append(f"confidence_below_{cfg.gold_min_confidence}")
        elif conf_src not in {c.value for c in _TRUSTED_CONFIDENCE}:
            gold_block.append(f"confidence_source_{conf_src}")
        if not gold_block:
            return Tier.GOLD, []
        return Tier.SILVER, gold_block
    if accepted and overall is not None and overall >= float(cfg.silver_min_overall):
        return Tier.SILVER, [f"overall_below_{cfg.gold_min_overall}"]
    if accepted and overall is not None:
        reasons.append(f"overall_below_{cfg.silver_min_overall}")
    return Tier.BRONZE, reasons


def tier_at_least(tier: str | None, minimum: str) -> bool:
    if tier is None:
        return False
    return _TIER_ORDER.get(Tier(tier), 0) >= _TIER_ORDER.get(Tier(minimum), 0)


# ------------------------------------------------------------------------------------ cinematic
def scene_is_cinematic(scene: dict[str, Any], cfg: CinematicSubsetConfig) -> tuple[bool, list[str]]:
    """Criteria on a *scene* record; clips and frames inherit the verdict of their scene."""
    why: list[str] = []
    if not tier_at_least(scene.get("tier"), cfg.min_tier):
        why.append(f"tier_below_{cfg.min_tier}")
    duration = scene.get("duration")
    if duration is None:
        duration = float(scene.get("end_time", 0.0)) - float(scene.get("start_time", 0.0))
    if float(duration) < float(cfg.min_scene_duration):
        why.append("too_short")
    camera = scene.get("camera") or {}
    if cfg.require_known_camera and str(camera.get("movement") or "unknown") == "unknown":
        why.append("camera_unknown")
    m = scene.get("measurements") or {}
    aesthetic = m.get("aesthetic_score")
    if aesthetic is None:
        if cfg.require_aesthetic:
            why.append("no_aesthetic_score")
    elif float(aesthetic) < float(cfg.min_aesthetic):
        why.append(f"aesthetic_below_{cfg.min_aesthetic}")
    sharp = m.get("sharpness")
    if float(cfg.min_sharpness) > 0 and sharp is not None and float(sharp) < float(cfg.min_sharpness):
        why.append("soft_focus")
    return not why, why


def cinematic_scene_ids(records: dict[str, list[dict[str, Any]]], cfg: CinematicSubsetConfig) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for sc in records.get("scenes", []):
        ok, _ = scene_is_cinematic(sc, cfg)
        if ok:
            out.add((str(sc.get("video_id")), str(sc.get("scene_id"))))
    return out


def _record_scene_ids(kind: str, rec: dict[str, Any]) -> list[str]:
    if kind in SCENE_LEVEL:
        return [str(rec.get("scene_id"))] if rec.get("scene_id") else []
    if kind in ("temporal_qa", "long_video_qa"):
        return [str(s) for s in ((rec.get("evidence") or {}).get("scene_ids") or [])]
    return [str(s) for s in (rec.get("scene_ids") or [])]


# ------------------------------------------------------------------------------------ driver
def annotate_records(records: dict[str, list[dict[str, Any]]], cfg: ExportConfig) -> dict[str, int]:
    """Fill split / tier / tier_reasons / subsets on every record dict in place, link frames to
    their clip and attach generation prompts to scene / clip records.

    Returns counts: {"cinematic": n_records_in_subset, "cinematic_scenes": n_scenes}.
    """
    from video_dataset.dataset.prompting import fill_frame_clip_ids, fill_generation_prompts

    fill_frame_clip_ids(records)
    fill_generation_prompts(records)
    for rows in records.values():
        for rec in rows:
            vid = str(rec.get("video_id") or "")
            rec["split"] = split_for_video(vid, cfg.split).value if cfg.split.enabled and vid else None
            if cfg.tiers.enabled:
                tier, reasons = tier_for_record(rec, cfg.tiers)
                rec["tier"] = tier.value if tier else None
                rec["tier_reasons"] = reasons
            else:
                rec["tier"] = None
                rec["tier_reasons"] = []
            rec["subsets"] = []
    counts = {"cinematic": 0, "cinematic_scenes": 0}
    if not cfg.cinematic.enabled:
        return counts
    good = cinematic_scene_ids(records, cfg.cinematic)
    counts["cinematic_scenes"] = len(good)
    for kind, rows in records.items():
        for rec in rows:
            sids = _record_scene_ids(kind, rec)
            vid = str(rec.get("video_id"))
            if sids and all((vid, s) in good for s in sids):
                rec["subsets"] = ["cinematic"]
                counts["cinematic"] += 1
    return counts


def subset_records(records: dict[str, list[dict[str, Any]]], name: str) -> dict[str, list[dict[str, Any]]]:
    return {kind: [r for r in rows if name in (r.get("subsets") or [])] for kind, rows in records.items()}


def split_manifest(records: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    """{split: sorted video ids} over every record."""
    out: dict[str, set[str]] = {s.value: set() for s in Split}
    for rows in records.values():
        for rec in rows:
            sp = rec.get("split")
            if sp in out and rec.get("video_id"):
                out[sp].add(str(rec["video_id"]))
    return {k: sorted(v) for k, v in out.items()}
