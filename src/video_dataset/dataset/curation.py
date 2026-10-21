"""Curation of exported records: split assignment, quality tiers, subsets, duplicate-video handling,
sequences and prompt variants.

Everything here works on plain record dicts (the JSON form of the record schemas) so the same
code annotates fresh per-video exports and records merged back from earlier final files.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from video_dataset.config import CinematicSubsetConfig, ExportConfig, LicensingConfig, SplitConfig, TierConfig
from video_dataset.schemas.common import SCHEMA_VERSION, ConfidenceSource
from video_dataset.schemas.curation import Split, Tier
from video_dataset.schemas.quality import QUALITY_COMPONENTS, ValidationStatus
from video_dataset.validation.policy import enforce_exported, strict_ok, verification_summary

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
LICENSED_SUBSET = "licensed"
STRICT_SUBSET = "strict"
RIGHTS_STATUSES = ("declared_cc_by", "restricted", "unknown", "removed")


def rights_status_for(license_status: str | None, source_type: str | None = None) -> str:
    """Rights status of a source from its normalised license. The platform's declaration is the
    only evidence available, so a CC-BY video is ``declared_cc_by`` (not "verified"), the standard
    YouTube license is ``restricted`` (viewable, not redistributable), a local file or anything
    else is ``unknown``."""
    st = str(license_status or "").lower()
    if st == "creative_commons":
        return "declared_cc_by"
    if st in ("standard", "other"):
        return "restricted"
    return "unknown"


def duration_band(duration: float | None, bands: dict[str, list[float]]) -> str | None:
    if duration is None:
        return None
    for name, (lo, hi) in bands.items():
        if float(lo) <= float(duration) < float(hi):
            return name
    return None


# ------------------------------------------------------------------------------------ split
def split_hash(video_id: str, cfg: SplitConfig) -> float:
    """Deterministic position in [0, 1) from sha1(salt | id)."""
    h = hashlib.sha1(f"{cfg.salt}|{video_id}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)


def split_for_video(video_id: str, cfg: SplitConfig) -> Split:
    """Pure hash bucket of one id: the fractions carve the unit interval (no minimums applied)."""
    u = split_hash(video_id, cfg)
    test = max(0.0, float(cfg.test_fraction))
    val = max(0.0, float(cfg.validation_fraction))
    if u < test:
        return Split.TEST
    if u < test + val:
        return Split.VALIDATION
    return Split.TRAIN


def assign_splits(group_ids: Iterable[str], cfg: SplitConfig) -> dict[str, Split]:
    """Split of every source group in ``group_ids``.

    Each group gets its hash bucket (``split_for_video``). When that leaves the test or validation
    split below ``min_test_videos`` / ``min_validation_videos``, the groups with the lowest hash
    among the remaining train groups are moved over (test first, then validation), so the result is
    still a deterministic function of (salt, set of ids). Train always keeps at least one group; a
    single-group export is all train. A group with a hash under the fractions never leaves its eval
    split, so the hash-based core is stable as the pool grows; only the top-up groups can move back
    to train once the fractions alone satisfy the minimums.
    """
    ids = sorted({str(g) for g in group_ids if g})
    if not ids:
        return {}
    out = {g: split_for_video(g, cfg) for g in ids}
    n = len(ids)
    if n == 1:
        return {ids[0]: Split.TRAIN}
    budget = n - 1  # groups that may sit in eval splits
    want = {Split.TEST: max(0, int(cfg.min_test_videos)), Split.VALIDATION: max(0, int(cfg.min_validation_videos))}
    have = {sp: sum(1 for v in out.values() if v == sp) for sp in (Split.TEST, Split.VALIDATION)}
    # minimums that do not fit beside one train group are reduced, validation first
    while want[Split.TEST] + want[Split.VALIDATION] > budget:
        if want[Split.VALIDATION] >= want[Split.TEST] and want[Split.VALIDATION] > 0:
            want[Split.VALIDATION] -= 1
        else:
            want[Split.TEST] -= 1
    train_sorted = sorted((g for g in ids if out[g] == Split.TRAIN), key=lambda g: (split_hash(g, cfg), g))
    for sp in (Split.TEST, Split.VALIDATION):
        while have[sp] < want[sp] and len(train_sorted) > 1:
            g = train_sorted.pop(0)
            out[g] = sp
            have[sp] += 1
    return out


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


def cap_tier(tier: Tier | None, cap: Tier, reasons: list[str], reason: str) -> tuple[Tier | None, list[str]]:
    """Lower ``tier`` to ``cap`` when it is above it, recording why."""
    if tier is not None and _TIER_ORDER[tier] > _TIER_ORDER[cap]:
        return cap, [*reasons, reason]
    if tier is not None and reason not in reasons and _TIER_ORDER[tier] <= _TIER_ORDER[cap] and tier != cap:
        return tier, reasons
    return tier, reasons


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
    if scene.get("quality_flags"):
        why.append("quality_flags")
    if scene.get("canonical_video_id"):
        why.append("near_duplicate_video")
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


# ------------------------------------------------------------------------------------ video info
def video_info_from_manifests(manifests: list[dict[str, Any]], similarity_threshold: float = 0.8, fingerprint_enabled: bool = True) -> dict[str, dict[str, Any]]:
    """Per-video curation facts gathered from the per-video manifests: license status, source quality
    flags and the canonical video id of near-duplicates (from the stored fingerprints, plus what the
    FRAME_EXTRACTION stage already noticed against the state DB)."""
    from video_dataset.deduplication.videos import near_duplicate_videos

    info: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, list[str]] = {}
    for m in manifests:
        vid = m.get("video_id")
        if not vid:
            continue
        info[str(vid)] = {
            "license_status": m.get("license_status"),
            "quality_flags": list(m.get("quality_flags") or []),
            "canonical_video_id": m.get("near_duplicate_of") or None,
        }
        if m.get("fingerprint"):
            fingerprints[str(vid)] = list(m["fingerprint"])
    if fingerprint_enabled and len(fingerprints) > 1:
        for vid, canon in near_duplicate_videos(fingerprints, similarity_threshold).items():
            info.setdefault(vid, {})["canonical_video_id"] = canon
    # resolve chains (a -> b -> c) and never let a video be its own canonical
    for vid, d in info.items():
        canon = d.get("canonical_video_id")
        seen = {vid}
        while canon and canon in info and info[canon].get("canonical_video_id") and canon not in seen:
            seen.add(canon)
            canon = info[canon]["canonical_video_id"]
        d["canonical_video_id"] = canon if canon and canon != vid else None
    return info


# ------------------------------------------------------------------------------------ driver
def annotate_records(
    records: dict[str, list[dict[str, Any]]],
    cfg: ExportConfig,
    video_info: dict[str, dict[str, Any]] | None = None,
    licensing: LicensingConfig | None = None,
    seed: int | str = 1234,
) -> dict[str, int]:
    """Fill split / tier / tier_reasons / subsets / canonical_video_id on every record dict in place,
    link frames to their clip, attach generation prompts (+ variants) to scene / clip records and
    (re)build the ``sequences`` records.

    ``video_info`` (per video id): ``canonical_video_id`` for near-duplicate videos (their records
    get the canonical video's split and a bronze cap), ``license_status`` (the ``licensed`` subset)
    and ``quality_flags`` of the source. Returns counts.
    """
    from video_dataset.dataset.prompting import fill_frame_clip_ids, fill_generation_prompts
    from video_dataset.dataset.sequences import build_sequences
    from video_dataset.dataset.variations import fill_prompt_variants

    video_info = video_info or {}
    allowed = {str(a).lower() for a in ((licensing.allowed if licensing else []) or [])}
    licensed_subset = bool(licensing.subset) if licensing else True

    fill_frame_clip_ids(records)
    fill_generation_prompts(records)
    records["sequences"] = build_sequences(records, cfg.sequences) if cfg.sequences.enabled else []
    fill_prompt_variants(records, int(cfg.prompt_variants), seed)

    counts = {"cinematic": 0, "cinematic_scenes": 0, "sequences": len(records["sequences"]), "duplicate_videos": 0, "licensed_videos": 0, "dropped_duplicate_records": 0, "strict": 0, "dropped_rights_records": 0, "policy_downgraded": 0}
    enforce = bool(getattr(cfg, "enforce_policy", True))
    group_ids: set[str] = set()
    for rows in records.values():
        for rec in rows:
            vid = str(rec.get("video_id") or "")
            canonical = video_info.get(vid, {}).get("canonical_video_id") or None
            if vid and canonical != vid:
                group_ids.add(canonical or vid)
            elif vid:
                group_ids.add(vid)
    splits = assign_splits(group_ids, cfg.split) if cfg.split.enabled else {}
    dup_videos: set[str] = set()
    licensed_videos: set[str] = set()
    release_rights = {str(r).lower() for r in ((licensing.release_rights if licensing else []) or [])}
    strict_cfg = getattr(cfg, "strict", None)
    bands = dict(getattr(cfg, "duration_bands", {}) or {})
    for kind, rows in list(records.items()):
        kept_rows: list[dict[str, Any]] = []
        for rec in rows:
            vid = str(rec.get("video_id") or "")
            vinfo = video_info.get(vid, {})
            canonical = vinfo.get("canonical_video_id") or None
            if canonical == vid:
                canonical = None
            rec["canonical_video_id"] = canonical
            if canonical:
                dup_videos.add(vid)
                if cfg.exclude_duplicate_videos:
                    counts["dropped_duplicate_records"] += 1
                    continue
            rights = str(vinfo.get("rights_status") or rights_status_for(vinfo.get("license_status")))
            if getattr(cfg, "require_rights", False) and rights not in release_rights:
                counts["dropped_rights_records"] += 1
                continue
            split_id = canonical or vid
            rec["source_group_id"] = split_id or None
            rec["schema_version"] = SCHEMA_VERSION
            rec["coverage"] = dict(vinfo.get("coverage") or rec.get("coverage") or {})
            rec["split"] = splits[split_id].value if cfg.split.enabled and split_id in splits else None
            if enforce and enforce_exported(rec):
                counts["policy_downgraded"] += 1
            v = rec.get("validation")
            if isinstance(v, dict):
                v["verification"] = verification_summary(rec)
            q = rec.get("quality")
            if isinstance(q, dict):
                q["missing_components"] = [c for c in QUALITY_COMPONENTS if q.get(c) is None]
            if cfg.tiers.enabled:
                tier, reasons = tier_for_record(rec, cfg.tiers)
                if canonical:
                    tier, reasons = cap_tier(tier, Tier.BRONZE, reasons, "near_duplicate_video")
                if rec.get("quality_flags"):
                    tier, reasons = cap_tier(tier, Tier.SILVER, reasons, "quality_flags")
                rec["tier"] = tier.value if tier else None
                rec["tier_reasons"] = reasons
            else:
                rec["tier"] = None
                rec["tier_reasons"] = []
            rec["subsets"] = []
            status = str(vinfo.get("license_status") or "").lower()
            if licensed_subset and status and status in allowed:
                rec["subsets"].append(LICENSED_SUBSET)
                licensed_videos.add(vid)
            if rights in release_rights:
                rec["subsets"].append(f"rights_{rights}")
            if bands and kind in ("scenes", "clips", "sequences"):
                d = rec.get("duration")
                if d is None and rec.get("start_time") is not None and rec.get("end_time") is not None:
                    d = float(rec["end_time"]) - float(rec["start_time"])
                band = duration_band(d, bands)
                if band:
                    rec["subsets"].append(f"duration_{band}")
            if strict_cfg is not None and getattr(strict_cfg, "enabled", False):
                ok, _why = strict_ok(rec)
                targets = set(getattr(strict_cfg, "generation_targets", []) or [])
                if ok and kind in ("scenes", "clips", "sequences") and targets and rec.get("generation_target") not in targets:
                    ok = False
                if ok:
                    rec["subsets"].append(STRICT_SUBSET)
                    counts["strict"] += 1
            kept_rows.append(rec)
        records[kind] = kept_rows
    counts["duplicate_videos"] = len(dup_videos)
    counts["licensed_videos"] = len(licensed_videos)
    if not cfg.cinematic.enabled:
        return counts
    good = cinematic_scene_ids(records, cfg.cinematic)
    counts["cinematic_scenes"] = len(good)
    for kind, rows in records.items():
        for rec in rows:
            sids = _record_scene_ids(kind, rec)
            vid = str(rec.get("video_id"))
            if sids and all((vid, s) in good for s in sids):
                rec["subsets"] = [*rec["subsets"], "cinematic"]
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


def split_leaks(records: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    """Videos / source groups whose records sit in more than one split: {id: [splits]}. Empty = no
    leakage. Checked per video id and per ``source_group_id`` (near-duplicate clusters)."""
    by_video: dict[str, set[str]] = {}
    by_group: dict[str, set[str]] = {}
    for rows in records.values():
        for rec in rows:
            sp = rec.get("split")
            if not sp:
                continue
            if rec.get("video_id"):
                by_video.setdefault(str(rec["video_id"]), set()).add(str(sp))
            grp = rec.get("source_group_id") or rec.get("canonical_video_id") or rec.get("video_id")
            if grp:
                by_group.setdefault(str(grp), set()).add(str(sp))
    leaks = {k: sorted(v) for k, v in by_video.items() if len(v) > 1}
    leaks.update({f"group:{k}": sorted(v) for k, v in by_group.items() if len(v) > 1})
    return leaks
