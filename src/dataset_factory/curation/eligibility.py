"""Rights resolution and T2V / I2V eligibility. Every verdict carries its reasons; T2V and I2V
are decided independently."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset_factory.config import DedupRules, FactoryConfig, RightsRules
from dataset_factory.schema.canonical import CanonicalRecord


def load_rights_evidence(rules: RightsRules, base: Path) -> dict[str, dict[str, Any]]:
    if not rules.evidence_file:
        return {}
    p = Path(rules.evidence_file)
    if not p.is_absolute():
        p = base / p
    out: dict[str, dict[str, Any]] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                for key in ("canonical_video_id", "video_id"):
                    if row.get(key):
                        out[str(row[key])] = row
    return out


def resolve_rights(rec: CanonicalRecord, rules: RightsRules, evidence: dict[str, dict[str, Any]], video_meta: dict[str, Any] | None = None) -> None:
    """Explicit evidence wins; otherwise the platform's declared licence maps through
    ``status_map``; absence of evidence is ``review_required``, never ``cleared``."""
    ev = evidence.get(str(rec.canonical_video_id)) or evidence.get(str(rec.original_video_id))
    if ev and str(ev.get("status")) in ("cleared", "restricted", "review_required", "unknown"):
        rec.source_rights_status = str(ev["status"])
        rec.source_rights_evidence = f"evidence_file:{ev.get('evidence') or ''}".rstrip(":")
        return
    meta = video_meta or {}
    declared = meta.get("rights_status") or rec.source_payload.get("rights_status") or meta.get("license_status") or rec.source_license
    if declared:
        mapped = rules.status_map.get(str(declared).lower())
        rec.source_rights_status = mapped or "review_required"
        rec.source_rights_evidence = f"platform_declared:{declared}"
        if rec.source_license is None:
            rec.source_license = str(declared)
    else:
        rec.source_rights_status = "review_required"
        rec.source_rights_evidence = "none"


def rights_allowed(rec: CanonicalRecord, rules: RightsRules) -> tuple[bool, bool]:
    """(allowed, via_override)."""
    if rec.source_rights_status in rules.export_allowed:
        return True, False
    if rec.source_rights_status in rules.export_override:
        return True, True
    return False, False


def decide(rec: CanonicalRecord, cfg: FactoryConfig) -> None:
    """Fill validation_status, t2v_eligible / i2v_eligible and their reasons in place."""
    reasons: list[str] = []
    t2v: list[str] = []
    i2v: list[str] = []
    if rec.media_validation_status != "passed":
        t2v.append(f"media:{rec.media_validation_status}")
    if rec.caption_quality_status != "passed":
        t2v.append(f"caption:{rec.caption_quality_status}")
    if not rec.caption_final:
        t2v.append("caption:missing")
    allowed, override = rights_allowed(rec, cfg.rights)
    if not allowed:
        t2v.append(f"rights:{rec.source_rights_status}")
    elif override:
        reasons.append(f"rights_override:{rec.source_rights_status}")
    if rec.duplicate_kind == "exact" and rec.duplicate_canonical is False:
        t2v.append("duplicate:exact")
    if rec.duplicate_kind == "near" and _exclude_near(cfg.dedup):
        t2v.append("duplicate:near_candidate")
    if rec.visual_quality_status == "failed":
        t2v.append("visual_quality:failed")
    rec.t2v_reasons = t2v
    rec.t2v_eligible = not t2v
    i2v = list(t2v)
    if rec.start_frame_status != "passed":
        i2v.append(f"start_frame:{rec.start_frame_status}")
    if rec.start_frame_clip_sha256 and rec.video_sha256 and rec.start_frame_clip_sha256 != rec.video_sha256:
        i2v.append("start_frame:clip_changed_since_extraction")
    if rec.caption_final and "no_motion_described" in rec.caption_quality_reasons:
        i2v.append("caption:no_motion_described")
    rec.i2v_reasons = i2v
    rec.i2v_eligible = not i2v
    statuses = [rec.media_validation_status, rec.caption_quality_status]
    if "failed" in statuses or rec.visual_quality_status == "failed":
        rec.validation_status = "failed"
    elif rec.media_validation_status == "not_checked":
        rec.validation_status = "not_checked"
    elif "needs_review" in statuses or rec.source_rights_status == "review_required" or rec.duplicate_kind == "near":
        rec.validation_status = "needs_review"
    else:
        rec.validation_status = "passed"
    rec.validation_reasons = reasons + sorted(set(rec.media_validation_reasons + rec.caption_quality_reasons))[:12]


def _exclude_near(rules: DedupRules) -> bool:
    return rules.action == "exclude"
