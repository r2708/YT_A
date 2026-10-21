"""Media integrity validation of one record: asset present and readable, probed metadata
plausible and consistent with the source row, required frames decodable, frame-level quality
metrics. Writes measured facts (sha256, dimensions, fingerprint) onto the record."""

from __future__ import annotations

from pathlib import Path

from dataset_factory.config import FactoryConfig
from dataset_factory.media.decode import decode_sampled_frames
from dataset_factory.media.fingerprints import sha256_of
from dataset_factory.media.probe import probe
from dataset_factory.media.quality import measure
from dataset_factory.schema.canonical import CanonicalRecord


def resolve_asset(rec: CanonicalRecord, cfg: FactoryConfig) -> Path | None:
    if not rec.video_path:
        return None
    p = Path(rec.video_path)
    if p.is_absolute():
        return p
    root = cfg.source.media_root
    return (Path(root) / p) if root else None


def asset_signature(path: Path | None) -> str:
    if path is None or not path.exists():
        return "absent"
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def validate_media(rec: CanonicalRecord, cfg: FactoryConfig) -> CanonicalRecord:
    rules = cfg.media
    path = resolve_asset(rec, cfg)
    reasons: list[str] = []
    claimed = {"width": rec.source_payload.get("_claimed_width", rec.width), "height": rec.source_payload.get("_claimed_height", rec.height)}
    if path is None:
        rec.media_validation_status = "not_checked"
        rec.media_validation_reasons = ["asset_unavailable:no_media_root" if rec.video_path else "asset_unavailable:no_video_path"]
        rec.visual_quality_status = "not_checked"
        return rec
    pr = probe(path)
    if not pr.ok:
        rec.media_validation_status = "not_checked" if pr.error == "asset_missing" else "failed"
        rec.media_validation_reasons = [f"asset_unavailable:{pr.error}" if pr.error == "asset_missing" else str(pr.error)]
        rec.visual_quality_status = "not_checked"
        return rec
    rec.video_sha256 = sha256_of(path)
    rec.source_payload.setdefault("_claimed_width", rec.width)  # what the source row said, kept for the mismatch check on re-runs
    rec.source_payload.setdefault("_claimed_height", rec.height)
    rec.width, rec.height, rec.fps, rec.frame_count, rec.codec = pr.width, pr.height, pr.fps, pr.frame_count, pr.codec
    probed_duration = float(pr.duration or 0.0)
    if probed_duration <= 0:
        reasons.append("zero_duration")
    if probed_duration and probed_duration < rules.min_duration_seconds:
        reasons.append(f"too_short:{probed_duration:.2f}s<{rules.min_duration_seconds}")
    if probed_duration > rules.max_duration_seconds:
        reasons.append(f"too_long:{probed_duration:.2f}s>{rules.max_duration_seconds}")
    if (pr.width or 0) < rules.min_width or (pr.height or 0) < rules.min_height:
        reasons.append(f"too_small:{pr.width}x{pr.height}")
    if pr.width and pr.height:
        ar = max(pr.width / pr.height, pr.height / pr.width)
        if ar > rules.max_aspect_ratio:
            reasons.append(f"implausible_aspect_ratio:{ar:.2f}")
    if pr.fps is not None and (pr.fps < rules.min_fps or pr.fps > rules.max_fps):
        reasons.append(f"fps_out_of_range:{pr.fps:.2f}")
    # consistency with what the source row claimed
    for key, probed in (("width", pr.width), ("height", pr.height)):
        c = claimed.get(key)
        if c and probed and abs(int(c) - int(probed)) / max(1, int(probed)) > rules.metadata_tolerance_fraction:
            reasons.append(f"{key}_mismatch:record={c},probed={probed}")
    span = rec.duration_seconds
    if span is not None and probed_duration and abs(span - probed_duration) > rules.duration_tolerance_seconds:
        reasons.append(f"duration_mismatch:record={span:.2f},probed={probed_duration:.2f}")
    frames, n_frames, _fps = decode_sampled_frames(path, rules.sample_frames)
    if not frames:
        reasons.append("decode_failed")
        rec.visual_quality_status = "failed"
        rec.visual_quality_reasons = ["no_frames_decoded"]
    else:
        vq = measure(frames, rules)
        rec.visual_quality_status, rec.visual_quality_score, rec.visual_quality_reasons = vq.status, vq.score, vq.reasons
        rec.video_fingerprint = vq.fingerprint
        if vq.status == "failed":
            reasons.extend(vq.reasons)
        if rec.frame_count in (None, 0) and n_frames:
            rec.frame_count = n_frames
    rec.media_validation_reasons = reasons
    rec.media_validation_status = "failed" if reasons else ("needs_review" if rec.visual_quality_status == "needs_review" else "passed")
    if rec.duration_seconds is None and probed_duration:
        rec.duration_seconds = round(probed_duration, 3)
    return rec
