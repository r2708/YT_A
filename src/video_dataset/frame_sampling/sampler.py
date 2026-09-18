"""Choose which timestamps of a scene deserve a full-resolution frame."""

from __future__ import annotations

from dataclasses import dataclass, field

from video_dataset.config import FrameSamplingConfig
from video_dataset.schemas.scene import SamplingReason, ScanSample, Scene, SceneScan


@dataclass
class Candidate:
    timestamp: float
    reason: SamplingReason
    change: float | None = None
    motion: float | None = None
    frame_index: int = 0
    priority: int = 0  # higher = keep first when trimming to max_frames
    tags: list[str] = field(default_factory=list)


def _nearest_sample(samples: list[ScanSample], t: float) -> ScanSample | None:
    if not samples:
        return None
    return min(samples, key=lambda s: abs(s.timestamp - t))


def _far_enough(t: float, chosen: list[Candidate], gap: float) -> bool:
    return all(abs(t - c.timestamp) >= gap for c in chosen)


def select_frames(scene: Scene, scan: SceneScan | None, cfg: FrameSamplingConfig, fps: float) -> list[Candidate]:
    duration = max(0.0, scene.duration)
    samples = scan.samples if scan else []
    min_gap = max(0.05, cfg.min_frame_gap_seconds)
    max_frames = max(1, cfg.max_frames_per_scene)
    min_frames = max(1, min(cfg.min_frames_per_scene, max_frames))
    chosen: list[Candidate] = []

    def add(t: float, reason: SamplingReason, priority: int, change: float | None = None, motion: float | None = None) -> None:
        t = min(max(t, scene.start_time), max(scene.start_time, scene.end_time - 1e-3))
        if len(chosen) >= max_frames or not _far_enough(t, chosen, min_gap * 0.5):
            return
        s = _nearest_sample(samples, t)
        chosen.append(
            Candidate(
                timestamp=round(t, 3),
                reason=reason,
                change=change if change is not None else (s.change if s else None),
                motion=motion if motion is not None else (s.flow_mag if s else None),
                priority=priority,
            )
        )

    # 1) scene boundary frames (slightly inside the shot so we never grab the cut frame itself)
    offset = min(cfg.boundary_offset_seconds, duration / 4.0)
    mid = scene.start_time + duration / 2.0
    if cfg.scene_boundary_frames and duration >= 3 * min_gap:
        add(scene.start_time + offset, SamplingReason.SCENE_START, priority=3)
        add(mid, SamplingReason.SCENE_MIDDLE, priority=3)
        add(scene.end_time - offset, SamplingReason.SCENE_END, priority=3)
    else:
        add(mid, SamplingReason.SCENE_MIDDLE, priority=3)

    # 2) motion / visual-change aware extras
    if cfg.motion_aware and samples:
        change_cands = sorted((s for s in samples[1:] if s.change >= cfg.change_threshold), key=lambda s: -s.change)
        for s in change_cands:
            if len(chosen) >= max_frames:
                break
            if _far_enough(s.timestamp, chosen, min_gap):
                add(s.timestamp, SamplingReason.VISUAL_CHANGE, priority=2, change=s.change, motion=s.flow_mag)
        mags = [s.flow_mag for s in samples if s.flow_mag is not None]
        if mags and len(mags) >= 3:
            sorted_m = sorted(mags)
            median = sorted_m[len(sorted_m) // 2]
            thresh = max(0.5, median * 1.5)
            for i in range(1, len(samples) - 1):
                s = samples[i]
                if s.flow_mag is None or s.flow_mag < thresh:
                    continue
                left, right = samples[i - 1].flow_mag or 0.0, samples[i + 1].flow_mag or 0.0
                if s.flow_mag >= left and s.flow_mag >= right and _far_enough(s.timestamp, chosen, min_gap):
                    add(s.timestamp, SamplingReason.MOTION_PEAK, priority=1, change=s.change, motion=s.flow_mag)
                if len(chosen) >= max_frames:
                    break

    # 3) uniform fill up to min_frames, always inserting into the largest temporal gap
    guard = 0
    while len(chosen) < min_frames and guard < 50:
        guard += 1
        pts = sorted([scene.start_time, *[c.timestamp for c in chosen], scene.end_time])
        gaps = [(pts[i + 1] - pts[i], (pts[i] + pts[i + 1]) / 2.0) for i in range(len(pts) - 1)]
        gap, t = max(gaps)
        if gap < min_gap:
            break
        before = len(chosen)
        add(t, SamplingReason.UNIFORM, priority=0)
        if len(chosen) == before:
            break

    # 4) trim to max_frames by priority, then restore temporal order
    if len(chosen) > max_frames:
        chosen.sort(key=lambda c: (-c.priority, -(c.change or 0.0)))
        chosen = chosen[:max_frames]
    chosen.sort(key=lambda c: c.timestamp)
    for c in chosen:
        fi = int(round(c.timestamp * fps))
        c.frame_index = min(max(fi, scene.start_frame), max(scene.start_frame, scene.end_frame - 1))
    return chosen
