"""Scene detection with PySceneDetect (content cuts + luminance fades), plus a fixed-interval fallback.

Boundaries are real visual transitions. Over-long shots can optionally be split for analysis,
but those splits are explicitly marked `TransitionType.SPLIT` so nothing downstream mistakes them for cuts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from video_dataset.config import SceneDetectionConfig
from video_dataset.schemas.scene import Scene, SceneDetectionResult, TransitionType
from video_dataset.schemas.video import MediaInfo
from video_dataset.utils.ids import scene_id as make_scene_id
from video_dataset.utils.logging import get_logger

log = get_logger("scene_detection")


class SceneDetector(Protocol):
    name: str

    def detect(self, video_path: Path, info: MediaInfo, video_id: str) -> SceneDetectionResult: ...


def _frames_to_seconds(frame: int, fps: float) -> float:
    return frame / fps if fps else 0.0


class PySceneDetector:
    name = "pyscenedetect"

    def __init__(self, config: SceneDetectionConfig):
        self.config = config

    def detect(self, video_path: Path, info: MediaInfo, video_id: str) -> SceneDetectionResult:
        from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, ThresholdDetector, open_video
        from scenedetect.stats_manager import StatsManager

        cfg = self.config
        fps = info.fps
        min_len = max(1, int(round(cfg.min_scene_len_seconds * fps)))

        video = open_video(str(video_path), backend="opencv")
        stats = StatsManager()
        manager = SceneManager(stats)
        if cfg.detector == "adaptive":
            manager.add_detector(AdaptiveDetector(adaptive_threshold=cfg.adaptive_threshold, min_scene_len=min_len))
        elif cfg.detector == "threshold":
            manager.add_detector(ThresholdDetector(threshold=cfg.fade_threshold, min_scene_len=min_len, add_final_scene=True))
        else:
            manager.add_detector(ContentDetector(threshold=cfg.threshold, min_scene_len=min_len))
        if cfg.detect_fades and cfg.detector != "threshold":
            manager.add_detector(ThresholdDetector(threshold=cfg.fade_threshold, min_scene_len=min_len, add_final_scene=True))
        if cfg.downscale:
            manager.auto_downscale = False
            manager.downscale = int(cfg.downscale)

        manager.detect_scenes(video=video, frame_skip=int(cfg.frame_skip or 0), show_progress=False)
        raw = manager.get_scene_list(start_in_scene=True)

        boundaries: list[tuple[int, int]] = []
        for start_tc, end_tc in raw:
            s = int(getattr(start_tc, "frame_num", None) or start_tc.get_frames())
            e = int(getattr(end_tc, "frame_num", None) or end_tc.get_frames())
            if e > s:
                boundaries.append((s, e))
        total_frames = int(info.frame_count) if info.frame_count else int(round(info.duration * fps))
        if not boundaries:
            boundaries = [(0, max(1, total_frames))]

        scenes: list[Scene] = []
        for idx, (s, e) in enumerate(boundaries):
            score = None
            transition = TransitionType.START if idx == 0 else TransitionType.CUT
            if idx > 0:
                score = self._content_val(stats, s)
                if score is not None and score < cfg.threshold and cfg.detect_fades and cfg.detector != "adaptive":
                    transition = TransitionType.FADE
                elif cfg.detector == "adaptive":
                    transition = TransitionType.ADAPTIVE
            scenes.append(
                Scene(
                    scene_id=make_scene_id(idx),
                    video_id=video_id,
                    index=idx,
                    start_time=round(_frames_to_seconds(s, fps), 3),
                    end_time=round(min(_frames_to_seconds(e, fps), info.duration) if info.duration else _frames_to_seconds(e, fps), 3),
                    start_frame=s,
                    end_frame=e,
                    transition_in=transition,
                    detector=self.name,
                    detector_score=round(score, 3) if score is not None else None,
                )
            )
        scenes = _snap_to_duration(scenes, info.duration, fps)
        scenes = split_long_scenes(scenes, cfg.max_scene_duration, fps, video_id)
        return SceneDetectionResult(video_id=video_id, detector=self.name, threshold=cfg.threshold, duration=info.duration, scenes=scenes)

    @staticmethod
    def _content_val(stats, frame: int) -> float | None:  # type: ignore[no-untyped-def]
        for key in ("content_val",):
            try:
                if stats.metrics_exist(frame, [key]):
                    vals = stats.get_metrics(frame, [key])
                    if vals and vals[0] is not None:
                        return float(vals[0])
            except Exception:
                continue
        return None


class FixedIntervalDetector:
    """Fallback only (PySceneDetect unavailable). Boundaries are artificial and marked SPLIT."""

    name = "fixed_interval"

    def __init__(self, interval_seconds: float = 10.0):
        self.interval = interval_seconds

    def detect(self, video_path: Path, info: MediaInfo, video_id: str) -> SceneDetectionResult:
        scenes: list[Scene] = []
        t = 0.0
        idx = 0
        while t < info.duration:
            end = min(t + self.interval, info.duration)
            scenes.append(
                Scene(
                    scene_id=make_scene_id(idx),
                    video_id=video_id,
                    index=idx,
                    start_time=round(t, 3),
                    end_time=round(end, 3),
                    start_frame=int(round(t * info.fps)),
                    end_frame=int(round(end * info.fps)),
                    transition_in=TransitionType.START if idx == 0 else TransitionType.SPLIT,
                    detector=self.name,
                )
            )
            idx += 1
            t = end
        return SceneDetectionResult(video_id=video_id, detector=self.name, threshold=None, duration=info.duration, scenes=scenes)


def _snap_to_duration(scenes: list[Scene], duration: float, fps: float) -> list[Scene]:
    if not scenes:
        return scenes
    out = [s.model_copy() for s in scenes]
    out[0] = out[0].model_copy(update={"start_time": 0.0, "start_frame": 0})
    if duration and out[-1].end_time < duration - (1.0 / fps if fps else 0.05):
        out[-1] = out[-1].model_copy(update={"end_time": round(duration, 3), "end_frame": int(round(duration * fps))})
    return out


def split_long_scenes(scenes: list[Scene], max_duration: float | None, fps: float, video_id: str) -> list[Scene]:
    """Split shots longer than `max_duration` into equal parts (marked SPLIT) and renumber."""
    if not max_duration or max_duration <= 0:
        return scenes
    out: list[Scene] = []
    for sc in scenes:
        if sc.duration <= max_duration * 1.25:
            out.append(sc)
            continue
        n = int(sc.duration // max_duration) + 1
        part = sc.duration / n
        for k in range(n):
            st = sc.start_time + k * part
            en = sc.start_time + (k + 1) * part if k < n - 1 else sc.end_time
            out.append(
                sc.model_copy(
                    update={
                        "start_time": round(st, 3),
                        "end_time": round(en, 3),
                        "start_frame": int(round(st * fps)),
                        "end_frame": int(round(en * fps)),
                        "transition_in": sc.transition_in if k == 0 else TransitionType.SPLIT,
                        "split_from": sc.scene_id,
                        "detector_score": sc.detector_score if k == 0 else None,
                    }
                )
            )
    renumbered: list[Scene] = []
    for idx, sc in enumerate(out):
        renumbered.append(sc.model_copy(update={"scene_id": make_scene_id(idx), "index": idx, "video_id": video_id}))
    return renumbered


def create_scene_detector(config: SceneDetectionConfig) -> SceneDetector:
    try:
        import scenedetect  # noqa: F401

        return PySceneDetector(config)
    except ImportError:  # pragma: no cover - scenedetect is a core dependency
        log.warning("PySceneDetect not installed; using fixed-interval fallback (boundaries are NOT real cuts)")
        return FixedIntervalDetector(config.max_scene_duration or 10.0)
