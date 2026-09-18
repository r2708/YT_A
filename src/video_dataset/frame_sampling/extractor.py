"""Extract chosen frames at full resolution with sequential seeking (never loads the video into RAM)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video_dataset.frame_sampling.sampler import Candidate
from video_dataset.schemas.scene import Frame, Scene
from video_dataset.utils.ffmpeg import run_ffmpeg
from video_dataset.utils.ids import frame_id as make_frame_id
from video_dataset.utils.logging import get_logger

log = get_logger("frame_sampling.extract")


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if max_side <= 0 or m <= max_side:
        return img
    scale = max_side / float(m)
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


class FrameExtractor:
    def __init__(self, video_path: Path, fps: float, max_side: int = 1280, jpeg_quality: int = 90, ffmpeg_path: str | None = None):
        self.video_path = video_path
        self.fps = fps
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self.ffmpeg_path = ffmpeg_path

    def _ffmpeg_fallback(self, timestamp: float, out: Path) -> bool:
        try:
            run_ffmpeg(["-ss", f"{timestamp:.3f}", "-i", str(self.video_path), "-frames:v", "1", "-q:v", "2", str(out)], ffmpeg_path=self.ffmpeg_path)
            return out.exists() and out.stat().st_size > 0
        except Exception as exc:
            log.warning("ffmpeg frame fallback failed at %.3fs: %s", timestamp, exc)
            return False

    def extract(self, video_id: str, plan: list[tuple[Scene, list[Candidate]]], out_root: Path) -> list[Frame]:
        requests: list[tuple[Scene, Candidate, int]] = []
        for scene, cands in plan:
            for k, c in enumerate(cands):
                requests.append((scene, c, k))
        requests.sort(key=lambda r: (r[1].frame_index, r[1].timestamp))

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.video_path}")
        frames: list[Frame] = []
        pos = 0  # index of the next frame cap.read() would return
        try:
            for scene, cand, k in requests:
                target = cand.frame_index
                if target < pos or target - pos > 120:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if pos < 0 or abs(pos - target) > 5:
                        pos = target
                while pos < target:
                    if not cap.grab():
                        break
                    pos += 1
                ok, img = cap.read()
                pos += 1
                scene_dir = out_root / scene.scene_id
                scene_dir.mkdir(parents=True, exist_ok=True)
                fid = make_frame_id(scene.index, k)
                out = scene_dir / f"{fid}_t{cand.timestamp:09.3f}.jpg"
                if not ok or img is None:
                    if not self._ffmpeg_fallback(cand.timestamp, out):
                        log.warning("could not extract frame %s at %.3fs", fid, cand.timestamp)
                        continue
                    img = cv2.imread(str(out))  # type: ignore[assignment]
                    if img is None:
                        continue
                    img = resize_max_side(img, self.max_side)
                    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                else:
                    img = resize_max_side(img, self.max_side)
                    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                h, w = img.shape[:2]
                frames.append(
                    Frame(
                        frame_id=fid,
                        video_id=video_id,
                        scene_id=scene.scene_id,
                        timestamp=cand.timestamp,
                        frame_index=cand.frame_index,
                        frame_path=str(out),
                        width=w,
                        height=h,
                        sampling_reason=cand.reason,
                        change_score=cand.change,
                        motion_score=cand.motion,
                    )
                )
        finally:
            cap.release()
        frames.sort(key=lambda f: (f.timestamp, f.frame_id))
        return frames
