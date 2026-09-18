"""Intelligent frame sampling: low-res scan pass -> change/motion-aware selection -> full-res extraction."""

from video_dataset.frame_sampling.extractor import FrameExtractor
from video_dataset.frame_sampling.sampler import Candidate, select_frames
from video_dataset.frame_sampling.scan import scan_video

__all__ = ["Candidate", "FrameExtractor", "scan_video", "select_frames"]
