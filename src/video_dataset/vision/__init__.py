"""Multimodal scene analysis: a swappable VisionAnalyzer interface with heuristic, local HF and API backends."""

from video_dataset.vision.base import AnalysisContext, VisionAnalyzer, create_vision_analyzer
from video_dataset.vision.heuristic import HeuristicVisionAnalyzer

__all__ = ["AnalysisContext", "HeuristicVisionAnalyzer", "VisionAnalyzer", "create_vision_analyzer"]
