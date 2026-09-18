"""OCR engines and cross-frame text merging."""

from video_dataset.ocr.base import OCREngine, RawDetection, create_ocr_engine
from video_dataset.ocr.merge import merge_detections

__all__ = ["OCREngine", "RawDetection", "create_ocr_engine", "merge_detections"]
