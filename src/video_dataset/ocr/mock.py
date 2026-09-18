"""Mock OCR engine for tests: returns detections registered per image path (or nothing)."""

from __future__ import annotations

from pathlib import Path

from video_dataset.ocr.base import RawDetection


class MockOCREngine:
    name = "mock"

    def __init__(self, detections: dict[str, list[RawDetection]] | None = None):
        self.detections = detections or {}

    def detect(self, image_path: Path) -> list[RawDetection]:
        return list(self.detections.get(str(image_path), self.detections.get(image_path.name, [])))
