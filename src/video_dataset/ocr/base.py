from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from video_dataset.config import OCRConfig
from video_dataset.utils.logging import get_logger

log = get_logger("ocr")


@dataclass
class RawDetection:
    text: str
    confidence: float | None
    bbox: list[float] = field(default_factory=list)  # x_min, y_min, x_max, y_max


def bbox_from_points(points: list[list[float]] | list[tuple[float, float]]) -> list[float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


class OCREngine(Protocol):
    name: str

    def detect(self, image_path: Path) -> list[RawDetection]: ...


def create_ocr_engine(cfg: OCRConfig, device: str = "cpu") -> OCREngine | None:
    provider = (cfg.provider or "none").lower()
    if not cfg.enabled or provider == "none":
        return None
    if provider == "mock":
        from video_dataset.ocr.mock import MockOCREngine

        return MockOCREngine()
    candidates = [provider, "rapidocr", "easyocr", "paddleocr"]
    tried: set[str] = set()
    for name in candidates:
        if name in tried:
            continue
        tried.add(name)
        try:
            if name == "rapidocr":
                from video_dataset.ocr.rapidocr_adapter import RapidOCREngine

                return RapidOCREngine()
            if name == "easyocr":
                from video_dataset.ocr.easyocr_adapter import EasyOCREngine

                return EasyOCREngine(cfg.languages, gpu=device != "cpu")
            if name == "paddleocr":
                from video_dataset.ocr.paddleocr_adapter import PaddleOCREngine

                return PaddleOCREngine(cfg.languages[0] if cfg.languages else "en")
        except ImportError as exc:
            log.warning("OCR engine '%s' not installed (%s)", name, exc)
        except Exception as exc:
            log.warning("OCR engine '%s' failed to initialize: %s", name, exc)
    log.warning("no OCR engine available; OCR disabled")
    return None
