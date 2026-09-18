"""PaddleOCR adapter."""

from __future__ import annotations

from pathlib import Path

from video_dataset.ocr.base import RawDetection, bbox_from_points


class PaddleOCREngine:
    name = "paddleocr"

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)

    def detect(self, image_path: Path) -> list[RawDetection]:
        out: list[RawDetection] = []
        result = self.ocr.ocr(str(image_path), cls=True) or []
        for page in result:
            for item in page or []:
                try:
                    box, (text, conf) = item[0], item[1]
                except (TypeError, ValueError):
                    continue
                if str(text).strip():
                    out.append(RawDetection(text=str(text).strip(), confidence=round(float(conf), 4), bbox=bbox_from_points(box)))
        return out
