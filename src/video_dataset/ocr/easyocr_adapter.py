"""EasyOCR adapter (PyTorch based)."""

from __future__ import annotations

from pathlib import Path

from video_dataset.ocr.base import RawDetection, bbox_from_points


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, languages: list[str], gpu: bool = False):
        import easyocr

        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)

    def detect(self, image_path: Path) -> list[RawDetection]:
        out: list[RawDetection] = []
        for box, text, conf in self.reader.readtext(str(image_path)):
            if str(text).strip():
                out.append(RawDetection(text=str(text).strip(), confidence=round(float(conf), 4), bbox=bbox_from_points(box)))
        return out
