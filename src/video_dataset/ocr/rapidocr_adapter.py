"""RapidOCR (ONNX runtime) - lightweight, CPU friendly, bundled models."""

from __future__ import annotations

from pathlib import Path

from video_dataset.ocr.base import RawDetection, bbox_from_points


class RapidOCREngine:
    name = "rapidocr"

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:  # newer package name
            from rapidocr import RapidOCR  # type: ignore

        self.engine = RapidOCR()

    def detect(self, image_path: Path) -> list[RawDetection]:
        result, _elapse = self.engine(str(image_path))
        out: list[RawDetection] = []
        if not result:
            return out
        for item in result:
            try:
                box, text, score = item[0], str(item[1]), float(item[2])
            except (IndexError, TypeError, ValueError):
                continue
            if not text.strip():
                continue
            out.append(RawDetection(text=text.strip(), confidence=round(max(0.0, min(1.0, score)), 4), bbox=bbox_from_points(box)))
        return out
