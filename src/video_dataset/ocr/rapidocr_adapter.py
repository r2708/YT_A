"""RapidOCR (ONNX runtime) - lightweight, CPU friendly, bundled models."""

from __future__ import annotations

from pathlib import Path

from video_dataset.ocr.base import RawDetection, bbox_from_points


class RapidOCREngine:
    name = "rapidocr"

    def __init__(self, threads: int | None = None) -> None:
        """``threads`` caps the ONNX intra-op threads of this instance; several instances running on a
        thread pool must share the cores, otherwise they oversubscribe them and get slower."""
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:  # newer package name
            from rapidocr import RapidOCR  # type: ignore

        kwargs = {"intra_op_num_threads": int(threads), "inter_op_num_threads": 1} if threads and threads > 0 else {}
        try:
            self.engine = RapidOCR(**kwargs)
        except TypeError:  # a RapidOCR build without these parameters
            self.engine = RapidOCR()
        self.threads = threads

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
