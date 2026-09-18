"""Image encoding for API calls (resize to bound token cost, JPEG re-encode, base64)."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2


def encode_image_base64(path: Path, max_side: int = 1024, quality: int = 88) -> tuple[str, str]:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read image {path}")
    h, w = img.shape[:2]
    m = max(h, w)
    if max_side and m > max_side:
        scale = max_side / float(m)
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"cannot encode image {path}")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii"), "image/jpeg"
