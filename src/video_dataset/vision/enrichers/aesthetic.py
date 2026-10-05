"""LAION aesthetic predictor: a linear head on the L2-normalised CLIP image embedding.

Scores are on the LAION scale (~1 poor .. 10 excellent). The head is a few kilobytes and runs on
the embeddings CLIP already produced for the zero-shot attributes, so it costs nothing extra.
Weights: https://github.com/LAION-AI/aesthetic-predictor (sa_0_4_*_linear.pth).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from video_dataset.config import VisionConfig
from video_dataset.utils.logging import get_logger

log = get_logger("vision.aesthetic")

_BASE = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/"
HEADS: dict[str, tuple[str, str, int]] = {  # clip model -> (name, url, embedding dim)
    "openai/clip-vit-base-patch32": ("laion/sa_0_4_vit_b_32_linear", _BASE + "sa_0_4_vit_b_32_linear.pth", 512),
    "openai/clip-vit-base-patch16": ("laion/sa_0_4_vit_b_16_linear", _BASE + "sa_0_4_vit_b_16_linear.pth", 512),
    "openai/clip-vit-large-patch14": ("laion/sa_0_4_vit_l_14_linear", _BASE + "sa_0_4_vit_l_14_linear.pth", 768),
}


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "video_dataset" / "aesthetic"


class AestheticHead:
    def __init__(self, weight: np.ndarray, bias: float, name: str):
        w = np.asarray(weight, dtype=np.float32).reshape(-1)
        self.weight = w
        self.bias = float(bias)
        self.name = name
        self.dim = int(w.shape[0])

    @classmethod
    def from_state_dict(cls, sd: dict[str, Any], name: str) -> AestheticHead:
        w = sd["weight"]
        b = sd["bias"]
        w = w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
        b = b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b)
        return cls(w, float(np.asarray(b).reshape(-1)[0]), name)

    @classmethod
    def load(cls, clip_model: str, weights: str | None = None, cache_dir: str | None = None) -> AestheticHead:
        """Load the head for ``clip_model`` from ``weights`` (path or URL) or the known LAION file."""
        name = f"custom:{Path(weights).name}" if weights else None
        url_or_path = weights
        if not url_or_path:
            if clip_model not in HEADS:
                raise ValueError(f"no aesthetic head known for CLIP model '{clip_model}'; set vision.aesthetic_weights")
            name, url_or_path, _dim = HEADS[clip_model]
        path = Path(url_or_path)
        if str(url_or_path).startswith(("http://", "https://")):
            cache = Path(cache_dir) if cache_dir else default_cache_dir()
            cache.mkdir(parents=True, exist_ok=True)
            path = cache / Path(str(url_or_path)).name
            if not path.exists():
                log.info("downloading aesthetic head %s -> %s", url_or_path, path)
                tmp = path.with_suffix(".tmp")
                urllib.request.urlretrieve(str(url_or_path), tmp)
                tmp.replace(path)
        if not path.exists():
            raise FileNotFoundError(f"aesthetic weights not found: {path}")
        import torch

        sd = torch.load(path, map_location="cpu")
        return cls.from_state_dict(sd, name or path.stem)

    def score(self, embeddings: np.ndarray) -> list[float]:
        """Scores for a (N, D) array of CLIP image embeddings (normalised here)."""
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.dim:
            raise ValueError(f"embedding dim {x.shape[1]} does not match the aesthetic head ({self.dim})")
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        return [round(float(v), 3) for v in (x @ self.weight + self.bias)]

    def summarize(self, embeddings: np.ndarray) -> dict[str, Any]:
        scores = self.score(embeddings)
        return {
            "model": self.name,
            "mean": round(float(np.mean(scores)), 3),
            "min": round(float(min(scores)), 3),
            "max": round(float(max(scores)), 3),
            "per_frame": scores,
        }


def head_from_config(cfg: VisionConfig) -> AestheticHead | None:
    """The configured head, or None (with a warning) when it is disabled or cannot be loaded."""
    if not getattr(cfg, "aesthetic", False):
        return None
    try:
        head = AestheticHead.load(cfg.clip_model, cfg.aesthetic_weights, cfg.aesthetic_cache_dir)
        log.info("aesthetic head %s loaded (dim %d)", head.name, head.dim)
        return head
    except Exception as exc:
        log.warning("aesthetic scoring disabled: %s", str(exc)[:200])
        return None
