from __future__ import annotations

from typing import Any, Protocol

from video_dataset.config import VisionConfig
from video_dataset.schemas.scene import Frame, Scene
from video_dataset.utils.logging import get_logger

log = get_logger("vision.enrichers")


class Enricher(Protocol):
    name: str

    def enrich(self, scene: Scene, frames: list[Frame]) -> dict[str, Any]: ...


def create_enrichers(cfg: VisionConfig, device: str) -> list[Enricher]:
    out: list[Enricher] = []
    for name in cfg.enrichers or []:
        if name == "clip":
            try:
                from video_dataset.vision.enrichers.clip import CLIPZeroShotEnricher

                out.append(CLIPZeroShotEnricher(cfg.clip_model, device))
            except Exception as exc:
                log.warning("CLIP enricher unavailable: %s", exc)
        else:
            log.warning("unknown enricher '%s'", name)
    return out
