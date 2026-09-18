"""Optional grounded enrichers that add measured signals to scene analyses (e.g. CLIP zero-shot)."""

from video_dataset.vision.enrichers.base import Enricher, create_enrichers

__all__ = ["Enricher", "create_enrichers"]
