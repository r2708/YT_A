"""Caption <-> frames alignment measured with CLIP (no generative model).

For every captioned scene of a video the image embedding of its frames is compared with the text
embedding of *every* scene caption of that video. The scene's own caption should be among the best
matches; when it ranks in the bottom part of the list, the caption describes some other shot better
than this one - a mismatch worth a review. The rank is used rather than an absolute similarity
because CLIP scores are not calibrated across videos and the measurement-only captions of this
pipeline are stylistically uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_dataset.utils.logging import get_logger

log = get_logger("validation.alignment")


@dataclass
class AlignmentResult:
    score: float  # cosine similarity between the scene's frames and its own caption
    rank: int  # 1 = own caption is the best match among the video's captions
    candidates: int
    best_other: str | None  # scene_id whose caption matched these frames best when it was not the own one
    rank_fraction: float  # rank / candidates
    best_score: float = 0.0  # similarity of the best-matching caption
    margin: float = 0.0  # best_score - score (0 when the own caption is the best match)

    def misaligned(self, max_rank_fraction: float, min_margin: float) -> bool:
        """Below the rank cut-off AND beaten by a clear margin (uniform, measurement-style captions
        of one video differ by a few thousandths, which is noise, not a mismatch)."""
        return self.rank_fraction > max_rank_fraction and self.margin >= min_margin


def _as_tensor(feats):  # type: ignore[no-untyped-def]
    return getattr(feats, "pooler_output", feats)


class CLIPCaptionAligner:
    """Wraps a loaded CLIP model (``CLIPZeroShotEnricher`` or anything with ``model``, ``processor``,
    ``device``, ``dtype`` and ``torch``) and scores captions against frame sets."""

    def __init__(self, enricher: Any):
        self.enricher = enricher
        self.torch = enricher.torch
        self.model = enricher.model
        self.processor = enricher.processor
        self.device = enricher.device
        self.dtype = getattr(enricher, "dtype", None)

    def _image_embedding(self, paths: list[Path]):  # type: ignore[no-untyped-def]
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        if self.dtype is not None and self.dtype != self.torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with self.torch.no_grad():
            img = _as_tensor(self.model.get_image_features(**inputs)).float()
        img = img / img.norm(dim=-1, keepdim=True)
        mean = img.mean(dim=0, keepdim=True)
        return mean / mean.norm(dim=-1, keepdim=True)

    def _text_embeddings(self, texts: list[str]):  # type: ignore[no-untyped-def]
        inputs = self.processor(text=[t[:300] for t in texts], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        with self.torch.no_grad():
            txt = _as_tensor(self.model.get_text_features(**inputs)).float()
        return txt / txt.norm(dim=-1, keepdim=True)

    def align(self, frames_by_scene: dict[str, list[Path]], captions: dict[str, str], max_frames: int = 3) -> dict[str, AlignmentResult]:
        """Alignment of every scene that has both frames and a caption."""
        ids = [sid for sid in captions if captions[sid] and frames_by_scene.get(sid)]
        if len(ids) < 2:
            return {}
        txt = self._text_embeddings([captions[sid] for sid in ids])
        out: dict[str, AlignmentResult] = {}
        for i, sid in enumerate(ids):
            paths = [p for p in frames_by_scene[sid] if p.exists()]
            if not paths:
                continue
            step = max(1, len(paths) // max_frames)
            img = self._image_embedding(paths[::step][:max_frames])
            sims = (img @ txt.T)[0].tolist()
            own = sims[i]
            order = sorted(range(len(ids)), key=lambda j: -sims[j])
            rank = order.index(i) + 1
            best_other = ids[order[0]] if order[0] != i else None
            best = float(sims[order[0]])
            out[sid] = AlignmentResult(
                score=round(float(own), 4), rank=rank, candidates=len(ids), best_other=best_other, rank_fraction=round(rank / len(ids), 3),
                best_score=round(best, 4), margin=round(max(0.0, best - float(own)), 4),
            )
        return out


def alignment_text(analysis: Any, mode: str = "prompt") -> str | None:
    """The text CLIP compares with a shot's frames.

    ``prompt``: the natural-language generation prompt built from the annotation ("A 3-second wide
    shot of a car and a person outdoors in daylight, slow pan to the right, dim lighting") - the
    measurement summary is a poor CLIP text (hex colours, numbers, a fixed sentence skeleton).
    Shots without a detected object have nothing verifiable in them and return None, so the check
    is skipped for them rather than judged on lighting words alone. ``summary``: the raw summary."""
    if str(mode or "prompt").lower() != "prompt":
        return analysis.summary or None
    if not analysis.objects:
        return None
    from video_dataset.dataset.prompting import (
        generation_prompt,  # local import: dataset -> validation would be circular
    )

    rec = {
        "start_time": analysis.start_time,
        "end_time": analysis.end_time,
        "environment": analysis.environment.model_dump(mode="json"),
        "objects": [o.name for o in analysis.objects],
        "object_details": [o.model_dump(mode="json") for o in analysis.objects],
        "people": analysis.people.model_dump(mode="json") if analysis.people else None,
        "actions": list(analysis.actions),
        "camera": analysis.camera.model_dump(mode="json"),
        "subject_motion": analysis.subject_motion.model_dump(mode="json") if analysis.subject_motion else None,
        "measurements": analysis.measurements.model_dump(mode="json") if analysis.measurements else {},
    }
    try:
        return generation_prompt(rec) or analysis.summary or None
    except Exception:  # a partial annotation must never break validation
        return analysis.summary or None


def find_loaded_clip(models: dict[str, Any]) -> Any | None:
    """A CLIP enricher already loaded by the vision stage (cv_models analyzer or the `clip` enricher)."""
    for key, obj in models.items():
        if key.startswith("vision:") and getattr(obj, "clip", None) is not None:
            return obj.clip
        if key.startswith("enrichers:"):
            for enr in obj or []:
                if getattr(enr, "name", "") == "clip":
                    return enr
    return None
