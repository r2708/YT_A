"""CLIP zero-shot attribute scoring. Produces real softmax probabilities over small label sets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.schemas.scene import Frame, Scene
from video_dataset.schemas.vision import CameraAngle, Measurements, SceneAnalysis, Setting, ShotType
from video_dataset.utils.device import torch_dtype
from video_dataset.vision.base import select_evenly
from video_dataset.vision.enrichers.aesthetic import AestheticHead

LABEL_SETS: dict[str, dict[str, str]] = {
    "setting": {"indoor": "a photo taken indoors", "outdoor": "a photo taken outdoors"},
    "time_of_day": {"day": "a photo taken during the day", "night": "a photo taken at night"},
    "shot_type": {
        "wide": "a wide shot showing a whole environment",
        "medium": "a medium shot of a subject",
        "close_up": "a close-up shot of a subject",
    },
    "people_present": {"yes": "a photo with people in it", "no": "a photo with no people in it"},
    "camera_angle": {"aerial": "an aerial drone photo from high above", "ground": "a photo taken from ground level"},
}


def _as_tensor(feats):  # type: ignore[no-untyped-def]
    """transformers>=5 returns BaseModelOutputWithPooling from get_*_features; 4.x returns a tensor."""
    return getattr(feats, "pooler_output", feats)


class CLIPZeroShotEnricher:
    name = "clip"

    def __init__(self, model_name: str, device: str = "cpu", max_frames: int = 3, aesthetic: AestheticHead | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.aesthetic = aesthetic
        self.dtype = torch_dtype(device) if device != "cpu" else torch.float32
        self.model = CLIPModel.from_pretrained(model_name, dtype=self.dtype).to(device).eval()  # type: ignore[arg-type]
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.max_frames = max_frames
        self.model_name = model_name
        self._text_cache: dict[str, Any] = {}

    def _text_features(self, key: str, prompts: list[str]):  # type: ignore[no-untyped-def]
        if key not in self._text_cache:
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                feats = _as_tensor(self.model.get_text_features(**inputs))
            self._text_cache[key] = feats / feats.norm(dim=-1, keepdim=True)
        return self._text_cache[key]

    def enrich(self, scene: Scene, frames: list[Frame]) -> dict[str, Any]:
        from PIL import Image

        chosen = [Path(f.frame_path) for f in select_evenly(frames, self.max_frames) if Path(f.frame_path).exists()]
        if not chosen:
            return {}
        images = [Image.open(p).convert("RGB") for p in chosen]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        if self.dtype != self.torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with self.torch.no_grad():
            img = _as_tensor(self.model.get_image_features(**inputs))
        per_frame = img / img.norm(dim=-1, keepdim=True)
        img = per_frame.mean(dim=0, keepdim=True)
        img = img / img.norm(dim=-1, keepdim=True)
        out: dict[str, Any] = {"model": self.model_name, "frames": len(chosen), "attributes": {}}
        if self.aesthetic is not None:
            try:
                out["aesthetic"] = self.aesthetic.summarize(per_frame.float().cpu().numpy())
            except Exception as exc:  # pragma: no cover - head / model mismatch
                out["aesthetic_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        for key, labels in LABEL_SETS.items():
            txt = self._text_features(key, list(labels.values()))
            probs = (100.0 * img.float() @ txt.float().T).softmax(dim=-1)[0].tolist()
            names = list(labels.keys())
            best = max(range(len(names)), key=lambda i: probs[i])
            out["attributes"][key] = {"label": names[best], "score": round(float(probs[best]), 4), "scores": {n: round(float(p), 4) for n, p in zip(names, probs)}}
        return out


def _confident(enrichment: dict[str, Any] | None, key: str, min_score: float) -> str | None:
    attrs = (enrichment or {}).get("attributes") or {}
    item = attrs.get(key)
    if item and float(item.get("score", 0.0)) >= min_score:
        return str(item["label"])
    return None


def apply_clip_attributes(analysis: SceneAnalysis, enrichment: dict[str, Any] | None, min_score: float = 0.75) -> None:
    """Fill fields the analyzer left unknown with CLIP labels that clear ``min_score``. Never overrides
    a value the analyzer already reported."""
    setting = _confident(enrichment, "setting", min_score)
    if setting and analysis.environment.setting == Setting.UNKNOWN:
        analysis.environment.setting = Setting(setting)
    tod = _confident(enrichment, "time_of_day", min_score)
    if tod and not analysis.environment.time_of_day:
        analysis.environment.time_of_day = tod
    shot = _confident(enrichment, "shot_type", min_score)
    if shot and analysis.camera.shot_type == ShotType.UNKNOWN:
        analysis.camera.shot_type = ShotType(shot)
    angle = _confident(enrichment, "camera_angle", min_score)
    if angle == "aerial" and analysis.camera.camera_angle == CameraAngle.UNKNOWN:
        analysis.camera.camera_angle = CameraAngle.AERIAL
        if analysis.camera.is_aerial is None:
            analysis.camera.is_aerial = True


def apply_clip_aesthetic(analysis: SceneAnalysis, enrichment: dict[str, Any] | None) -> None:
    """Copy the aesthetic score of the CLIP enrichment into the scene measurements."""
    aes = (enrichment or {}).get("aesthetic") or {}
    if aes.get("mean") is None:
        return
    update = {"aesthetic_score": float(aes["mean"]), "aesthetic_model": aes.get("model")}
    analysis.measurements = analysis.measurements.model_copy(update=update) if analysis.measurements is not None else Measurements(**update)


def clip_phrase(enrichment: dict[str, Any] | None, min_score: float = 0.75) -> str:
    """One factual sentence from the confident CLIP labels, or '' when none clears ``min_score``."""
    words: list[str] = []
    setting = _confident(enrichment, "setting", min_score)
    if setting:
        words.append("indoors" if setting == "indoor" else "outdoors")
    tod = _confident(enrichment, "time_of_day", min_score)
    if tod:
        words.append("during the day" if tod == "day" else "at night")
    shot = _confident(enrichment, "shot_type", min_score)
    if shot:
        words.append(f"framed as a {shot.replace('_', '-')} shot")
    if _confident(enrichment, "camera_angle", min_score) == "aerial":
        words.append("from an aerial viewpoint")
    if not words:
        return ""
    return "CLIP classifies the shot as " + ", ".join(words) + "."
