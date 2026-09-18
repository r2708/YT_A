"""CLIP zero-shot attribute scoring. Produces real softmax probabilities over small label sets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_dataset.schemas.scene import Frame, Scene
from video_dataset.utils.device import torch_dtype
from video_dataset.vision.base import select_evenly

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


class CLIPZeroShotEnricher:
    name = "clip"

    def __init__(self, model_name: str, device: str = "cpu", max_frames: int = 3):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
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
                feats = self.model.get_text_features(**inputs)
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
            img = self.model.get_image_features(**inputs)
        img = (img / img.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
        img = img / img.norm(dim=-1, keepdim=True)
        out: dict[str, Any] = {"model": self.model_name, "frames": len(chosen), "attributes": {}}
        for key, labels in LABEL_SETS.items():
            txt = self._text_features(key, list(labels.values()))
            probs = (100.0 * img.float() @ txt.float().T).softmax(dim=-1)[0].tolist()
            names = list(labels.keys())
            best = max(range(len(names)), key=lambda i: probs[i])
            out["attributes"][key] = {"label": names[best], "score": round(float(probs[best]), 4), "scores": {n: round(float(p), 4) for n, p in zip(names, probs)}}
        return out
