"""Local Hugging Face VLM analyzer (SmolVLM2, Qwen2-VL / Qwen2.5-VL, LLaVA-OneVision, Idefics3, ...).

Uses AutoModelForImageTextToText + the processor's chat template so most image-text-to-text
checkpoints work without model-specific code. Handles OOM by halving the number of images.
"""

from __future__ import annotations

from pathlib import Path

from video_dataset.config import VisionConfig
from video_dataset.schemas.scene import Clip, Frame, Scene
from video_dataset.schemas.vision import FrameAnalysis, SceneAnalysis, VerificationResult
from video_dataset.utils.device import empty_cache, is_oom_error, resolve_device, torch_dtype
from video_dataset.utils.logging import get_logger
from video_dataset.vision.base import AnalysisContext, VisionAnalyzer, existing_paths, select_evenly
from video_dataset.vision.parsing import (
    extract_json,
    frame_analysis_from_dict,
    scene_analysis_from_dict,
    verification_from_dict,
)
from video_dataset.vision.prompts import SYSTEM_PROMPT, frame_prompt, scene_prompt, verify_prompt

log = get_logger("vision.hf")

_JSON_HINT = (
    "\nRespond with a single JSON object and nothing else. Use these keys: summary, environment{location, setting, weather, "
    "lighting, time_of_day, background}, objects[{name, attributes, location, count}], people{count, actions, clothing, "
    "body_position, interactions}, actions[], camera{shot_type, camera_angle, movement, zoom, stability, is_aerial}, "
    "visual_style{composition, lighting, color, depth_of_field, framing, perspective, motion, transitions}, temporal_progression, confidence."
)


class HFVisionAnalyzer(VisionAnalyzer):
    supports_verification = True
    is_generative = True

    def __init__(self, cfg: VisionConfig, device_pref: str = "auto"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not cfg.model:
            raise ValueError("vision.model must be set for provider 'hf' (e.g. HuggingFaceTB/SmolVLM2-256M-Video-Instruct)")
        self.cfg = cfg
        self.name = "hf"
        self.model = cfg.model
        self.device = resolve_device(device_pref)
        dtype = torch_dtype(self.device, cfg.torch_dtype)
        log.info("loading VLM %s on %s (%s)", cfg.model, self.device, dtype)
        self.processor = AutoProcessor.from_pretrained(cfg.model)  # type: ignore[arg-type]
        self.net = AutoModelForImageTextToText.from_pretrained(cfg.model, dtype=dtype)  # type: ignore[arg-type,misc]
        self.net.to(self.device)  # type: ignore[arg-type]
        self.net.eval()
        self.torch = torch
        self.supports_video = bool(cfg.use_video_input)

    # ------------------------------------------------------------------ generation
    def _load_images(self, image_paths: list[Path]):  # type: ignore[no-untyped-def]
        """Downscale before tokenizing: VLM processors tile large images into many crops (memory!)."""
        from PIL import Image

        out = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            m = max(img.size)
            if self.cfg.image_max_side and m > self.cfg.image_max_side:
                scale = self.cfg.image_max_side / float(m)
                img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
            out.append(img)
        return out

    def _generate(self, prompt: str, image_paths: list[Path], video_path: Path | None = None, max_new_tokens: int | None = None) -> str:
        torch = self.torch
        full_prompt = SYSTEM_PROMPT + "\n\n" + prompt
        if video_path is not None:
            messages = [{"role": "user", "content": [{"type": "video", "path": str(video_path)}, {"type": "text", "text": full_prompt}]}]
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        else:
            images = self._load_images(image_paths)
            messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": full_prompt}]}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        for k, v in list(inputs.items()):
            if hasattr(v, "dtype") and v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                inputs[k] = v.to(self.net.dtype)
        with torch.no_grad():
            out = self.net.generate(  # type: ignore[misc]
                **inputs,
                max_new_tokens=int(max_new_tokens or self.cfg.max_new_tokens),
                do_sample=self.cfg.temperature > 0,
                temperature=max(1e-3, float(self.cfg.temperature)) if self.cfg.temperature > 0 else None,
            )
        prompt_len = inputs["input_ids"].shape[1]
        text_out = self.processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0]
        return text_out.strip()

    def _generate_with_oom_backoff(self, prompt: str, paths: list[Path], video: Path | None = None) -> str:
        paths = list(paths)
        while True:
            try:
                return self._generate(prompt, paths, video)
            except Exception as exc:
                if is_oom_error(exc) and (len(paths) > 1 or video is not None):
                    empty_cache(self.device)
                    if video is not None:
                        log.warning("OOM on video input; falling back to frames")
                        video = None
                        continue
                    paths = select_evenly_paths(paths, max(1, len(paths) // 2))
                    log.warning("OOM; retrying with %d images", len(paths))
                    continue
                raise

    # ------------------------------------------------------------------ interface
    def analyze_scene(self, scene: Scene, frames: list[Frame], context: AnalysisContext) -> SceneAnalysis:
        chosen = select_evenly(frames, int(self.cfg.max_images_per_request))
        paths = existing_paths(chosen)
        if not paths:
            raise FileNotFoundError(f"no frame files for {scene.scene_id}")
        measured = context.measurements.camera_motion_label if context.measurements else None
        prompt = scene_prompt(len(paths), [f.timestamp for f in chosen], scene.duration, context.transcript_text, context.ocr_texts, measured, context.previous_summary) + _JSON_HINT
        video = Path(context.extra["clip_path"]) if (self.supports_video and context.extra.get("clip_path") and Path(context.extra["clip_path"]).exists()) else None
        text = self._generate_with_oom_backoff(prompt, paths, video)
        data = extract_json(text)
        if data is None:
            log.warning("%s returned non-JSON for %s; keeping raw text as summary", self.model, scene.scene_id)
            data = {"summary": text[:1500]}
            analysis = scene_analysis_from_dict(data, scene, frames, self.name, self.model)
            analysis.errors.append("unparseable_json")
        else:
            analysis = scene_analysis_from_dict(data, scene, frames, self.name, self.model)
        analysis.frame_ids = [f.frame_id for f in chosen]
        return analysis

    def analyze_clip(self, clip: Clip, scene: Scene, frames: list[Frame], context: AnalysisContext) -> SceneAnalysis:
        context.extra["clip_path"] = clip.clip_path
        return self.analyze_scene(scene, frames, context)

    def analyze_frame(self, frame: Frame, context: AnalysisContext) -> FrameAnalysis:
        paths = existing_paths([frame])
        if not paths:
            raise FileNotFoundError(frame.frame_path)
        text = self._generate_with_oom_backoff(frame_prompt(frame.timestamp) + "\nRespond with JSON only.", paths)
        data = extract_json(text) or {"caption": text[:500]}
        return frame_analysis_from_dict(data, frame, self.name, self.model)

    def verify(self, claim: str, frames: list[Frame], context: AnalysisContext) -> VerificationResult:
        chosen = select_evenly(frames, int(self.cfg.max_images_per_request))
        paths = existing_paths(chosen)
        if not paths:
            return VerificationResult(claim=claim, verifier=self.name, rationale="no frames")
        try:
            text = self._generate_with_oom_backoff(verify_prompt(claim, [f.timestamp for f in chosen]) + "\nRespond with JSON only.", paths)
        except Exception as exc:
            return VerificationResult(claim=claim, verifier=self.name, rationale=f"verifier error: {exc}"[:200])
        return verification_from_dict(extract_json(text), claim, f"{self.name}:{self.model}", [f.frame_id for f in chosen])

    def close(self) -> None:
        del self.net
        empty_cache(self.device)


def select_evenly_paths(paths: list[Path], k: int) -> list[Path]:
    if k >= len(paths):
        return paths
    step = (len(paths) - 1) / max(1, k - 1)
    return [paths[int(round(i * step))] for i in range(k)] if k > 1 else [paths[len(paths) // 2]]
