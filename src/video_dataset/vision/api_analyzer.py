"""Vision analyzer backed by an API/LLM client (Anthropic Claude or any OpenAI-compatible VLM server)."""

from __future__ import annotations

from video_dataset.config import VisionConfig
from video_dataset.llm.base import LLMClient
from video_dataset.schemas.scene import Frame, Scene
from video_dataset.schemas.vision import FrameAnalysis, SceneAnalysis, VerificationResult
from video_dataset.utils.logging import get_logger
from video_dataset.utils.retry import retry_call
from video_dataset.vision.base import AnalysisContext, VisionAnalyzer, existing_paths, select_evenly
from video_dataset.vision.parsing import (
    extract_json,
    frame_analysis_from_dict,
    scene_analysis_from_dict,
    verification_from_dict,
)
from video_dataset.vision.prompts import (
    FRAME_SCHEMA,
    SCENE_SCHEMA,
    SYSTEM_PROMPT,
    VERIFY_SCHEMA,
    frame_prompt,
    scene_prompt,
    verify_prompt,
)

log = get_logger("vision.api")


class APIVisionAnalyzer(VisionAnalyzer):
    supports_verification = True
    is_generative = True

    def __init__(self, client: LLMClient, cfg: VisionConfig):
        self.client = client
        self.cfg = cfg
        self.name = f"api:{client.name}"
        self.model = client.model

    def _call(self, prompt: str, images, schema) -> dict | None:  # type: ignore[no-untyped-def]
        def once() -> dict | None:
            resp = self.client.complete(
                prompt, system=SYSTEM_PROMPT, images=images, json_schema=schema,
                max_tokens=int(self.cfg.max_new_tokens), temperature=float(self.cfg.temperature),
            )
            if resp.refusal:
                raise RuntimeError("model refused the request")
            data = extract_json(resp.text)
            if data is None:
                raise ValueError(f"model output was not JSON: {resp.text[:200]!r}")
            return data

        return retry_call(once, retries=int(self.cfg.retries), base_delay=1.5, label=f"{self.name} call")

    def analyze_scene(self, scene: Scene, frames: list[Frame], context: AnalysisContext) -> SceneAnalysis:
        chosen = select_evenly(frames, int(self.cfg.max_images_per_request))
        paths = existing_paths(chosen)
        if not paths:
            raise FileNotFoundError(f"no frame files for {scene.scene_id}")
        measured = context.measurements.camera_motion_label if context.measurements else None
        prompt = scene_prompt(len(paths), [f.timestamp for f in chosen], scene.duration, context.transcript_text, context.ocr_texts, measured, context.previous_summary)
        data = self._call(prompt, paths, SCENE_SCHEMA)
        analysis = scene_analysis_from_dict(data or {}, scene, frames, self.name, self.model)
        analysis.frame_ids = [f.frame_id for f in chosen]
        return analysis

    def analyze_frame(self, frame: Frame, context: AnalysisContext) -> FrameAnalysis:
        paths = existing_paths([frame])
        if not paths:
            raise FileNotFoundError(frame.frame_path)
        data = self._call(frame_prompt(frame.timestamp), paths, FRAME_SCHEMA)
        return frame_analysis_from_dict(data or {}, frame, self.name, self.model)

    def verify(self, claim: str, frames: list[Frame], context: AnalysisContext) -> VerificationResult:
        chosen = select_evenly(frames, int(self.cfg.max_images_per_request))
        paths = existing_paths(chosen)
        if not paths:
            return VerificationResult(claim=claim, verifier=self.name, rationale="no frames")
        try:
            data = self._call(verify_prompt(claim, [f.timestamp for f in chosen]), paths, VERIFY_SCHEMA)
        except Exception as exc:
            log.warning("verification failed: %s", exc)
            return VerificationResult(claim=claim, verifier=self.name, rationale=f"verifier error: {exc}"[:200])
        return verification_from_dict(data, claim, self.name, [f.frame_id for f in chosen])
