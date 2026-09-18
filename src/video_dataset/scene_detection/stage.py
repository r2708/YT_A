"""SCENE_DETECTION stage."""

from __future__ import annotations

from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.scene_detection.detector import create_scene_detector
from video_dataset.stages import Stage
from video_dataset.utils.io import write_json_atomic


def scene_detection_stage(ctx: VideoContext) -> StageOutput:
    detector = create_scene_detector(ctx.config.scene_detection)
    result = detector.detect(ctx.video_path, ctx.media_info, ctx.video_id)
    out = ctx.paths.scenes_file(ctx.video_id)
    write_json_atomic(out, result)
    n_split = sum(1 for s in result.scenes if s.transition_in == "split")
    n_fade = sum(1 for s in result.scenes if s.transition_in == "fade")
    ctx.logger(Stage.SCENE_DETECTION).info("%d scenes (%d fades, %d artificial splits)", len(result.scenes), n_fade, n_split)
    return StageOutput(
        artifact_path=str(out),
        metrics={"scenes": len(result.scenes), "fades": n_fade, "splits": n_split, "detector": result.detector},
        message=f"{len(result.scenes)} scenes detected",
    )
