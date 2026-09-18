"""DOWNLOAD stage: fetch the video, write metadata.json + download.json, register hash for duplicate detection."""

from __future__ import annotations

from video_dataset.downloader.local import LocalFileSource
from video_dataset.downloader.youtube import YouTubeDownloader
from video_dataset.pipeline.context import StageOutput, VideoContext
from video_dataset.schemas.video import DownloadRecord, DownloadStatus
from video_dataset.stages import Stage
from video_dataset.utils.io import file_sha256, write_json_atomic


class DownloadFailed(RuntimeError):
    pass


def download_stage(ctx: VideoContext) -> StageOutput:
    log = ctx.logger(Stage.DOWNLOAD)
    item = ctx.input_item
    dest_dir = ctx.paths.video_dir(ctx.video_id)
    meta_path = ctx.paths.metadata_file(ctx.video_id)

    if item is None:
        # Re-run on an already registered video: reuse what is on disk.
        if meta_path.exists() and ctx.paths.source_candidates(ctx.video_id):
            return StageOutput(artifact_path=str(meta_path), message="existing download reused", metrics={"status": "existing"})
        raise DownloadFailed("no input item and no existing download on disk")

    source = LocalFileSource() if item.kind == "local" else YouTubeDownloader(ctx.config.download, ctx.config.project.ffmpeg_path)
    result = source.fetch(item, ctx.video_id, dest_dir)

    record = DownloadRecord(
        video_id=ctx.video_id,
        url=item.url,
        status=result.status,
        video_path=str(result.video_path) if result.video_path else None,
        metadata_path=str(meta_path) if result.metadata else None,
        error=result.error,
        attempts=result.attempts,
    )

    if result.status in (DownloadStatus.FAILED, DownloadStatus.UNAVAILABLE) or result.video_path is None:
        write_json_atomic(ctx.paths.download_record_file(ctx.video_id), record)
        raise DownloadFailed(f"{result.status}: {result.error}")

    assert result.metadata is not None
    sha = file_sha256(result.video_path)
    record.sha256 = sha
    dup_owner = ctx.db.register_file_hash(sha, ctx.video_id)
    if dup_owner:
        log.warning("video file is byte-identical to %s (duplicate video)", dup_owner)
    write_json_atomic(meta_path, result.metadata)
    write_json_atomic(ctx.paths.download_record_file(ctx.video_id), record)
    ctx.db.update_video_meta(ctx.video_id, title=result.metadata.title, duration=result.metadata.duration)
    ctx.invalidate()

    msg = "already downloaded" if result.status == DownloadStatus.SKIPPED_EXISTING else f"{result.video_path.name} ({result.video_path.stat().st_size / 1e6:.1f} MB)"
    return StageOutput(
        artifact_path=str(meta_path),
        metrics={"status": result.status, "size_bytes": result.video_path.stat().st_size, "duplicate_of": dup_owner},
        message=msg,
    )
