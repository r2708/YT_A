"""Where every artifact lives on disk. One place to change the layout."""

from __future__ import annotations

from pathlib import Path


class DataPaths:
    SUBDIRS = (
        "input",
        "downloads",
        "audio",
        "scenes",
        "frames",
        "clips",
        "transcripts",
        "ocr",
        "annotations",
        "qa",
        "validated",
        "final",
        "logs",
    )

    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)

    def ensure_all(self) -> None:
        for sub in self.SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # --- top-level dirs ---
    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def downloads_dir(self) -> Path:
        return self.root / "downloads"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def scenes_dir(self) -> Path:
        return self.root / "scenes"

    @property
    def frames_root(self) -> Path:
        return self.root / "frames"

    @property
    def clips_root(self) -> Path:
        return self.root / "clips"

    @property
    def transcripts_dir(self) -> Path:
        return self.root / "transcripts"

    @property
    def ocr_dir(self) -> Path:
        return self.root / "ocr"

    @property
    def annotations_root(self) -> Path:
        return self.root / "annotations"

    @property
    def qa_dir(self) -> Path:
        return self.root / "qa"

    @property
    def validated_dir(self) -> Path:
        return self.root / "validated"

    @property
    def final_dir(self) -> Path:
        return self.root / "final"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    # --- per-video artifacts ---
    def video_dir(self, video_id: str) -> Path:
        return self.downloads_dir / video_id

    def video_file(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "video.mp4"

    def metadata_file(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "metadata.json"

    def download_record_file(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "download.json"

    def media_info_file(self, video_id: str) -> Path:
        return self.video_dir(video_id) / "media_info.json"

    def audio_file(self, video_id: str) -> Path:
        return self.audio_dir / f"{video_id}.wav"

    def audio_events_file(self, video_id: str) -> Path:
        return self.audio_dir / f"{video_id}.events.json"

    def scenes_file(self, video_id: str) -> Path:
        return self.scenes_dir / f"{video_id}.json"

    def frames_dir(self, video_id: str) -> Path:
        return self.frames_root / video_id

    def frames_file(self, video_id: str) -> Path:
        return self.frames_dir(video_id) / "frames.json"

    def clips_dir(self, video_id: str) -> Path:
        return self.clips_root / video_id

    def transcript_file(self, video_id: str) -> Path:
        return self.transcripts_dir / f"{video_id}.json"

    def ocr_file(self, video_id: str) -> Path:
        return self.ocr_dir / f"{video_id}.json"

    def annotations_dir(self, video_id: str) -> Path:
        return self.annotations_root / video_id

    def vision_file(self, video_id: str) -> Path:
        return self.annotations_dir(video_id) / "scenes.json"

    def timeline_file(self, video_id: str) -> Path:
        return self.annotations_dir(video_id) / "timeline.json"

    def qa_file(self, video_id: str) -> Path:
        return self.qa_dir / f"{video_id}.json"

    def validated_file(self, video_id: str) -> Path:
        return self.validated_dir / f"{video_id}.json"

    def video_log_file(self, video_id: str) -> Path:
        return self.logs_dir / f"{video_id}.log"

    def source_candidates(self, video_id: str) -> list[Path]:
        d = self.video_dir(video_id)
        if not d.exists():
            return []
        return sorted(p for p in d.glob("source.*") if p.suffix.lower() not in {".part", ".json", ".ytdl"})
