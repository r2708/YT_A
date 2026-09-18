from pathlib import Path

import pandas as pd

from video_dataset.dataset.export import (
    DATASET_FILES,
    normalize_record,
    write_combined_jsonl,
    write_jsonl_files,
    write_parquet,
)
from video_dataset.dataset.stats import compute_statistics, format_statistics
from video_dataset.schemas.common import ConfidenceSource
from video_dataset.schemas.dataset import EventRecord
from video_dataset.schemas.qa import Evidence, QARecord, QAType
from video_dataset.schemas.quality import ValidationInfo, ValidationStatus
from video_dataset.utils.io import read_jsonl


def test_jsonl_and_parquet_export(tmp_path: Path):
    qa = QARecord(question_id="qa_v_000001", video_id="vid_v", type=QAType.DURATION, question="How long?", answer="About 6 seconds.", evidence=Evidence(start_time=1.0, end_time=7.0, event_ids=["event_0001"], scene_ids=["scene_001"]), confidence=0.8, confidence_source=ConfidenceSource.DERIVED_MIN, validation=ValidationInfo(status=ValidationStatus.ACCEPTED))
    ev = EventRecord(record_id="event_v_event_0001", video_id="vid_v", event_id="event_0001", event_type="action", start_time=1.0, end_time=7.0, event="A car drives.", source="vision", confidence=0.8, relations=[{"relation": "BEFORE", "event_b": "event_0002"}])
    records = {"temporal_qa": [qa], "events": [ev]}
    counts = write_jsonl_files(records, tmp_path)
    assert counts["temporal_qa"] == 1 and counts["events"] == 1 and counts["frames"] == 0
    assert all((tmp_path / f"{n}.jsonl").exists() for n in DATASET_FILES)
    rows = list(read_jsonl(tmp_path / "temporal_qa.jsonl"))
    assert rows[0]["question_id"] == "qa_v_000001" and rows[0]["evidence"]["event_ids"] == ["event_0001"]
    n = write_parquet(records, tmp_path / "dataset.parquet")
    df = pd.read_parquet(tmp_path / "dataset.parquet")
    assert n == 2 and len(df) == 2
    qa_row = df[df.record_type == "temporal_qa"].iloc[0]
    assert qa_row["answer"] == "About 6 seconds." and qa_row["start_time"] == 1.0 and qa_row["validation_status"] == "accepted"
    ev_row = df[df.record_type == "event"].iloc[0]
    assert ev_row["text"] == "A car drives." and "relations" in ev_row["payload"]


def test_normalize_record_frame_uses_timestamp():
    rec = {"record_id": "frame_x", "video_id": "v", "scene_id": "scene_001", "timestamp": 3.5, "caption": "hi", "frame_path": "f.jpg"}
    n = normalize_record("frame", rec)
    assert n["start_time"] == 3.5 and n["end_time"] == 3.5 and n["media_path"] == "f.jpg" and n["text"] == "hi"


def test_statistics():
    stats = compute_statistics(
        [{"video_id": "a", "duration": 120.0, "status": "done"}],
        {"temporal_qa": [{"type": "duration", "difficulty": "easy"}], "long_video_qa": [], "events": [{"event_type": "action", "relations": [1, 2]}], "scenes": [], "frames": [], "clips": [], "video_descriptions": []},
        [{"qa": {"accepted": 1, "review": 0, "rejected": 0, "duplicate": 0}}],
    )
    assert stats["videos"] == 1 and stats["temporal_relations"] == 2 and stats["qa_by_type"] == {"duration": 1}
    text = format_statistics(stats)
    assert "Temporal QA" in text and "Accepted" in text


def test_combined_jsonl_contains_every_record_with_type(tmp_path: Path):
    qa = QARecord(question_id="qa_v_000001", video_id="vid_v", type=QAType.DURATION, question="How long?", answer="About 6 seconds.", evidence=Evidence(start_time=1.0, end_time=7.0, event_ids=["event_0001"]))
    ev = EventRecord(record_id="event_v_event_0001", video_id="vid_v", event_id="event_0001", event_type="action", start_time=1.0, end_time=7.0, event="A car drives.", source="vision")
    n = write_combined_jsonl({"temporal_qa": [qa], "events": [ev], "frames": []}, tmp_path / "dataset.jsonl")
    rows = list(read_jsonl(tmp_path / "dataset.jsonl"))
    assert n == 2 and [r["record_type"] for r in rows] == ["event", "temporal_qa"]
    assert rows[1]["question"] == "How long?" and rows[0]["event"] == "A car drives."


def test_package_dataset_bundles_files_and_media(tmp_path: Path):
    import zipfile

    from video_dataset.config import load_config
    from video_dataset.dataset.package import package_dataset

    cfg = load_config(None, {"project.data_dir": str(tmp_path)})
    final = cfg.export_dir
    final.mkdir(parents=True)
    media = tmp_path / "frames" / "vid_x" / "scene_001" / "frame_001_000.jpg"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"\xff\xd8\xff\xd9")
    (final / "dataset.jsonl").write_text('{"record_type": "frame", "record_id": "frame_x", "frame_path": "frames/vid_x/scene_001/frame_001_000.jpg"}\n{"record_type": "clip", "clip_path": "clips/vid_x/missing.mp4"}\n')
    (final / "statistics.json").write_text("{}")
    out = tmp_path / "bundle.zip"
    summary = package_dataset(cfg, out, include_media=True)
    names = zipfile.ZipFile(out).namelist()
    assert "dataset.jsonl" in names and "statistics.json" in names
    assert "frames/vid_x/scene_001/frame_001_000.jpg" in names
    assert summary["media_files"] == 1 and summary["media_missing"] == 1
    summary2 = package_dataset(cfg, tmp_path / "bundle2.zip", include_media=False)
    assert summary2["media_files"] == 0 and "frames/vid_x/scene_001/frame_001_000.jpg" not in zipfile.ZipFile(tmp_path / "bundle2.zip").namelist()
