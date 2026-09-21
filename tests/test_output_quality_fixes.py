"""Regression tests for the output-quality fixes: OCR watermark/fragment filtering, long-range QA
guard, minimum scene length after detector merging, and summarised video descriptions."""

from __future__ import annotations

from video_dataset.config import TemporalConfig
from video_dataset.dataset.builders import DatasetBuilder
from video_dataset.ocr.merge import classify_tracks
from video_dataset.scene_detection.detector import enforce_min_scene_length
from video_dataset.schemas.ocr import OCRResult, OCRTrack
from video_dataset.schemas.scene import Frame, Scene, TransitionType
from video_dataset.temporal.events import EventExtractor


def _track(i: int, text: str, first: float, last: float, scenes: list[str], frames: list[str] | None = None) -> OCRTrack:
    return OCRTrack(
        track_id=f"ocr_{i:04d}", text=text, normalized_text=text.lower().replace("@", "").strip(), first_seen=first, last_seen=last,
        frame_ids=frames or [f"f{i}"], scene_ids=scenes, detection_ids=[f"d{i}"], num_detections=1, mean_confidence=0.9,
    )


def _watermark_tracks() -> list[OCRTrack]:
    tracks = []
    i = 1
    for s in range(1, 21):  # handle visible in every one of 20 scenes, sometimes read without the @
        tracks.append(_track(i, "@sharmajikaladkaa" if s % 2 else "sharmajikaladkaa", s * 10.0, s * 10.0 + 2, [f"scene_{s:03d}"]))
        i += 1
    for s, txt in [(1, "COMED"), (2, "COM"), (3, "COME"), (4, "COMEDY"), (5, "COM"), (6, "COMED"), (7, "COMEDY"), (8, "COM")]:  # sign read in pieces
        tracks.append(_track(i, txt, s * 10.0 + 1, s * 10.0 + 2, [f"scene_{s:03d}"]))
        i += 1
    tracks.append(_track(i, "Thanks For Watching", 195.0, 199.0, ["scene_020"], ["f_end"]))
    i += 1
    tracks.append(_track(i, "Thanks For", 195.0, 195.0, ["scene_020"], ["f_end"]))  # partial read on the same frame
    i += 1
    tracks.append(_track(i, "EXIT", 42.0, 43.0, ["scene_004"]))  # a real, transient text
    i += 1
    tracks.append(_track(i, "sharmajika", 50.0, 50.0, ["scene_005"]))  # partial read of the watermark
    i += 1
    for s, txt in [(9, "CON"), (10, "COML"), (11, "EOMED"), (12, "COI")]:  # one-character misreads of the sign
        tracks.append(_track(i, txt, s * 10.0 + 1, s * 10.0 + 2, [f"scene_{s:03d}"]))
        i += 1
    tracks.append(_track(i, "CAT", 155.0, 156.0, ["scene_015"]))  # 2 edits from "com": a real word, kept
    return tracks


def test_classify_tracks_flags_overlays_and_fragments():
    tracks, overlays = classify_tracks(_watermark_tracks(), total_scenes=20)
    assert overlays == ["comedy", "sharmajikaladkaa"]
    by_text = {}
    for t in tracks:
        by_text.setdefault(t.text, []).append(t)
    assert all(t.is_static_overlay and t.overlay_family == "sharmajikaladkaa" for t in by_text["@sharmajikaladkaa"] + by_text["sharmajikaladkaa"])
    assert all(t.is_static_overlay for txt in ("COMED", "COM", "COME", "COMEDY") for t in by_text[txt])
    assert by_text["Thanks For"][0].is_fragment and not by_text["Thanks For Watching"][0].is_fragment
    assert not by_text["sharmajika"][0].is_event_worthy  # prefix of the watermark -> joins its family
    exit_track = by_text["EXIT"][0]
    assert exit_track.is_event_worthy and not exit_track.is_static_overlay and not exit_track.is_fragment
    assert all(by_text[t][0].is_fragment for t in ("CON", "COML", "EOMED", "COI")), {t: by_text[t][0].is_fragment for t in ("CON", "COML", "EOMED", "COI")}
    assert by_text["CAT"][0].is_event_worthy
    assert [t.track_id for t in tracks] == sorted(t.track_id for t in tracks)  # order preserved by id
    again, overlays2 = classify_tracks(tracks, total_scenes=20)
    assert overlays2 == overlays and [(t.is_static_overlay, t.is_fragment) for t in again] == [(t.is_static_overlay, t.is_fragment) for t in tracks]


def test_classify_tracks_small_videos_and_transient_text():
    # 3 scenes only: nothing can be an overlay (below static_overlay_min_scenes)
    tracks = [_track(1, "LOGO", 1.0, 2.0, ["scene_001"]), _track(2, "LOGO", 5.0, 6.0, ["scene_002"]), _track(3, "LOGO", 9.0, 10.0, ["scene_003"])]
    out, overlays = classify_tracks(tracks, total_scenes=3)
    assert overlays == [] and all(t.is_event_worthy for t in out)
    # a title card seen in 2 of 20 scenes stays an event
    out, overlays = classify_tracks([_track(1, "CHAPTER 2", 1.0, 2.0, ["scene_001"]), _track(2, "CHAPTER 2", 50.0, 51.0, ["scene_009"])], total_scenes=20)
    assert overlays == [] and all(t.is_event_worthy for t in out)


def test_ocr_events_skip_overlays_by_default():
    frames = [Frame(frame_id=f"f{i}", video_id="v", scene_id=f"scene_{s:03d}", frame_index=i, timestamp=float(i), frame_path="x.jpg", width=16, height=9) for i, s in enumerate(range(1, 21), start=1)]
    ocr = OCRResult(video_id="v", engine="mock", frames_processed=20, tracks=_watermark_tracks())
    ex = EventExtractor(TemporalConfig(), scene_threshold=27.0)
    events = ex._ocr_events("v", ocr, frames, 200.0)
    texts = sorted(e.entities[0] for e in events)
    assert texts == ["CAT", "EXIT", "Thanks For Watching"], texts
    ex_all = EventExtractor(TemporalConfig(include_static_overlay_text=True), scene_threshold=27.0)
    assert len(ex_all._ocr_events("v", ocr, frames, 200.0)) == len(ocr.tracks)


def _scene(idx: int, start: float, end: float, fps: float = 25.0, transition: TransitionType = TransitionType.CUT) -> Scene:
    return Scene(scene_id=f"scene_{idx + 1:03d}", video_id="v", index=idx, start_time=start, end_time=end, start_frame=int(start * fps), end_frame=int(end * fps), transition_in=transition)


def test_enforce_min_scene_length_merges_sub_minimum_scenes():
    scenes = [_scene(0, 0.0, 10.0, transition=TransitionType.START), _scene(1, 10.0, 10.12), _scene(2, 10.12, 20.0, transition=TransitionType.FADE), _scene(3, 20.0, 20.4), _scene(4, 20.4, 30.0)]
    out = enforce_min_scene_length(scenes, 0.6)
    assert [(s.scene_id, s.index, s.start_time, s.end_time) for s in out] == [("scene_001", 0, 0.0, 10.12), ("scene_002", 1, 10.12, 20.4), ("scene_003", 2, 20.4, 30.0)]
    assert out[0].transition_in == TransitionType.START and out[1].transition_in == TransitionType.FADE
    assert out[0].end_frame == int(10.12 * 25)
    # first scene too short: absorbed into the next one, keeping the START transition
    out = enforce_min_scene_length([_scene(0, 0.0, 0.2, transition=TransitionType.START), _scene(1, 0.2, 5.0)], 0.6)
    assert len(out) == 1 and out[0].start_time == 0.0 and out[0].end_time == 5.0 and out[0].transition_in == TransitionType.START
    # nothing to do -> same objects back
    ok = [_scene(0, 0.0, 5.0), _scene(1, 5.0, 9.0)]
    assert enforce_min_scene_length(ok, 0.6) is ok and enforce_min_scene_length(ok, 0) is ok


def test_description_summarise_collapses_repeats():
    s = DatasetBuilder._summarize
    assert s(["dim", "dim", "dim"], 3) == "dim throughout"
    assert s(["dim", "dark", "dim", "dim"], 4) == "mostly dim (3 of 4 shots), dark (1 of 4 shots)"
    assert s(["the camera is static", "the camera makes a complex movement", "the camera is static"], 3) == "mostly the camera is static (2 of 3 shots), the camera makes a complex movement (1 of 3 shots)"
    assert s([], 3) is None and s(["", ""], 2) is None
    assert s(["bright"], 1) == "bright"
