from pathlib import Path

from video_dataset.utils.ids import frame_id, make_video_id, qa_id, scene_id
from video_dataset.utils.urls import classify_input, collect_inputs, extract_youtube_id, read_url_file


def test_extract_youtube_id_variants():
    cases = {
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=10": "dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://m.youtube.com/watch?feature=share&v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123": "dQw4w9WgXcQ",
        "dQw4w9WgXcQ": "dQw4w9WgXcQ",
    }
    for url, vid in cases.items():
        assert extract_youtube_id(url) == vid, url
    assert extract_youtube_id("https://example.com/video") is None


def test_video_id_is_stable_across_url_variants():
    a = make_video_id(youtube_id=extract_youtube_id("https://youtu.be/dQw4w9WgXcQ"))
    b = make_video_id(youtube_id=extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=5"))
    assert a == b and a.startswith("vid_") and len(a) == 16
    assert make_video_id(youtube_id="abcdefghijk") != a


def test_id_helpers():
    assert scene_id(0) == "scene_001"
    assert frame_id(2, 5) == "frame_003_005"
    assert qa_id("vid_abc", 7) == "qa_abc_000007"


def test_read_url_file_and_directory(tmp_path: Path):
    f = tmp_path / "urls.txt"
    f.write_text("# comment\nhttps://youtu.be/dQw4w9WgXcQ\n\nhttps://www.youtube.com/watch?v=dQw4w9WgXcQ\nhttps://www.youtube.com/watch?v=aaaaaaaaaaa\nnot a url\n")
    items = read_url_file(f)
    assert [i.youtube_id for i in items] == ["dQw4w9WgXcQ", "dQw4w9WgXcQ", "aaaaaaaaaaa"]
    collected = collect_inputs(str(tmp_path))
    assert [i.youtube_id for i in collected] == ["dQw4w9WgXcQ", "aaaaaaaaaaa"]  # deduped
    single = collect_inputs("https://youtu.be/dQw4w9WgXcQ")
    assert len(single) == 1 and single[0].url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_classify_local_video(synthetic_video: Path):
    item = classify_input(str(synthetic_video))
    assert item is not None and item.kind == "local" and item.local_path == str(synthetic_video.resolve())
