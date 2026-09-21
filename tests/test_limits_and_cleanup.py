"""URL validation, disk guard, rate limiter, download limits and the cleanup planner."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_dataset.config import LimitsConfig, load_config
from video_dataset.downloader.youtube import YouTubeDownloader
from video_dataset.errors import InsufficientDiskSpace, VideoRejected
from video_dataset.llm.base import create_llm_client
from video_dataset.stages import Stage
from video_dataset.storage import cleanup as cl
from video_dataset.storage.paths import DataPaths
from video_dataset.storage.state_db import StateDB
from video_dataset.utils.disk import ensure_free_disk, free_gb
from video_dataset.utils.ratelimit import RateLimitedLLMClient, RateLimiter
from video_dataset.utils.retry import retry_call
from video_dataset.utils.urls import InvalidURL, classify_input, url_problem, validate_url

# --------------------------------------------------------------------------- URL validation


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "http://example.com/video.mp4",
        "https://vimeo.com/123456",
    ],
)
def test_validate_url_accepts_public_http(url: str):
    assert validate_url(url) == url


@pytest.mark.parametrize(
    "url, fragment",
    [
        ("ftp://example.com/a.mp4", "scheme"),
        ("file:///etc/passwd", "scheme"),
        ("javascript:alert(1)", "scheme"),
        ("https://user:pw@youtube.com/watch?v=dQw4w9WgXcQ", "credentials"),
        ("http://127.0.0.1/x", "public"),
        ("http://10.0.0.5/x", "public"),
        ("http://169.254.169.254/latest/meta-data", "public"),
        ("http://[::1]/x", "public"),
        ("http://localhost/x", "not allowed"),
        ("http://intranet/x", "fully qualified"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ evil", "whitespace"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ\n&x=1", "whitespace"),
        ("", "empty"),
        ("https://" + "a" * 3000 + ".com", "longer"),
    ],
)
def test_validate_url_rejects(url: str, fragment: str):
    with pytest.raises(InvalidURL) as exc:
        validate_url(url)
    assert fragment in str(exc.value)


def test_allowed_domains_suffix_match():
    allowed = ["youtube.com", "youtu.be"]
    assert url_problem("https://www.youtube.com/watch?v=dQw4w9WgXcQ", allowed) is None
    assert url_problem("https://m.youtube.com/watch?v=dQw4w9WgXcQ", allowed) is None
    assert url_problem("https://youtu.be/dQw4w9WgXcQ", allowed) is None
    assert "allowed_domains" in (url_problem("https://notyoutube.com/watch?v=dQw4w9WgXcQ", allowed) or "")
    assert "allowed_domains" in (url_problem("https://youtube.com.evil.net/watch?v=dQw4w9WgXcQ", allowed) or "")
    assert url_problem("https://anything.example/x", []) is None  # empty list = no restriction


def test_classify_input_drops_invalid_urls():
    assert classify_input("https://user:pw@youtube.com/watch?v=dQw4w9WgXcQ") is None
    assert classify_input("http://127.0.0.1/watch?v=dQw4w9WgXcQ") is None
    assert classify_input("ftp://example.com/a") is None
    ok = classify_input("dQw4w9WgXcQ")  # bare ids still map to the canonical URL
    assert ok is not None and ok.kind == "youtube" and url_problem(ok.url) is None


# --------------------------------------------------------------------------- disk guard


def test_ensure_free_disk(tmp_path: Path):
    assert free_gb(tmp_path) > 0
    assert ensure_free_disk(tmp_path, 0.0) > 0
    assert ensure_free_disk(tmp_path / "does" / "not" / "exist", 0.0) > 0  # walks up to an existing parent
    with pytest.raises(InsufficientDiskSpace) as exc:
        ensure_free_disk(tmp_path, 10**9, label="download")  # a billion GB floor
    assert "download" in str(exc.value) and "min_free_disk_gb" in str(exc.value)
    with pytest.raises(InsufficientDiskSpace):
        ensure_free_disk(tmp_path, 0.0, needed_bytes=10**18)


def test_disk_guard_is_not_retried_by_runner(tmp_path: Path, synthetic_video: Path):
    from video_dataset.pipeline.runner import PipelineRunner

    cfg = load_config(None, {"project.data_dir": str(tmp_path / "d"), "limits.min_free_disk_gb": "1000000000", "pipeline.stage_retries": "3", "project.log_level": "ERROR"})
    runner = PipelineRunner(cfg)
    try:
        item = classify_input(str(synthetic_video))
        assert item is not None
        (vid, _), = runner.register_inputs([item])
        res = runner.run_batch([(vid, item)], until=Stage.PREPROCESS)[0]
        assert not res.completed and res.failed_stage == Stage.DOWNLOAD
        assert "GB free" in (res.error or "")
        assert runner.db.get_stage(vid, Stage.DOWNLOAD).attempts == 1  # NonRetryableError: no retries burned
    finally:
        runner.close()


# --------------------------------------------------------------------------- download limits


def _dl(**limits):  # type: ignore[no-untyped-def]
    cfg = load_config(None, {"project.data_dir": "data"})
    return YouTubeDownloader(cfg.download, None, LimitsConfig(**limits))


def test_check_limits_duration_and_size(tmp_path: Path):
    dl = _dl(max_duration_seconds=7200, max_file_size_gb=1.0, min_free_disk_gb=0)
    assert dl.check_limits({"duration": 100, "filesize": 10_000}, tmp_path) is None
    assert "max_duration_seconds" in (dl.check_limits({"duration": 7201}, tmp_path) or "")
    assert "max_file_size_gb" in (dl.check_limits({"filesize_approx": 2e9}, tmp_path) or "")
    assert _dl(max_duration_seconds=None).check_limits({"duration": 10**7}, tmp_path) is None
    huge = _dl(min_free_disk_gb=0).check_limits({"filesize": 10**18}, tmp_path)
    assert huge is not None and "not enough disk" in huge


def test_match_filter_rejection_is_reported_as_rejected(tmp_path: Path, monkeypatch):
    """yt-dlp's match_filter hook must turn into DownloadStatus.REJECTED without a retry."""
    import yt_dlp

    calls = {"n": 0}

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            calls["n"] += 1
            info = {"id": "dQw4w9WgXcQ", "title": "t", "duration": 99999, "webpage_url": url}
            mf = self.opts.get("match_filter")
            assert mf is not None
            assert mf(info, incomplete=True) is None  # never rejects on partial metadata
            assert mf(info) is not None
            return info  # yt-dlp returns the info dict without downloading when the filter rejects

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    dl = _dl(max_duration_seconds=7200)
    item = classify_input("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert item is not None
    res = dl.fetch(item, "vid_test", tmp_path / "dl")
    assert res.status.value == "rejected" and "max_duration_seconds" in (res.error or "")
    assert calls["n"] == 1 and res.attempts == 1


def test_preprocess_rejects_over_long_video(tmp_path: Path, synthetic_video: Path):
    from video_dataset.pipeline.runner import PipelineRunner

    cfg = load_config(None, {"project.data_dir": str(tmp_path / "d"), "limits.max_duration_seconds": "5", "limits.min_free_disk_gb": "0", "project.log_level": "ERROR"})
    runner = PipelineRunner(cfg)
    try:
        item = classify_input(str(synthetic_video))
        assert item is not None
        (vid, _), = runner.register_inputs([item])
        res = runner.run_batch([(vid, item)], until=Stage.PREPROCESS)[0]
        assert not res.completed and res.failed_stage == Stage.PREPROCESS
        assert "max_duration_seconds" in (res.error or "")
    finally:
        runner.close()


def test_download_stage_rejects_disallowed_domain(tmp_path: Path):
    from video_dataset.downloader.stage import DownloadRejected, download_stage
    from video_dataset.pipeline.context import VideoContext

    cfg = load_config(None, {"project.data_dir": str(tmp_path / "d"), "download.allowed_domains": "[youtube.com]", "limits.min_free_disk_gb": "0"})
    paths = DataPaths(cfg.data_dir)
    paths.ensure_all()
    db = StateDB(cfg.db_path)
    try:
        item = classify_input("https://vimeo.com/123456")
        assert item is not None and item.kind == "url"
        ctx = VideoContext(video_id="vid_x", config=cfg, paths=paths, db=db, input_item=item)
        with pytest.raises(DownloadRejected) as exc:
            download_stage(ctx)
        assert "allowed_domains" in str(exc.value)
        assert isinstance(exc.value, VideoRejected)
        assert paths.download_record_file("vid_x").exists()
    finally:
        db.close()


def test_retry_call_no_retry_on():
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise VideoRejected("nope")

    with pytest.raises(VideoRejected):
        retry_call(boom, retries=5, base_delay=0.0, retry_on=(Exception,), no_retry_on=(VideoRejected,))
    assert calls["n"] == 1


# --------------------------------------------------------------------------- rate limiter


def test_rate_limiter_token_bucket():
    now = [0.0]
    slept: list[float] = []

    def clock():
        return now[0]

    def sleeper(s: float):
        slept.append(s)
        now[0] += s

    rl = RateLimiter(60.0, burst=2, clock=clock, sleeper=sleeper)  # 1 token / s, 2 stored
    assert rl.acquire() == 0.0 and rl.acquire() == 0.0
    waited = rl.acquire()  # bucket empty -> must wait ~1 s
    assert waited == pytest.approx(1.0) and slept == [pytest.approx(1.0)]
    now[0] += 10
    assert rl.acquire() == 0.0  # refilled (capped at burst)
    assert rl.waits == 1 and rl.total_wait_seconds == pytest.approx(1.0)


def test_rate_limited_client_wraps_mock():
    plain = create_llm_client("mock")
    assert not isinstance(plain, RateLimitedLLMClient)
    limited = create_llm_client("mock", requests_per_minute=600)
    assert isinstance(limited, RateLimitedLLMClient)
    assert limited.name == plain.name and limited.model == plain.model
    resp = limited.complete("hello", max_tokens=10)
    assert resp.text is not None
    # same provider+model share one bucket
    again = create_llm_client("mock", requests_per_minute=600)
    assert isinstance(again, RateLimitedLLMClient) and again.limiter is limited.limiter


# --------------------------------------------------------------------------- cleanup planner


def _touch(p: Path, size: int = 10) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    return p


def test_cleanup_plans(tmp_path: Path):
    paths = DataPaths(tmp_path / "data")
    paths.ensure_all()
    db = StateDB(paths.root / "state.db")
    try:
        vid, orphan = "vid_aaaaaaaaaaaa", "vid_bbbbbbbbbbbb"
        db.upsert_video(vid, "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        for st in Stage:
            db.start_stage(vid, st)
            db.finish_stage(vid, st, None, {}, "h", 0.1)
        src = _touch(paths.video_dir(vid) / "source.mkv", 100)
        canon = _touch(paths.video_file(vid), 100)
        wav = _touch(paths.audio_file(vid), 50)
        frame = _touch(paths.frames_dir(vid) / "scene_001" / "f.jpg", 5)
        _touch(paths.video_dir(orphan) / "source.mp4", 30)
        _touch(paths.scenes_file(orphan), 3)
        log = _touch(paths.logs_dir / f"{vid}.log", 7)

        inter = cl.plan_intermediate(paths, db)
        assert set(inter.paths) == {src, wav} and inter.bytes == 150
        assert canon not in inter.paths and frame not in inter.paths

        orphans = cl.plan_orphans(paths, db)
        assert paths.video_dir(orphan) in orphans.paths and paths.scenes_file(orphan) in orphans.paths
        assert paths.video_dir(vid) not in orphans.paths

        full = cl.plan_video(paths, vid)
        assert paths.video_dir(vid) in full.paths and paths.frames_dir(vid) in full.paths and log in full.paths

        # not exported yet -> intermediate keeps everything and explains why
        db.reset_stages(vid, Stage.EXPORT)
        kept = cl.plan_intermediate(paths, db)
        assert kept.paths == [] and any("EXPORT not done" in n for n in kept.notes)

        removed, freed = cl.apply_plan(inter)
        assert removed == 2 and freed == 150 and not src.exists() and not wav.exists() and canon.exists()
        usage = cl.data_dir_usage(paths)
        assert usage["downloads"] >= 100 and "frames" in usage
    finally:
        db.close()


def test_clean_cli_dry_run_and_report(tmp_path: Path):
    from typer.testing import CliRunner

    from video_dataset import cli

    cli.state.config = None
    cli.state.config_path = None
    cli.state.overrides = {}
    cli.state.log_level = None
    data = str(tmp_path / "d")
    runner = CliRunner()
    r = runner.invoke(cli.app, ["--data-dir", data, "clean", "--report"])
    assert r.exit_code == 0, r.output
    assert "disk usage" in r.output and "free:" in r.output
    r = runner.invoke(cli.app, ["--data-dir", data, "clean"])
    assert r.exit_code == 0 and "Nothing to do" in r.output
    r = runner.invoke(cli.app, ["--data-dir", data, "clean", "--logs", "--orphans", "--dry-run"])
    assert r.exit_code == 0, r.output
