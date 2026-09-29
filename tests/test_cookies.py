"""Cookie handling must never block downloads of public videos.

On Windows, yt-dlp's ``--cookies-from-browser chrome`` fails from a background process (DPAPI refuses to
decrypt, Chrome locks its cookie database). The pipeline probes the browser once, logs a warning and
continues without cookies; a failure that still reaches yt-dlp is retried once without cookies.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from video_dataset.config import DownloadConfig, load_config
from video_dataset.downloader import youtube as yt
from video_dataset.downloader.youtube import YouTubeDownloader, is_cookie_failure, parse_browser_spec
from video_dataset.utils.urls import classify_input

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    yt.reset_cookie_probe_cache()
    yield
    yt.reset_cookie_probe_cache()


def _dl(**download):  # type: ignore[no-untyped-def]
    return YouTubeDownloader(DownloadConfig(**download))


# --------------------------------------------------------------------------- spec parsing


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("chrome", ("chrome", None, None, None)),
        ("Chrome", ("chrome", None, None, None)),
        ("chrome:Profile 1", ("chrome", "Profile 1", None, None)),
        ("firefox::personal", ("firefox", None, None, "personal")),
        ("chrome+basictext:Default", ("chrome", "Default", "BASICTEXT", None)),
        (["chrome"], ("chrome", None, None, None)),
        (("edge", "Profile 2"), ("edge", "Profile 2", None, None)),
        (("edge", None, None, None), ("edge", None, None, None)),
        ("", None),
        (None, None),
        ([], None),
    ],
)
def test_parse_browser_spec(spec, expected):
    assert parse_browser_spec(spec) == expected


@pytest.mark.parametrize(
    "msg",
    [
        "ERROR: Could not copy Chrome cookie database. See https://github.com/yt-dlp/yt-dlp/issues/7271 for more info",
        "Failed to decrypt with DPAPI",
        "failed to load cookies",
        "could not find chrome cookies database in \"C:\\Users\\x\\AppData\\Local\\Google\\Chrome\\User Data\"",
        'unsupported browser: "netscape"',
    ],
)
def test_is_cookie_failure_markers(msg):
    assert is_cookie_failure(msg)
    assert not is_cookie_failure("ERROR: [youtube] abc: Video unavailable")
    assert not is_cookie_failure("HTTP Error 403: Forbidden")


# --------------------------------------------------------------------------- preflight probe


def test_browser_cookie_failure_is_dropped_with_warning(tmp_path: Path, monkeypatch, caplog):
    def boom(*a, **k):
        raise PermissionError(13, "Could not copy Chrome cookie database")

    import yt_dlp.cookies

    monkeypatch.setattr(yt_dlp.cookies, "extract_cookies_from_browser", boom)
    caplog.set_level(logging.WARNING)
    dl = _dl(cookies_from_browser="chrome")
    opts = dl.build_options(tmp_path)
    assert "cookiesfrombrowser" not in opts
    assert "cookiefile" not in opts
    assert any("could not load cookies from browser 'chrome'" in r.getMessage() for r in caplog.records)
    assert any("Continuing WITHOUT cookies" in r.getMessage() for r in caplog.records)

    # Probed once per process: a second downloader (next video) does not retry the extraction.
    caplog.clear()
    calls = {"n": 0}

    def count(*a, **k):
        calls["n"] += 1
        raise RuntimeError("should not be called again")

    monkeypatch.setattr(yt_dlp.cookies, "extract_cookies_from_browser", count)
    opts2 = _dl(cookies_from_browser="chrome").build_options(tmp_path)
    assert "cookiesfrombrowser" not in opts2 and calls["n"] == 0


def test_empty_jar_counts_as_failure(tmp_path: Path, monkeypatch, caplog):
    """Windows DPAPI: every cookie fails to decrypt and yt-dlp returns an *empty* jar without raising."""
    import yt_dlp.cookies

    monkeypatch.setattr(yt_dlp.cookies, "extract_cookies_from_browser", lambda *a, **k: yt_dlp.cookies.YoutubeDLCookieJar())
    caplog.set_level(logging.WARNING)
    opts = _dl(cookies_from_browser="chrome").build_options(tmp_path)
    assert "cookiesfrombrowser" not in opts
    assert any("DPAPI" in r.getMessage() for r in caplog.records)


def test_browser_cookies_kept_when_extraction_works(tmp_path: Path, monkeypatch):
    import http.cookiejar

    import yt_dlp.cookies

    def ok(browser, profile=None, logger=None, *, keyring=None, container=None):
        assert (browser, profile, keyring, container) == ("chrome", "Profile 1", None, None)
        jar = yt_dlp.cookies.YoutubeDLCookieJar()
        jar.set_cookie(http.cookiejar.Cookie(0, "a", "b", None, False, ".youtube.com", True, True, "/", True, True, None, False, None, None, {}))
        return jar

    monkeypatch.setattr(yt_dlp.cookies, "extract_cookies_from_browser", ok)
    opts = _dl(cookies_from_browser="chrome:Profile 1").build_options(tmp_path)
    assert opts["cookiesfrombrowser"] == ("chrome", "Profile 1", None, None)


def test_extra_args_string_spec_is_normalized_and_probed(tmp_path: Path, monkeypatch):
    """`extra_args: {cookiesfrombrowser: chrome}` (a bare string) must not be handed to yt-dlp raw."""
    import yt_dlp.cookies

    seen = {}

    def boom(browser, *a, **k):
        seen["browser"] = browser
        raise FileNotFoundError("could not find chrome cookies database")

    monkeypatch.setattr(yt_dlp.cookies, "extract_cookies_from_browser", boom)
    opts = _dl(extra_args={"cookiesfrombrowser": "chrome"}).build_options(tmp_path)
    assert seen["browser"] == "chrome"
    assert "cookiesfrombrowser" not in opts


def test_cookies_optional_false_raises(tmp_path: Path, monkeypatch):
    import yt_dlp.cookies

    monkeypatch.setattr(yt_dlp.cookies, "extract_cookies_from_browser", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("locked")))
    with pytest.raises(RuntimeError, match="could not load cookies from browser 'chrome'"):
        _dl(cookies_from_browser="chrome", cookies_optional=False).build_options(tmp_path)


def test_missing_cookies_file_is_dropped(tmp_path: Path, caplog):
    caplog.set_level(logging.WARNING)
    opts = _dl(cookies_file=str(tmp_path / "nope.txt")).build_options(tmp_path)
    assert "cookiefile" not in opts
    assert any("does not exist" in r.getMessage() for r in caplog.records)
    with pytest.raises(FileNotFoundError):
        _dl(cookies_file=str(tmp_path / "nope.txt"), cookies_optional=False).build_options(tmp_path)
    f = tmp_path / "cookies.txt"
    f.write_text("# Netscape HTTP Cookie File\n")
    assert _dl(cookies_file=str(f)).build_options(tmp_path)["cookiefile"] == str(f)


def test_config_and_env_expose_browser_cookies(monkeypatch):
    monkeypatch.setenv("YT_DLP_COOKIES_FROM_BROWSER", "edge")
    cfg = load_config(None, {"project.data_dir": "data"})
    assert cfg.download.cookies_from_browser == "edge"
    assert cfg.download.cookies_optional is True


# --------------------------------------------------------------------------- runtime fallback in fetch


def test_fetch_retries_without_cookies_when_ytdlp_fails_loading_them(tmp_path: Path, monkeypatch, caplog):
    """yt-dlp loads cookies lazily, so a cookie error surfaces as DownloadError inside extract_info."""
    import yt_dlp
    from yt_dlp.utils import DownloadError

    # Bypass the preflight so the option reaches yt-dlp.
    monkeypatch.setattr(yt, "probe_browser_cookies", lambda spec: (True, "1 cookies"))
    dest = tmp_path / "dl"
    calls: list[dict] = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = dict(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            calls.append(self.opts)
            if "cookiesfrombrowser" in self.opts:
                raise DownloadError("ERROR: Could not copy Chrome cookie database. See https://github.com/yt-dlp/yt-dlp/issues/7271 for more info")
            dest.mkdir(parents=True, exist_ok=True)
            media = dest / "source.mp4"
            media.write_bytes(b"\0" * 10)
            return {"id": "dQw4w9WgXcQ", "title": "t", "duration": 10, "webpage_url": url, "requested_downloads": [{"filepath": str(media)}]}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    caplog.set_level(logging.WARNING)
    dl = _dl(cookies_from_browser="chrome", retries=1)
    item = classify_input(URL)
    assert item is not None
    res = dl.fetch(item, "vid_test", dest)
    assert res.status.value == "done", res.error
    assert len(calls) == 2 and "cookiesfrombrowser" in calls[0] and "cookiesfrombrowser" not in calls[1]
    assert res.attempts == 1  # the cookie-less rerun is not a download retry
    assert any("retrying" in r.getMessage() and "without cookies" in r.getMessage() for r in caplog.records)


def test_fetch_cookie_failure_is_fatal_when_not_optional(tmp_path: Path, monkeypatch):
    import yt_dlp
    from yt_dlp.utils import DownloadError

    monkeypatch.setattr(yt, "probe_browser_cookies", lambda spec: (True, "1 cookies"))

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            raise DownloadError("ERROR: Failed to decrypt with DPAPI")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    dl = _dl(cookies_from_browser="chrome", cookies_optional=False, retries=1)
    item = classify_input(URL)
    assert item is not None
    res = dl.fetch(item, "vid_test", tmp_path / "dl")
    assert res.status.value == "failed" and "DPAPI" in (res.error or "")


def test_non_cookie_errors_are_not_masked(tmp_path: Path, monkeypatch):
    """A genuine download failure must still be reported, not turned into a cookie retry."""
    import yt_dlp
    from yt_dlp.utils import DownloadError

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
            raise DownloadError("ERROR: [youtube] dQw4w9WgXcQ: Video unavailable")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    dl = _dl(retries=1)
    item = classify_input(URL)
    assert item is not None
    res = dl.fetch(item, "vid_test", tmp_path / "dl")
    assert res.status.value == "unavailable" and calls["n"] == 1
