"""YouTube challenge solving: yt-dlp needs a JS runtime + solver script or it drops every video format.

yt-dlp enables only deno by default. The downloader must hand it the configured runtimes (node first,
since that is what is installed here) and allow the EJS solver script to be fetched when the bundled
package is missing, while ``download.extra_args`` keeps the last word.
"""

from __future__ import annotations

from pathlib import Path

from video_dataset.config import DownloadConfig
from video_dataset.downloader.youtube import YouTubeDownloader, js_challenge_options


def _opts(tmp_path: Path, **download):  # type: ignore[no-untyped-def]
    return YouTubeDownloader(DownloadConfig(**download)).build_options(tmp_path)


def test_defaults_enable_node_and_ejs_fetch(tmp_path: Path):
    opts = _opts(tmp_path)
    assert list(opts["js_runtimes"]) == ["node", "deno", "bun"]
    assert opts["js_runtimes"]["node"] == {}
    assert opts["remote_components"] == {"ejs:github"}
    # the format selector must still prefer <=1080p mp4 (no android-client downgrade to 360p)
    assert "bestvideo[height<=1080][ext=mp4]" in opts["format"]
    assert "player_client" not in str(opts.get("extractor_args", {}))


def test_runtime_spec_with_path_and_case():
    cfg = DownloadConfig(js_runtimes=["Node:/opt/node/bin/node", " deno "], remote_components=[" ejs:github ", ""])
    out = js_challenge_options(cfg)
    assert out["js_runtimes"] == {"node": {"path": "/opt/node/bin/node"}, "deno": {}}
    assert out["remote_components"] == {"ejs:github"}


def test_windows_style_path_keeps_drive_letter():
    out = js_challenge_options(DownloadConfig(js_runtimes=[r"node:C:\Program Files\nodejs\node.exe"]))
    assert out["js_runtimes"] == {"node": {"path": r"C:\Program Files\nodejs\node.exe"}}


def test_empty_lists_leave_yt_dlp_defaults(tmp_path: Path):
    opts = _opts(tmp_path, js_runtimes=[], remote_components=[])
    assert "js_runtimes" not in opts
    assert "remote_components" not in opts


def test_extra_args_override(tmp_path: Path):
    opts = _opts(tmp_path, extra_args={"js_runtimes": {"deno": {}}, "remote_components": set()})
    assert opts["js_runtimes"] == {"deno": {}}
    assert opts["remote_components"] == set()
