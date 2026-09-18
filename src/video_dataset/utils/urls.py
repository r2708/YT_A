"""YouTube URL parsing and input collection (single URL, URL file, directory of URL files, local video)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mpg", ".mpeg", ".ts"}
URL_FILE_EXTENSIONS = {".txt", ".urls", ".list", ".csv"}

_YT_ID = r"([A-Za-z0-9_-]{11})"
_PATTERNS = [
    re.compile(r"(?:youtube\.com|youtube-nocookie\.com)/watch\?(?:.*&)?v=" + _YT_ID),
    re.compile(r"youtu\.be/" + _YT_ID),
    re.compile(r"youtube\.com/(?:shorts|embed|live|v)/" + _YT_ID),
    re.compile(r"youtube\.com/attribution_link\?.*v(?:%3D|=)" + _YT_ID),
]
_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_youtube_id(url: str) -> str | None:
    url = url.strip()
    for pat in _PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    if _BARE_ID.match(url):
        return url
    return None


def canonical_youtube_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"


def is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))


def is_local_video(text: str) -> bool:
    p = Path(text.strip()).expanduser()
    return p.suffix.lower() in VIDEO_EXTENSIONS and p.exists() and p.is_file()


@dataclass(frozen=True)
class InputItem:
    raw: str
    kind: str  # "youtube" | "url" | "local"
    youtube_id: str | None = None
    local_path: str | None = None

    @property
    def url(self) -> str:
        if self.kind == "youtube" and self.youtube_id:
            return canonical_youtube_url(self.youtube_id)
        if self.kind == "local" and self.local_path:
            return Path(self.local_path).resolve().as_uri()
        return self.raw


def classify_input(text: str) -> InputItem | None:
    text = text.strip()
    if not text or text.startswith("#"):
        return None
    yt = extract_youtube_id(text)
    if yt and (is_url(text) or _BARE_ID.match(text)):
        return InputItem(raw=text, kind="youtube", youtube_id=yt)
    if is_local_video(text):
        return InputItem(raw=text, kind="local", local_path=str(Path(text).expanduser().resolve()))
    if is_url(text):
        return InputItem(raw=text, kind="url")
    return None


def read_url_file(path: str | Path) -> list[InputItem]:
    items: list[InputItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        # allow simple CSV: first column is the URL
        candidate = line.split(",")[0] if "," in line and is_url(line.split(",")[0]) else line
        item = classify_input(candidate)
        if item:
            items.append(item)
    return items


def collect_inputs(source: str) -> list[InputItem]:
    """Accepts a URL, a bare YouTube id, a local video, a URL file or a directory of URL files."""
    item = classify_input(source)
    if item is not None and item.kind in ("youtube", "local", "url"):
        return [item]
    p = Path(source).expanduser()
    if p.is_dir():
        items: list[InputItem] = []
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in URL_FILE_EXTENSIONS:
                items.extend(read_url_file(f))
            elif f.suffix.lower() in VIDEO_EXTENSIONS:
                items.append(InputItem(raw=str(f), kind="local", local_path=str(f.resolve())))
        return dedupe_inputs(items)
    if p.is_file():
        return dedupe_inputs(read_url_file(p))
    raise FileNotFoundError(f"Input '{source}' is not a URL, a video file, a URL file or a directory")


def dedupe_inputs(items: list[InputItem]) -> list[InputItem]:
    seen: set[str] = set()
    out: list[InputItem] = []
    for it in items:
        key = it.youtube_id or it.local_path or it.raw
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out
