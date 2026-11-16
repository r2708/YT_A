"""One loader for the T2V and V2V datasets (spec section 11).

Metadata (``data/<split>.jsonl``) is read as a stream of records; media is fetched only for the
batch being built, verified against the sha256 in the record and kept in a size-bounded LRU cache.
Files of the active batch are pinned and released with ``loader.release(batch)``. The cache index is
rebuilt from the directory on start, so a crash leaves nothing to repair by hand.

    loader = UnifiedLoader.from_config(config, split="train")
    for batch in loader.batches(steps=1000):
        ...  # batch["task"], batch["samples"][i]["record"], ["media"]["target"] / ["source"] (local paths)
        loader.release(batch)

Decoding and the model-specific frame sampling belong to the trainer: this loader hands out
verified, seekable local files, which is what video decoders need."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from video_dataset.utils.io import file_sha256
from video_dataset.utils.logging import get_logger

log = get_logger("v3.loader")

TASK_REPO = {"text_to_video": "t2v", "video_to_video": "v2v"}


class CacheFull(RuntimeError):
    """The files pinned by active batches alone exceed the cache limit."""


class AssetCache:
    def __init__(self, root: Path, max_bytes: int):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lru: OrderedDict[str, int] = OrderedDict()  # relative path -> size, least recently used first
        self._pins: dict[str, int] = {}
        self.peak_bytes = 0
        self.recover()

    def recover(self) -> None:
        """Rebuild the index from disk; partial downloads of a crashed run are removed."""
        self._lru.clear()
        files = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            if p.name.endswith(".part") or ".tmp" in p.name:
                p.unlink(missing_ok=True)
                continue
            files.append((p.stat().st_atime, str(p.relative_to(self.root)), p.stat().st_size))
        for _atime, rel, size in sorted(files):
            self._lru[rel] = size
        shutil.rmtree(self.root / ".incoming", ignore_errors=True)

    @property
    def bytes(self) -> int:
        return sum(self._lru.values())

    def _key(self, repo_id: str, path: str) -> str:
        return f"{repo_id.replace('/', '__')}/{path}"

    def _evict(self, need: int) -> None:
        for key in list(self._lru):
            if self.bytes + need <= self.max_bytes:
                return
            if self._pins.get(key):
                continue
            (self.root / key).unlink(missing_ok=True)
            del self._lru[key]
        if self.bytes + need > self.max_bytes:
            raise CacheFull(f"cache limit {self.max_bytes / 1e9:.2f} GB is too small for the active batch (pinned {self.bytes / 1e9:.2f} GB + {need / 1e9:.2f} GB); raise v3.cache.max_gb or lower the batch size")

    def get(self, ref: dict[str, Any], fetch: Callable[[str, str, Path], Path]) -> Path:
        """Local path of a media reference ({repo_id, path, sha256}); fetched, verified and pinned."""
        key = self._key(ref["repo_id"], ref["path"])
        local = self.root / key
        if key in self._lru and local.exists():
            self._lru.move_to_end(key)
            os.utime(local, None)
        else:
            incoming = self.root / ".incoming" / f"{time.time_ns()}"
            incoming.mkdir(parents=True, exist_ok=True)
            try:
                got = fetch(ref["repo_id"], ref["path"], incoming)
                sha = file_sha256(got)
                if ref.get("sha256") and sha != ref["sha256"]:
                    raise ValueError(f"checksum mismatch for {ref['repo_id']}:{ref['path']} (expected {ref['sha256'][:12]}, got {sha[:12]})")
                size = got.stat().st_size
                self._evict(size)
                local.parent.mkdir(parents=True, exist_ok=True)
                os.replace(got, local)
            finally:
                shutil.rmtree(incoming, ignore_errors=True)
            self._lru[key] = size
        self._pins[key] = self._pins.get(key, 0) + 1
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return local

    def release(self, ref: dict[str, Any]) -> None:
        key = self._key(ref["repo_id"], ref["path"])
        if self._pins.get(key):
            self._pins[key] -= 1
            if not self._pins[key]:
                del self._pins[key]


class RepoSource:
    """Reads one dataset repository: a local directory (a staged or cloned repo) or the Hub."""

    def __init__(self, repo_id: str, local_dir: Path | None = None, token: str | None = None, revision: str | None = None):
        self.repo_id, self.local_dir, self.token, self.revision = repo_id, Path(local_dir) if local_dir else None, token, revision

    def fetch(self, path: str, dest_dir: Path) -> Path:
        if self.local_dir is not None:
            src = self.local_dir / path
            if not src.exists():
                raise FileNotFoundError(f"{self.repo_id}:{path} not found under {self.local_dir}")
            dst = dest_dir / Path(path).name
            shutil.copy2(src, dst)
            return dst
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.repo_id, path, repo_type="dataset", revision=self.revision, token=self.token, local_dir=str(dest_dir)))

    def records(self, split: str, scratch: Path) -> Iterator[dict[str, Any]]:
        """Stream the records of one split (only this small JSONL file is fetched, never the repo)."""
        rel = f"data/{split}.jsonl"
        try:
            path = self.local_dir / rel if self.local_dir is not None else self.fetch(rel, scratch)
        except Exception as exc:
            log.warning("%s has no %s (%s)", self.repo_id, rel, type(exc).__name__)
            return
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


class UnifiedLoader:
    def __init__(self, sources: dict[str, RepoSource], cache: AssetCache, split: str = "train", task_weights: dict[str, float] | None = None,
                 batch_size: dict[str, int] | None = None, seed: int = 1234, primary_only: bool = True):
        self.sources = sources  # repo_id -> RepoSource (every repository a media reference may point at)
        self.cache, self.split, self.primary_only = cache, split, primary_only
        self.task_weights = {t: float(w) for t, w in (task_weights or {"text_to_video": 0.7, "video_to_video": 0.3}).items() if float(w) > 0}
        self.batch_size = batch_size or {}
        self.rng = random.Random(seed)
        self.task_repo: dict[str, str] = {}
        self.stats: dict[str, Any] = {"batches": {}, "samples": {}, "failed": []}

    @classmethod
    def from_config(cls, config: Any, split: str = "train", local_dirs: dict[str, Path] | None = None) -> UnifiedLoader:
        token = os.environ.get(config.v3.repos.token_env) or None
        repos = {"text_to_video": config.v3.repos.t2v, "video_to_video": config.v3.repos.v2v}
        sources = {rid: RepoSource(rid, (local_dirs or {}).get(rid), token) for rid in repos.values()}
        cache_dir = Path(config.v3.cache.dir) if config.v3.cache.dir else config.v3_dir / "cache"
        loader = cls(sources, AssetCache(cache_dir, int(float(config.v3.cache.max_gb) * 1e9)), split, dict(config.v3.training.task_weights), dict(config.v3.training.batch_size), int(config.v3.training.seed))
        loader.task_repo = repos
        return loader

    def _fetch(self, repo_id: str, path: str, dest: Path) -> Path:
        if repo_id not in self.sources:
            raise KeyError(f"no source configured for repository {repo_id}")
        return self.sources[repo_id].fetch(path, dest)

    def _stream(self, task: str) -> Iterator[dict[str, Any]]:
        repo_id = self.task_repo.get(task)
        if not repo_id or repo_id not in self.sources:
            return
        for rec in self.sources[repo_id].records(self.split, self.cache.root / ".incoming" / "metadata"):
            if rec.get("quality_status") != "accepted" or rec.get("task") != task:
                continue  # only accepted records train
            if self.primary_only and rec.get("primary_training") is False:
                continue
            yield rec

    def _load(self, rec: dict[str, Any]) -> dict[str, Any] | None:
        refs = {"target": rec["target_media"]}
        if rec["task"] == "video_to_video":
            refs["source"] = rec["source_media"]
        media: dict[str, Path] = {}
        held: list[dict[str, Any]] = []
        try:
            for role, ref in refs.items():
                media[role] = self.cache.get(ref, self._fetch)
                held.append(ref)
        except CacheFull:
            for ref in held:
                self.cache.release(ref)
            raise
        except Exception as exc:  # recorded and counted, never skipped silently
            for ref in held:
                self.cache.release(ref)
            self.stats["failed"].append({"sample_id": rec.get("sample_id"), "error": f"{type(exc).__name__}: {exc}"})
            log.warning("sample %s not loaded: %s", rec.get("sample_id"), exc)
            return None
        return {"record": rec, "media": media, "_refs": held}

    def batches(self, steps: int | None = None, cycle: bool = False) -> Iterator[dict[str, Any]]:
        """Batches of one task each, the task drawn by ``task_weights``. One pass over both datasets
        by default; ``cycle`` restarts an exhausted task (use with ``steps``)."""
        streams = {t: self._stream(t) for t in self.task_weights}
        live = set(streams)
        produced = 0
        while live and (steps is None or produced < steps):
            tasks = sorted(live)
            task = self.rng.choices(tasks, weights=[self.task_weights[t] for t in tasks])[0]
            want = max(1, int(self.batch_size.get(task, 1)))
            samples: list[dict[str, Any]] = []
            while len(samples) < want:
                rec = next(streams[task], None)
                if rec is None:
                    if cycle and self.stats["samples"].get(task):
                        streams[task] = self._stream(task)
                        continue
                    live.discard(task)
                    break
                loaded = self._load(rec)
                if loaded is not None:
                    samples.append(loaded)
                    self.stats["samples"][task] = self.stats["samples"].get(task, 0) + 1
            if not samples:
                continue
            produced += 1
            self.stats["batches"][task] = self.stats["batches"].get(task, 0) + 1
            yield {"task": task, "samples": samples}
        self.stats["cache_peak_bytes"] = self.cache.peak_bytes

    def release(self, batch: dict[str, Any]) -> None:
        """Unpin the batch's files once it has been consumed; they stay cached until evicted."""
        for sample in batch["samples"]:
            for ref in sample.pop("_refs", []):
                self.cache.release(ref)
