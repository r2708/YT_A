"""Aggregate the finished jobs, stage the T2V and V2V repositories, upload them and verify the
remote copy (spec sections 9, 10, 12, 13 and the cleanup rule of section 15).

Layout of each staged repository:

    README.md  dataset_info.json  schemas/  manifests/  data/<split>.jsonl  media/<bucket>/  quality/

Nothing is marked verified from a successful upload call alone: every staged file is compared with
the hash the Hub stores (LFS sha256) or is downloaded again and re-hashed. Only then are
``metadata_upload_verified`` / ``visual_asset_upload_verified`` set and cleanup allowed."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from video_dataset.utils.io import ensure_dir, file_sha256, read_jsonl, write_json_atomic, write_jsonl
from video_dataset.utils.logging import get_logger
from video_dataset.v3 import V3_SCHEMA_VERSION, legacy, rights
from video_dataset.v3.media import near_duplicate
from video_dataset.v3.pipeline import load_job_records, open_store

log = get_logger("v3.publish")

SPLITS = ("train", "validation", "test")
READINESS = ("pending", "rights_blocked", "asset_missing", "decode_failed", "annotation_mismatch", "pair_invalid", "needs_human_review", "accepted", "rejected")
_PRIVATE_ASSET_KEYS = ("local_path", "fingerprint", "upload_status")


class PublishError(RuntimeError):
    pass


class RepoClient(Protocol):
    repo_id: str

    def ensure_repo(self) -> str: ...
    def upload_dir(self, local_dir: Path, message: str) -> str: ...
    def remote_files(self) -> dict[str, str | None]: ...  # path -> sha256 the Hub stores (None: not an LFS file)
    def download(self, path: str, dest_dir: Path) -> Path: ...


class HubRepo:
    def __init__(self, repo_id: str, token: str | None, private: bool = True):
        from huggingface_hub import HfApi

        if not repo_id or "/" not in repo_id:
            raise PublishError("repository id must be '<user-or-org>/<dataset-name>'")
        self.repo_id, self.private, self.token = repo_id, private, token
        self.api = HfApi(token=token)

    def ensure_repo(self) -> str:
        return str(self.api.create_repo(self.repo_id, repo_type="dataset", private=self.private, exist_ok=True))

    def upload_dir(self, local_dir: Path, message: str) -> str:
        info = self.api.upload_folder(repo_id=self.repo_id, folder_path=str(local_dir), repo_type="dataset", commit_message=message)
        return str(getattr(info, "commit_url", None) or info)

    def remote_files(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for item in self.api.list_repo_tree(self.repo_id, repo_type="dataset", recursive=True, expand=True):
            if item.__class__.__name__ != "RepoFile":
                continue
            lfs = getattr(item, "lfs", None)
            out[item.path] = (lfs or {}).get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        return out

    def download(self, path: str, dest_dir: Path) -> Path:
        return Path(self.api.hf_hub_download(self.repo_id, path, repo_type="dataset", local_dir=str(dest_dir), force_download=True))


# ----------------------------------------------------------------------------- aggregate
def aggregate(config: Any) -> dict[str, Any]:
    """Collect every finished job, apply the cross-source gates (rights still approved, duplicates,
    split leakage) and return the registries both repositories are built from."""
    rights_path = rights.default_rights_path(config)
    manifest = rights.load_rights(rights_path)
    store = open_store(config)
    try:
        jobs = load_job_records(config, store)
    finally:
        store.close()
    assets: dict[str, dict[str, Any]] = {}
    t2v: list[dict[str, Any]] = []
    v2v: list[dict[str, Any]] = []
    rights_rows: dict[str, dict[str, Any]] = {}
    skipped: Counter[str] = Counter()
    group_split: dict[str, str] = {}
    leaks: list[str] = []
    for job in sorted(jobs, key=lambda j: j["source_video_id"]):
        vid = job["source_video_id"]
        entry = manifest.get(vid)
        blocked = not rights.is_approved(entry, rights_path.parent)
        for reason, n in (job.get("skipped") or {}).items():
            skipped[reason] += n
        if not blocked:
            rights_rows[vid] = rights.public_row(entry or {})
        group = job.get("source_group_id") or vid
        split = job.get("split")
        if split and group_split.setdefault(group, split) != split:
            leaks.append(group)
        for asset in job.get("assets") or []:
            assets.setdefault(asset["asset_id"], asset)  # one physical asset = one row, however often it is referenced
        for rec in job.get("t2v") or []:
            t2v.append({**rec, "quality_status": "rights_blocked", "quality_reasons": ["rights are no longer approved"]} if blocked else rec)
        for rec in job.get("v2v") or []:
            v2v.append({**rec, "quality_status": "rights_blocked", "quality_reasons": ["rights are no longer approved"]} if blocked else rec)
    for rec in t2v + v2v:  # a group seen in two splits never trains: fail closed
        if rec["source_group_id"] in leaks and rec["quality_status"] == "accepted":
            rec.update(quality_status="rejected", quality_reasons=["source group appears in more than one split"])
    # duplicates: exact (same bytes) and near (all sampled frame hashes within the hamming limit)
    limit = int(config.v3.clips.near_duplicate_hamming)
    kept: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rec in t2v:
        if rec["quality_status"] != "accepted":
            continue
        asset = assets.get(rec["target_asset_id"]) or {}
        dup = next((k for k, a in kept if a["sha256"] == asset.get("sha256") or near_duplicate(a.get("fingerprint") or [], asset.get("fingerprint") or [], limit)), None)
        if dup is not None:
            rec.update(quality_status="rejected", quality_reasons=[f"duplicate of {dup['sample_id']}"], duplicate_of=dup["sample_id"])
        else:
            kept.append((rec, asset))
    accepted_clips = {r["clip_id"] for r in t2v if r["quality_status"] == "accepted"}
    for pair in v2v:
        if pair["quality_status"] == "accepted" and pair["clip_id"] not in accepted_clips:
            pair.update(quality_status="rejected", quality_reasons=["its source clip is not an accepted asset"])
    for rec in t2v + v2v:
        if rec["quality_status"] == "accepted" and rec.get("split") not in SPLITS:
            rec.update(quality_status="needs_human_review", quality_reasons=["no split assigned to the source group"])
    return {"assets": assets, "t2v": t2v, "v2v": v2v, "rights": rights_rows, "skipped": dict(skipped), "leaking_groups": sorted(set(leaks)), "sources": [j["source_video_id"] for j in jobs]}


def _quality_report(task: str, records: list[dict[str, Any]], bundle: dict[str, Any]) -> dict[str, Any]:
    accepted = [r for r in records if r["quality_status"] == "accepted"]
    durations = sorted(float(r.get("duration_seconds") or (r.get("temporal_alignment") or {}).get("duration_seconds") or 0) for r in accepted)
    return {
        "task": task, "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "schema_version": V3_SCHEMA_VERSION,
        "records": len(records), "by_status": dict(Counter(r["quality_status"] for r in records)),
        "accepted_by_split": dict(Counter(str(r.get("split")) for r in accepted if r.get("primary_training", True))),
        "accepted_by_bucket": dict(Counter(str(r.get("resolution_bucket")) for r in accepted)),
        "accepted_by_transform": dict(Counter(r["transform"] for r in accepted if r.get("transform"))),
        "source_videos": len({r["source_video_id"] for r in accepted}), "source_groups": len({r["source_group_id"] for r in accepted}),
        "duration_seconds": {"min": durations[0], "median": durations[len(durations) // 2], "max": durations[-1], "total": round(sum(durations), 2)} if durations else None,
        "legacy_clips_not_candidates": bundle["skipped"], "leaking_groups": bundle["leaking_groups"],
        "rejections": [{"sample_id": r["sample_id"], "status": r["quality_status"], "reasons": r.get("quality_reasons")} for r in records if r["quality_status"] != "accepted"],
    }


# ----------------------------------------------------------------------------- schemas / card
def _schema(title: str, required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title, "version": V3_SCHEMA_VERSION, "type": "object", "required": required, "properties": properties, "additionalProperties": True}


_MEDIA_REF = {"type": "object", "required": ["repo_id", "path", "sha256"], "properties": {"repo_id": {"type": "string"}, "path": {"type": "string"}, "sha256": {"type": "string"}}}
_STATUS = {"type": "string", "enum": list(READINESS)}
SCHEMAS = {
    "t2v_schema.json": _schema("T2V sample", ["sample_id", "task", "source_group_id", "source_video_id", "clip_id", "prompt", "target_asset_id", "target_media", "rights_evidence_id", "quality_status", "split"], {
        "task": {"const": "text_to_video"}, "prompt": {"type": "string", "minLength": 1}, "target_media": _MEDIA_REF, "quality_status": _STATUS, "split": {"enum": list(SPLITS)},
        "timeline": {"type": "array"}, "cinematography": {"type": "object"}, "resolution_bucket": {"type": "string"}, "upscaled": {"type": "boolean"},
    }),
    "v2v_schema.json": _schema("V2V pair", ["sample_id", "task", "source_group_id", "source_asset_id", "source_media", "instruction", "target_asset_id", "target_media", "preserve", "modify", "temporal_alignment", "rights_evidence_id", "quality_status", "split"], {
        "task": {"const": "video_to_video"}, "instruction": {"type": "string", "minLength": 1}, "source_media": _MEDIA_REF, "target_media": _MEDIA_REF, "quality_status": _STATUS,
        "split": {"enum": list(SPLITS)}, "preserve": {"type": "array", "items": {"type": "string"}}, "modify": {"type": "array", "items": {"type": "string"}}, "temporal_alignment": {"type": "object", "required": ["alignment_status"]},
    }),
    "media_schema.json": _schema("Media asset", ["asset_id", "source_video_id", "media_type", "repo_id", "path", "sha256", "bytes", "duration_seconds", "resolution_bucket", "rights_evidence_id", "decode_status"], {
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "upscaled": {"type": "boolean"}, "aspect_ratio_preserved": {"type": "boolean"},
    }),
}


def _card(task: str, repo_id: str, config: Any, report: dict[str, Any], splits: list[str], shell: bool) -> str:
    name = "Text-to-Video" if task == "t2v" else "Video-to-Video"
    configs = "\n".join(f"      - split: {s}\n        path: data/{s}.jsonl" for s in splits)
    front = f"---\nlicense: other\ntask_categories:\n  - {'text-to-video' if task == 't2v' else 'video-to-video'}\n" + (f"configs:\n  - config_name: default\n    data_files:\n{configs}\n" if splits else "") + "---\n"
    res = config.resolution
    body = f"""
# {repo_id.split('/')[-1]}

{name} training records built from the annotation dataset `{config.v3.legacy.repo_id or config.upload.repo_id}` (kept unchanged; revision: {legacy.legacy_revision(config)}).
{"**Repository shell: schemas and card only, no records or media yet.**" if shell else ""}

## What a record is

{"A prompt paired with a real target clip (`target_media`)." if task == "t2v" else "A real source clip (`source_media`), an editing instruction and the real transformed target clip (`target_media`). Source clips are referenced from the T2V repository by `repo_id` + `path` + `sha256` and are not duplicated here."}
Every media reference carries the sha256 of the file; `manifests/media_manifest.jsonl` is the asset registry (one row per physical file).

## Contents

| | |
|---|---|
| records | {report['records']} ({report['by_status'].get('accepted', 0)} accepted) |
| accepted by split | {json.dumps(report['accepted_by_split'])} |
| accepted by resolution bucket | {json.dumps(report['accepted_by_bucket'])} |
| source videos / groups | {report['source_videos']} / {report['source_groups']} |

Only `accepted` records are in `data/`. The reasons for every other record are in `quality/{task}_quality_report.json`.

## Preprocessing

- Resolution: target short edge {res.target} px. Larger sources are downscaled (Lanczos), aspect ratio preserved, no crop or pad; smaller sources keep their native size in their own bucket and are {"upscaled and labelled `upscaled: true`" if res.allow_upscale else "never upscaled"}. Sources under {res.experimental_min} px are excluded.
- Frame rate: the source frame rate is kept (`frame_sampling_profile: {config.v3.frame_sampling_profile}`); a model-specific sampling profile still has to be applied by the trainer.
- Clips are frame-exact re-encodes (libx264, CRF {config.v3.clips.crf}) of the annotated time spans. No latents are published: they are model-specific.
{"- Targets are deterministic FFmpeg edits (" + ", ".join(config.v3.v2v.transforms) + "); the exact filter is in `transform_parameters`. Slow motion re-times the same frames to a lower frame rate (no interpolated or repeated frames). No generative edits are included." if task == "v2v" else "- Prompts and cinematography labels are the existing measurement-based annotations (`annotation_origin: reused_legacy`); they were not regenerated."}

## Splits

Assigned per source group (channel + near-duplicate videos) before any clip or edit was derived, so every clip and every edit of one source is in a single split.

## Rights

Only sources with an `approved` entry in the rights manifest (explicit download / ML-training / derivative-processing permissions and stored evidence) are included; `manifests/rights_manifest.jsonl` lists the decision and attribution per source. Evidence documents are kept by the maintainer and are not published.

## Limitations

- Annotations are heuristic / measurement-based and inherit the limits of the source dataset.
- Automated pair and clip checks are measurements, not a human review.
- This card is not legal advice.
"""
    return front + body


# ----------------------------------------------------------------------------- staging
def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _public_asset(asset: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in asset.items() if k not in _PRIVATE_ASSET_KEYS}


def previously_verified(config: Any) -> set[tuple[str, str, str]]:
    """(repo_id, path, sha256) of every asset an earlier publish verified on the Hub."""
    path = config.v3_dir / "registry" / "media_manifest.jsonl"
    if not path.exists():
        return set()
    return {(r["repo_id"], r["path"], r["sha256"]) for r in read_jsonl(path) if r.get("upload_status") == "verified"}


def stage(config: Any, bundle: dict[str, Any], shell: bool = False) -> dict[str, Path]:
    """Build both repository trees under <work_dir>/staging (media hard-linked, nothing copied twice).
    An asset whose local file was already cleaned up after a verified upload is not staged again:
    it is on the Hub, and an upload never removes remote files."""
    done = previously_verified(config)
    repos = {"t2v": config.v3.repos.t2v, "v2v": config.v3.repos.v2v}
    out: dict[str, Path] = {}
    private = bool(config.v3.repos.private)
    rights_path = rights.default_rights_path(config)
    manifest = rights.load_rights(rights_path)
    for task in ("t2v", "v2v"):
        root = config.v3_dir / "staging" / task
        if root.exists():
            shutil.rmtree(root)
        ensure_dir(root)
        records = [] if shell else bundle[task]
        accepted = [r for r in records if r["quality_status"] == "accepted"]
        used: dict[str, dict[str, Any]] = {}
        for rec in accepted:
            for key in ("target_asset_id", "source_asset_id"):
                asset = bundle["assets"].get(rec.get(key) or "")
                if asset is None:
                    continue
                used[asset["asset_id"]] = asset
        for asset in used.values():
            if asset["repo_id"] != repos[task]:
                continue  # referenced from the other repository, stored only there
            ok, why = rights.may_publish_media(manifest.get(asset["source_video_id"]) or {}, private)
            if not ok:
                raise PublishError(f"{asset['source_video_id']}: {why}; make the repositories private or record redistribution_allowed")
            src = Path(asset["local_path"])
            if not src.exists() and (asset["repo_id"], asset["path"], asset["sha256"]) in done:
                continue
            if not src.exists():
                raise PublishError(f"asset file missing: {src} ({asset['asset_id']}); re-run `video-dataset v3 run`")
            if file_sha256(src) != asset["sha256"]:
                raise PublishError(f"asset changed on disk since it was registered: {src}")
            _link(src, root / asset["path"])
        splits = []
        for split in SPLITS:
            rows = [r for r in accepted if r.get("split") == split and r.get("primary_training", True)]
            if rows:
                write_jsonl(root / "data" / f"{split}.jsonl", rows)
                splits.append(split)
        experimental = [r for r in accepted if not r.get("primary_training", True)]
        if experimental:
            write_jsonl(root / "data" / "experimental_lowres.jsonl", experimental)
        write_jsonl(root / "manifests" / ("media_manifest.jsonl" if task == "t2v" else "pair_manifest.jsonl"), [_public_asset(a) for a in used.values()] if task == "t2v" else [
            {"sample_id": r["sample_id"], "source_asset_id": r["source_asset_id"], "source_media": r["source_media"], "target_asset_id": r["target_asset_id"], "target_media": r["target_media"], "transform": r["transform"], "split": r["split"]} for r in accepted
        ])
        if task == "v2v":
            write_jsonl(root / "manifests" / "media_manifest.jsonl", [_public_asset(a) for a in used.values()])
        vids = {r["source_video_id"] for r in accepted}
        write_jsonl(root / "manifests" / "rights_manifest.jsonl", [row for vid, row in sorted(bundle["rights"].items()) if vid in vids])
        for name, schema in SCHEMAS.items():
            if name.startswith("media") or name.startswith(task):
                write_json_atomic(root / "schemas" / name, schema)
        report = _quality_report(task, records, bundle)
        write_json_atomic(root / "quality" / f"{task}_quality_report.json", report)
        write_json_atomic(root / "dataset_info.json", {
            "name": repos[task], "task": "text_to_video" if task == "t2v" else "video_to_video", "schema_version": V3_SCHEMA_VERSION, "generated_at": report["generated_at"],
            "splits": {s: sum(1 for r in accepted if r.get("split") == s and r.get("primary_training", True)) for s in splits}, "records": len(accepted), "assets_stored_here": sum(1 for a in used.values() if a["repo_id"] == repos[task]),
            "bytes_stored_here": sum(int(a["bytes"]) for a in used.values() if a["repo_id"] == repos[task]), "resolution": config.resolution.model_dump(mode="json"),
            "legacy_dataset": config.v3.legacy.repo_id or config.upload.repo_id, "legacy_dataset_revision": legacy.legacy_revision(config), "shell": shell,
        })
        (root / "README.md").write_text(_card(task, repos[task], config, report, splits, shell), encoding="utf-8")
        out[task] = root
    return out


def staged_checksums(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)).replace(os.sep, "/"): file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def verify_remote(client: RepoClient, root: Path, readback: int, scratch: Path) -> dict[str, Any]:
    """Compare every staged file with the remote copy. LFS files: the sha256 the Hub stores; other
    files: downloaded and re-hashed. ``readback`` media files are additionally downloaded in full."""
    local = staged_checksums(root)
    remote = client.remote_files()
    missing = [p for p in local if p not in remote]
    mismatched: list[str] = []
    verified: list[str] = []
    media = [p for p in local if p.startswith("media/") and p in remote]
    reread = set(media[: max(0, int(readback))])
    for path, sha in local.items():
        if path in missing:
            continue
        if remote[path] and path not in reread:
            (verified if remote[path] == sha else mismatched).append(path)
            continue
        if remote[path] and remote[path] != sha:
            mismatched.append(path)
            continue
        try:
            got = file_sha256(client.download(path, ensure_dir(scratch)))
        except Exception as exc:
            log.warning("read-back of %s failed: %s", path, exc)
            mismatched.append(path)
            continue
        (verified if got == sha else mismatched).append(path)
    shutil.rmtree(scratch, ignore_errors=True)
    return {"ok": not missing and not mismatched, "files": len(local), "verified": len(verified), "read_back": len(reread), "missing_on_remote": missing, "mismatched": mismatched, "verified_paths": verified}


# ----------------------------------------------------------------------------- publish
def publish(config: Any, clients: dict[str, RepoClient] | None = None, dry_run: bool = False, shell: bool = False) -> dict[str, Any]:
    """Stage, upload, verify. ``dry_run`` stops after staging; ``shell`` publishes card + schemas
    only. Job flags and cleanup permission are set only after a full, verified publish."""
    bundle = aggregate(config)
    roots = stage(config, bundle, shell=shell)
    summary: dict[str, Any] = {"staged": {t: str(p) for t, p in roots.items()}, "dry_run": dry_run, "shell": shell,
                               "t2v_accepted": sum(1 for r in bundle["t2v"] if r["quality_status"] == "accepted"), "v2v_accepted": sum(1 for r in bundle["v2v"] if r["quality_status"] == "accepted")}
    if dry_run:
        return summary
    if clients is None:
        token = os.environ.get(config.v3.repos.token_env)
        if not token:
            raise PublishError(f"{config.v3.repos.token_env} is not set (write token, in the environment or .env)")
        clients = {"t2v": HubRepo(config.v3.repos.t2v, token, bool(config.v3.repos.private)), "v2v": HubRepo(config.v3.repos.v2v, token, bool(config.v3.repos.private))}
    results: dict[str, Any] = {}
    store = open_store(config)
    try:
        for task in ("t2v", "v2v"):  # T2V first: V2V references its source clips
            client = clients[task]
            client.ensure_repo()
            commit = client.upload_dir(roots[task], f"V3 {'shell' if shell else 'release'} {dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M UTC')}")
            results[task] = {"repo_id": client.repo_id, "commit": commit}
        if not shell:
            for vid in bundle["sources"]:
                if not store.reached(vid, "UPLOADED"):
                    store.advance(vid, "UPLOADED", "upload calls returned; not yet verified")
        all_ok = True
        verified_paths: dict[str, set[str]] = {}
        for task in ("t2v", "v2v"):
            check = verify_remote(clients[task], roots[task], int(config.v3.verify_readback_files), config.v3_dir / "staging" / f".readback_{task}")
            verified_paths[clients[task].repo_id] = set(check.pop("verified_paths"))
            results[task]["verification"] = check
            all_ok = all_ok and check["ok"]
        summary["repos"] = results
        summary["verified"] = all_ok
        write_json_atomic(config.v3_dir / "registry" / "last_publish.json", {**summary, "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")})
        if shell:
            return summary
        metadata_ok = all(not any(p for p in results[t]["verification"]["missing_on_remote"] + results[t]["verification"]["mismatched"] if not p.startswith("media/")) for t in results)
        accepted_assets: dict[str, list[dict[str, Any]]] = {}
        for rec in bundle["t2v"] + bundle["v2v"]:
            if rec["quality_status"] == "accepted":
                accepted_assets.setdefault(rec["source_video_id"], []).append(bundle["assets"][rec["target_asset_id"]])
        registry: list[dict[str, Any]] = []
        done = previously_verified(config)

        def _verified(a: dict[str, Any]) -> bool:
            return a["path"] in verified_paths.get(a["repo_id"], set()) or (a["repo_id"], a["path"], a["sha256"]) in done

        for vid in bundle["sources"]:
            mine = list({a["asset_id"]: a for a in accepted_assets.get(vid, [])}.values())
            visual_ok = all(_verified(a) for a in mine)
            for a in mine:
                registry.append({**_public_asset(a), "upload_status": "verified" if _verified(a) else "unverified"})
            if store.reached(vid, "COMPLETE"):
                continue  # already verified and cleaned by an earlier publish
            flags = {"metadata_upload_verified": bool(metadata_ok), "visual_asset_upload_verified": bool(visual_ok and metadata_ok) if mine else False,
                     "visual_assets": "published" if mine else "none", "cleanup_allowed": bool(metadata_ok and visual_ok)}
            if metadata_ok and visual_ok:
                store.advance(vid, "REMOTE_ASSETS_VERIFIED", f"{len(mine)} asset(s) verified on the Hub", **flags)
                store.advance(vid, "CLEANUP_ALLOWED", "metadata and visual assets verified remotely")
            else:
                store.set_flags(vid, **flags)
        write_jsonl(config.v3_dir / "registry" / "media_manifest.jsonl", registry)
    finally:
        store.close()
    return summary
