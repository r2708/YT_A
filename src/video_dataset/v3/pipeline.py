"""The V3 job for one source video (Changes 2-5):

    rights gate -> acquire -> probe -> decode check -> match legacy annotations -> extract clips
    -> T2V records + V2V pairs

Every step is checkpointed in the job store and its artifacts live under ``<work_dir>/jobs/<id>/``,
so an interrupted run resumes without repeating finished work. Existing annotations are reused; no
annotation model is run here. Uploading, remote verification and cleanup are in publish.py /
cleanup.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from video_dataset.utils.io import file_sha256, read_json, write_json_atomic
from video_dataset.utils.logging import get_logger
from video_dataset.v3 import V3_SCHEMA_VERSION, legacy, media, rights, v2v
from video_dataset.v3.resolution import choose_resolution, resolution_metadata
from video_dataset.v3.state import JobStore

log = get_logger("v3.pipeline")


def job_dir(config: Any, vid: str) -> Path:
    return config.v3_dir / "jobs" / vid


def open_store(config: Any) -> JobStore:
    return JobStore(config.v3_dir / "state.sqlite")


def _short(record_id: str) -> str:
    return record_id[5:] if record_id.startswith("clip_") else record_id


def clip_eligibility(clip: dict[str, Any], cfg: Any) -> str | None:
    """Why a legacy clip is not a training-clip candidate (None = candidate)."""
    status = (clip.get("validation") or {}).get("status")
    if status not in cfg.statuses:
        return f"legacy_status_{status}"
    if cfg.generation_targets and clip.get("generation_target") not in cfg.generation_targets:
        return f"generation_target_{clip.get('generation_target')}"
    if cfg.require_boundary_check and clip.get("boundary_check") != "passed":
        return f"boundary_check_{clip.get('boundary_check')}"
    duration = float(clip.get("end_time") or 0) - float(clip.get("start_time") or 0)
    if duration < float(cfg.min_duration):
        return "too_short"
    if duration > float(cfg.max_duration):
        return "too_long"
    if not str(clip.get("generation_prompt") or "").strip():
        return "no_prompt"
    return None


def match_source(row: dict[str, Any], probe: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """Does the reacquired file carry the video the legacy annotations describe?"""
    legacy_duration = row.get("duration")
    result: dict[str, Any] = {
        "legacy_duration": legacy_duration, "duration": probe.get("duration"), "legacy_source_sha256": row.get("source_sha256"), "sha256": probe.get("sha256"),
        "identical_file": bool(row.get("source_sha256")) and row.get("source_sha256") == probe.get("sha256"),
    }
    if legacy_duration is None:
        result.update(status="unverified", reason="the legacy record has no duration to compare with")
    else:
        delta = abs(float(probe.get("duration") or 0) - float(legacy_duration))
        result["duration_delta_seconds"] = round(delta, 3)
        ok = delta <= tolerance
        result.update(status="matched" if ok else "mismatch", reason=None if ok else f"duration differs by {delta:.2f}s (tolerance {tolerance}s): not the same edit of the video")
    result["method"] = "identical_file_sha256" if result["identical_file"] else "duration_and_timestamps"
    return result


class SourceJob:
    def __init__(self, config: Any, store: JobStore, row: dict[str, Any], clips: list[dict[str, Any]], entry: dict[str, Any] | None, rights_dir: Path):
        self.config, self.store, self.row, self.clips, self.entry, self.rights_dir = config, store, row, clips, entry, rights_dir
        self.vid = row["source_video_id"]
        self.dir = job_dir(config, self.vid)
        self.ffmpeg, self.ffprobe = config.project.ffmpeg_path, config.project.ffprobe_path

    # ------------------------------------------------------------------ driver
    def run(self, force: bool = False) -> str:
        store, vid = self.store, self.vid
        store.discover(vid)
        if force:
            store.reset(vid)
        job = store.get(vid) or {}
        if job.get("failure") == "FAILED_REVIEW" and not force:
            return "FAILED_REVIEW"
        if job.get("failure") == "FAILED_RETRYABLE" and int(job.get("attempts") or 0) >= int(self.config.v3.acquisition.max_attempts) and not force:
            return "FAILED_RETRYABLE"
        if not rights.is_approved(self.entry, self.rights_dir):
            problems = rights.entry_problems(self.entry, self.rights_dir) if self.entry else ["no entry in the rights manifest"]
            # no visual training asset is produced for this source; its annotations stay metadata-only
            store.advance(vid, "RIGHTS_PENDING", "; ".join(problems), visual_assets="none")
            return "RIGHTS_PENDING"
        if store.reached(vid, "TASK_ASSETS_CREATED"):
            return (store.get(vid) or {})["state"]
        try:
            store.advance(vid, "RIGHTS_APPROVED", None, visual_assets="pending", rights_evidence_id=rights.evidence_id(self.entry or {}))
            source = self._acquire()
            probe = self._probe(source)
            self._decode(source)
            match = self._match(probe)
            if match["status"] == "mismatch":
                self._write_records(probe, match, [], [], [], {"annotation_mismatch": len(self.clips)})
                store.fail(vid, f"annotation_mismatch: {match['reason']}", retryable=False)
                return "FAILED_REVIEW"
            self._build(source, probe, match)
        except media.AcquisitionError as exc:
            store.fail(vid, f"acquisition: {exc}", retryable=exc.retryable)
            return "FAILED_RETRYABLE" if exc.retryable else "FAILED_REVIEW"
        except Exception as exc:  # recorded, never skipped silently
            log.exception("v3 job %s failed", vid)
            store.fail(vid, f"{type(exc).__name__}: {exc}"[:800], retryable=True)
            return "FAILED_RETRYABLE"
        return "TASK_ASSETS_CREATED"

    # ------------------------------------------------------------------ steps
    def _acquire(self) -> Path:
        source, method = media.acquire(self.config, self.entry or {}, self.row, self.dir)
        self.store.advance(self.vid, "ACQUIRED", f"{method}: {source.name}")
        return source

    def _probe(self, source: Path) -> dict[str, Any]:
        path = self.dir / "probe.json"
        if path.exists() and (cached := read_json(path)).get("size_bytes") == source.stat().st_size:
            probe = cached
        else:
            probe = media.probe(source, self.ffprobe)
            write_json_atomic(path, probe)
        self.store.advance(self.vid, "PROBED", f"{probe['width']}x{probe['height']} {probe['fps']:.3f} fps {probe['duration']:.1f}s sha256 {probe['sha256'][:12]}")
        return probe

    def _decode(self, source: Path) -> None:
        marker = self.dir / "decode.json"
        mode = self.config.v3.acquisition.decode_check
        size = source.stat().st_size
        if not (marker.exists() and (prev := read_json(marker)).get("size_bytes") == size and prev.get("mode") == mode and prev.get("ok")):
            ok, err = media.decode_check(source, mode, self.ffmpeg)
            write_json_atomic(marker, {"ok": ok, "mode": mode, "size_bytes": size, "error": err})
            if not ok:
                raise media.AcquisitionError(f"decode_failed: {err}", retryable=False)
        self.store.advance(self.vid, "DECODE_VALIDATED", f"decode check: {mode}")

    def _match(self, probe: dict[str, Any]) -> dict[str, Any]:
        match = match_source(self.row, probe, float(self.config.v3.acquisition.duration_tolerance_seconds))
        if match["status"] != "mismatch":
            self.store.advance(self.vid, "LEGACY_ANNOTATIONS_MATCHED", f"{match['status']} ({match['method']})")
        return match

    def _build(self, source: Path, probe: dict[str, Any], match: dict[str, Any]) -> None:
        cfg, ccfg = self.config, self.config.v3.clips
        decision = choose_resolution(int(probe["width"]), int(probe["height"]), cfg.resolution)
        res_meta = resolution_metadata(int(probe["width"]), int(probe["height"]), probe.get("fps"), decision, probe.get("fps"), cfg.v3.frame_sampling_profile)
        entry = self.entry or {}
        ev_id = rights.evidence_id(entry)
        revision = legacy.legacy_revision(cfg)
        skipped: dict[str, int] = {}
        assets: list[dict[str, Any]] = []
        t2v: list[dict[str, Any]] = []
        pairs: list[dict[str, Any]] = []
        duration = float(probe.get("duration") or 0.0)
        for clip in sorted(self.clips, key=lambda c: (float(c.get("start_time") or 0), str(c.get("record_id")))):
            reason = clip_eligibility(clip, ccfg)
            if reason is None and decision.excluded:
                reason = "excluded_lowres"
            if reason is None and float(clip.get("end_time") or 0) > duration + float(ccfg.duration_tolerance_seconds):
                reason = "annotation_mismatch_outside_source"
            if reason is not None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            asset, record = self._clip(source, probe, clip, decision, res_meta, match, ev_id, revision)
            assets.append(asset)
            t2v.append(record)
            if cfg.v3.v2v.enabled and record["quality_status"] == "accepted" and entry.get("derivative_processing_allowed") is True:
                for name in v2v.pick_transforms(clip["record_id"], list(cfg.v3.v2v.transforms), int(cfg.v3.v2v.pairs_per_clip)):
                    tgt_asset, pair = self._pair(asset, record, clip, name, ev_id)
                    assets.append(tgt_asset)
                    pairs.append(pair)
        self.store.advance(self.vid, "CLIPS_EXTRACTED", f"{len(t2v)} clip(s), bucket {decision.bucket}")
        self._write_records(probe, match, assets, t2v, pairs, skipped)
        accepted = sum(1 for r in t2v if r["quality_status"] == "accepted")
        self.store.advance(self.vid, "TASK_ASSETS_CREATED", f"t2v {accepted}/{len(t2v)} accepted, v2v {sum(1 for p in pairs if p['quality_status'] == 'accepted')}/{len(pairs)} accepted", visual_assets="created" if accepted else "none")

    def _clip(self, source: Path, probe: dict[str, Any], clip: dict[str, Any], decision: Any, res_meta: dict[str, Any], match: dict[str, Any], ev_id: str, revision: str) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg, ccfg = self.config, self.config.v3.clips
        rid = str(clip["record_id"])
        start, end = float(clip["start_time"]), float(clip["end_time"])
        out = self.dir / "clips" / f"{rid}.mp4"
        key = f"{start:.3f}|{end:.3f}|{decision.target_width}x{decision.target_height}|{ccfg.crf}|{ccfg.preset}|{bool(ccfg.keep_audio)}|{probe['sha256']}"
        sidecar = out.with_suffix(".json")
        if not (out.exists() and out.stat().st_size > 0 and sidecar.exists() and read_json(sidecar).get("key") == key):
            media.extract_clip(source, out, start, end, decision, int(probe["width"]), int(probe["height"]), ccfg, bool(probe.get("has_audio")), self.ffmpeg)
            write_json_atomic(sidecar, {"key": key})
        m = media.measure_clip(out, self.ffprobe)
        sha = file_sha256(out)
        reasons: list[str] = []
        status = "accepted"
        expected = end - start
        if not m.ok:
            status, reasons = "decode_failed", [m.error or "clip does not decode"]
        else:
            fps = float(probe.get("fps") or m.fps or 0)
            if abs(m.duration - expected) > float(ccfg.duration_tolerance_seconds):
                reasons.append(f"duration {m.duration:.3f}s differs from the annotated span {expected:.3f}s")
            if fps and abs(m.frame_count - round(expected * fps)) > int(ccfg.frame_count_tolerance):
                reasons.append(f"frame count {m.frame_count} differs from the expected {round(expected * fps)}")
            if reasons:
                status = "annotation_mismatch"
            else:
                if m.black_fraction > float(ccfg.max_black_fraction):
                    reasons.append(f"black frames: {m.black_fraction:.0%} of sampled frames")
                if float(ccfg.min_sharpness) > 0 and m.sharpness < float(ccfg.min_sharpness):
                    reasons.append(f"blurry: sharpness {m.sharpness:.1f} < {ccfg.min_sharpness}")
                if float(ccfg.min_motion) > 0 and m.motion < float(ccfg.min_motion):
                    reasons.append(f"frozen: motion {m.motion:.4f} < {ccfg.min_motion}")
                cuts = media.count_cuts(out, float(cfg.scene_detection.threshold), float(cfg.scene_detection.min_scene_len_seconds)) if ccfg.single_shot_check else None
                if cuts:
                    reasons.append(f"not a single shot: PySceneDetect found {cuts} cut(s) inside the clip")
                if reasons:
                    status = "rejected"
        prompt_check = "not_run"
        if ccfg.prompt_check:
            prompt_check, why = media.prompt_problem(str(clip.get("generation_prompt") or ""), cfg.qa.spacy_model)
            if why and status == "accepted":
                status, reasons = "needs_human_review", [why]
        asset_id = f"asset_{sha[:16]}"
        path_in_repo = f"media/{decision.bucket}/{rid}.mp4"
        asset = {
            "asset_id": asset_id, "role": "t2v_target", "source_video_id": self.vid, "clip_id": rid, "media_type": "video",
            "repo_id": cfg.v3.repos.t2v, "path": path_in_repo, "local_path": str(out), "sha256": sha, "bytes": out.stat().st_size,
            "duration_seconds": round(m.duration, 3), "frame_count": m.frame_count, "start_seconds": start, "end_seconds": end,
            **res_meta, "rights_evidence_id": ev_id, "decode_status": "passed" if m.ok else "failed", "upload_status": "pending",
            "source_sha256": probe["sha256"], "fingerprint": m.hashes, "measurements": {"black_fraction": round(m.black_fraction, 3), "sharpness": round(m.sharpness, 2), "motion": round(m.motion, 5)},
        }
        camera = clip.get("camera") or {}
        record = {
            "sample_id": f"t2v_{_short(rid)}", "task": "text_to_video", "schema_version": V3_SCHEMA_VERSION,
            "source_group_id": self.row.get("source_group_id") or self.vid, "source_video_id": self.vid, "clip_id": rid,
            "prompt": str(clip.get("generation_prompt") or "").strip(), "prompt_variants": list(clip.get("prompt_variants") or []),
            "target_asset_id": asset_id, "target_media": {"repo_id": cfg.v3.repos.t2v, "path": path_in_repo, "sha256": sha},
            "timeline": [{"start_seconds": 0.0, "end_seconds": round(expected, 3), "description": clip.get("description")}],
            "cinematography": {
                "shot_type": camera.get("shot_type"), "camera_angle": camera.get("camera_angle"), "camera_movement": camera.get("movement"),
                "camera_speed": camera.get("speed"), "lighting": ((clip.get("lighting") or {}).get("level") if isinstance(clip.get("lighting"), dict) else None),
            },
            "source_start_seconds": start, "source_end_seconds": end, "duration_seconds": round(m.duration, 3), "frame_count": m.frame_count,
            **res_meta, "primary_training": bool(decision.include_in_primary_training),
            "technical_profile": cfg.v3.frame_sampling_profile, "rights_evidence_id": ev_id,
            "annotation_origin": "reused_legacy", "legacy_dataset_revision": revision, "legacy_record_id": rid,
            "legacy_validation_status": (clip.get("validation") or {}).get("status"),
            "annotation_validation": "passed" if status not in ("annotation_mismatch", "decode_failed") else "failed", "prompt_check": prompt_check,
            "visual_grounding_status": "legacy_measurements_reused" if match["identical_file"] else "legacy_measurements_reused_timestamps_matched",
            "last_validated_against_asset": asset_id,
            "quality_status": status, "quality_reasons": reasons, "split": self.row.get("split"),
        }
        return asset, record

    def _pair(self, src_asset: dict[str, Any], record: dict[str, Any], clip: dict[str, Any], name: str, ev_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg = self.config
        spec = v2v.TRANSFORMS[name]
        rid = str(clip["record_id"])
        src = Path(src_asset["local_path"])
        out = self.dir / "targets" / f"{rid}__{name}.mp4"
        key = f"{src_asset['sha256']}|{name}|{spec['filter']}|{cfg.v3.clips.crf}|{cfg.v3.clips.preset}"
        sidecar = out.with_suffix(".json")
        if not (out.exists() and out.stat().st_size > 0 and sidecar.exists() and read_json(sidecar).get("key") == key):
            v2v.apply_transform(src, out, name, cfg.v3.clips, self.ffmpeg)
            write_json_atomic(sidecar, {"key": key})
        sm, tm = media.measure_clip(src, self.ffprobe), media.measure_clip(out, self.ffprobe)
        status, checks = v2v.validate_pair(src, out, name, sm, tm, cfg.v3.v2v, float(cfg.v3.clips.duration_tolerance_seconds), int(cfg.v3.clips.frame_count_tolerance))
        sha = file_sha256(out)
        asset_id = f"asset_{sha[:16]}"
        bucket = src_asset["resolution_bucket"]
        path_in_repo = f"media/{bucket}/{rid}__{name}.mp4"
        keep = ("source_width", "source_height", "source_fps", "training_width", "training_height", "training_fps", "resolution_bucket", "resolution_action", "aspect_ratio_preserved", "upscaled", "crop_applied", "pad_applied", "frame_sampling_profile")
        asset = {
            "asset_id": asset_id, "role": "v2v_target", "source_video_id": self.vid, "clip_id": rid, "media_type": "video",
            "repo_id": cfg.v3.repos.v2v, "path": path_in_repo, "local_path": str(out), "sha256": sha, "bytes": out.stat().st_size,
            "duration_seconds": round(tm.duration, 3), "frame_count": tm.frame_count, **{k: src_asset.get(k) for k in keep},
            "rights_evidence_id": ev_id, "decode_status": "passed" if tm.ok else "failed", "upload_status": "pending",
            "derived_from_asset_id": src_asset["asset_id"], "transform": name, "fingerprint": tm.hashes,
        }
        speed = float(spec["speed_factor"])
        pair = {
            "sample_id": f"v2v_{_short(rid)}__{name}", "task": "video_to_video", "schema_version": V3_SCHEMA_VERSION,
            "source_group_id": record["source_group_id"], "source_video_id": self.vid, "clip_id": rid,
            "source_asset_id": src_asset["asset_id"], "source_media": {"repo_id": src_asset["repo_id"], "path": src_asset["path"], "sha256": src_asset["sha256"]},
            "instruction": spec["instruction"], "transform": name, "transform_method": "ffmpeg_filter", "transform_parameters": spec["filter"],
            "target_asset_id": asset_id, "target_media": {"repo_id": cfg.v3.repos.v2v, "path": path_in_repo, "sha256": sha},
            "preserve": list(spec["preserve"]), "modify": list(spec["modify"]), "source_prompt": record["prompt"],
            "temporal_alignment": {
                "source_start_seconds": 0.0, "target_start_seconds": 0.0, "duration_seconds": round(sm.duration, 3), "target_duration_seconds": round(tm.duration, 3),
                "speed_factor": speed, "source_fps": round(sm.fps, 3), "target_fps": round(tm.fps, 3), "temporal_method": spec.get("temporal_method"), "alignment_status": checks.get("alignment_status", "failed"),
            },
            "pair_checks": checks, "resolution_bucket": bucket, "primary_training": record["primary_training"],
            "rights_evidence_id": ev_id, "quality_status": status, "quality_reasons": [] if status == "accepted" else [f"{k}=false" for k, v in checks.items() if v is False] or [str(checks.get("error"))],
            "split": record["split"],
        }
        return asset, pair

    def _write_records(self, probe: dict[str, Any], match: dict[str, Any], assets: list[dict[str, Any]], t2v: list[dict[str, Any]], pairs: list[dict[str, Any]], skipped: dict[str, int]) -> None:
        from video_dataset.utils.device import runtime_info

        write_json_atomic(self.dir / "records.json", {
            "schema_version": V3_SCHEMA_VERSION, "source_video_id": self.vid, "source_group_id": self.row.get("source_group_id") or self.vid, "split": self.row.get("split"),
            "source": {k: probe.get(k) for k in ("sha256", "duration", "fps", "width", "height", "video_codec", "pix_fmt", "has_audio", "size_bytes", "frame_count")},
            "match": match, "legacy_clips": len(self.clips), "skipped": skipped, "assets": assets, "t2v": t2v, "v2v": pairs,
            "runtime": runtime_info(self.config.project.device),
        })


def run(config: Any, video_ids: list[str] | None = None, limit: int | None = None, force: bool = False) -> dict[str, Any]:
    """Process every source of the legacy index (or ``video_ids``). Sources without an approved rights
    entry stop at RIGHTS_PENDING and are never downloaded."""
    index = legacy.load_source_index(config)
    rights_path = rights.default_rights_path(config)
    manifest = rights.load_rights(rights_path)
    wanted = set(video_ids or [])
    rows = [r for r in index if not wanted or r["source_video_id"] in wanted]
    approved = [r for r in rows if rights.is_approved(manifest.get(r["source_video_id"]), rights_path.parent)]
    if limit:
        approved = approved[: int(limit)]
    approved_ids = {r["source_video_id"] for r in approved}
    clips_by_video: dict[str, list[dict[str, Any]]] = {}
    if approved_ids:
        for clip in legacy.load_records(legacy.legacy_root(config), "clips"):
            if clip.get("video_id") in approved_ids:
                clips_by_video.setdefault(clip["video_id"], []).append(clip)
    store = open_store(config)
    results: dict[str, str] = {}
    try:
        for row in rows:
            vid = row["source_video_id"]
            if vid not in approved_ids and rights.is_approved(manifest.get(vid), rights_path.parent):
                continue  # approved but beyond --limit: left for the next run
            job = SourceJob(config, store, row, clips_by_video.get(vid, []), manifest.get(vid), rights_path.parent)
            results[vid] = job.run(force=force and vid in approved_ids)
            if vid in approved_ids:
                log.info("v3 %s -> %s", vid, results[vid])
        summary = store.summary()
    finally:
        store.close()
    return {"sources": len(rows), "approved": len(approved_ids), "results": results, "states": summary}


def load_job_records(config: Any, store: JobStore) -> list[dict[str, Any]]:
    out = []
    for job in store.all():
        path = job_dir(config, job["source_video_id"]) / "records.json"
        if path.exists() and not job["failure"] and store.reached(job["source_video_id"], "TASK_ASSETS_CREATED"):
            out.append(json.loads(path.read_text(encoding="utf-8")))
    return out
