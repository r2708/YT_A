"""Stage functions behind the CLI. Each one streams records from the state store, skips work
whose inputs have not changed, checkpoints after every record and returns measured counts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataset_factory import PIPELINE_VERSION
from dataset_factory.captions.cache import ProviderCache, cache_key
from dataset_factory.captions.heuristics import REVIEW_METHOD, SCORE_BASIS, check_caption
from dataset_factory.captions.providers import ReviewRequest, make_provider, parse_provider_json
from dataset_factory.captions.reuse import choose_caption
from dataset_factory.config import FactoryConfig
from dataset_factory.curation.deduplicate import group_duplicates
from dataset_factory.curation.eligibility import decide, load_rights_evidence, resolve_rights
from dataset_factory.curation.splits import assign, load_persisted, persist
from dataset_factory.curation.validate import asset_signature, resolve_asset, validate_media
from dataset_factory.exports.canonical import write_canonical
from dataset_factory.exports.release import export_version, promote, staging_dir, write_release_manifest
from dataset_factory.exports.wan_i2v import write_i2v
from dataset_factory.exports.wan_t2v import write_t2v
from dataset_factory.media.frame_extract import extract_start_frame, verify_pair
from dataset_factory.reports.summary import summarize, write_summary
from dataset_factory.schema.canonical import CanonicalRecord, Keyframe
from dataset_factory.schema.source_mapping import map_source_row
from dataset_factory.source.local_source import iter_local_rows
from dataset_factory.state import StateStore

Progress = Callable[[str], None] | None


def _h(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def open_state(cfg: FactoryConfig) -> StateStore:
    return StateStore(cfg.path("state_db"))


def _records(store: StateStore, limit: int | None) -> Iterator[CanonicalRecord]:
    yield from store.iter_records(limit=limit)


# ------------------------------------------------------------------ ingest
def ingest(cfg: FactoryConfig, *, limit: int | None = None, source_root: Path | None = None, limit_shards: int | None = None, log: Progress = None) -> dict[str, Any]:
    """Map source rows into the manifest. Unchanged rows (same source hash) keep their processed
    state; changed rows are replaced and every downstream step re-runs for them."""
    if cfg.source.kind == "huggingface" and source_root is None:
        from dataset_factory.source.huggingface_source import ingest_shards, iter_source_rows

        raw = cfg.path("raw_dir") / cfg.source.repo_id.replace("/", "__")
        shards = ingest_shards(cfg, raw, limit_shards)
        rows = iter_source_rows(cfg, shards)
        source_name = cfg.source.repo_id
    else:
        root = source_root or Path(cfg.source.local_dir or "")
        if not root or not Path(root).exists():
            raise FileNotFoundError(f"local source directory not found: {root or '(source.local_dir unset)'}")
        rows = iter_local_rows(cfg, Path(root))
        source_name = str(root)
    counts = {"seen": 0, "new": 0, "changed": 0, "unchanged": 0}
    with open_state(cfg) as store:
        for rtype, row, meta in rows:
            if limit and counts["seen"] >= limit:
                break
            counts["seen"] += 1
            rec = map_source_row(row, cfg, source_dataset=source_name, source_config=rtype, video_meta=meta)
            if meta:
                rec.source_payload["_video_meta"] = {k: meta.get(k) for k in ("url", "license", "license_status", "rights_status", "channel", "title", "canonical_video_id") if meta.get(k) is not None}
            old = store.source_hash(rec.record_id)
            if old == rec.source_record_hash:
                counts["unchanged"] += 1
                continue
            counts["changed" if old else "new"] += 1
            store.upsert(rec)
            if counts["seen"] % 500 == 0:
                store.commit()
                if log:
                    log(f"ingested {counts['seen']} rows")
        store.commit()
        store.set_meta("last_ingest", {**counts, "source": source_name, "at": datetime.now(UTC).isoformat(timespec='seconds')})
        counts["total_records"] = store.count()
    return counts


# ------------------------------------------------------------------ validate
def validate(cfg: FactoryConfig, *, limit: int | None = None, force: bool = False, log: Progress = None) -> dict[str, Any]:
    rules_hash = cfg.section_hash("media")
    counts = {"checked": 0, "skipped_current": 0, "passed": 0, "failed": 0, "needs_review": 0, "not_checked": 0}
    with open_state(cfg) as store:
        for n, rec in enumerate(_records(store, limit), start=1):
            sig = asset_signature(resolve_asset(rec, cfg))
            ih = _h(sig, rules_hash, PIPELINE_VERSION, rec.source_record_hash, cfg.source.media_root)
            if not force and store.step_current(rec.record_id, "media", ih):
                counts["skipped_current"] += 1
                continue
            validate_media(rec, cfg)
            store.upsert(rec)
            store.mark_step(rec.record_id, "media", ih, rec.media_validation_status, {"reasons": rec.media_validation_reasons})
            counts["checked"] += 1
            counts[rec.media_validation_status] = counts.get(rec.media_validation_status, 0) + 1
            if n % 100 == 0:
                store.commit()
                if log:
                    log(f"validated {n} records")
        store.commit()
    return counts


# ------------------------------------------------------------------ frames
def extract_frames(cfg: FactoryConfig, *, limit: int | None = None, force: bool = False, log: Progress = None) -> dict[str, Any]:
    rules_hash = cfg.section_hash("frames", "media")
    counts = {"extracted": 0, "skipped_current": 0, "skipped_no_media": 0, "failed": 0}
    frames_dir = cfg.path("frames_dir")
    with open_state(cfg) as store:
        for n, rec in enumerate(_records(store, limit), start=1):
            if rec.media_validation_status not in ("passed", "needs_review") or not rec.video_sha256:
                counts["skipped_no_media"] += 1
                continue
            ih = _h(rec.video_sha256, rules_hash, PIPELINE_VERSION)
            if not force and store.step_current(rec.record_id, "frame", ih) and rec.start_frame_path and (cfg.data_root / rec.start_frame_path).exists():
                counts["skipped_current"] += 1
                continue
            clip = resolve_asset(rec, cfg)
            res = extract_start_frame(clip, frames_dir / rec.record_id.replace(":", "_"), "start", cfg.frames, cfg.media)  # type: ignore[arg-type]
            rec.start_frame_status = res.status
            rec.start_frame_reasons = res.reasons
            rec.start_frame_extraction = res.settings
            if res.status == "passed" and res.path:
                rec.start_frame_path = res.path.relative_to(cfg.data_root).as_posix()
                rec.start_frame_timestamp_seconds = res.timestamp_seconds
                rec.start_frame_sha256, rec.start_frame_width, rec.start_frame_height = res.sha256, res.width, res.height
                rec.start_frame_dhash = res.dhash
                rec.start_frame_clip_sha256 = rec.video_sha256
                rec.keyframe_paths = [Keyframe(path=rec.start_frame_path, timestamp_seconds=float(res.timestamp_seconds or 0.0), sha256=res.sha256, width=res.width, height=res.height)]
                counts["extracted"] += 1
            else:
                counts["failed"] += 1
            store.upsert(rec)
            store.mark_step(rec.record_id, "frame", ih, res.status, {"reasons": res.reasons})
            if n % 50 == 0:
                store.commit()
                if log:
                    log(f"frames: {counts['extracted']} extracted")
        store.commit()
    return counts


# ------------------------------------------------------------------ captions
def review_captions(cfg: FactoryConfig, *, limit: int | None = None, max_calls: int | None = None, dry_run: bool = False, select_all: bool = False, offline: bool | None = None, log: Progress = None) -> dict[str, Any]:
    offline_mode = cfg.offline if offline is None else offline
    provider = make_provider(cfg.provider, offline_mode)
    budget = cfg.provider.max_calls if max_calls is None else max_calls
    cache = ProviderCache(cfg.path("cache_dir") / "captions.sqlite") if cfg.provider.cache else None
    counts = {"records": 0, "rules_passed": 0, "rules_failed": 0, "rules_review": 0, "provider_calls": 0, "cache_hits": 0, "provider_applied": 0, "provider_invalid": 0, "would_call": 0}
    rules_hash = cfg.section_hash("captions", "mapping", "provider")
    with open_state(cfg) as store:
        dup: dict[str, int] = {}
        for rec in store.iter_records():
            text, _src = choose_caption(rec, cfg.mapping.caption_priority)
            if text:
                dup[text] = dup.get(text, 0) + 1
        for n, rec in enumerate(_records(store, limit), start=1):
            counts["records"] += 1
            text, src = choose_caption(rec, cfg.mapping.caption_priority)
            chk = check_caption(text, cfg.captions, duplicate_count=dup.get(text or "", 0), actions=len(rec.actions))
            rec.caption_final, rec.caption_source = text, src
            rec.caption_quality_status, rec.caption_quality_score, rec.caption_quality_reasons = chk.status, chk.score, chk.reasons
            rec.caption_quality_score_basis = SCORE_BASIS
            rec.caption_review_method, rec.caption_model, rec.caption_prompt_version = REVIEW_METHOD, None, None
            counts[{"passed": "rules_passed", "failed": "rules_failed", "needs_review": "rules_review"}[chk.status]] += 1
            wants_provider = provider.name != "none" and text and (select_all or chk.status != "passed")
            if wants_provider:
                key = cache_key(asset=rec.video_sha256 or rec.source_record_hash, text=text, operation="review", template_version=cfg.provider.template_version, provider=provider.name, model=provider.model, settings=cfg.provider.settings)
                cached = cache.get(key) if cache else None
                if cached is not None:
                    counts["cache_hits"] += 1
                    _apply_provider(rec, cached, provider.name, cfg)
                    counts["provider_applied"] += 1
                elif dry_run:
                    counts["would_call"] += 1
                elif budget and counts["provider_calls"] >= budget:
                    rec.caption_quality_reasons.append("provider_budget_exhausted")
                else:
                    counts["provider_calls"] += 1
                    res = provider.review(ReviewRequest(rec, text, [rec.start_frame_path] if rec.start_frame_path else []))
                    out, why = (res.output, None) if res.output is not None else parse_provider_json(res.raw) if res.raw else (None, res.error or "no_output")
                    if out is None:
                        counts["provider_invalid"] += 1
                        rec.caption_quality_status = "needs_review"
                        rec.caption_quality_reasons.append(f"provider_output_invalid:{why}")
                    else:
                        result = out.model_dump()
                        if cache:
                            cache.put(key, provider=provider.name, model=provider.model, operation="review", raw_output=res.raw, result=result)
                        _apply_provider(rec, result, provider.name, cfg)
                        counts["provider_applied"] += 1
            if not dry_run:
                store.upsert(rec)
                store.mark_step(rec.record_id, "caption", _h(rules_hash, rec.source_record_hash, PIPELINE_VERSION), rec.caption_quality_status)
            if n % 500 == 0:
                store.commit()
                if log:
                    log(f"captions: {n} checked")
        if not dry_run:
            store.commit()
    if cache:
        cache.close()
    return counts


def _apply_provider(rec: CanonicalRecord, result: dict[str, Any], name: str, cfg: FactoryConfig) -> None:
    """A validated provider verdict refines the record; the original caption stays untouched."""
    verdict = str(result.get("verdict"))
    rec.caption_review_method = f"provider:{name}"
    rec.caption_model = cfg.provider.model
    rec.caption_prompt_version = cfg.provider.template_version
    if verdict == "supported" and result.get("caption"):
        rec.caption_final = str(result["caption"])
        rec.caption_source = f"provider:{name}"
        if rec.caption_quality_status == "needs_review" and not any(r.startswith(("template_residue", "shared_by")) for r in rec.caption_quality_reasons):
            rec.caption_quality_status = "passed"
    elif verdict == "contradicted":
        rec.caption_quality_status = "failed"
        rec.caption_quality_reasons.append("provider:contradicted")
    else:
        rec.caption_quality_status = "needs_review"
        rec.caption_quality_reasons.append("provider:uncertain")


# ------------------------------------------------------------------ deduplicate
def deduplicate(cfg: FactoryConfig, log: Progress = None) -> dict[str, Any]:
    with open_state(cfg) as store:
        light = [CanonicalRecord(record_id=r.record_id, source_dataset=r.source_dataset, video_sha256=r.video_sha256, video_fingerprint=r.video_fingerprint, canonical_video_id=r.canonical_video_id, original_video_id=r.original_video_id) for r in store.iter_records()]
        groups = group_duplicates(light, cfg.dedup)
        counts = {"records": len(light), "in_groups": len(groups), "exact_groups": len({g["group"] for g in groups.values() if g["kind"] == "exact"}), "near_groups": len({g["group"] for g in groups.values() if g["kind"] == "near"})}
        for rec in list(store.iter_records()):
            g = groups.get(rec.record_id)
            new = (g["group"], g["kind"], g["canonical"]) if g else (None, None, None)
            if (rec.duplicate_group_id, rec.duplicate_kind, rec.duplicate_canonical) != new:
                rec.duplicate_group_id, rec.duplicate_kind, rec.duplicate_canonical = new  # type: ignore[assignment]
                flags = [f for f in rec.quality_flags if not f.startswith("duplicate_")]
                if g:
                    flags.append(f"duplicate_{g['kind']}" + ("" if g["canonical"] else "_of_group"))
                rec.quality_flags = flags
                store.upsert(rec)
        store.commit()
        store.set_meta("duplicate_links", sorted({tuple(sorted((a.canonical_video_id or "", b.canonical_video_id or ""))) for a in light for b in light if a.record_id < b.record_id and a.record_id in groups and b.record_id in groups and groups[a.record_id]["group"] == groups[b.record_id]["group"] and a.canonical_video_id != b.canonical_video_id}))
    return counts


# ------------------------------------------------------------------ curate
def curate(cfg: FactoryConfig, log: Progress = None) -> dict[str, Any]:
    evidence = load_rights_evidence(cfg.rights, cfg.data_root)
    version = export_version(cfg)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    counts = {"records": 0, "t2v_eligible": 0, "i2v_eligible": 0, "rights": {}, "validation": {}, "split_changes": 0}
    with open_state(cfg) as store:
        groups = store.video_ids()
        persisted = load_persisted(cfg.path("curated_dir") / cfg.splits.persist_file)
        links = [(str(x[0]), str(x[1])) for x in (store.get_meta("duplicate_links") or []) if all(x)]
        assignment, changes = assign(groups, cfg.splits, persisted, links)
        persist(cfg.path("curated_dir") / cfg.splits.persist_file, assignment, cfg.splits, changes)
        counts["split_changes"] = len(changes)
        for n, rec in enumerate(store.iter_records(), start=1):
            resolve_rights(rec, cfg.rights, evidence, rec.source_payload.get("_video_meta"))
            if rec.start_frame_path and rec.start_frame_status == "passed":
                clip, frame = resolve_asset(rec, cfg), cfg.data_root / rec.start_frame_path
                if clip is not None and rec.start_frame_timestamp_seconds is not None:
                    ok, why = verify_pair(clip, frame, rec.start_frame_timestamp_seconds, rec.start_frame_dhash, cfg.frames.consistency_hamming)
                    if not ok:
                        rec.start_frame_status = "failed"
                        rec.start_frame_reasons = [why]
            rec.split = assignment.get(rec.canonical_video_id or "")
            decide(rec, cfg)
            rec.dataset_version, rec.processed_at_utc, rec.pipeline_version, rec.processing_config_hash = version, now, PIPELINE_VERSION, cfg.config_hash
            store.upsert(rec)
            counts["records"] += 1
            counts["t2v_eligible"] += int(rec.t2v_eligible)
            counts["i2v_eligible"] += int(rec.i2v_eligible)
            counts["rights"][rec.source_rights_status] = counts["rights"].get(rec.source_rights_status, 0) + 1
            counts["validation"][rec.validation_status] = counts["validation"].get(rec.validation_status, 0) + 1
            if n % 500 == 0:
                store.commit()
                if log:
                    log(f"curated {n} records")
        store.commit()
        store.set_meta("last_curate", {"at": now, "version": version, **{k: v for k, v in counts.items() if not isinstance(v, dict)}})
    return counts


# ------------------------------------------------------------------ export
def export(cfg: FactoryConfig, target_name: str, *, overwrite: bool = False, log: Progress = None) -> dict[str, Any]:
    target = cfg.exports.targets.get(target_name)
    if target is None:
        raise KeyError(f"unknown export target '{target_name}'; configured: {', '.join(cfg.exports.targets)}")
    version = export_version(cfg)
    final = cfg.path("exports_dir") / target_name / version
    staging = staging_dir(final)
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    overrides_used: list[str] = []
    with open_state(cfg) as store:
        if target.kind == "canonical":
            counts: dict[str, Any] = write_canonical(store.iter_records(), staging)
        else:
            want = (lambda r: r.t2v_eligible) if target.kind == "t2v" else (lambda r: r.i2v_eligible)

            def selected() -> Iterator[CanonicalRecord]:
                for r in store.iter_records():
                    if not want(r):
                        continue
                    if target.require_rights and r.source_rights_status not in cfg.rights.export_allowed:
                        if r.source_rights_status in cfg.rights.export_override:
                            overrides_used.append(r.record_id)
                        else:
                            continue
                    yield r

            resolve = lambda r: resolve_asset(r, cfg)  # noqa: E731
            resolve_frame = lambda r: (cfg.data_root / r.start_frame_path) if r.start_frame_path else None  # noqa: E731
            counts = write_t2v(selected(), staging, target, resolve) if target.kind == "t2v" else write_i2v(selected(), staging, target, resolve, resolve_frame)
    # every exported path must exist and be readable
    missing = []
    data_file = staging / target.data_file
    if data_file.exists():
        for line in data_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for key in ("video", "image"):
                if row.get(key):
                    p = (staging / row[key]) if target.packaging != "reference" else (staging / row[key]).resolve()
                    if not p.exists():
                        missing.append(row[key])
    extra: dict[str, Any] = {"adapter": counts.get("adapter"), "packaging": target.packaging, "target_config": target.model_dump(mode="json"), "rights_override_records": len(overrides_used), "missing_paths": missing[:20]}
    if overrides_used:
        extra["WARNING"] = f"{len(overrides_used)} record(s) exported under rights.export_override {cfg.rights.export_override}: not cleared rights"
    write_release_manifest(staging, cfg, target=target_name, version=version, counts=counts, extra=extra)
    if missing:
        raise RuntimeError(f"export aborted: {len(missing)} referenced path(s) missing, see {staging}/release_manifest.json")
    promote(staging, final, overwrite)
    return {"target": target_name, "version": version, "path": str(final), "counts": counts, "rights_override_records": len(overrides_used)}


# ------------------------------------------------------------------ report
def report(cfg: FactoryConfig) -> dict[str, Any]:
    with open_state(cfg) as store:
        summary = summarize(store.iter_records())
        extra = {"last_ingest": store.get_meta("last_ingest"), "last_curate": store.get_meta("last_curate"), "pipeline_version": PIPELINE_VERSION, "config_hash": cfg.config_hash, "media_root": cfg.source.media_root}
    js, md = write_summary(summary, cfg.path("reports_dir"), extra)
    return {"summary": summary, "json": str(js), "markdown": str(md)}
