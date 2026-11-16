"""Rights and provenance gate (spec section 3 / Change 2).

The rights manifest is a JSONL file with one entry per source video. Only an entry whose status is
``approved`` *and* whose explicit booleans and evidence are complete lets a source through; a public
URL, a successful download or a platform licence label never does. The pipeline never writes
``approved`` itself: `rights-init` creates every entry as ``metadata_only`` and a person edits it."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from video_dataset.utils.ids import stable_hash
from video_dataset.utils.io import ensure_dir, read_jsonl

STATUSES = ("approved", "permission_pending", "metadata_only", "rejected", "expired_or_revoked")
REQUIRED_TRUE = ("download_authorized", "ml_training_allowed", "derivative_processing_allowed")
REQUIRED_TEXT = ("creator_or_rightsholder", "source_license", "evidence_uri", "verified_at", "verified_by")
PIPELINE_ACTION = {
    "approved": "eligible for visual processing",
    "permission_pending": "not downloaded, not used for visual training",
    "metadata_only": "annotations only; kept out of T2V / V2V",
    "rejected": "not reacquired, not used",
    "expired_or_revoked": "new processing blocked; follow the removal process (video-dataset remove-source)",
}


def default_rights_path(config: Any) -> Path:
    if config.v3.rights_manifest:
        p = Path(config.v3.rights_manifest)
        return p if p.is_absolute() else (Path.cwd() / p)
    return config.v3_dir / "rights" / "rights_manifest.jsonl"


def evidence_id(entry: dict[str, Any]) -> str:
    return str(entry.get("rights_evidence_id") or f"rights_{stable_hash(entry.get('source_video_id'), length=12)}")


def load_rights(path: Path) -> dict[str, dict[str, Any]]:
    """{source_video_id: entry}. A missing file is an empty manifest: nothing is approved."""
    if not Path(path).exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        vid = row.get("source_video_id")
        if vid:
            out[str(vid)] = row
    return out


def _date(value: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def entry_problems(entry: dict[str, Any], base_dir: Path | None = None, today: dt.date | None = None) -> list[str]:
    """Why an entry that claims ``approved`` is not usable. Empty list = complete."""
    problems: list[str] = []
    status = str(entry.get("rights_status") or "")
    if status not in STATUSES:
        return [f"rights_status '{status}' is not one of {', '.join(STATUSES)}"]
    if status != "approved":
        return [f"rights_status is {status}"]
    for key in REQUIRED_TRUE:
        if entry.get(key) is not True:  # explicit boolean true; "yes" / 1 / null do not count
            problems.append(f"{key} is not true")
    for key in REQUIRED_TEXT:
        if not str(entry.get(key) or "").strip():
            problems.append(f"{key} is empty")
    if entry.get("verified_at") and _date(entry["verified_at"]) is None:
        problems.append("verified_at is not a date (YYYY-MM-DD)")
    uri = str(entry.get("evidence_uri") or "").strip()
    if uri and not uri.lower().startswith(("http://", "https://")) and base_dir is not None:
        p = Path(uri)
        if not (p if p.is_absolute() else base_dir / p).exists():
            problems.append(f"evidence file not found: {uri} (relative to {base_dir})")
    expires = _date(entry.get("expires_at")) if entry.get("expires_at") else None
    if expires and expires < (today or dt.date.today()):
        problems.append(f"permission expired on {expires.isoformat()} (set rights_status: expired_or_revoked)")
    return problems


def is_approved(entry: dict[str, Any] | None, base_dir: Path | None = None) -> bool:
    return bool(entry) and not entry_problems(entry or {}, base_dir)


def may_publish_media(entry: dict[str, Any], private_repo: bool) -> tuple[bool, str | None]:
    """Media of a source whose redistribution is not allowed never goes to a public repository."""
    if entry.get("redistribution_allowed") is True:
        return True, None
    if private_repo:
        return True, None
    return False, "redistribution_allowed is not true and the target repository is public"


def new_entry(row: dict[str, Any]) -> dict[str, Any]:
    """A fresh manifest entry for a legacy source: metadata_only, every permission false."""
    declared = row.get("license_status") or "unknown"
    return {
        "source_video_id": row["source_video_id"],
        "source_url": row.get("source_url"),
        "creator_or_rightsholder": row.get("channel") or "",
        "source_license": "",
        "platform_declared_license": declared,  # evidence to check, not a permission
        "rights_status": "metadata_only",
        "download_authorized": False,
        "ml_training_allowed": False,
        "derivative_processing_allowed": False,
        "redistribution_allowed": False,
        "commercial_use_allowed": False,
        "attribution_text": "",
        "evidence_uri": "",
        "verified_at": "",
        "verified_by": "",
        "expires_at": None,
        "local_path": None,  # optional: a file supplied by the rightsholder, used instead of a download
        "notes": "",
    }


def init_rights(index_rows: list[dict[str, Any]], path: Path) -> tuple[int, int]:
    """Add an entry for every source that has none. Existing entries are never changed.
    Returns (added, total)."""
    existing = load_rights(path)
    order = list(existing.values())
    added = 0
    for row in index_rows:
        if row["source_video_id"] not in existing:
            order.append(new_entry(row))
            added += 1
    ensure_dir(Path(path).parent)
    tmp = Path(path).with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for entry in order:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return added, len(order)


def rights_report(index_rows: list[dict[str, Any]], rights: dict[str, dict[str, Any]], base_dir: Path | None = None) -> dict[str, Any]:
    counts = {s: 0 for s in STATUSES}
    counts["no_entry"] = 0
    incomplete: list[dict[str, Any]] = []
    eligible: list[str] = []
    for row in index_rows:
        entry = rights.get(row["source_video_id"])
        if entry is None:
            counts["no_entry"] += 1
            continue
        status = str(entry.get("rights_status") or "")
        counts[status if status in counts else "no_entry"] += 1
        if status == "approved":
            problems = entry_problems(entry, base_dir)
            if problems:
                incomplete.append({"source_video_id": row["source_video_id"], "problems": problems})
            else:
                eligible.append(row["source_video_id"])
    return {"sources": len(index_rows), "by_status": counts, "eligible": eligible, "approved_but_incomplete": incomplete}


def public_row(entry: dict[str, Any]) -> dict[str, Any]:
    """What a published rights_manifest.jsonl carries: the decision and attribution, not the evidence
    document or internal notes."""
    keep = (
        "source_video_id", "source_url", "creator_or_rightsholder", "source_license", "rights_status", "download_authorized",
        "ml_training_allowed", "derivative_processing_allowed", "redistribution_allowed", "commercial_use_allowed",
        "attribution_text", "verified_at", "expires_at",
    )
    out = {k: entry.get(k) for k in keep}
    out["rights_evidence_id"] = evidence_id(entry)
    return out
