"""Versioned, repeatable migrations of exported record dicts to the current ``SCHEMA_VERSION``.

Every published record declares one ``schema_version``. Records written by an earlier pipeline
are brought forward one step at a time (``1.x`` -> ``2.0.0`` -> ``2.1.0`` -> ``2.2.0`` -> ``2.3.0``),
each step a small pure function registered in ``MIGRATIONS`` and covered by a test, so a rebuild of
an old shard is deterministic and never loses data:

* renamed fields are mapped (``start`` / ``end`` -> ``start_time`` / ``end_time``, ``question_id``
  -> ``record_id``);
* fields the current schema does not know are moved verbatim into ``extensions`` (never dropped);
* new fields get their explicit "not evaluated" defaults (``boundary_check: not_run``, quality
  ``missing_components``, ``verification``, ``grounding_method: not_assessed`` ...), never a value
  that looks like a measurement.

``migrate_record`` returns the migrated dict plus the list of steps applied; ``upgrade_record`` in
``dataset.export`` runs it before validating against the pydantic model. A record whose version is
*newer* than this code is left untouched and reported, because a downgrade would invent fields.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from video_dataset.schemas.common import SCHEMA_VERSION

Migration = Callable[[str, dict[str, Any]], dict[str, Any]]

# Versions this code knows how to read, oldest first. "0.0.0" stands for a record without a version
# field (the first exports); every other entry is a SCHEMA_VERSION that was once current.
KNOWN_VERSIONS = ("0.0.0", "2.0.0", "2.1.0", "2.2.0", "2.3.0")

_LEGACY_QUALITY_KEYS = ("grounding", "temporal_accuracy", "description_quality")


def _version_tuple(v: str | None) -> tuple[int, int, int]:
    if not v:
        return (0, 0, 0)
    parts = str(v).split(".")
    out: list[int] = []
    for p in parts[:3]:
        try:
            out.append(int(p))
        except ValueError:
            out.append(0)
    while len(out) < 3:
        out.append(0)
    return (out[0], out[1], out[2])


def _normalise_version(v: str | None) -> str:
    """Map a record's declared version to the closest known step (anything below 2.0.0 is legacy)."""
    t = _version_tuple(v)
    if t < (2, 0, 0):
        return "0.0.0"
    for known in reversed(KNOWN_VERSIONS):
        if _version_tuple(known) <= t:
            return known
    return "0.0.0"


# ----------------------------------------------------------------------------- steps
def _to_2_0_0(kind: str, rec: dict[str, Any]) -> dict[str, Any]:
    """Legacy (unversioned) -> 2.0.0: field renames and the common envelope."""
    out = dict(rec)
    if kind == "scenes":
        if "start" in out and "start_time" not in out:
            out["start_time"] = out.pop("start")
        if "end" in out and "end_time" not in out:
            out["end_time"] = out.pop("end")
        if "duration" not in out and out.get("start_time") is not None and out.get("end_time") is not None:
            out["duration"] = round(float(out["end_time"]) - float(out["start_time"]), 3)
    if kind in ("temporal_qa", "long_video_qa") and not out.get("record_id") and out.get("question_id"):
        out["record_id"] = out["question_id"]
    v = out.get("validation")
    if isinstance(v, dict):
        v.setdefault("issues", [])
        v.setdefault("checks", {})
        if "status" not in v:
            v["status"] = "review"
    return out


def _to_2_1_0(kind: str, rec: dict[str, Any]) -> dict[str, Any]:
    """2.0.0 -> 2.1.0: policy fields, clip boundary states, claim lists (all 'not evaluated')."""
    out = dict(rec)
    v = out.get("validation")
    if isinstance(v, dict):
        v.setdefault("reason_codes", [])
        v.setdefault("policy_version", None)  # validated before the policy existed: stays None, the policy downgrades it
        v.setdefault("semantic_verification", None)
    if kind == "clips":
        out.setdefault("boundary_check", "not_run")
        out.setdefault("exact", True)
    if kind in ("scenes", "clips"):
        out.setdefault("claims", [])
    if kind == "events":
        out.setdefault("boundary_precision", None)
    return out


def _to_2_2_0(kind: str, rec: dict[str, Any]) -> dict[str, Any]:
    """2.1.0 -> 2.2.0: verification summary, quality missing components, coverage, source group."""
    out = dict(rec)
    v = out.get("validation")
    if isinstance(v, dict):
        v.setdefault("verification", {})
        v.setdefault("reevaluated", None)
    q = out.get("quality")
    if isinstance(q, dict):
        q.setdefault("components_used", [k for k in _LEGACY_QUALITY_KEYS if q.get(k) is not None])
        q.setdefault("missing_components", [k for k in _LEGACY_QUALITY_KEYS if q.get(k) is None])
    out.setdefault("coverage", {})
    out.setdefault("source_group_id", out.get("canonical_video_id") or out.get("video_id"))
    return out


def _to_2_3_0(kind: str, rec: dict[str, Any]) -> dict[str, Any]:
    """2.2.0 -> 2.3.0: provenance / rights / temporal validation / continuity slots, grounding
    method, status history, media status, OCR detections, description evidence."""
    out = dict(rec)
    v = out.get("validation")
    if isinstance(v, dict):
        v.setdefault("history", [])
    q = out.get("quality")
    if isinstance(q, dict):
        q.setdefault("grounding_method", None if q.get("grounding") is not None else "not_assessed")
        q.setdefault("scoring_version", None)  # scored under an older rule set; the version is unknown
    out.setdefault("provenance", None)
    out.setdefault("rights", None)
    out.setdefault("temporal_validation", None)
    out.setdefault("continuity", None)
    out.setdefault("confidence_calibration", None)
    if kind in ("frames", "clips"):
        out.setdefault("asset_id", None)
        out.setdefault("media_status", None)
    if kind == "frames":
        out.setdefault("ocr_detections", [])
    if kind == "clips":
        out.setdefault("boundary_repair", None)
        out.setdefault("frame_count", None)
        out.setdefault("frame_count_check", None)
        out.setdefault("stream_integrity", None)
        out.setdefault("object_tracks", [])
    if kind == "scenes":
        out.setdefault("object_tracks", [])
        out.setdefault("track_scope", None)
        out.setdefault("lighting", None)
        out.setdefault("color", None)
    if kind == "video_descriptions":
        out.setdefault("evidence", {"scene_ids": list(out.get("scene_ids") or []), "clip_ids": [], "frame_ids": [], "timestamps": []})
        out.setdefault("claims", [])
        out.setdefault("facts", {"observed": [], "inferred": []})
        out.setdefault("grounding_assessed", False)
    cam = out.get("camera")
    if isinstance(cam, dict):
        cam.setdefault("movement_ambiguous", False)
        cam.setdefault("movement_alternatives", [])
        cam.setdefault("camera_motion_source", None)
    return out


MIGRATIONS: dict[str, tuple[str, Migration]] = {
    "0.0.0": ("2.0.0", _to_2_0_0),
    "2.0.0": ("2.1.0", _to_2_1_0),
    "2.1.0": ("2.2.0", _to_2_2_0),
    "2.2.0": ("2.3.0", _to_2_3_0),
}


# ----------------------------------------------------------------------------- driver
def known_fields(model: Any) -> set[str]:
    """Field names (and validation aliases) the pydantic model accepts."""
    names: set[str] = set()
    for name, field in getattr(model, "model_fields", {}).items():
        names.add(name)
        alias = getattr(field, "validation_alias", None)
        choices = getattr(alias, "choices", None)
        if choices:
            names.update(str(c) for c in choices)
        elif isinstance(alias, str):
            names.add(alias)
    return names


def move_unknown_to_extensions(rec: dict[str, Any], accepted: set[str]) -> list[str]:
    """Move top-level keys the model does not know into ``rec["extensions"]`` (in place). Returns
    the moved keys. ``record_type`` is left alone (the exporter re-adds it)."""
    moved: list[str] = []
    ext = rec.get("extensions")
    if not isinstance(ext, dict):
        ext = {}
    for key in list(rec):
        if key in accepted or key in ("record_type", "extensions"):
            continue
        ext[key] = rec.pop(key)
        moved.append(key)
    if ext or "extensions" in accepted:
        rec["extensions"] = ext
    return moved


def migrate_record(kind: str, rec: dict[str, Any], target: str = SCHEMA_VERSION) -> tuple[dict[str, Any], list[str]]:
    """Apply every step from the record's declared version up to ``target``. Returns (record,
    steps) where steps are ``"<from>-><to>"`` strings; an already-current record returns []. A
    record newer than ``target`` is returned unchanged with the step ``"newer:<version>"``."""
    declared = rec.get("schema_version")
    if _version_tuple(declared) > _version_tuple(target):
        return rec, [f"newer:{declared}"]
    current = _normalise_version(declared)
    out = rec
    steps: list[str] = []
    while _version_tuple(current) < _version_tuple(target):
        nxt = MIGRATIONS.get(current)
        if nxt is None:
            break
        to, fn = nxt
        out = fn(kind, out)
        steps.append(f"{current}->{to}")
        current = to
    if steps or declared != target:
        out["schema_version"] = target
    return out, steps


def migration_path(from_version: str | None, target: str = SCHEMA_VERSION) -> list[str]:
    """The step names ``migrate_record`` would apply for a record of ``from_version``."""
    current = _normalise_version(from_version)
    out: list[str] = []
    while _version_tuple(current) < _version_tuple(target) and current in MIGRATIONS:
        to, _ = MIGRATIONS[current]
        out.append(f"{current}->{to}")
        current = to
    return out
