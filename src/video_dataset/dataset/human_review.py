"""Stratified human verification with double annotation, agreement statistics and adjudication.

The automatic validation (policy engine, claims, boundary check, camera-motion agreement) decides
what is *accepted*; this module measures how often those decisions are right by drawing a
deterministic stratified sample for humans, assigning two reviewers to a share of it, scoring
inter-annotator agreement and turning the verdicts into the ``reviews/labels.jsonl`` file that
``apply_review_labels`` (see ``review.py``) and ``video-dataset calibrate`` already consume.

Files (all under ``final/reviews/``):

``sample.jsonl`` - one row per sampled record, written by ``write_sample``::

    {"sample_id": "3f9a1c2b0d4e", "record_id": "...", "record_type": "clip", "video_id": "...",
     "stratum": {"record_type": "clip", "tier": "gold", "failure_category": "accepted"},
     "stratum_size": 412, "assigned_to": ["reviewer_a", "reviewer_b"], "double_annotated": true,
     "evidence": {"start_time": 1.5, "end_time": 4.0, "media": "clips/....mp4",
                  "frame_ids": [...], "text": "...", "answer": null,
                  "camera_movement": "pan_left", "boundary_precision": null},
     "protocol_version": "review-protocol-2026.11",
     "label": null, "reviewer": null, "note": null,
     "camera_movement_label": null, "boundary_ok": null, "boundary_error_seconds": null}

Reviewers fill ``label`` (accept | reject | correct), ``reviewer``, optionally ``note``,
``corrections``, ``camera_movement_label`` (family or fine label such as ``pan_left``),
``boundary_ok`` (true / false) and ``boundary_error_seconds``. A double-annotated row is
duplicated, one copy per reviewer. ``read_labelled`` reads the filled file (and the legacy
``labels.jsonl`` rows that carry no reviewer, which get reviewer ``unknown``).

``labels.jsonl`` - final verdict per record, written by ``write_adjudicated``: the ``review.py``
shape (``record_id``, ``label``, ``note``, ``corrections``) plus ``reviewer`` (the single reviewer,
``consensus`` or ``adjudicated``) and ``reviewers`` (everyone who labelled the record).

All statistics return plain numbers or ``None`` (never NaN) so they serialise into the release
report unchanged.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from video_dataset.schemas.vision import MOVEMENT_TAXONOMY, CameraMovement
from video_dataset.utils.io import read_jsonl, write_jsonl

PROTOCOL_VERSION = "review-protocol-2026.11"
SAMPLE_FILE = Path("reviews") / "sample.jsonl"
LABELS_FILE = Path("reviews") / "labels.jsonl"
LABELS = ("accept", "reject", "correct")
DEFAULT_STRATA: tuple[str, ...] = ("record_type", "tier", "failure_category")
_Z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
_RECORD_TYPE_BY_KEY = {
    "frames": "frame", "clips": "clip", "scenes": "scene", "sequences": "sequence", "events": "event",
    "temporal_qa": "temporal_qa", "long_video_qa": "long_video_qa", "video_descriptions": "video_description",
}


def _dict(rec: dict[str, Any], key: str) -> dict[str, Any]:
    val = rec.get(key)
    return val if isinstance(val, dict) else {}


# --------------------------------------------------------------------------------------------
# strata
# --------------------------------------------------------------------------------------------


def failure_category(rec: dict[str, Any]) -> str:
    """Why the automatic validation did (not) accept a record, as one coarse bucket.

    Order of precedence: human_labelled, contradicted_claim, boundary, camera, semantic,
    missing_check, confidence, accepted, other.
    """
    v = rec.get("validation") or {}
    codes = [str(c) for c in (v.get("reason_codes") or [])]
    cam = _dict(rec, "camera")
    if any(c.startswith("human:") for c in codes):
        return "human_labelled"
    if any(c.get("status") == "contradicted" for c in rec.get("claims") or [] if isinstance(c, dict)):
        return "contradicted_claim"
    if any("clip_boundaries" in c or "timestamps" in c for c in codes):
        return "boundary"
    if any("camera_motion_agreement" in c for c in codes) or bool(cam.get("movement_ambiguous")):
        return "camera"
    if any("semantic_verification" in c for c in codes):
        return "semantic"
    if any("missing_check:" in c for c in codes):
        return "missing_check"
    if any("confidence:" in c for c in codes):
        return "confidence"
    if v.get("status") == "accepted" and not codes:
        return "accepted"
    return "other"


def _record_type(rec: dict[str, Any], key: str) -> str:
    return str(rec.get("record_type") or _RECORD_TYPE_BY_KEY.get(key) or key)


def stratum_of(rec: dict[str, Any], strata: tuple[str, ...] = DEFAULT_STRATA, record_type: str | None = None) -> dict[str, str]:
    """The stratum values of one record; unknown keys are read from the record (``None`` -> "none")."""
    out: dict[str, str] = {}
    for name in strata:
        if name == "record_type":
            out[name] = record_type or str(rec.get("record_type") or "unknown")
        elif name == "failure_category":
            out[name] = failure_category(rec)
        elif name == "status":
            out[name] = str((rec.get("validation") or {}).get("status") or "none")
        else:
            val = rec.get(name)
            out[name] = "none" if val is None else str(val)
    return out


def _stratum_key(stratum: dict[str, str]) -> str:
    return "|".join(f"{k}={v}" for k, v in stratum.items())


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of ``n`` over strata with at least one per non-empty stratum.

    Floors of the proportional shares first, then the remainder by largest fractional part; each
    quota is capped by the stratum size. When ``n`` is smaller than the number of strata the
    largest strata get one each.
    """
    keys = [k for k in sorted(sizes) if sizes[k] > 0]
    total = sum(sizes[k] for k in keys)
    n = min(n, total)
    if n <= 0 or not keys:
        return dict.fromkeys(keys, 0)
    if n < len(keys):
        chosen = sorted(keys, key=lambda k: (-sizes[k], k))[:n]
        return {k: (1 if k in chosen else 0) for k in keys}
    exact = {k: n * sizes[k] / total for k in keys}
    alloc = {k: max(1, min(sizes[k], int(math.floor(exact[k])))) for k in keys}
    remaining = n - sum(alloc.values())
    # more than n handed out through the minimum-one rule: take back from the largest surpluses
    while remaining < 0:
        donors = [k for k in keys if alloc[k] > 1]
        if not donors:
            break
        k = max(donors, key=lambda k: (alloc[k] - exact[k], alloc[k], k))
        alloc[k] -= 1
        remaining += 1
    order = sorted(keys, key=lambda k: (-(exact[k] - math.floor(exact[k])), -sizes[k], k))
    while remaining > 0:
        progressed = False
        for k in order:
            if remaining == 0:
                break
            if alloc[k] < sizes[k]:
                alloc[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def _short_id(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]


def _evidence(rec: dict[str, Any]) -> dict[str, Any]:
    ev = _dict(rec, "evidence")
    cam = _dict(rec, "camera")
    family, direction = cam.get("camera_movement"), cam.get("movement_direction")
    if not family and cam.get("movement"):
        try:
            fam_enum, direction = MOVEMENT_TAXONOMY[CameraMovement(str(cam["movement"]))]
            family = fam_enum.value
        except (ValueError, KeyError):
            family = None
    movement = f"{family}_{direction}" if family and direction else (str(family) if family else None)
    return {
        "start_time": rec.get("start_time", ev.get("start_time", rec.get("timestamp"))),
        "end_time": rec.get("end_time", ev.get("end_time", rec.get("timestamp"))),
        "media": rec.get("clip_path") or rec.get("frame_path"),
        "frame_ids": list(rec.get("frame_ids") or ev.get("frame_ids") or [])[:8],
        "text": rec.get("generation_prompt") or rec.get("summary") or rec.get("description") or rec.get("question") or rec.get("event"),
        "answer": rec.get("answer"),
        "camera_movement": movement,
        "boundary_precision": rec.get("boundary_precision"),
    }


def stratified_sample(
    records: dict[str, list[dict[str, Any]]],
    n: int,
    *,
    strata: tuple[str, ...] = DEFAULT_STRATA,
    per_video_cap: int | None = None,
    double_fraction: float = 0.2,
    reviewers: tuple[str, ...] = ("reviewer_a", "reviewer_b"),
    seed: int | str = 1234,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Draw a deterministic stratified sample of ``n`` records for human verification.

    Allocation is proportional to stratum size with at least one record per non-empty stratum
    (remainder by largest fractional part). ``per_video_cap`` limits records per ``video_id``
    across the whole sample; when a stratum cannot fill its quota under the cap the shortfall is
    taken from other strata so the sample still reaches ``n`` whenever possible. A
    ``double_fraction`` share of the rows (chosen with the seeded RNG) is assigned to every
    reviewer; the remaining rows are handed out round-robin. ``split`` restricts the population.
    """
    rng = random.Random(str(seed))
    pool: dict[str, list[tuple[str, dict[str, Any], dict[str, str]]]] = {}
    for key, recs in records.items():
        rtype = _RECORD_TYPE_BY_KEY.get(key, key)
        for rec in recs:
            if split is not None and str(rec.get("split") or "") != split:
                continue
            if not rec.get("record_id"):
                continue
            stratum = stratum_of(rec, strata, _record_type(rec, key))
            pool.setdefault(_stratum_key(stratum), []).append((rtype, rec, stratum))
    for key in pool:
        pool[key].sort(key=lambda t: str(t[1].get("record_id")))
        rng.shuffle(pool[key])
    sizes = {k: len(v) for k, v in pool.items()}
    alloc = _allocate(sizes, n)

    chosen: list[tuple[str, dict[str, Any], dict[str, str]]] = []
    seen: set[str] = set()
    per_video: Counter[str] = Counter()
    cursor: dict[str, int] = dict.fromkeys(pool, 0)

    def take(key: str, want: int) -> int:
        got = 0
        items = pool[key]
        while got < want and cursor[key] < len(items):
            rtype, rec, stratum = items[cursor[key]]
            cursor[key] += 1
            rid = str(rec["record_id"])
            vid = str(rec.get("video_id") or "")
            if rid in seen or (per_video_cap is not None and per_video[vid] >= per_video_cap):
                continue
            seen.add(rid)
            per_video[vid] += 1
            chosen.append((rtype, rec, stratum))
            got += 1
        return got

    for key in sorted(pool):
        take(key, alloc.get(key, 0))
    # fill pass: strata that could not meet their quota (per-video cap) give way to the others
    shortfall = min(n, sum(sizes.values())) - len(chosen)
    for key in sorted(pool, key=lambda k: (-sizes[k], k)):
        if shortfall <= 0:
            break
        shortfall -= take(key, shortfall)

    rows: list[dict[str, Any]] = []
    for rtype, rec, stratum in chosen:
        rid = str(rec["record_id"])
        rows.append({
            "sample_id": _short_id(PROTOCOL_VERSION, seed, rid),
            "record_id": rid,
            "record_type": rtype,
            "video_id": rec.get("video_id"),
            "stratum": stratum,
            "stratum_size": sizes[_stratum_key(stratum)],
            "assigned_to": [],
            "double_annotated": False,
            "evidence": _evidence(rec),
            "protocol_version": PROTOCOL_VERSION,
            "label": None,
            "reviewer": None,
            "note": None,
            "corrections": {},
            "camera_movement_label": None,
            "boundary_ok": None,
            "boundary_error_seconds": None,
        })
    rows.sort(key=lambda r: (_stratum_key(r["stratum"]), r["record_id"]))
    n_double = int(round(double_fraction * len(rows))) if len(reviewers) > 1 else 0
    double_idx = set(rng.sample(range(len(rows)), min(n_double, len(rows)))) if n_double else set()
    single = 0
    for i, row in enumerate(rows):
        if i in double_idx:
            row["assigned_to"] = list(reviewers)
            row["double_annotated"] = True
        else:
            row["assigned_to"] = [reviewers[single % len(reviewers)]] if reviewers else []
            single += 1
    return rows


def sample_size(population: int, margin: float = 0.05, confidence: float = 0.95, p: float = 0.5) -> int:
    """Cochran sample size for a proportion with finite population correction."""
    if population <= 0:
        return 0
    z = _Z.get(round(confidence, 2))
    if z is None:
        raise ValueError(f"confidence must be one of {sorted(_Z)}, got {confidence}")
    n0 = (z * z) * p * (1.0 - p) / (margin * margin)
    n = n0 / (1.0 + (n0 - 1.0) / population)
    return max(1, min(population, int(math.ceil(n))))


# --------------------------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------------------------


def write_sample(rows: list[dict[str, Any]], final_dir: Path) -> Path:
    path = Path(final_dir) / SAMPLE_FILE
    write_jsonl(path, rows)
    return path


def read_sample(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    return list(read_jsonl(path)) if path.exists() else []


def _norm_label(value: Any) -> str | None:
    lab = str(value or "").strip().lower()
    return lab if lab in LABELS else None


def read_labelled(path: Path) -> list[dict[str, Any]]:
    """Rows of ``sample.jsonl`` (or a legacy ``labels.jsonl``) that carry a valid label.

    Rows without ``reviewer`` get reviewer ``unknown``; labels are lower-cased.
    """
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        lab = _norm_label(row.get("label"))
        rid = str(row.get("record_id") or "").strip()
        if not lab or not rid:
            continue
        r = dict(row)
        r["record_id"] = rid
        r["label"] = lab
        r["reviewer"] = str(row.get("reviewer") or "unknown").strip() or "unknown"
        out.append(r)
    return out


# --------------------------------------------------------------------------------------------
# agreement
# --------------------------------------------------------------------------------------------


def _kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa over (label_a, label_b) pairs. None when there are no pairs; never NaN."""
    if not pairs:
        return None
    n = len(pairs)
    classes = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(1 for a, b in pairs if a == b) / n
    row = Counter(a for a, _ in pairs)
    col = Counter(b for _, b in pairs)
    pe = sum((row[c] / n) * (col[c] / n) for c in classes)
    if pe >= 1.0 - 1e-12:
        return 1.0 if po >= 1.0 - 1e-12 else 0.0
    return round((po - pe) / (1.0 - pe), 4)


def _by_record(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """record_id -> reviewer -> row (a reviewer's last row for a record wins)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        lab = _norm_label(r.get("label"))
        rid = str(r.get("record_id") or "")
        if not rid or not lab:
            continue
        out.setdefault(rid, {})[str(r.get("reviewer") or "unknown")] = {**r, "label": lab}
    return out


def _pairs(by_record: dict[str, dict[str, dict[str, Any]]], field: str, transform: Any = None) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Per reviewer pair, the (value_a, value_b) label pairs where both reviewers gave ``field``."""
    out: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for reviews in by_record.values():
        names = sorted(reviews)
        for a, b in combinations(names, 2):
            va, vb = reviews[a].get(field), reviews[b].get(field)
            if va is None or vb is None or va == "" or vb == "":
                continue
            va, vb = str(va).strip().lower(), str(vb).strip().lower()
            if transform:
                va, vb = transform(va), transform(vb)
            out.setdefault((a, b), []).append((va, vb))
    return out


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    m = len(s) // 2
    return round(s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0, 4)


def _pairwise_kappa(pairs_by_reviewers: dict[tuple[str, str], list[tuple[str, str]]]) -> float | None:
    ks = [k for k in (_kappa(p) for p in pairs_by_reviewers.values()) if k is not None]
    return _mean(ks)


def _percent(pairs_by_reviewers: dict[tuple[str, str], list[tuple[str, str]]]) -> float | None:
    flat = [p for ps in pairs_by_reviewers.values() for p in ps]
    return round(sum(1 for a, b in flat if a == b) / len(flat), 4) if flat else None


def _binary(lab: str) -> str:
    return "accept" if lab == "accept" else "not_accept"


def _agreement_block(by_record: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    multi = {rid: rv for rid, rv in by_record.items() if len(rv) >= 2}
    label_pairs = _pairs(multi, "label")
    binary_pairs = _pairs(multi, "label", _binary)
    disagreements = sorted(rid for rid, rv in multi.items() if len({r["label"] for r in rv.values()}) > 1)
    return {
        "records": len(by_record),
        "records_multi_reviewed": len(multi),
        "percent_agreement": _percent(label_pairs),
        "kappa_binary": _pairwise_kappa(binary_pairs),
        "kappa_multiclass": _pairwise_kappa(label_pairs),
        "disagreements": disagreements,
        "disagreement_rate": round(len(disagreements) / len(multi), 4) if multi else None,
    }


def _camera_family(value: str) -> str:
    try:
        return MOVEMENT_TAXONOMY[CameraMovement(value)][0].value
    except ValueError:
        return value.split("_")[0] if value else "unknown"


def agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Inter-annotator agreement over labelled sample rows (see module docstring for the fields).

    Kappa is Cohen's kappa per reviewer pair, averaged over pairs when more than two reviewers
    labelled the same records; with two reviewers it is the standard statistic.
    """
    by_record = _by_record(rows)
    multi = {rid: rv for rid, rv in by_record.items() if len(rv) >= 2}
    out = _agreement_block(by_record)

    cam_pairs = _pairs(multi, "camera_movement_label")
    fam_pairs = _pairs(multi, "camera_movement_label", _camera_family)
    out["camera_agreement"] = {
        "n_pairs": sum(len(p) for p in cam_pairs.values()),
        "percent_agreement": _percent(cam_pairs),
        "kappa": _pairwise_kappa(cam_pairs),
        "family_percent_agreement": _percent(fam_pairs),
    }

    audited = [r for r in rows if _norm_label(r.get("label")) and r.get("boundary_ok") is not None]
    errors: list[float] = []
    for r in audited:
        try:
            if r.get("boundary_error_seconds") is not None:
                errors.append(abs(float(r["boundary_error_seconds"])))
        except (TypeError, ValueError):
            continue
    out["boundary_audit"] = {
        "n": len(audited),
        "share_ok": round(sum(1 for r in audited if bool(r["boundary_ok"])) / len(audited), 4) if audited else None,
        "mean_error_seconds": _mean(errors),
        "median_error_seconds": _median(errors),
    }

    def breakdown(getter: Any) -> dict[str, Any]:
        groups: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for rid, rv in by_record.items():
            key = getter(next(iter(rv.values())))
            groups.setdefault(key, {})[rid] = rv
        result = {}
        for key in sorted(groups):
            block = _agreement_block(groups[key])
            block.pop("disagreements", None)
            result[key] = block
        return result

    out["by_record_type"] = breakdown(lambda r: str(r.get("record_type") or "unknown"))
    out["by_stratum"] = {
        "failure_category": breakdown(lambda r: str((r.get("stratum") or {}).get("failure_category") or "unknown")),
    }
    return out


# --------------------------------------------------------------------------------------------
# adjudication
# --------------------------------------------------------------------------------------------


def adjudicate(rows: list[dict[str, Any]], adjudications: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Final label per record: unanimous reviewers win, an explicit adjudication overrides,
    otherwise the record is flagged ``needs_adjudication`` with ``label`` None.

    The rows are shaped like ``review.py`` labels so ``apply_review_labels`` can consume them.
    """
    adjudications = {str(k): v for k, v in (adjudications or {}).items()}
    out: dict[str, dict[str, Any]] = {}
    for rid, reviews in sorted(_by_record(rows).items()):
        names = sorted(reviews)
        labels = {reviews[n]["label"] for n in names}
        notes = [str(reviews[n].get("note")).strip() for n in names if reviews[n].get("note")]
        corrections: dict[str, Any] = {}
        for n in names:
            c = reviews[n].get("corrections")
            if isinstance(c, dict):
                corrections.update(c)
        row: dict[str, Any] = {
            "record_id": rid, "label": None, "note": " | ".join(notes) or None, "corrections": corrections,
            "reviewer": None, "reviewers": names, "needs_adjudication": False,
        }
        adj = _norm_label(adjudications.get(rid))
        if adj:
            row["label"], row["reviewer"] = adj, "adjudicated"
        elif len(labels) == 1:
            row["label"] = next(iter(labels))
            row["reviewer"] = names[0] if len(names) == 1 else "consensus"
        else:
            row["needs_adjudication"] = True
            row["votes"] = {n: reviews[n]["label"] for n in names}
        if row["label"] != "correct":
            row["corrections"] = {}
        out[rid] = row
    for rid, lab in adjudications.items():  # adjudications for records nobody labelled
        norm = _norm_label(lab)
        if rid not in out and norm:
            out[rid] = {"record_id": rid, "label": norm, "note": None, "corrections": {}, "reviewer": "adjudicated", "reviewers": [], "needs_adjudication": False}
    return out


def write_adjudicated(final_dir: Path, labels: dict[str, dict[str, Any]]) -> Path:
    """Merge the adjudicated labels into ``reviews/labels.jsonl`` (new rows win, unresolved skipped)."""
    path = Path(final_dir) / LABELS_FILE
    merged: dict[str, dict[str, Any]] = {}
    if path.exists():
        for row in read_jsonl(path):
            rid = str(row.get("record_id") or "")
            if rid:
                merged[rid] = row
    for rid, row in labels.items():
        if _norm_label(row.get("label")):
            merged[str(rid)] = {k: v for k, v in row.items() if k not in ("needs_adjudication", "votes")}
    write_jsonl(path, [merged[k] for k in sorted(merged)])
    return path


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------


def verification_report(sample_rows: list[dict[str, Any]], labelled_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary of the human verification round for the release report."""
    labelled_ids = {str(r.get("record_id")) for r in labelled_rows if _norm_label(r.get("label"))}
    strata: dict[str, dict[str, Any]] = {}
    for r in sample_rows:
        key = _stratum_key(r.get("stratum") or {})
        s = strata.setdefault(key, {"sampled": 0, "labelled": 0, "population": r.get("stratum_size")})
        s["sampled"] += 1
        if str(r.get("record_id")) in labelled_ids:
            s["labelled"] += 1
    agg = agreement(labelled_rows)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "sample_size": len(sample_rows),
        "population_hint": sum(int(s["population"] or 0) for s in strata.values()),
        "double_annotated": sum(1 for r in sample_rows if r.get("double_annotated")),
        "labelled": len(labelled_ids),
        "coverage_by_stratum": {k: strata[k] for k in sorted(strata)},
        "agreement": agg,
        "unresolved_disagreements": len(agg.get("disagreements") or []),
    }
