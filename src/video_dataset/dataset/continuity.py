"""Shot-to-shot continuity links on exported records (issue doc P3-07).

For every video the scenes are ordered by ``start_time`` and each one gets a ``continuity`` block:
its neighbours, the transitions on both sides, whether the next shot starts where this one ends,
the sequence record it belongs to, and whether subjects (object classes) and location (setting +
time of day) carry over to the next shot. Clips and frames copy the block of their scene so a
consumer never has to join files to know what came before and after a shot.

Everything is derived from fields the pipeline already measured; where a field is unknown on
either side the continuity value is ``None``, never a guess. ``check_continuity`` verifies that the
links resolve inside the same video and that adjacency agrees with the timestamps.
"""

from __future__ import annotations

from typing import Any

from video_dataset.dataset.tasks import sub_dict

CONTINUITY_METHOD = "export_rules_2026.11"
ADJACENCY_TOLERANCE_SECONDS = 0.5
_UNKNOWN = (None, "", "unknown")


def _env(rec: dict[str, Any], key: str) -> str | None:
    v = sub_dict(rec, "environment").get(key)
    v = str(v).strip().lower() if v is not None else None
    return None if v in _UNKNOWN else v


def _objects(rec: dict[str, Any]) -> set[str]:
    return {str(o).strip().lower() for o in (rec.get("objects") or []) if o}


def _adjacent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return abs(float(b.get("start_time") or 0.0) - float(a.get("end_time") or 0.0)) <= ADJACENCY_TOLERANCE_SECONDS


def _subject_continuity(a: dict[str, Any], b: dict[str, Any]) -> bool | None:
    oa, ob = _objects(a), _objects(b)
    if not oa or not ob:
        return None
    return bool(oa & ob)


def _location_continuity(a: dict[str, Any], b: dict[str, Any]) -> bool | None:
    sa, sb = _env(a, "setting"), _env(b, "setting")
    ta, tb = _env(a, "time_of_day"), _env(b, "time_of_day")
    if None in (sa, sb, ta, tb):
        return None
    return sa == sb and ta == tb


def _sequence_index(records: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], str]:
    """(video_id, scene_id) -> sequence_id of the (first) sequence that contains the scene."""
    out: dict[tuple[str, str], str] = {}
    for seq in records.get("sequences", []):
        vid, sid = str(seq.get("video_id") or ""), str(seq.get("sequence_id") or "")
        if not vid or not sid:
            continue
        for scene in seq.get("scene_ids") or []:
            out.setdefault((vid, str(scene)), sid)
    return out


def _scenes_by_video(records: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    by_video: dict[str, list[dict[str, Any]]] = {}
    for rec in records.get("scenes", []):
        if rec.get("start_time") is None or not rec.get("scene_id"):
            continue
        by_video.setdefault(str(rec.get("video_id") or ""), []).append(rec)
    for scenes in by_video.values():
        scenes.sort(key=lambda r: (float(r["start_time"]), str(r.get("scene_id"))))
    return by_video


def fill_continuity(records: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    """Set ``continuity`` on every scene (in place) and copy it to the clips and frames of the same
    (video_id, scene_id). ``subject_continuity`` / ``location_continuity`` / ``temporal_adjacency``
    / ``transition_out`` describe the link to the *next* shot; ``same_sequence_as_prev`` the link
    to the previous one. Returns ``{"scenes": n, "clips": n, "frames": n}``."""
    seq_of = _sequence_index(records)
    by_scene: dict[tuple[str, str], dict[str, Any]] = {}
    counts = {"scenes": 0, "clips": 0, "frames": 0}
    for vid, scenes in _scenes_by_video(records).items():
        for i, rec in enumerate(scenes):
            prev = scenes[i - 1] if i > 0 else None
            nxt = scenes[i + 1] if i + 1 < len(scenes) else None
            sid = str(rec["scene_id"])
            seq_id = seq_of.get((vid, sid))
            prev_seq = seq_of.get((vid, str(prev["scene_id"]))) if prev else None
            rec["continuity"] = {
                "prev_scene_id": str(prev["scene_id"]) if prev else None,
                "next_scene_id": str(nxt["scene_id"]) if nxt else None,
                "transition_in": rec.get("transition_in"),
                "transition_out": nxt.get("transition_in") if nxt else None,
                "temporal_adjacency": _adjacent(rec, nxt) if nxt else False,
                "sequence_id": seq_id,
                "same_sequence_as_prev": (seq_id is not None and seq_id == prev_seq) if prev else None,
                "subject_continuity": _subject_continuity(rec, nxt) if nxt else None,
                "location_continuity": _location_continuity(rec, nxt) if nxt else None,
                "continuity_method": CONTINUITY_METHOD,
            }
            by_scene[(vid, sid)] = rec["continuity"]
            counts["scenes"] += 1
    for kind in ("clips", "frames"):
        for rec in records.get(kind, []):
            block = by_scene.get((str(rec.get("video_id") or ""), str(rec.get("scene_id") or "")))
            if block is not None:
                rec["continuity"] = dict(block)
                counts[kind] += 1
    return counts


def check_continuity(records: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Problems with the continuity blocks: neighbour / sequence links that do not resolve within
    the same video, links that are not mutual, adjacency flags that disagree with the timestamps,
    and clips / frames whose block differs from their scene's. Empty = consistent."""
    problems: list[str] = []
    by_video = _scenes_by_video(records)
    scene_index = {(vid, str(s["scene_id"])): s for vid, scenes in by_video.items() for s in scenes}
    sequences = {(str(s.get("video_id") or ""), str(s.get("sequence_id") or "")): s for s in records.get("sequences", [])}
    for vid, scenes in by_video.items():
        for rec in scenes:
            sid = str(rec["scene_id"])
            c = rec.get("continuity")
            if not isinstance(c, dict):
                problems.append(f"{vid}/{sid}: no continuity block")
                continue
            for key in ("prev_scene_id", "next_scene_id"):
                other = c.get(key)
                if other is None:
                    continue
                target = scene_index.get((vid, str(other)))
                if target is None:
                    problems.append(f"{vid}/{sid}: {key} {other} does not resolve in the same video")
                    continue
                back = (target.get("continuity") or {}).get("next_scene_id" if key == "prev_scene_id" else "prev_scene_id")
                if str(back) != sid:
                    problems.append(f"{vid}/{sid}: {key} {other} does not link back")
                if key == "next_scene_id":
                    expected = _adjacent(rec, target)
                    if bool(c.get("temporal_adjacency")) != expected:
                        problems.append(f"{vid}/{sid}: temporal_adjacency {c.get('temporal_adjacency')} disagrees with timestamps ({expected})")
                    if c.get("transition_out") != target.get("transition_in"):
                        problems.append(f"{vid}/{sid}: transition_out {c.get('transition_out')} != next transition_in {target.get('transition_in')}")
            if c.get("next_scene_id") is None and c.get("temporal_adjacency"):
                problems.append(f"{vid}/{sid}: temporal_adjacency without a next scene")
            seq_id = c.get("sequence_id")
            if seq_id is not None:
                seq = sequences.get((vid, str(seq_id)))
                if seq is None:
                    problems.append(f"{vid}/{sid}: sequence_id {seq_id} does not resolve in the same video")
                elif sid not in {str(s) for s in (seq.get("scene_ids") or [])}:
                    problems.append(f"{vid}/{sid}: sequence {seq_id} does not contain the scene")
    for kind in ("clips", "frames"):
        for rec in records.get(kind, []):
            where = (str(rec.get("video_id") or ""), str(rec.get("scene_id") or ""))
            scene = scene_index.get(where)
            if scene is None or not isinstance(scene.get("continuity"), dict):
                continue
            if rec.get("continuity") != scene["continuity"]:
                problems.append(f"{where[0]}/{kind}/{rec.get('record_id')}: continuity differs from scene {where[1]}")
    return problems
