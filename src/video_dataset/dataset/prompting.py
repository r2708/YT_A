"""Generation prompts and record cross-links, computed on exported record dicts.

``summary`` / ``description`` are the raw analysis text ("A 6.5-second shot with bright lighting
and a desaturated ..."). ``generation_prompt`` is the same shot phrased as a text-to-video prompt.
It is assembled from the structured fields only (shot size, subject, environment, camera taxonomy,
measured lighting / colour, lens estimates when a VLM gave them), so it never claims anything the
record does not already state. Runs at aggregate / re-annotate time, so old shards get it too.
"""

from __future__ import annotations

from typing import Any

from video_dataset.schemas.vision import SubjectMotion
from video_dataset.vision.subject_motion import subject_motion_phrase

_SHOT_WORDS = {"extreme_wide": "extreme wide", "wide": "wide", "medium": "medium", "close_up": "close-up", "extreme_close_up": "extreme close-up"}
_ANGLE_WORDS = {"low": "from a low angle", "high": "from a high angle", "overhead": "from directly overhead", "aerial": "from the air", "dutch": "with a tilted (dutch) frame"}
_LENS_WORDS = {"wide_angle": "wide-angle lens", "normal": "normal lens", "portrait": "portrait lens", "telephoto": "telephoto lens", "anamorphic": "anamorphic lens", "fisheye": "fisheye lens", "macro": "macro lens"}
_PLURAL_IRREGULAR = {"person": "people", "man": "men", "woman": "women", "child": "children", "mouse": "mice", "sheep": "sheep", "fish": "fish"}


def _plural(noun: str) -> str:
    if noun in _PLURAL_IRREGULAR:
        return _PLURAL_IRREGULAR[noun]
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def _article(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


_TIME_PHRASES = {"day": "in daylight", "daytime": "in daylight", "daylight": "in daylight", "night": "at night", "nighttime": "at night", "evening": "in the evening", "morning": "in the morning", "afternoon": "in the afternoon"}


def _time_phrase(time_of_day: str) -> str:
    t = time_of_day.strip().lower()
    return _TIME_PHRASES.get(t, f"at {t}")


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _subject_phrase(rec: dict[str, Any]) -> str | None:
    """'a car and two people' from object_details (count, prominence) / objects / people."""
    details = rec.get("object_details") or []
    names: list[tuple[str, int, float]] = []
    seen: set[str] = set()
    for o in details:
        if not isinstance(o, dict) or not o.get("name"):
            continue
        name = str(o["name"]).strip().lower()
        if name in seen:
            continue
        seen.add(name)
        names.append((name, int(o.get("count") or 1), float(o.get("area_fraction") or o.get("frame_fraction") or 0.0)))
    for name in rec.get("objects") or []:
        n = str(name).strip().lower()
        if n and n not in seen:
            seen.add(n)
            names.append((n, 1, 0.0))
    people = rec.get("people") if isinstance(rec.get("people"), dict) else None
    if people and people.get("count") and "person" not in seen and "people" not in seen:
        names.append(("person", int(people["count"]), 1.0))
    names.sort(key=lambda t: -t[2])
    parts = [f"{count} {_plural(name)}" if count > 1 else _article(name) for name, count, _ in names[:3]]
    return _join(parts) or None


def camera_phrase(cam: dict[str, Any] | None) -> str | None:
    """'slow pan to the left', 'tracking shot moving right following a person', 'static camera' ..."""
    if not isinstance(cam, dict):
        return None
    family, direction, speed = cam.get("camera_movement"), cam.get("movement_direction"), cam.get("movement_speed") or cam.get("speed")
    if not family or family == "unknown":
        return None
    tracked = f" following {_article(str(cam['tracked_subject']))}" if cam.get("tracked_subject") else ""
    base = {
        "static": "static camera",
        "pan": f"pan to the {direction}" if direction else "pan",
        "tilt": f"tilt {direction}" if direction else "tilt",
        "dolly": f"dolly {direction}" if direction else "dolly move",
        "zoom": f"zoom {direction}" if direction else "zoom",
        "tracking": ("tracking shot" + (f" moving {direction}" if direction else "")) + tracked,
        "orbit": f"camera orbiting the subject {direction}" if direction else "camera orbiting the subject",
        "crane": f"crane {direction}" if direction else "crane move",
        "handheld": "handheld camera",
        "fpv": "first-person camera",
        "drone": "drone shot",
        "complex": "complex camera movement",
    }.get(str(family))
    if not base:
        return None
    if speed and family not in ("static", "handheld", "fpv", "drone", "complex"):
        base = f"{speed} {base}"
    return base


def generation_prompt(rec: dict[str, Any]) -> str | None:
    """Text-to-video prompt for a scene / clip record dict (None when nothing beyond timing is known)."""
    cam = rec.get("camera") if isinstance(rec.get("camera"), dict) else {}
    env = rec.get("environment") if isinstance(rec.get("environment"), dict) else {}
    meas = rec.get("measurements") if isinstance(rec.get("measurements"), dict) else {}
    duration = rec.get("duration")
    if duration is None and rec.get("start_time") is not None and rec.get("end_time") is not None:
        duration = float(rec["end_time"]) - float(rec["start_time"])

    shot = _SHOT_WORDS.get(str(cam.get("shot_size") or ""))
    head = f"A {duration:.0f}-second " if duration else "A "
    head += f"{shot} shot" if shot else "shot"
    subject = _subject_phrase(rec)
    if subject:
        head += f" of {subject}"
    where = []
    if env.get("location"):
        where.append(f"in {_article(str(env['location']).strip())}" if not str(env["location"]).lower().startswith(("a ", "an ", "the ")) else f"in {env['location']}")
    elif env.get("setting") in ("indoor", "outdoor"):
        where.append("indoors" if env["setting"] == "indoor" else "outdoors")
    if env.get("time_of_day"):
        where.append(_time_phrase(str(env["time_of_day"])))
    if env.get("weather"):
        where.append(f"in {env['weather']} weather")
    if where:
        head += " " + " ".join(where)

    details: list[str] = []
    angle = _ANGLE_WORDS.get(str(cam.get("camera_angle") or ""))
    if angle:
        details.append(angle)
    movement = camera_phrase(cam)
    if movement:
        details.append(movement)
    sm = rec.get("subject_motion")
    motion_phrase = None
    if isinstance(sm, dict) and sm.get("label") and sm["label"] not in ("static", "tracked_by_camera"):
        try:
            motion_phrase = subject_motion_phrase(SubjectMotion.model_validate(sm))
        except Exception:  # a legacy / partial subject_motion dict: skip the phrase rather than fail the export
            motion_phrase = None
    if motion_phrase:
        details.append(motion_phrase)
    light = env.get("lighting") or meas.get("lighting_level")
    if light:
        details.append(f"{str(light).strip()} lighting")
    temp = meas.get("color_temperature")
    if temp and temp != "neutral":
        details.append(f"{temp} tones")
    sat = meas.get("saturation_mean")
    if isinstance(sat, (int, float)):
        if sat < 50:
            details.append("desaturated colours")
        elif sat >= 120:
            details.append("highly saturated colours")
    if cam.get("depth_of_field"):
        details.append(f"{cam['depth_of_field']} depth of field")
    if cam.get("lens_type") or cam.get("focal_length_mm"):
        lens = _LENS_WORDS.get(str(cam.get("lens_type") or ""), "lens")
        focal = f"{cam['focal_length_mm']:.0f}mm " if cam.get("focal_length_mm") else ""
        details.append(f"shot on a {focal}{lens}")
    if cam.get("stabilization") in ("stabilized", "gimbal", "steadicam", "tripod"):
        details.append(f"{cam['stabilization']} camera")
    # visible actions, minus camera-movement sentences and the subject-motion sentence (already covered above)
    actions = [str(a).strip().rstrip(".") for a in (rec.get("actions") or []) if a and not str(a).lower().startswith(("camera ", "the camera"))]
    actions = [a for a in actions if not motion_phrase or a.lower() != motion_phrase.lower()]

    if not subject and not where and not details and not actions:
        return None
    text = head
    if actions:
        text += "; " + "; ".join(a[0].lower() + a[1:] for a in actions[:3])
    if details:
        text += ", " + ", ".join(details)
    return text.strip() + "."


def fill_generation_prompts(records: dict[str, list[dict[str, Any]]]) -> int:
    """Set ``generation_prompt`` on scene records and on their clip records (a clip inherits the
    prompt of its scene; a clip without a scene record gets one from its own fields). In place."""
    n = 0
    by_scene: dict[tuple[str, str], str | None] = {}
    for rec in records.get("scenes", []):
        rec["generation_prompt"] = generation_prompt(rec)
        by_scene[(str(rec.get("video_id")), str(rec.get("scene_id")))] = rec["generation_prompt"]
        n += rec["generation_prompt"] is not None
    for rec in records.get("clips", []):
        key = (str(rec.get("video_id")), str(rec.get("scene_id")))
        rec["generation_prompt"] = by_scene[key] if key in by_scene else generation_prompt(rec)
        n += rec["generation_prompt"] is not None
    return n


def fill_frame_clip_ids(records: dict[str, list[dict[str, Any]]]) -> int:
    """Link every frame record to the clip that contains it (video -> clip -> frame). When a scene
    has sub-clips, the whole-scene clip (shortest id) wins. In place; returns the number linked."""
    clip_of: dict[tuple[str, str], str] = {}
    for rec in sorted(records.get("clips", []), key=lambda r: (len(str(r.get("clip_id") or "")), str(r.get("clip_id") or "")), reverse=True):
        for fid in rec.get("frame_ids") or []:
            clip_of[(str(rec.get("video_id")), str(fid))] = str(rec["clip_id"])
    n = 0
    for rec in records.get("frames", []):
        cid = clip_of.get((str(rec.get("video_id")), str(rec.get("frame_id"))))
        if cid:
            rec["clip_id"] = cid
            n += 1
        else:
            rec.setdefault("clip_id", None)
    return n
