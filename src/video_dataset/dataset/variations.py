"""Prompt variations without a language model: the same structured facts rendered through
alternative sentence templates and a small synonym dictionary.

Every variant is built from the record's fields (shot size, subject, environment, camera taxonomy,
measured lighting / colour, subject motion) exactly like ``generation_prompt``; only the wording
and clause order differ, so a variant never adds or drops a fact. Deterministic per record (seeded
with the record id), so re-exports reproduce the same variants.
"""

from __future__ import annotations

import random
from typing import Any

from video_dataset.dataset.prompting import _ANGLE_WORDS, _article, _subject_phrase, _time_phrase
from video_dataset.schemas.vision import SubjectMotion
from video_dataset.vision.subject_motion import subject_motion_phrase

# synonym dictionary: canonical phrase -> alternatives (all equally factual)
SHOT_WORDS: dict[str, list[str]] = {
    "extreme_wide": ["extreme wide shot", "very wide shot", "extreme long shot"],
    "wide": ["wide shot", "wide framing", "long shot"],
    "medium": ["medium shot", "mid shot", "medium framing"],
    "close_up": ["close-up", "close-up shot", "close shot"],
    "extreme_close_up": ["extreme close-up", "tight close-up", "extreme close shot"],
}
CAMERA_WORDS: dict[str, list[str]] = {
    "static": ["static camera", "fixed camera", "locked-off camera", "stationary camera"],
    "pan": ["pan to the {d}", "camera pans {d}", "panning {d}", "a {d}ward pan"],
    "tilt": ["tilt {d}", "camera tilts {d}", "tilting {d}"],
    "zoom": ["zoom {d}", "camera zooms {d}", "zooming {d}"],
    "dolly": ["dolly {d}", "camera dollies {d}", "dolly move {d}ward"],
    "tracking": ["tracking shot moving {d}", "camera tracks {d}", "camera travels {d} following the subject", "moving camera heading {d}"],
    "orbit": ["camera orbiting the subject {d}", "orbit shot {d}", "camera circles the subject {d}"],
    "crane": ["crane {d}", "camera cranes {d}", "crane move {d}ward"],
    "handheld": ["handheld camera", "handheld camera work", "hand-held, unsteady camera"],
    "fpv": ["first-person camera", "point-of-view camera", "FPV camera"],
    "drone": ["drone shot", "aerial drone camera", "drone footage"],
    "complex": ["complex camera movement", "combined camera move", "multi-axis camera movement"],
}
SPEED_WORDS: dict[str, list[str]] = {"slow": ["slow", "gentle", "gradual"], "moderate": ["moderate", "medium-speed", "medium-paced"], "fast": ["fast", "quick", "rapid"]}
LIGHT_WORDS: dict[str, list[str]] = {
    "bright": ["bright lighting", "brightly lit", "high-key lighting"],
    "normal": ["normal lighting", "evenly lit", "balanced lighting"],
    "dim": ["dim lighting", "dimly lit", "low-key lighting"],
    "dark": ["dark lighting", "very low light", "mostly dark"],
}
TEMP_WORDS: dict[str, list[str]] = {"warm": ["warm tones", "warm colour palette", "warm-toned image"], "cool": ["cool tones", "cool colour palette", "cool-toned image"]}
SAT_WORDS: dict[str, list[str]] = {"desaturated": ["desaturated colours", "muted colours", "washed-out colour"], "saturated": ["highly saturated colours", "vivid colours", "strongly saturated colour"]}
MOTION_WORDS: dict[str, list[str]] = {
    "left": ["the {n} moves left", "the {n} travels to the left", "the {n} heads left"],
    "right": ["the {n} moves right", "the {n} travels to the right", "the {n} heads right"],
    "up": ["the {n} moves up", "the {n} rises in the frame", "the {n} moves upward"],
    "down": ["the {n} moves down", "the {n} descends in the frame", "the {n} moves downward"],
    "toward_camera": ["the {n} approaches the camera", "the {n} comes toward the camera", "the {n} grows larger as it nears the camera"],
    "away_from_camera": ["the {n} moves away from the camera", "the {n} recedes into the distance", "the {n} shrinks as it moves away"],
}
DURATION_WORDS = ["A {d}-second", "A {d}s", "A roughly {d}-second"]


def _pick(rng: random.Random, options: list[str]) -> str:
    return options[rng.randrange(len(options))] if options else ""


def prompt_facts(rec: dict[str, Any]) -> dict[str, Any]:
    """The facts ``generation_prompt`` is built from, as a dict (None where the record has nothing)."""
    cam = rec.get("camera") if isinstance(rec.get("camera"), dict) else {}
    env = rec.get("environment") if isinstance(rec.get("environment"), dict) else {}
    meas = rec.get("measurements") if isinstance(rec.get("measurements"), dict) else {}
    duration = rec.get("duration")
    if duration is None and rec.get("start_time") is not None and rec.get("end_time") is not None:
        duration = float(rec["end_time"]) - float(rec["start_time"])
    sm = rec.get("subject_motion") if isinstance(rec.get("subject_motion"), dict) else None
    motion = None
    motion_phrase = None
    if sm and sm.get("label") not in (None, "static", "tracked_by_camera", "panned_past"):
        try:
            m = SubjectMotion.model_validate(sm)
            motion = {"noun": "person" if m.subject == "person" else m.subject, "direction": m.direction or None}
            motion_phrase = subject_motion_phrase(m).lower()
        except Exception:
            motion = None
    sat = meas.get("saturation_mean")
    where: list[str] = []
    if env.get("location"):
        loc = str(env["location"]).strip()
        where.append(f"in {loc}" if loc.lower().startswith(("a ", "an ", "the ")) else f"in {_article(loc)}")
    elif env.get("setting") in ("indoor", "outdoor"):
        where.append("indoors" if env["setting"] == "indoor" else "outdoors")
    if env.get("time_of_day"):
        where.append(_time_phrase(str(env["time_of_day"])))
    if env.get("weather"):
        where.append(f"in {env['weather']} weather")
    actions = [str(a).strip().rstrip(".") for a in (rec.get("actions") or []) if a and not str(a).lower().startswith(("camera ", "the camera"))]
    return {
        "duration": float(duration) if duration else None,
        "shot": str(cam.get("shot_size")) if cam.get("shot_size") in SHOT_WORDS else None,
        "subject": _subject_phrase(rec),
        "where": where,
        "angle": _ANGLE_WORDS.get(str(cam.get("camera_angle") or "")),
        "family": str(cam.get("camera_movement")) if cam.get("camera_movement") in CAMERA_WORDS else None,
        "direction": cam.get("movement_direction"),
        "speed": cam.get("movement_speed") or cam.get("speed"),
        "tracked": cam.get("tracked_subject"),
        "motion": motion,
        "light": str(env.get("lighting") or meas.get("lighting_level") or "").strip().lower() or None,
        "temp": meas.get("color_temperature") if meas.get("color_temperature") in TEMP_WORDS else None,
        "saturation": ("desaturated" if sat < 50 else ("saturated" if sat >= 120 else None)) if isinstance(sat, (int, float)) else None,
        "dof": cam.get("depth_of_field"),
        "stabilization": cam.get("stabilization") if cam.get("stabilization") in ("stabilized", "gimbal", "steadicam", "tripod") else None,
        "actions": [a for a in actions if not motion_phrase or a.lower() != motion_phrase][:3],
    }


def _camera_text(f: dict[str, Any], rng: random.Random) -> str | None:
    if not f["family"]:
        return None
    template = _pick(rng, CAMERA_WORDS[f["family"]])
    if "{d}" in template:
        if not f["direction"]:
            template = template.split(" {d}")[0].replace(" to the", "").replace(" {d}ward", "")
        template = template.replace("{d}", str(f["direction"] or ""))
    text = template
    if f["speed"] and f["family"] not in ("static", "handheld", "fpv", "drone", "complex"):
        speed = _pick(rng, SPEED_WORDS.get(str(f["speed"]), [str(f["speed"])]))
        text = f"a {speed} {text[2:]}" if text.startswith("a ") else f"{speed} {text}"
    if f["tracked"] and f["family"] == "tracking":
        text += f" following {_article(str(f['tracked']))}"
    return text


def _detail_texts(f: dict[str, Any], rng: random.Random, include_camera: bool = True) -> list[str]:
    out: list[str] = []
    if f["angle"]:
        out.append(f["angle"])
    cam = _camera_text(f, rng) if include_camera else None
    if cam:
        out.append(cam)
    if f["motion"] and f["motion"]["direction"] in MOTION_WORDS:
        out.append(_pick(rng, MOTION_WORDS[f["motion"]["direction"]]).replace("{n}", f["motion"]["noun"]))
    if f["light"]:
        out.append(_pick(rng, LIGHT_WORDS.get(f["light"], [f"{f['light']} lighting"])))
    if f["temp"]:
        out.append(_pick(rng, TEMP_WORDS[f["temp"]]))
    if f["saturation"]:
        out.append(_pick(rng, SAT_WORDS[f["saturation"]]))
    if f["dof"]:
        out.append(f"{f['dof']} depth of field")
    if f["stabilization"]:
        out.append(f"{f['stabilization']} camera")
    return out


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def render_variant(f: dict[str, Any], style: int, rng: random.Random) -> str | None:
    """One rendering of the facts. Styles: 0 'camera-first', 1 'subject-first', 2 'flowing with
    and', 3 'duration-last'. Returns None when the facts are too thin for that style."""
    shot = _pick(rng, SHOT_WORDS[f["shot"]]) if f["shot"] else "shot"
    dur = f"{f['duration']:.0f}" if f["duration"] else None
    subject = f["subject"]
    where = " ".join(f["where"])
    actions = "; ".join(a[0].lower() + a[1:] for a in f["actions"])
    if style == 0:
        cam = _camera_text(f, rng)
        if not cam:
            return None
        rest = _detail_texts(f, rng, include_camera=False)
        dur_phrase = _pick(rng, DURATION_WORDS).replace("{d}", dur) if dur else "a"
        head = f"{_cap(cam)}: {dur_phrase[:1].lower()}{dur_phrase[1:]} {shot}"
        if subject:
            head += f" of {subject}"
        if where:
            head += f" {where}"
        tail = (", " + ", ".join(rest) if rest else "") + ("; " + actions if actions else "")
        return (head + tail).strip() + "."
    if style == 1:
        if not subject:
            return None
        head = _cap(subject)
        if where:
            head += f" {where}"
        head += f", {shot}" + (f" lasting {dur} seconds" if dur else "")
        details = _detail_texts(f, rng)
        text = head + (": " + "; ".join(details) if details else "")
        if actions:
            text += f". {_cap(actions)}"
        return text.strip() + "."
    if style == 2:
        details = [d for d in _detail_texts(f, rng) if d != f["angle"]]
        if not details and not subject:
            return None
        head = (f"{_pick(rng, DURATION_WORDS).replace('{d}', dur)} " if dur else "A ") + shot
        if subject:
            head += f" showing {subject}"
        if where:
            head += f" {where}"
        if f["angle"]:
            head += f", {f['angle']}"
        if details:
            head += " with " + (", ".join(details[:-1]) + " and " + details[-1] if len(details) > 1 else details[0])
        if actions:
            head += f"; {actions}"
        return _cap(head.strip()) + "."
    details = _detail_texts(f, rng)
    if not details and not subject and not where:
        return None
    head = _cap(shot) + (f" of {subject}" if subject else "") + (f" {where}" if where else "")
    if details:
        head += ", " + ", ".join(details)
    if actions:
        head += f"; {actions}"
    if dur:
        head += f" ({dur} seconds)"
    return head.strip() + "."


def prompt_variants(rec: dict[str, Any], n: int, seed: int | str = 1234) -> list[str]:
    """Up to ``n`` distinct variants of the record's generation prompt (never the prompt itself)."""
    if n <= 0:
        return []
    base = rec.get("generation_prompt")
    if not base:
        return []
    facts = prompt_facts(rec)
    rng = random.Random(f"{seed}:{rec.get('video_id')}:{rec.get('record_id') or rec.get('scene_id')}")
    out: list[str] = []
    seen = {base}
    styles = [0, 1, 2, 3]
    attempts = 0
    while len(out) < n and attempts < 4 * max(n, 4):
        style = styles[attempts % len(styles)]
        attempts += 1
        text = render_variant(facts, style, rng)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def fill_prompt_variants(records: dict[str, list[dict[str, Any]]], n: int, seed: int | str = 1234) -> int:
    """Set ``prompt_variants`` on scene, clip and sequence record dicts in place. Clips reuse the
    variants of their scene. Returns the number of records that received at least one variant."""
    count = 0
    by_scene: dict[tuple[str, str], list[str]] = {}
    for rec in records.get("scenes", []):
        rec["prompt_variants"] = prompt_variants(rec, n, seed)
        by_scene[(str(rec.get("video_id")), str(rec.get("scene_id")))] = rec["prompt_variants"]
        count += bool(rec["prompt_variants"])
    for rec in records.get("clips", []):
        key = (str(rec.get("video_id")), str(rec.get("scene_id")))
        rec["prompt_variants"] = list(by_scene[key]) if key in by_scene else prompt_variants(rec, n, seed)
        count += bool(rec["prompt_variants"])
    for rec in records.get("sequences", []):
        rec["prompt_variants"] = sequence_prompt_variants(rec, n, seed)
        count += bool(rec["prompt_variants"])
    return count


def sequence_prompt_variants(rec: dict[str, Any], n: int, seed: int | str = 1234) -> list[str]:
    """Variants of a sequence prompt: the shot prompts stay as they are; the framing sentence varies."""
    base = rec.get("generation_prompt")
    if not base or n <= 0:
        return []
    rng = random.Random(f"{seed}:{rec.get('video_id')}:{rec.get('record_id')}")
    shots = rec.get("shot_prompts") or []
    if not shots:
        return []
    dur = float(rec.get("duration") or 0.0)
    subjects = rec.get("subjects") or []
    subj = (", ".join(subjects[:3])) if subjects else None
    heads = [
        f"A continuous {dur:.0f}-second sequence of {len(shots)} shots" + (f" featuring {subj}" if subj else ""),
        f"{len(shots)} consecutive shots over {dur:.0f} seconds" + (f", following {subj}" if subj else ""),
        f"A {dur:.0f}-second multi-shot sequence" + (f" with {subj}" if subj else ""),
        f"Sequence of {len(shots)} shots, {dur:.0f} seconds in total" + (f", subject: {subj}" if subj else ""),
    ]
    joiners = [" Then: ", " Next: ", " Followed by: "]
    out: list[str] = []
    seen = {base}
    for i in range(4 * n):
        head = heads[i % len(heads)]
        joiner = joiners[rng.randrange(len(joiners))]
        text = head + ". " + f"Shot 1: {shots[0]}" + "".join(f"{joiner}shot {k + 2}: {s}" for k, s in enumerate(shots[1:]))
        if text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) >= n:
            break
    return out
