"""Vision / multimodal analysis schemas."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from video_dataset.schemas.common import BaseSchema, ConfidenceSource, StrEnum

UNKNOWN = "unknown"


class Setting(StrEnum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ShotType(StrEnum):
    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    UNKNOWN = "unknown"


class CameraAngle(StrEnum):
    EYE_LEVEL = "eye_level"
    LOW = "low"
    HIGH = "high"
    OVERHEAD = "overhead"
    AERIAL = "aerial"
    DUTCH = "dutch"
    UNKNOWN = "unknown"


class CameraMovement(StrEnum):
    """Fine-grained camera movement label: family + direction in one token. The standardized
    taxonomy (family / direction) is derived from it, see ``MOVEMENT_TAXONOMY``."""

    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"  # the camera body moves forward (a VLM can tell; optical flow reports zoom_in for both)
    DOLLY_OUT = "dolly_out"
    ZOOM_IN = "zoom_in"  # optical flow: content expands (lens zoom or forward move)
    ZOOM_OUT = "zoom_out"
    TRACKING = "tracking"  # follows the subject, direction not resolved
    TRACKING_LEFT = "tracking_left"
    TRACKING_RIGHT = "tracking_right"
    TRACKING_FORWARD = "tracking_forward"
    TRACKING_BACKWARD = "tracking_backward"
    ORBIT = "orbit"  # arcs around the subject, direction not resolved
    ORBIT_CLOCKWISE = "orbit_clockwise"
    ORBIT_COUNTERCLOCKWISE = "orbit_counterclockwise"
    CRANE = "crane"  # vertical boom / jib move, direction not resolved
    CRANE_UP = "crane_up"
    CRANE_DOWN = "crane_down"
    HANDHELD = "handheld"
    FPV = "fpv"  # first-person / POV camera
    DRONE = "drone"  # aerial drone flight
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class CameraMovementFamily(StrEnum):
    """Standardized movement vocabulary; ``movement_direction`` carries the qualifier."""

    STATIC = "static"
    PAN = "pan"  # left | right
    TILT = "tilt"  # up | down
    DOLLY = "dolly"  # in | out
    TRACKING = "tracking"  # left | right | forward | backward
    ORBIT = "orbit"  # clockwise | counterclockwise
    CRANE = "crane"  # up | down
    ZOOM = "zoom"  # in | out
    HANDHELD = "handheld"
    FPV = "fpv"
    DRONE = "drone"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


_F = CameraMovementFamily
MOVEMENT_TAXONOMY: dict[CameraMovement, tuple[CameraMovementFamily, str | None]] = {
    CameraMovement.STATIC: (_F.STATIC, None),
    CameraMovement.PAN_LEFT: (_F.PAN, "left"),
    CameraMovement.PAN_RIGHT: (_F.PAN, "right"),
    CameraMovement.TILT_UP: (_F.TILT, "up"),
    CameraMovement.TILT_DOWN: (_F.TILT, "down"),
    CameraMovement.DOLLY_IN: (_F.DOLLY, "in"),
    CameraMovement.DOLLY_OUT: (_F.DOLLY, "out"),
    CameraMovement.ZOOM_IN: (_F.ZOOM, "in"),
    CameraMovement.ZOOM_OUT: (_F.ZOOM, "out"),
    CameraMovement.TRACKING: (_F.TRACKING, None),
    CameraMovement.TRACKING_LEFT: (_F.TRACKING, "left"),
    CameraMovement.TRACKING_RIGHT: (_F.TRACKING, "right"),
    CameraMovement.TRACKING_FORWARD: (_F.TRACKING, "forward"),
    CameraMovement.TRACKING_BACKWARD: (_F.TRACKING, "backward"),
    CameraMovement.ORBIT: (_F.ORBIT, None),
    CameraMovement.ORBIT_CLOCKWISE: (_F.ORBIT, "clockwise"),
    CameraMovement.ORBIT_COUNTERCLOCKWISE: (_F.ORBIT, "counterclockwise"),
    CameraMovement.CRANE: (_F.CRANE, None),
    CameraMovement.CRANE_UP: (_F.CRANE, "up"),
    CameraMovement.CRANE_DOWN: (_F.CRANE, "down"),
    CameraMovement.HANDHELD: (_F.HANDHELD, None),
    CameraMovement.FPV: (_F.FPV, None),
    CameraMovement.DRONE: (_F.DRONE, None),
    CameraMovement.COMPLEX: (_F.COMPLEX, None),
    CameraMovement.UNKNOWN: (_F.UNKNOWN, None),
}
_BY_TAXONOMY: dict[tuple[CameraMovementFamily, str | None], CameraMovement] = {v: k for k, v in MOVEMENT_TAXONOMY.items()}
_DIRECTION_ALIASES = {"counter_clockwise": "counterclockwise", "counter-clockwise": "counterclockwise", "anticlockwise": "counterclockwise", "forwards": "forward", "backwards": "backward", "inward": "in", "outward": "out"}


def _as_enum(value: Any, enum_cls: type[Any], default: Any) -> Any:
    if value is None:
        return default
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        return default


def movement_from_taxonomy(family: Any, direction: Any) -> CameraMovement:
    """(family, direction) -> fine-grained label; the family alone when the direction is unknown or
    does not exist for that family (e.g. ``("pan", None)`` has no member and yields UNKNOWN)."""
    fam = _as_enum(family, CameraMovementFamily, CameraMovementFamily.UNKNOWN)
    d = str(direction).strip().lower().replace(" ", "_") if direction else None
    d = _DIRECTION_ALIASES.get(d, d) if d else None
    return _BY_TAXONOMY.get((fam, d)) or _BY_TAXONOMY.get((fam, None)) or CameraMovement.UNKNOWN


_CAMERA_HEIGHT = {"low": "low", "eye_level": "eye_level", "high": "high", "overhead": "overhead", "aerial": "aerial"}
_CAMERA_DISTANCE = {"extreme_wide": "very_far", "wide": "far", "medium": "medium", "close_up": "near", "extreme_close_up": "very_near"}


def derive_camera_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Fill the standardized / derived camera fields of a CameraAnnotation dict from its primary
    fields. ``movement`` is the source of truth for the taxonomy; when it is unknown but a family
    (+ direction) was given, ``movement`` is reconstructed from them instead. Pure: returns ``d``."""
    movement = _as_enum(d.get("movement"), CameraMovement, CameraMovement.UNKNOWN)
    given_family, given_direction = d.get("camera_movement"), d.get("movement_direction")
    if movement == CameraMovement.UNKNOWN and given_family:
        movement = movement_from_taxonomy(given_family, given_direction)
    if movement != CameraMovement.UNKNOWN:
        family, direction = MOVEMENT_TAXONOMY[movement]
        if direction is None and given_direction:  # "tracking" + direction "left" -> tracking_left
            refined = movement_from_taxonomy(family, given_direction)
            if MOVEMENT_TAXONOMY[refined][1] is not None:
                movement, direction = refined, MOVEMENT_TAXONOMY[refined][1]
        d["movement"], d["camera_movement"], d["movement_direction"] = movement.value, family.value, direction
    else:
        d["movement"] = CameraMovement.UNKNOWN.value
        d["camera_movement"] = _as_enum(given_family, CameraMovementFamily, None)
        d["camera_movement"] = d["camera_movement"].value if d["camera_movement"] else None
    shot = _as_enum(d.get("shot_type"), ShotType, ShotType.UNKNOWN)
    d["shot_size"] = shot.value if shot != ShotType.UNKNOWN else None
    d["camera_distance"] = _CAMERA_DISTANCE.get(shot.value) if shot != ShotType.UNKNOWN else d.get("camera_distance")
    angle = _as_enum(d.get("camera_angle"), CameraAngle, CameraAngle.UNKNOWN)
    if angle != CameraAngle.UNKNOWN and angle.value in _CAMERA_HEIGHT:
        d["camera_height"] = _CAMERA_HEIGHT[angle.value]
    if d.get("speed"):
        d["movement_speed"] = d["speed"]
    if d.get("stability"):
        d["stabilization"] = d["stability"]
    elif not d.get("stabilization") and movement in (CameraMovement.STATIC, CameraMovement.HANDHELD):
        d["stabilization"] = movement.value
    if d.get("is_aerial") is None and (movement == CameraMovement.DRONE or angle == CameraAngle.AERIAL):
        d["is_aerial"] = True
    if d.get("focal_length_mm") is not None:
        try:
            d["focal_length_mm"] = float(d["focal_length_mm"])
        except (TypeError, ValueError):
            d["focal_length_mm"] = None
    return d


def normalize_depth_of_field(text: Any) -> str | None:
    """Free text such as 'shallow, background blurred' -> shallow | medium | deep (None when unclear)."""
    if not text:
        return None
    low = str(text).lower()
    for word, label in (("shallow", "shallow"), ("deep", "deep"), ("medium", "medium"), ("moderate", "medium")):
        if word in low:
            return label
    return None


def camera_fields_from_style(camera: dict[str, Any] | None, style: dict[str, Any] | None) -> dict[str, Any] | None:
    """Copy depth_of_field / perspective from a VisualStyle dict into a CameraAnnotation dict when the
    camera does not have them yet (the VLM reports them under visual_style)."""
    if not isinstance(camera, dict) or not isinstance(style, dict):
        return camera
    if not camera.get("depth_of_field") and style.get("depth_of_field"):
        dof = normalize_depth_of_field(style["depth_of_field"])
        if dof:
            camera = {**camera, "depth_of_field": dof}
    if not camera.get("perspective") and style.get("perspective") and len(str(style["perspective"])) <= 60:
        camera = {**camera, "perspective": str(style["perspective"]).strip()}
    return camera


class Environment(BaseSchema):
    location: str | None = None
    setting: Setting = Setting.UNKNOWN
    weather: str | None = None
    lighting: str | None = None
    time_of_day: str | None = None
    background: str | None = None


class ObjectAnnotation(BaseSchema):
    name: str
    attributes: list[str] = Field(default_factory=list)
    location: str | None = None  # e.g. "center", "left foreground"
    count: int | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    # composition, measured from detector boxes (None when the provider reports no boxes)
    position: str | None = None  # 3x3 grid cell of the mean box centre: "center", "upper-left", "lower-right", ...
    center: list[float] | None = None  # mean normalised box centre [cx, cy] in 0..1
    area_fraction: float | None = Field(default=None, ge=0, le=1)  # mean box area / frame area
    scale: str | None = None  # tiny | small | medium | large | dominant (from area_fraction)
    frame_fraction: float | None = Field(default=None, ge=0, le=1)  # share of analysed frames the object appears in


class PersonAnnotation(BaseSchema):
    """People are described by count / visible action / clothing only. No identities."""

    count: int | None = None
    actions: list[str] = Field(default_factory=list)
    clothing: list[str] = Field(default_factory=list)
    body_position: str | None = None
    interactions: list[str] = Field(default_factory=list)


class CameraAnnotation(BaseSchema):
    """Camera description. The primary fields come from the analyzer (optical flow / VLM); the
    standardized block below is derived from them on validation (see ``derive_camera_fields``) so
    every record carries the same vocabulary whichever provider produced it. Lens / optics fields are
    estimates a VLM may give; nothing in the pipeline measures them, so they stay None otherwise."""

    shot_type: ShotType = ShotType.UNKNOWN
    camera_angle: CameraAngle = CameraAngle.UNKNOWN
    movement: CameraMovement = CameraMovement.UNKNOWN
    zoom: str | None = None
    stability: str | None = None  # "static" | "handheld" | "stabilized" | None
    is_aerial: bool | None = None
    speed: str | None = None  # slow | moderate | fast - measured from optical flow (fraction of the frame per second)
    tracked_subject: str | None = None  # set when movement == tracking: the detected object the camera follows
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    # --- standardized taxonomy, derived from the primary fields ---
    shot_size: str | None = None  # = shot_type under its standard name: extreme_wide | wide | medium | close_up | extreme_close_up
    camera_movement: CameraMovementFamily | None = None  # static | pan | tilt | dolly | tracking | orbit | crane | zoom | handheld | fpv | drone | complex
    movement_direction: str | None = None  # left | right | up | down | in | out | forward | backward | clockwise | counterclockwise
    movement_speed: str | None = None  # slow | moderate | fast (= speed; measured from optical flow)
    camera_height: str | None = None  # low | eye_level | high | overhead | aerial (from camera_angle)
    camera_distance: str | None = None  # very_far | far | medium | near | very_near (from shot size)
    stabilization: str | None = None  # static | handheld | stabilized | tripod | gimbal | steadicam (= stability)
    # --- lens / optics: VLM estimates only (lens_confidence_source says so); never measured ---
    lens_type: str | None = None  # wide_angle | normal | portrait | telephoto | anamorphic | fisheye | macro
    focal_length_mm: float | None = None  # full-frame-equivalent estimate, e.g. 24, 35, 50, 85
    depth_of_field: str | None = None  # shallow | medium | deep
    focus_type: str | None = None  # fixed | rack_focus | follow_focus | soft
    perspective: str | None = None  # e.g. "linear", "first person", "over the shoulder"
    lens_confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE

    @model_validator(mode="before")
    @classmethod
    def _derive(cls, data: Any) -> Any:
        return derive_camera_fields(dict(data)) if isinstance(data, dict) else data


class VisualStyle(BaseSchema):
    composition: str | None = None
    lighting: str | None = None
    color: str | None = None
    depth_of_field: str | None = None
    framing: str | None = None
    perspective: str | None = None
    motion: str | None = None
    transitions: str | None = None


class Measurements(BaseSchema):
    """Directly measured, model-free signals. Always grounded."""

    brightness_mean: float | None = None  # 0..255
    brightness_std: float | None = None
    contrast: float | None = None  # 0..1 (std / 128 clipped)
    saturation_mean: float | None = None  # 0..255
    colorfulness: float | None = None  # Hasler & Süsstrunk metric
    dominant_colors: list[str] = Field(default_factory=list)  # hex strings
    color_temperature: str | None = None  # warm | neutral | cool
    edge_density: float | None = None  # 0..1
    sharpness: float | None = None  # variance of Laplacian (normalized)
    motion_magnitude: float | None = None  # mean optical flow magnitude (analysis px/frame)
    camera_motion_label: str | None = None  # from optical flow interpretation
    camera_motion_consistency: float | None = None  # 0..1 fraction of samples agreeing with label
    camera_speed: str | None = None  # slow | moderate | fast (None when the camera is static / unmeasured)
    camera_speed_fraction_per_second: float | None = None  # mean global flow as fraction of frame width per second
    brightness_trend: str | None = None  # "brightening" | "darkening" | "stable"
    lighting_level: str | None = None  # dark | dim | normal | bright
    aesthetic_score: float | None = None  # LAION aesthetic predictor on CLIP embeddings, ~1 (poor) .. 10 (excellent)
    aesthetic_model: str | None = None  # which linear head produced it, e.g. "laion/sa_0_4_vit_b_32_linear"


class SubjectMotion(BaseSchema):
    """Motion of the main detected subject, separated from the camera motion.

    Box velocity is measured in frame coordinates (fraction of the frame per second) from detector
    boxes across the analysed frames; camera velocity is the global optical-flow translation in the
    same units. ``relative_velocity`` = box velocity - camera-induced velocity is the subject's own
    motion. When the subject stays put in the frame while the background streams, the camera is
    tracking it.
    """

    subject: str  # detected class name, e.g. "person"
    label: str  # static | moves_left | moves_right | moves_up | moves_down | approaches | recedes | tracked_by_camera | panned_past
    camera_relation: str  # camera_static | tracked | moves_with_camera | moves_against_camera | independent
    box_velocity: list[float] = Field(default_factory=list)  # [vx, vy] fraction of frame / s, frame coordinates
    camera_velocity: list[float] = Field(default_factory=list)  # [vx, vy] apparent background motion, same units
    relative_velocity: list[float] = Field(default_factory=list)  # box - camera
    scale_change_per_second: float | None = None  # d(area_fraction)/dt; > 0 grows (approaches), < 0 shrinks
    n_frames: int = 0  # analysed frames in which the subject was detected
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.MEASUREMENT


class FrameAnalysis(BaseSchema):
    frame_id: str
    scene_id: str
    timestamp: float
    caption: str | None = None
    objects: list[ObjectAnnotation] = Field(default_factory=list)
    people: PersonAnnotation | None = None
    actions: list[str] = Field(default_factory=list)
    environment: Environment | None = None
    camera: CameraAnnotation | None = None
    measurements: Measurements | None = None
    provider: str | None = None
    model: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE


class VerificationVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"  # verifier could not judge -> no confidence contribution


class VerificationResult(BaseSchema):
    claim: str
    verdict: VerificationVerdict = VerificationVerdict.UNKNOWN
    score: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None
    verifier: str | None = None
    evidence_frame_ids: list[str] = Field(default_factory=list)


class SceneAnalysis(BaseSchema):
    scene_id: str
    video_id: str
    start_time: float
    end_time: float
    summary: str | None = None
    environment: Environment = Field(default_factory=Environment)
    objects: list[ObjectAnnotation] = Field(default_factory=list)
    people: PersonAnnotation | None = None
    actions: list[str] = Field(default_factory=list)
    camera: CameraAnnotation = Field(default_factory=CameraAnnotation)
    subject_motion: SubjectMotion | None = None  # measured from detector boxes + optical flow (cv_models)
    visual_style: VisualStyle = Field(default_factory=VisualStyle)
    temporal_progression: str | None = None  # what changes from the start to the end of the scene
    frame_analyses: list[FrameAnalysis] = Field(default_factory=list)
    measurements: Measurements | None = None
    frame_ids: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_source: ConfidenceSource = ConfidenceSource.UNAVAILABLE
    verification: VerificationResult | None = None
    enrichments: dict[str, Any] = Field(default_factory=dict)  # e.g. CLIP zero-shot scores
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _camera_from_style(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("camera"), dict):
            data = {**data, "camera": camera_fields_from_style(data["camera"], data.get("visual_style"))}
        return data


class VisionResult(BaseSchema):
    video_id: str
    provider: str
    model: str | None = None
    scenes: list[SceneAnalysis] = Field(default_factory=list)
