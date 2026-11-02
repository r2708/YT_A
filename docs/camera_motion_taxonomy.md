# Camera motion taxonomy (operational definitions)

Vocabulary used by every record (`camera.camera_movement` + `camera.movement_direction`, derived from
the fine label `camera.movement`; see `src/video_dataset/schemas/vision.py`). Labels are produced by
optical-flow measurement (`camera_motion_source: optical_flow`), optionally refined by a subject
track (`optical_flow+subject_track`). Nothing in the pipeline estimates camera pose; `camera.pose`
stays `null` unless an estimator with a documented method fills it.

Conventions: flow is the apparent motion of image content at analysis resolution; `+x` right,
`+y` down. A camera pan **left** makes content stream **right** (`flow_dx > 0`); a tilt **up** makes
content stream **down**. Positive divergence = content expands. Speed is the mean global flow as a
fraction of the frame width per second (`measurements.camera_speed_fraction_per_second`).

| family | definition (what must be true in the pixels) | positive example | negative example (looks alike) | measurement limit |
|---|---|---|---|---|
| `static` | global flow magnitude below `temporal.min_camera_motion` for the whole shot; subjects may move | locked-off interview, a car crossing a fixed frame | a slow pan (`pan`), a frozen still image (`frozen` quality flag) | sub-pixel drift is static |
| `pan` (`left` / `right`) | coherent horizontal translation of the *background*; verticals stay vertical; parallax small | a landscape swept left to right | `tracking` (background streams while a subject stays centred), handheld sway | cannot tell a pan from a lateral dolly without parallax |
| `tilt` (`up` / `down`) | coherent vertical translation of the background | tilting up a building | `crane` (vertical camera travel: parallax), `zoom` with off-centre content | same as pan |
| `zoom` (`in` / `out`) | radial expansion / contraction about the frame centre with little translation | lens zoom into a face | `dolly` (forward move: parallax between fore- and background), a subject approaching a static camera | **ambiguous with dolly**: flow alone cannot separate them; such labels carry `movement_ambiguous` and stay in review |
| `dolly` (`in` / `out`) | camera body moves along its axis: radial flow **with** parallax (near objects expand faster) | a push-in on a table | `zoom` | VLM-only distinction; measurement labels it `zoom` + ambiguous |
| `tracking` (`left` / `right` / `forward` / `backward`) | the camera travels with a subject: background streams, the subject's box stays in place (`subject_motion.label: tracked_by_camera`) and the global flow is non-zero | following a walking person | `pan` past a static subject (`panned_past`), a subject moving across a static frame | needs **both** the flow and a detector track; never inferred from the subject box alone |
| `orbit` (`clockwise` / `counterclockwise`) | the camera arcs around a subject: background streams while the subject stays centred and its visible side changes | product turntable shot | `tracking` sideways | VLM / pose only; flow labels it `tracking` + ambiguous |
| `crane` (`up` / `down`) | vertical camera travel with parallax | jib rising over a crowd | `tilt` | VLM / pose only |
| `handheld` | large flow magnitude without a coherent global direction (jitter), or `stability: handheld` | run-and-gun footage | `complex` (coherent but changing), subject motion in a static frame | detector of coherence only |
| `fpv` | first-person camera (helmet / body mount) | GoPro ride | `handheld` | label only (VLM); never measured |
| `drone` | aerial flight; `camera_angle: aerial` | top-down coast flyover | `crane`, `tracking` from a vehicle | label only (VLM) or angle-derived |
| `complex` | several families in one shot (`motion_segments` disagree) or a coherent but changing direction | whip pan into a tilt | `handheld` | reported with its segments |
| `unknown` | no measurement (no scan, too short) or consistency below `temporal.min_camera_consistency` | | | the honest default; never guessed |

## What the checks do

* `camera_motion_agreement` (`validation/stage.py`) compares the label family with the measured flow
  family. Agreement with consistent flow -> `passed`. Disagreement, a look-alike pair (zoom/dolly,
  tracking/pan, tracking/tilt), or flow consistency below the threshold -> `failed` + `camera.movement_ambiguous: true`
  with `movement_alternatives`; the record goes to `review`.
* `subject_motion` keeps three entities apart: `world_motion` (background flow), the camera (label +
  `camera_motion_source`) and the subject (`relative_velocity`, `camera_relation`).
* A confusion matrix against human labels is produced by `video-dataset calibrate --camera-labels <file>`
  (`dataset/camera_eval.py`): rows = human family, columns = measured family, per-class precision /
  recall / F1, most confused pairs. The human labels come from the stratified review sample
  (`video-dataset review-sample`, column `camera_movement_label`).
* `tests/test_controlled_motion.py` renders a synthetic pan, zoom and static clip and asserts the
  measured labels and the sign convention above.

## Stabilisation labels

`camera.stabilization` is one of `static | handheld | stabilized | tripod | gimbal | steadicam |
vehicle_mount | drone | unknown`, with `stabilization_evidence` (`movement_label`, `optical_flow_jitter`,
`vlm`) and `stabilization_confidence`. Measurement supports `static`, `handheld` and `drone` only;
the others require external evidence and otherwise stay `unknown`.
