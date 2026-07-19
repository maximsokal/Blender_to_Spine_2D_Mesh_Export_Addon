# B4 camera-render projection

## Purpose

Blender object baking has no camera-ray bake type. A UV `COMBINED` bake cannot
preserve appearance that depends on the active camera or ray type, including
Fresnel, Layer Weight, Light Path, refraction, transmission, volume and
render-evaluated displacement.

B4 therefore performs a real render from the active Blender camera and projects
the decoded screen-space result into Spine. The synthetic `CAMERA_COMBINED`
pass in `CameraProjectionPlan` is metadata only and is never sent to
`bpy.ops.object.bake`.

## Automatic routing

```text
renderer-effective material graph
    |
    +-- LOCAL / AUXILIARY / SCENE
    |     -> BakePlan
    |     -> object UV bake
    |
    +-- CAMERA / VOLUME / render displacement
          -> CameraProjectionPlan
          -> active-camera render
```

`CameraProjectionPlan` remains a frozen `BakePlan` subtype. Existing consumers
retain the normal source ID, settings, frame tasks, output paths and
`BakeExecutionResult` contracts.

The real render operator is called only through
`blender_adapter/bake_executor.py::_call_render_operator`, preserving one
failure-injection boundary.

## Physical execution ownership

The former `camera_projection_executor_core.py` mixed runtime validation,
reversible Scene mutation, rendering, coverage processing, crop rewrite,
reservation, commit and result construction.

Physical ownership is now:

```text
camera_projection_error.py
  -> shared CameraProjectionExecutionError

camera_projection_validation.py
  -> immutable request validation
  -> renderer and output-policy resolution
  -> bpy / Context / Scene resolution
  -> View Layer and Scene-context validation
  -> reservation-order validation
  -> CameraProjectionRuntime

camera_projection_state.py
  -> capture and restore Scene/frame/visibility state
  -> source-only camera visibility mutation
  -> per-frame Scene render configuration
  -> timeline evaluation

camera_projection_execution.py
  -> consume CameraProjectionRuntime and existing reservations
  -> enter one reversible state scope
  -> render every full-frame task
  -> validate each staged file
  -> no coverage, crop, reservation or commit

camera_projection_image.py
  -> staged image decode
  -> deterministic 8-bit alpha extraction
  -> pixel-buffer crop
  -> straight/premultiplied conversion
  -> staged image rewrite

camera_projection_postprocess.py
  -> fixed-size sequence coverage union
  -> morphology, crop, contour and triangulation layout
  -> rewrite all staged frames with one stable crop
  -> no Scene mutation, operator access, reservation or commit

camera_projection_output.py
  -> caller-owned reservation
  -> detailed and full-frame staging
  -> atomic transaction and exactly one commit
  -> strict committed-path order validation
  -> typed BakeExecutionResult

camera_projection_executor_core.py
  -> compatibility re-exports only

camera_projection_executor.py
  -> stable public facade
```

The compatibility core no longer owns an implementation. Historical private
names such as `_render_to_reservations`, `_reserve` and
`_build_execution_result` resolve to functions in `camera_projection_output.py`.

## Validation-before-mutation contract

A complete B4 request is validated before `AtomicFileTransaction.reserve()` and
before any Blender Scene mutation:

1. source must be a Blender Mesh;
2. plan must be `CameraProjectionPlan`;
3. source identity must match `plan.source_object_id`;
4. execution settings must be `BakeExecutionSettings` or `None`;
5. execution renderer must match the analyzed Scene renderer;
6. output format, dynamic range, tone mapping and alpha representation must be
   compatible;
7. frame tasks must be non-empty, contiguous and have unique output paths;
8. Context, Scene and active camera must exist;
9. source must be available in the required View Layer;
10. Object, Scene, World, camera and light snapshots must still match the plan.

`None` is handled explicitly. Falsy values are never silently replaced with
default settings, Context or Scene values.

Direct execution validates before creating the atomic transaction. Caller-owned
staging validates before the first reservation.

## Reversible render execution

For every static or sequence frame:

1. capture render settings, current frame, `hide_render` and
   `visible_camera`;
2. expose the source to direct camera rays;
3. disable only direct camera visibility for other renderable objects;
4. preserve their diffuse, glossy, transmission and shadow participation;
5. disable Scene Compositor and Sequencer execution without mutating their
   node/data structures;
6. set the planned timeline frame and update the View Layer;
7. configure renderer, dimensions, transparent film, output path and image
   format;
8. call the public render hook;
9. require the staged file to exist and be non-empty;
10. restore every captured Blender value in `finally`.

Blender background rendering can expose a zero-sized `Render Result` after a
successful render. The staged file is therefore the source of truth.

## Postprocessing occurs after state restoration

Coverage decode and crop rewrite no longer run while temporary Scene state is
active:

```text
render every full-frame task
-> leave preserve_camera_projection_state()
-> decode staged alpha coverage
-> build sequence union and layout
-> rewrite staged frames
```

This shortens the mutable Blender scope and guarantees that failures in contour
construction or image rewriting cannot leave temporary render settings active.

## Stable sequence-union crop

Detailed production staging uses one fixed-size accumulator:

```text
frame 1 coverage --+
frame 2 coverage --+--> max union --> cleanup --> crop + contour
...                 |
frame N coverage --+
```

Memory remains `O(width * height)`. Each decoded frame buffer is released after
`add_coverage()`.

The shared layout records:

- full dimensions and crop bounds;
- cropped dimensions and full-frame screen offset;
- alpha threshold and coverage cleanup statistics;
- simplified concave contour or deterministic convex fallback;
- exact triangle indices;
- frame count.

Every sequence frame is rewritten with identical dimensions and placement.

The historical reservations-only staging API intentionally keeps full-frame
images. It renders and restores state but performs no coverage decode and no
crop rewrite.

## Coverage, contour and triangulation

Production defaults remain:

```python
BakeExecutionSettings(
    projection_alpha_threshold=1.0 / 255.0,
    projection_contour_mode=ProjectionContourMode.SIMPLIFIED_CONCAVE,
    projection_contour_simplify_tolerance_pixels=1.0,
)
```

Coverage uses `HYSTERESIS_MORPHOLOGY` by default. Weak antialias coverage is
retained only when connected to a strong core; translucent-only objects use the
explicit weak-only fallback. Small detached components and bounded pinholes are
handled conservatively without generic closing that could bridge objects.

One outer component may produce a simplified concave contour. Holes remain
texture alpha. Disconnected outer components use a deterministic convex
fallback. Concave contours use deterministic ear clipping and must satisfy
positive orientation, `n - 2` triangles and exact total signed area.

`CameraProjectionLayout.hull` remains the compatibility field name.

## Output policy

```text
PNG / WEBP -> display-referred SDR -> Scene view transform
            -> straight alpha -> 8-bit

OPEN_EXR   -> scene-linear HDR -> no tone mapping
            -> premultiplied alpha -> 32-bit float
```

Invalid combinations fail before rendering. Crop rewrite reads Blender's
source `Image.alpha_mode`, performs explicit straight/premultiplied conversion,
normalizes zero-alpha RGB and does not clamp finite HDR RGB values.

## Atomic output lifecycle

Detailed staging:

```text
validate request
-> reserve caller-owned outputs
-> render all frames
-> restore Blender state
-> build layout and rewrite crops
-> return reservations + CameraProjectionLayout
```

Direct execution:

```text
validate request
-> create named atomic transaction
-> reserve
-> render
-> restore
-> postprocess
-> commit exactly once
-> require committed paths == reservation order
-> require committed paths == frame-task order
-> build BakeExecutionResult
```

Any render, decode, layout, crop, document-finalization or commit failure rolls
back the caller's complete JSON plus texture transaction. Existing outputs are
restored byte-for-byte and temporary work files are removed according to the
diagnostics policy.

## Post-render document finalization

The final layout exists only after every frame succeeds. A1 output services use
the detailed staging API, rebuild the B4 attachment from that layout, compose
typed Spine documents, serialize staged JSON and commit JSON plus textures
together.

Serialized JSON is never patched in place.

## Grouped B4 boundary

`grouped_camera_projection_executor.py` now imports the shared error,
validation, render hook and state helpers from their physical modules. Its
grouped visibility, coverage and caller-owned output pipeline remain unchanged
in this slice.

A later independent slice may decompose grouped B4 without changing the
single-object B4 contract.

## Validation state

For this decomposition slice:

- all new and replaced production modules compile;
- source architecture tests verify validation/execution/postprocess/output
  boundaries;
- tests verify render state ends before postprocessing;
- tests verify validation precedes reservation and transaction creation;
- tests verify direct execution contains exactly one commit;
- tests verify compatibility private aliases remain;
- automatic GitHub Actions were not triggered.

The complete pytest suite and real Blender 4.4 matrices remain manual release
gates for the final candidate.
