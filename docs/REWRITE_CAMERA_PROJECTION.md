# B4 camera-render projection

## Purpose

Blender object baking has no camera-ray bake type. A UV `COMBINED` bake cannot
preserve appearance that depends on the active camera or ray type, including
Fresnel, Layer Weight, Light Path, refraction, transmission, volume and
render-evaluated displacement.

B4 performs a real render from the active Blender camera and projects the
decoded screen-space result into Spine. The synthetic `CAMERA_COMBINED` pass in
`CameraProjectionPlan` is metadata only and is never sent to
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
retain source ID, settings, frame tasks, output paths and
`BakeExecutionResult`.

The real render operator is called only through
`blender_adapter/bake_executor.py::_call_render_operator`, preserving one
failure-injection boundary.

## Physical single-B4 ownership

```text
camera_projection_error.py
  -> shared CameraProjectionExecutionError

camera_projection_validation.py
  -> complete request and reservation validation
  -> renderer and output-policy resolution
  -> bpy / Context / Scene / View Layer validation
  -> CameraProjectionRuntime

camera_projection_state.py
  -> reversible Scene/frame/visibility state
  -> source-only camera visibility
  -> per-frame render configuration and timeline evaluation

camera_projection_execution.py
  -> full-frame rendering inside one reversible state scope
  -> staged-file verification
  -> no coverage, crop, reserve or commit

camera_projection_image.py
  -> staged decode and deterministic 8-bit alpha extraction
  -> pixel crop and straight/premultiplied conversion
  -> single/grouped staged image rewrite

camera_projection_postprocess.py
  -> ProjectionPostprocessRequest
  -> shared single/grouped sequence coverage engine
  -> morphology, crop, contour and triangulation
  -> rewrite all staged frames after render state restoration

camera_projection_output.py
  -> caller-owned reservation and staging
  -> direct named transaction and exactly one commit
  -> strict committed-path validation
  -> BakeExecutionResult

camera_projection_executor_core.py
  -> historical private compatibility re-exports only

camera_projection_executor.py
  -> stable public facade
```

The compatibility core no longer owns a second implementation. Historical
private names such as `_render_to_reservations`, `_reserve` and
`_build_execution_result` resolve to physical output functions.

## Shared single/grouped postprocess

`camera_projection_postprocess.py` now owns one immutable
`ProjectionPostprocessRequest` and one `process_projection_outputs()` engine.
Single B4 adapts `CameraProjectionRuntime`; grouped B4 adapts
`GroupedCameraProjectionRuntime`.

Both paths therefore use the same:

- alpha threshold;
- fixed-size sequence max-union;
- coverage cleanup;
- crop;
- contour and fallback policy;
- exact triangle validation;
- HDR/tone-mapping/alpha rewrite.

`rewrite_staged_image_with_crop()` explicitly accepts
`CameraProjectionPlan | GroupedCameraProjectionPlan`. This matches the existing
real use without changing pixel output.

## Single-B4 validation-before-mutation

A complete single B4 request is validated before
`AtomicFileTransaction.reserve()` and before Scene mutation:

1. source must be a Blender Mesh;
2. plan must be `CameraProjectionPlan`;
3. source identity must match the plan;
4. execution settings must be typed or `None`;
5. execution renderer must match the analyzed renderer;
6. output format, dynamic range, tone mapping and alpha must be compatible;
7. frame tasks must be non-empty, contiguous and path-unique;
8. Context, Scene and active camera must exist;
9. source must be available in the required View Layer;
10. Object, Scene, World, camera and light snapshots must still match.

`None` is handled explicitly. Falsy objects are not silently replaced.

Direct execution validates before creating its atomic transaction. Caller-owned
staging validates before the first reservation.

## Reversible render execution

For every static or sequence frame:

1. capture render settings, current frame, `hide_render` and `visible_camera`;
2. expose the source to direct camera rays;
3. disable only direct camera visibility for other renderable objects;
4. preserve dependency ray participation;
5. disable Compositor and Sequencer execution without mutating their data;
6. set the planned frame and update the View Layer;
7. configure renderer, dimensions, film, path and image format;
8. call the public render hook;
9. require a non-empty staged file;
10. restore all captured Blender values in `finally`.

Blender background rendering may expose a zero-sized `Render Result` after a
successful render. The staged file remains the source of truth.

## Postprocessing after state restoration

```text
render every full-frame task
-> leave preserve_camera_projection_state()
-> decode staged alpha coverage
-> build sequence union and layout
-> rewrite staged frames
```

Failures in coverage cleanup, contour construction, triangulation or image
rewrite occur after original Scene, frame and visibility state has returned.

## Stable sequence-union crop

Detailed staging uses one fixed-size accumulator:

```text
frame 1 coverage --+
frame 2 coverage --+--> max union --> cleanup --> crop + contour
...                 |
frame N coverage --+
```

Memory remains `O(width * height)`. Each decoded frame buffer is released after
`add_coverage()`.

The immutable layout records full dimensions, crop, screen offset, coverage
statistics, contour, triangle indices and frame count. Every sequence frame is
rewritten with identical dimensions and placement.

The historical reservations-only single-B4 API intentionally keeps full-frame
images and performs no coverage decode or crop rewrite.

## Coverage, contour and triangulation

Production defaults remain:

```python
BakeExecutionSettings(
    projection_alpha_threshold=1.0 / 255.0,
    projection_contour_mode=ProjectionContourMode.SIMPLIFIED_CONCAVE,
    projection_contour_simplify_tolerance_pixels=1.0,
)
```

Coverage uses `HYSTERESIS_MORPHOLOGY`. Weak antialias coverage is retained only
when connected to a strong core; translucent-only objects use the explicit
weak-only fallback. Small detached components and bounded pinholes are handled
without generic closing that could bridge objects.

One outer component may produce a simplified concave contour. Holes remain
texture alpha. Disconnected outer components use deterministic convex fallback.
Concave contours use deterministic ear clipping and must satisfy positive
orientation, `n - 2` triangles and exact total signed area.

`CameraProjectionLayout.hull` remains the compatibility field name.

## Output policy

```text
PNG / WEBP -> display-referred SDR -> Scene view transform
            -> straight alpha -> 8-bit

OPEN_EXR   -> scene-linear HDR -> no tone mapping
            -> premultiplied alpha -> 32-bit float
```

Invalid combinations fail before reservation and rendering. Crop rewrite reads
the staged `Image.alpha_mode`, performs explicit alpha conversion, normalizes
zero-alpha RGB and does not clamp finite HDR RGB.

## Atomic ownership

Single direct execution:

```text
validate request
-> create named atomic transaction
-> reserve
-> render and restore
-> postprocess
-> commit exactly once
-> require committed paths == reservation order == frame-task order
-> build BakeExecutionResult
```

Grouped execution is caller-owned:

```text
validate grouped request
-> use existing multi/mixed transaction
-> reserve grouped frames
-> render and restore
-> shared postprocess
-> return grouped reservations + layout
```

Grouped staging never creates or commits a transaction. Multi/mixed output owns
the one JSON plus individual plus grouped texture commit.

Any render, decode, layout, crop, document-finalization or commit failure rolls
back the complete caller transaction. Existing outputs are restored according to
the atomic diagnostics policy.

## Grouped B4 physical ownership

```text
grouped_camera_projection_validation.py
  -> GroupedCameraProjectionRuntime and reservation validation

grouped_camera_projection_visibility.py
  -> grouped camera visibility mutation

grouped_camera_projection_execution.py
  -> reversible grouped rendering only

grouped_camera_projection_postprocess.py
  -> adapter to the shared postprocess engine and grouped diagnostics

grouped_camera_projection_output.py
  -> caller-owned reserve/render/postprocess staging

grouped_camera_projection_executor.py
  -> compatibility re-exports only
```

See `docs/REWRITE_B4_GROUPED_CONNECTED.md`.

## Validation state

The latest decomposition validation includes:

- compilation of all new/replaced production modules;
- import-graph loading with Blender/domain stubs;
- focused ownership and ordering architecture tests;
- compatibility alias checks;
- validation-before-reservation checks;
- state-restoration-before-postprocess checks;
- proof that grouped output contains no transaction creation or commit.

Automatic GitHub Actions were not triggered. The complete pytest suite and real
Blender 4.4 matrices remain manual release gates.
