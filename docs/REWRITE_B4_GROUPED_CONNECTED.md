# Grouped connected B4 camera rendering

## Purpose

Independent B4 renders preserve one object's camera-dependent appearance, but
separate Spine slots still use one fixed draw order. They cannot reproduce
arbitrary per-pixel intersections when connected objects overlap or exchange
front/back order.

Grouped B4 renders every compatible connected camera-dependent object together.
Blender resolves depth, transparency, reflection, transmission and ray
visibility before the result becomes one root-bound Spine attachment.

## Output policy

`A1MultiObjectExportSettings.connected_b4_render_policy` accepts:

- `INDIVIDUAL_LAYERS`;
- `AUTO_GROUPED_CAMERA`, the production default;
- `GROUPED_CAMERA_REQUIRED`.

AUTO groups only a complete compatible connected B4 set. A mixed local/B4 set
falls back to individual layers because UV-baked and camera-rendered coordinate
spaces still have unresolved depth against one another. REQUIRED raises an
explicit policy error instead of falling back.

## Immutable compatibility gate

Every grouped source must:

- use `CameraProjectionPlan`;
- share renderer-specific `SceneBakeContext` and active camera;
- use identical dimensions, format, margin and frame range;
- use identical `BakeExecutionSettings`;
- use one image-relative directory;
- expose exactly one camera-projection visual slot;
- use unique source IDs;
- use collision-free individual and grouped output paths.

`GroupedCameraProjectionPlan` remains Blender-independent. Runtime validation
revalidates builder assumptions so a manually constructed grouped plan cannot
bypass dense frame indices, unique paths or the common render contract.

## Physical execution ownership

The former `grouped_camera_projection_executor.py` mixed validation, RNA
identity, visibility mutation, reservation, rendering, coverage union, contour
construction and crop rewrite.

Physical ownership is now:

```text
grouped_camera_projection_validation.py
  -> object name and Blender RNA identity
  -> complete grouped request validation
  -> per-source single-B4 runtime validation
  -> common Context, Scene, renderer and output policy
  -> grouped frame/path contract
  -> strict reservation-order validation
  -> GroupedCameraProjectionRuntime

grouped_camera_projection_visibility.py
  -> grouped source camera visibility
  -> direct-camera isolation for other renderable objects
  -> no validation, rendering, coverage or output lifecycle

grouped_camera_projection_execution.py
  -> consume GroupedCameraProjectionRuntime and existing reservations
  -> one reversible Scene/frame/visibility scope
  -> render every full-frame grouped task
  -> require non-empty staged files
  -> no coverage, crop, reserve or commit

camera_projection_postprocess.py
  -> shared ProjectionPostprocessRequest
  -> one single/grouped sequence coverage engine
  -> morphology, crop, contour, triangulation and image rewrite

grouped_camera_projection_postprocess.py
  -> adapt grouped runtime to the shared postprocess engine
  -> grouped ownership diagnostics
  -> no Scene mutation, render operator or output lifecycle

grouped_camera_projection_output.py
  -> GroupedCameraProjectionStageResult
  -> validate before caller-owned reservation
  -> reserve, render, restore and postprocess
  -> no transaction creation and no commit

grouped_camera_projection_executor.py
  -> compatibility re-exports only
```

Production multi/mixed output imports `grouped_camera_projection_output.py`
directly. Historical private names remain available through the compatibility
facade:

```text
_object_name
_rna_identity
_validate_group_runtime
_configure_group_camera_visibility
_reserve_group_outputs
```

## Validation-before-reservation

`stage_grouped_camera_projection_outputs()` performs the complete grouped
validation before `AtomicFileTransaction.reserve()`:

1. validate source tuple, grouped plan and execution settings;
2. validate dense frame tasks and unique grouped output paths;
3. reject collisions with individual source outputs;
4. reject duplicate Blender objects by RNA identity;
5. require source order to equal `plan.source_object_ids`;
6. resolve renderer and output policy;
7. validate every source with the single-B4 runtime validator;
8. require one Context, Scene, renderer and output policy;
9. validate the caller-owned transaction;
10. reserve grouped outputs in exact frame-task order.

Invalid HDR, tone-mapping, alpha or texture-format combinations therefore fail
before filesystem reservation or Scene mutation.

## Blender visibility

During grouped render:

- every grouped source receives `hide_render=False`;
- every grouped source receives `visible_camera=True` when available;
- other renderable objects receive only `visible_camera=False`;
- dependency objects retain diffuse, glossy, transmission, occlusion and shadow
  participation;
- lights and cameras are not hidden;
- every changed value is restored by the existing B4 state boundary.

Blender identity prefers `as_pointer()` and falls back to RNA name, then Python
identity. This avoids false mismatches from transient Python RNA wrappers.

## Reversible render and postprocess ordering

The grouped mutable Blender scope now contains rendering only:

```text
validate grouped request
-> reserve caller-owned outputs
-> enter preserve_camera_projection_state()
-> configure grouped visibility
-> render every full-frame staged file
-> leave preserve_camera_projection_state()
-> decode alpha coverage
-> max-union sequence coverage
-> cleanup, crop, contour and triangulate
-> rewrite every staged image with one layout
-> return GroupedCameraProjectionStageResult
```

Coverage decode and crop rewrite run only after Scene, frame and visibility state
has been restored. Render failure never starts postprocessing. Postprocess
failure occurs while the original Blender state is already active again.

## Shared postprocess engine

Single and grouped B4 now use the same `process_projection_outputs()` engine and
the same:

- deterministic 8-bit alpha decode;
- `O(width * height)` sequence max-union;
- hysteresis and conservative morphology;
- stable padded crop;
- simplified concave contour;
- convex fallback for disconnected outer components;
- deterministic exact triangulation;
- HDR, tone-mapping and straight/premultiplied-alpha rewrite.

The image crop writer explicitly accepts both `CameraProjectionPlan` and
`GroupedCameraProjectionPlan`. This formalizes the existing real call contract
without changing pixel semantics.

## Atomic ownership

Grouped staging never creates or commits a transaction.

Connected and mixed production output retain one outer transaction:

```text
prepare every source
-> reserve JSON
-> stage/finalize individual textures
-> validate and stage grouped B4 textures
-> compose typed connected document
-> apply grouped overlay
-> compose outer mixed document when required
-> serialize JSON
-> commit JSON + individual textures + grouped textures exactly once
```

The outer output service verifies committed paths against JSON, individual and
grouped reservation order. Any grouped render, decode, layout, overlay,
serialization or commit failure rolls back the complete export.

Individual B4 textures are still emitted as compatibility/debug artifacts. Their
visual slots are made transparent by the grouped overlay. Removing those hidden
files remains a separate compatibility decision.

## Spine document transformation

The original connected rig, bones, constraints, attachments and animations
remain in the document.

For every grouped source visual slot:

- setup color becomes `ffffff00`;
- color and attachment timelines are removed so the source cannot reappear;
- bone, constraint, deform, draw-order and event timelines remain.

One grouped slot is appended on the connected root and uses the final grouped
crop, contour, triangle indices and sequence metadata.

For mixed output, the grouped overlay is applied inside the connected subgroup
before composition with standalone objects.

## Intentional flattening boundary

The grouped attachment is root-bound and visually replaces individual connected
B4 slots. Runtime movement of individual source bones cannot independently move
the flattened grouped texture.

Relative source motion, camera motion and front/back changes must be present in
the rendered sequence. Retained source rig data remains available for
compatibility and nonvisual uses.

## Validation state

Focused validation for this decomposition includes:

- compilation of every new/replaced grouped production module;
- import-graph loading with Blender/domain stubs;
- ownership and ordering architecture tests;
- compatibility alias identity checks;
- validation-before-reservation checks;
- reversible-render-before-postprocess checks;
- proof that grouped output contains no transaction creation or commit;
- proof that production callers import the physical output owner.

The complete pytest suite and real Blender 4.4 grouped fixture remain manual
release gates. Automatic Actions triggers remain disabled.
