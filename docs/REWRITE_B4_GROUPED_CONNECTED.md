# Grouped connected B4 camera rendering

## Problem

Independent camera renders preserve one object's camera-dependent appearance, but several
connected objects exported as separate Spine slots still use one fixed draw order. That cannot
reproduce arbitrary per-pixel depth intersections when the objects overlap, cross, or exchange
front/back order.

Grouped B4 solves this specific case by rendering all compatible connected camera-dependent
objects together in Blender. Blender resolves the real depth buffer, transparency, reflection,
transmission and ray visibility before the result becomes one Spine attachment.

## Output policy

`A1MultiObjectExportSettings.connected_b4_render_policy` accepts:

- `INDIVIDUAL_LAYERS` — preserve the previous separate B4 slots;
- `AUTO_GROUPED_CAMERA` — production default; group only a complete compatible connected B4 set;
- `GROUPED_CAMERA_REQUIRED` — fail explicitly when grouping is not possible.

AUTO never groups a mixed B1/B2/B3/B4 connected set. Mixing UV-baked and camera-rendered
coordinate spaces would leave unresolved depth against the local layers, so AUTO falls back to
individual output and REQUIRED raises a structured error.

## Compatibility gate

Every connected component must:

- use `CameraProjectionPlan`;
- share one renderer-specific `SceneBakeContext` and active camera;
- use identical dimensions, output format, margin and sequence frame range;
- use identical `BakeExecutionSettings`;
- use one image-relative output directory;
- expose exactly one camera-projection visual slot;
- use unique source object IDs and collision-free grouped output paths.

Grouped planning is immutable and Blender-independent. Runtime validation reuses the existing
single-object B4 validation for every source.

## Blender visibility

During grouped render:

- every grouped source receives `hide_render=False` and `visible_camera=True`;
- other renderable objects receive only `visible_camera=False`;
- other objects remain available to reflection, transmission, diffuse, occlusion and shadow rays;
- lights and camera are not hidden;
- all render/frame/visibility state is restored through the existing B4 `finally` boundary.

Blender RNA identity uses `as_pointer()` when available rather than Python wrapper identity.
This avoids false mismatches when Blender creates transient RNA wrapper objects.

## Atomic execution

Connected and mixed production output uses this order:

```text
prepare every source
    -> reserve final JSON
    -> stage/finalize every individual texture
    -> validate grouped request
    -> reserve/render grouped B4 sequence
    -> derive grouped coverage/crop/contour
    -> compose typed connected document
    -> apply grouped overlay
    -> compose outer mixed document when required
    -> serialize JSON
    -> commit JSON, individual textures and grouped textures together
```

Any grouped render, decode, crop, contour, overlay, serialization or commit failure rolls back the
entire transaction.

Individual B4 textures remain emitted as compatibility/debug artifacts. Their source slots remain
in the typed document but become invisible. A later optimization may omit those hidden files only
after a separate output compatibility decision.

## Spine document transformation

The connected rig, bones, constraints, attachments and animations remain in the document.

For every grouped source visual slot:

- setup color becomes `ffffff00`;
- color and attachment timelines for that slot are removed so it cannot become visible again;
- bone, constraint, deform, draw-order and event timelines remain untouched.

One new slot is appended inside the connected subgroup:

```text
bone = root
attachment = grouped camera mesh
```

The attachment uses the grouped render crop, simplified concave contour, exact triangulation and
sequence metadata. Its source object order and group identity are recorded in document extras.

For mixed output, the overlay is added to the connected subgroup before the connected document is
composed with standalone components. Existing connected-versus-standalone draw order is therefore
preserved.

## Intentional flattening boundary

The grouped overlay is root-bound and visually replaces the individual connected B4 slots.
Runtime movement of individual source bones no longer moves the grouped image.

Relative source motion, camera motion and front/back changes must therefore be present in the B4
rendered sequence. The original bone/constraint data is retained for compatibility and other
nonvisual uses, but it is not a substitute for the flattened grouped visual sequence.

This is the deliberate tradeoff for depth-correct per-pixel appearance.

## Coverage and contour

Grouped rendering uses the same B4 policies as individual projection:

- max-unioned 8-bit alpha coverage;
- hysteresis and conservative morphology;
- stable sequence crop;
- simplified concave contour;
- safe convex fallback for disconnected alpha islands;
- exact deterministic triangulation.

One grouped render therefore has the same layout diagnostics and guarantees as one ordinary B4
source.

## Validation

Pure and application-level tests cover:

- compatible grouped-plan construction;
- frame/scene/output incompatibility rejection;
- AUTO and REQUIRED behavior;
- individual compatibility mode;
- root-bound grouped mesh construction;
- transparent source slots;
- hidden source-slot timeline removal;
- preservation of bone/deform/event animation data;
- sequence path and metadata;
- stable Blender RNA identity;
- grouped visibility isolation.

A dedicated manual Blender fixture creates two overlapping colored camera-dependent meshes and
validates:

- AUTO grouped selection;
- one cropped grouped PNG containing both source colors;
- one root-bound grouped mesh slot;
- transparent individual visual slots;
- attachment dimensions matching the decoded PNG;
- context/render/visibility restoration;
- absence of temporary datablock leaks.

The fixture is included in the manual-only `Blender 4.4 Camera Projection` workflow. It has not
been executed automatically on the current branch.

## Remaining limits

- mixed local and B4 connected sets remain individual layers;
- the grouped visual is flattened and cannot respond independently to source bones at runtime;
- individual hidden texture files are still emitted;
- real validation currently targets Cycles;
- Eevee and custom Compositor receive their own matrix in the next slice.
