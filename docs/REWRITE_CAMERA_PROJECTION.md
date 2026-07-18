# B4 camera-render projection

## Problem

Blender 4.4 object baking has no camera-ray bake type. A UV `COMBINED` pass cannot
preserve appearance whose value depends on the active camera or ray type:

- Fresnel and Layer Weight;
- Light Path;
- Glass and Refraction;
- Principled Transmission;
- camera-preserving reflections;
- Volume;
- material displacement whose final appearance must be evaluated by render.

The rewrite does not invent an `ACTIVE_CAMERA` bake mode and does not silently flatten
these materials through an incorrect UV bake. B4 uses a real still render from the active
Blender camera and projects that screen-space result into Spine.

## Automatic routing

`domain/baking/camera_projection.py` exposes:

```text
requires_camera_projection()
build_camera_projection_plan()
build_texture_plan()
CameraProjectionPlan
```

`build_texture_plan()` is the production planner used by `prepare_a1_object()`:

```text
reachable material graph
        |
        +-- LOCAL / SCENE / AUXILIARY --> BakePlan --> object UV bake
        |
        +-- CAMERA / VOLUME / render displacement
                                      --> CameraProjectionPlan
                                      --> active-camera render
```

The complete object is routed to camera projection when any used material slot requires
camera evaluation. Mixing UV-baked and screen-rendered pixels inside one object would mix two
coordinate spaces and is intentionally not attempted.

A camera projection plan requires:

- immutable `ObjectBakeContext`;
- immutable `SceneBakeContext`;
- an active camera snapshot;
- PNG, WEBP, or OPEN_EXR output so transparency is representable.

JPEG is rejected during planning.

## Recursive Shader Node Groups

`blender_adapter/shader_graph_analyzer.py` recursively enters every **reachable** Shader Node
Group. It does not scan all nodes inside every group. Traversal follows the actual socket path:

```text
outer Group output
    -> active internal Group Output input
    -> reachable internal links
    -> internal Group Input output
    -> matching outer Group input
```

This distinction prevents an unused Fresnel or Layer Weight connected to an unused group input
from incorrectly selecting B4.

The recursive analysis provides:

- instance-qualified IDs such as `Outer Instance::Inner Instance::Layer Weight`;
- explicit `ShaderNodeSnapshot.group_path` values;
- nested Camera, View, Reflection, Transmission, Image, Volume and Time discovery;
- nested node-tree animation detection;
- deterministic socket matching by identifier, then name, then interface position;
- recursive cycle detection;
- a maximum group depth of 64;
- analysis issues instead of infinite recursion or silent dependency invention.

Group datablock identity and group-node instance identity remain separate. Two instances of the
same node group therefore receive different reachable node IDs.

## Compatibility contract

`CameraProjectionPlan` is a frozen `BakePlan` subtype. Existing consumers continue to use:

- `source_object_id`;
- `settings`;
- `frame_tasks`;
- `representative_task`;
- attachment path and sequence helpers;
- atomic output reservations;
- `BakeExecutionResult`.

Its synthetic `CAMERA_COMBINED` pass is metadata only. It is never passed to
`bpy.ops.object.bake`. Runtime dispatch uses `bpy.ops.render.render(write_still=True)` through
the public failure-injection hook.

## Render execution layers

The B4 executor is split by responsibility:

- `camera_projection_state.py`: validation and reversible Scene/frame/visibility state;
- `camera_projection_image.py`: staged-image decoding, alpha-mask extraction and crop rewrite;
- `camera_projection_executor_core.py`: frame orchestration and atomic reservations;
- `camera_projection_executor.py`: stable public facade;
- `texture_executor.py`: object-bake/B4 dispatch without operator access.

The only real render operator access is:

```text
blender_adapter/bake_executor.py::_call_render_operator
```

For every static or sequence frame the executor:

1. validates source, Scene, World, camera and light identities against the immutable plan;
2. captures render settings, timeline frame, `hide_render`, and `visible_camera` values;
3. makes the source directly visible to the active camera;
4. disables only direct camera visibility for other renderable objects;
5. keeps their diffuse, glossy, transmission and shadow participation intact;
6. enables transparent film while retaining World lighting and reflection contribution;
7. renders the full frame to an atomic staged path;
8. validates and decodes the staged image Blender actually wrote;
9. extracts a binary alpha mask using
   `BakeExecutionSettings.projection_alpha_threshold`;
10. after every frame succeeds, derives one sequence-union layout;
11. rewrites every staged frame with the same crop dimensions;
12. restores every captured Blender value in `finally`.

The compatibility default is exactly `1 / 255`, so existing exports preserve their previous
crop and hull. One immutable threshold is used for every frame and is retained in
`CameraProjectionLayout.alpha_threshold`.

Blender 4.4 background mode exposes `Render Result` as a zero-sized image after a completed
render. The implementation therefore derives alpha from the staged image bytes, not from the
unreliable headless `Render Result` datablock.

## Source-only camera layer

Other meshes must remain available to reflection, refraction, occlusion and shadows, but they
must not be drawn directly into every exported object's texture. B4 changes only
`visible_camera` for other renderable objects.

It deliberately does **not** set `hide_render=True` on dependencies. This distinction is
required for Glass and reflective materials. All previous `hide_render` and `visible_camera`
values are restored on success and failure.

## Stable sequence-union crop

`domain/baking/projection_layout.py` owns the Blender-independent crop model:

```text
ProjectionPixelPoint
ProjectionCropBounds
CameraProjectionLayout
build_sequence_union_layout()
convex_hull()
```

All frame masks are unioned before geometry is created:

```text
frame 1 alpha mask --+
frame 2 alpha mask --+--> union alpha mask --> crop + convex hull
...                  |
frame N alpha mask --+
```

The crop is the union alpha bounding box expanded by `BakeSettings.margin_pixels`. For A1 this
is the existing export `bake_margin`, so no new per-material or B4-only UI switch was added.
The bounds are clamped to the original render dimensions.

The alpha cutoff is a global execution/output policy:

```python
BakeExecutionSettings(
    projection_alpha_threshold=1.0 / 255.0,
)
```

Finite values in `[0, 1]` are accepted. Booleans, NaN, infinities, non-numeric values, and
out-of-range values are rejected before rendering. Automation may also provide the optional
Scene attribute or Blender custom property `spine2d_projection_alpha_threshold`; missing
properties retain the compatibility default.

Every sequence frame receives exactly the same:

- cropped width and height;
- full-frame screen offset;
- UV mapping;
- convex hull;
- triangulation;
- attachment dimensions.

A frame cannot change attachment geometry or move the crop independently. Later frames are
therefore not clipped by a crop derived from the representative frame alone.

An all-transparent static render or sequence is rejected with a structured execution error.
A translucent-only render can also become empty when the selected threshold is higher than all
of its decoded alpha values.

## Screen-space convex hull

The alpha union contributes pixel-boundary points. A deterministic monotonic-chain algorithm
builds one counter-clockwise convex hull and removes collinear points.

The hull is triangulated as a fan:

```text
vertex count = H
triangle count = H - 2
triangle index count = (H - 2) * 3
```

The hull follows the visible alpha union, while UVs are normalized inside the padded crop. This
keeps transparent padding around the texture without expanding Spine geometry to a full crop
rectangle.

Blender image pixels use a bottom-left origin. Spine UV conversion is:

```text
u = (x - crop_min_x) / crop_width
v = 1 - (y - crop_min_y) / crop_height
```

Spine positions preserve full-frame camera placement:

```text
spine_x = x - full_width  / 2
spine_y = y - full_height / 2
```

Cropping therefore reduces texture dimensions without recentering the rendered object.

## Post-render document finalization

A camera layout does not exist during initial object preparation. Production output therefore
uses this transaction order:

```text
prepare immutable object/rig/initial document
        -> reserve JSON
        -> render every texture frame
        -> derive sequence-union crop and hull
        -> rewrite staged textures
        -> rebuild camera projection attachment
        -> compose typed Spine documents
        -> serialize staged JSON
        -> commit JSON and every texture together
```

`blender_adapter/a1_projection_finalization.py` rebuilds only the B4 attachment. Source geometry,
materials, rig planning and texture planning are not repeated.

The same post-render finalization is used by:

- single-object export;
- standalone multi-object export;
- connected multi-object export;
- mixed connected/standalone export.

Multi-object flows re-run the existing typed Spine composition from finalized in-memory
documents. Serialized JSON is never patched or merged.

The historical reservations-only `stage_bake_plan_outputs()` keeps full-frame B4 output for
external callers that serialize JSON before staging. Production output services use the
detailed `stage_texture_plan_outputs()` API and receive the exact `CameraProjectionLayout`.
This prevents cropped images from being paired with a stale full-frame attachment.

## Atomicity and rollback

JSON is reserved first to preserve public output ordering, but its bytes are written only after
all projection frames and the final layout succeed.

Failures during any of these stages roll back the entire transaction:

- render;
- staged-image decode;
- alpha-union construction;
- crop rewrite;
- hull generation;
- projection attachment rebuild;
- multi/mixed typed composition;
- JSON serialization;
- final commit.

Existing JSON and textures are restored byte-for-byte. Staged and backup files are removed.
Temporary Blender images and all Scene/context/visibility state are restored in `finally`.

## Blender 4.4 validation

The dedicated `Blender 4.4 Camera Projection` workflow runs real Cycles renders and decoded
image checks for:

- production Layer Weight/Fresnel selection and render;
- transparent background and visible source coverage;
- static crop smaller than the full render;
- attachment dimensions matching the decoded cropped PNG;
- convex hull UV/triangle invariants;
- Glass projection;
- Principled Volume projection;
- camera-dependent sequence frames with one stable union crop;
- timeline restoration;
- forced render failure and atomic JSON/PNG rollback;
- recursive nested Layer Weight groups;
- recursive nested Volume groups;
- unused group input precision;
- standalone multi-object cropped composition;
- connected multi-object cropped composition;
- mixed connected/standalone cropped composition;
- absence of temporary Blender datablocks.

Pure tests cover planner routing, recursive group traversal, group cycles, union masks, padding,
convex hull construction, UV conversion, all-transparent rejection, configurable alpha-policy
validation, single/multi bridge propagation, and architecture boundaries.

## Current boundaries

### Convex rather than concave geometry

The screen-space attachment is a convex hull. Deep concavities and internal transparent holes
remain inside the mesh and are represented by texture alpha. This is deliberate: a simple
convex polygon is deterministic and safe for Spine triangulation.

### Alpha threshold versus antialias coverage

The cutoff is now configurable and deterministic. Coverage-weighted antialias reconstruction,
contour simplification based on fractional coverage, and morphology-based fringe cleanup remain
separate output-policy work.

### Connected multi-object depth

Each B4 texture remains a source-only camera layer. Connected and mixed documents now carry
correct cropped textures and attachments, but one fixed Spine slot order still cannot reproduce
arbitrary per-pixel depth intersections between several layers. Grouped rendering or depth-aware
composition is required before arbitrary connected B4 visual parity is claimed.

### Camera movement versus Spine rig movement

B4 captures the rendered camera result. Camera motion is baked into texture frames, not
converted into Spine bone motion. This is intentional for appearance parity.

### Render engine scope

Real B4 validation targets Blender 4.4 Cycles. Eevee-specific parity and compositor-dependent
render pipelines need separate fixtures before being claimed.

### Color and HDR

PNG/WEBP use Blender's configured color-management transform. OPENEXR preserves higher
precision. Configurable tone mapping, premultiplied-alpha variants and HDR Spine runtime
behavior remain output-policy work.

## Next safe increments

1. optional concave/contour simplification policy if real fixtures justify it;
2. grouped multi-object camera rendering;
3. depth-aware connected composition;
4. Eevee and custom Compositor fixtures;
5. representative private `.blend` parity against accepted v0.23 JSON and images.

No increment should add per-material UI mode switches. Immutable analysis facts and registered
pipeline selection remain authoritative.
