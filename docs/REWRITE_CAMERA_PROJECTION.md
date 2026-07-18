# B4 camera-render projection

## Problem

Blender object baking has no camera-ray bake type. A UV `COMBINED` bake cannot preserve
appearance whose value depends on the active camera or ray type, including:

- Fresnel and Layer Weight;
- Light Path;
- Glass, Refraction and Principled Transmission;
- camera-preserving reflections;
- Volume;
- render-evaluated displacement.

The rewrite therefore does not invent an unsupported bake mode. B4 performs a real render
from the active Blender camera and projects the decoded screen-space result into Spine.

## Automatic routing

`domain/baking/camera_projection.py` owns the immutable camera plan:

```text
reachable material graph
        |
        +-- LOCAL / SCENE / AUXILIARY --> BakePlan --> object UV bake
        |
        +-- CAMERA / VOLUME / render displacement
                                      --> CameraProjectionPlan
                                      --> active-camera render
```

A camera plan requires immutable object and scene context, an active camera and an output
format that can represent alpha. JPEG is rejected during planning.

## Recursive Shader Node Groups

`blender_adapter/shader_graph_analyzer.py` recursively follows only reachable group sockets:

```text
outer Group output
    -> active internal Group Output input
    -> reachable internal links
    -> internal Group Input output
    -> matching outer Group input
```

The analyzer preserves group-instance identity, supports nested Image, Time, Camera, View,
Reflection, Transmission and Volume dependencies, respects renderer-specific Material Outputs
and muted-node bypasses, detects recursive cycles and applies a bounded traversal depth.
Unused group inputs do not leak camera dependencies.

## Compatibility contract

`CameraProjectionPlan` is a frozen `BakePlan` subtype. Existing consumers continue to use:

- `source_object_id`;
- `settings`;
- `frame_tasks`;
- `representative_task`;
- attachment path and sequence helpers;
- atomic output reservations;
- `BakeExecutionResult`.

Its synthetic `CAMERA_COMBINED` pass is metadata only and is never sent to
`bpy.ops.object.bake`. The real render operator is called only through the public
failure-injection hook in `blender_adapter/bake_executor.py`.

## Render execution layers

Responsibilities are separated as follows:

- `camera_projection_state.py`: runtime validation and reversible Scene/frame/visibility state;
- `camera_projection_image.py`: staged-image decode, alpha extraction and crop rewrite;
- `camera_projection_executor_core.py`: frame orchestration and atomic reservations;
- `domain/baking/projection_contour.py`: pure boundary extraction, simplification and
  triangulation;
- `domain/baking/projection_layout.py`: sequence union, crop and immutable layout;
- `camera_projection_executor.py`: stable public facade;
- `texture_executor.py`: typed object-bake/B4 dispatch without direct operator access.

For every static or sequence frame the executor:

1. validates source, Scene, World, camera and light identities against the immutable plan;
2. captures render settings, timeline frame, `hide_render` and `visible_camera` values;
3. makes the source directly camera-visible;
4. disables only direct camera visibility for other renderable objects;
5. keeps their diffuse, glossy, transmission and shadow participation intact;
6. enables transparent film while retaining World lighting and reflection contribution;
7. renders the full frame to an atomic staged path;
8. validates and decodes the staged image Blender actually wrote;
9. extracts an alpha mask using the configured output policy;
10. incrementally merges it into one fixed-size sequence-union buffer;
11. after every frame succeeds, derives one shared crop and contour;
12. rewrites every staged frame with the same crop dimensions;
13. restores every captured Blender value in `finally`.

Blender background rendering can expose an unusable zero-sized `Render Result` datablock after
a successful render. B4 therefore treats the staged image file as the source of truth.

## Source-only camera layer

Other objects may be required for reflection, refraction, occlusion and shadows, but they must
not be drawn directly into each exported object's texture. B4 changes only direct camera
visibility for other renderable objects and deliberately does not hide them from all render
rays. Previous visibility values are restored on success and failure.

## Stable sequence-union crop

All frame masks are unioned before geometry is created:

```text
frame 1 alpha mask --+
frame 2 alpha mask --+--> fixed union mask --> crop + contour + triangulation
...                  |
frame N alpha mask --+
```

The crop is the union-alpha bounding box expanded by `BakeSettings.margin_pixels`. In A1 this
is the existing `bake_margin`. Bounds are clamped to the original render dimensions.

Every frame receives exactly the same:

- cropped width and height;
- full-frame screen offset;
- UV mapping;
- outer contour;
- triangle topology;
- attachment dimensions.

Later frames therefore cannot be clipped by geometry derived only from the representative
frame.

## Alpha threshold

The immutable output setting is:

```python
BakeExecutionSettings(
    projection_alpha_threshold=1.0 / 255.0,
)
```

The compatibility default remains exactly `1 / 255`. Finite numeric values in `[0, 1]` are
accepted. Booleans, non-numeric values, NaN, infinities and out-of-range values are rejected
before rendering.

One value is shared by every sequence frame and is recorded in
`CameraProjectionLayout.alpha_threshold`. Automation may also provide the optional Scene/RNA
or Blender custom property `spine2d_projection_alpha_threshold`.

An all-transparent result after applying the threshold fails before atomic commit. A
translucent-only render may also become empty when every decoded alpha value is below the
selected threshold.

## Simplified concave screen-space contour

The production output policy is:

```python
BakeExecutionSettings(
    projection_contour_mode=ProjectionContourMode.SIMPLIFIED_CONCAVE,
    projection_contour_simplify_tolerance_pixels=1.0,
)
```

`ProjectionContourMode.CONVEX_HULL` remains an explicit compatibility mode.

### Boundary extraction

Every visible union pixel contributes oriented unit boundary edges only where its neighboring
pixel is transparent or outside the frame. The tracer follows deterministic turn priority:

1. left;
2. straight;
3. right;
4. reverse.

This keeps diagonal contacts as separate components instead of joining them through one shared
corner.

### Components and holes

One connected outer component becomes a simple concave contour. Internal hole loops are not
converted into polygon holes; they remain represented by transparent texture pixels.

Several disconnected outer components cannot be represented by one simple polygon without an
artificial bridge. B4 therefore uses a deterministic convex fallback that contains every
visible pixel. The layout records the actual contour mode, component count and fallback reason.

### Conservative simplification

Exact collinear points are removed first. The simplifier may then remove only shallow reflex
vertices whose distance from the replacement chord is within the configured tolerance.
Convex corners are never removed, so visible alpha coverage is not clipped.

A replacement chord is accepted only when it does not intersect a non-adjacent contour edge and
the final contour remains simple, counter-clockwise and triangulatable. Simplification may add a
small transparent overdraw area but never cuts into the visible mask.

### Triangulation

Convex contours retain the historical deterministic triangle fan. Concave contours use
deterministic ear clipping.

Every triangulation must satisfy:

```text
triangle count = contour vertex count - 2
all triangle signed areas > 0
sum(triangle signed areas) = contour signed area
```

Duplicate points, consecutive collinear points, clockwise boundaries, self-intersections and
degenerate ears are rejected.

`CameraProjectionLayout.hull` remains the compatibility field name and now contains the
selected simple outer contour. New code may use `layout.contour`. The application layer consumes
`layout.triangle_indices` and does not assume fan topology.

## UV and screen placement

Blender image pixels use a bottom-left origin. Spine UV conversion is unchanged:

```text
u = (x - crop_min_x) / crop_width
v = 1 - (y - crop_min_y) / crop_height
```

Spine positions preserve full-frame camera placement:

```text
spine_x = x - full_width  / 2
spine_y = y - full_height / 2
```

Cropping reduces texture dimensions without recentering the rendered object.

## Post-render document finalization

The final layout exists only after all renders succeed:

```text
prepare immutable object/rig/initial document
        -> reserve JSON
        -> render every texture frame
        -> derive sequence-union crop and contour
        -> triangulate contour
        -> rewrite staged textures
        -> rebuild camera projection attachment
        -> compose typed Spine documents
        -> serialize staged JSON
        -> commit JSON and every texture together
```

`blender_adapter/a1_projection_finalization.py` rebuilds only the B4 attachment. Source geometry,
material analysis, rig planning and texture planning are not repeated.

The same post-render finalization is used by single, standalone multi, connected multi and mixed
exports. Multi-object flows recompose typed in-memory `SpineDocument` values; serialized JSON is
never patched or merged.

The historical reservations-only staging API keeps full-frame output for external callers that
serialize JSON before staging. Production output services use the detailed staging API and
receive the exact `CameraProjectionLayout`.

## Atomicity and rollback

JSON is reserved first but written only after all frame renders, contour construction and
attachment finalization succeed.

Failures in any of these stages roll back the complete transaction:

- render;
- staged-image decode;
- alpha-union construction;
- crop calculation;
- contour extraction or simplification;
- triangulation;
- crop rewrite;
- attachment rebuild;
- typed multi/mixed composition;
- JSON serialization;
- final commit.

Existing JSON and texture files are restored byte-for-byte. Staged and backup files are removed.
Temporary Blender images and Scene/context/visibility state are restored in `finally`.

## Validation

The last complete automatic Blender 4.4 Cycles matrix before CI was switched to manual-only
covered Fresnel/Layer Weight, Glass, Volume, recursive groups, sequence union, crop/attachment
dimension parity, timeline restoration, rollback and single/multi/mixed composition.

The current concave-contour slice adds pure and application-level coverage for:

- exact L-shaped concavity;
- shallow-notch simplification;
- preservation of deeper concavity;
- holes retained as texture alpha;
- diagonal contacts and disconnected-component fallback;
- explicit convex compatibility mode;
- exact-area ear clipping;
- arbitrary triangulation edge topology;
- complete concave `MeshSnapshot` construction;
- 250 deterministic randomized binary masks.

The full pytest and real Blender headless matrices have not yet been rerun for the current HEAD.
Automatic Actions remain manual-only on this branch.

## Current boundaries

- disconnected alpha components still use one convex fallback mesh;
- fractional-coverage antialias reconstruction and morphology cleanup are the next output-policy
  slice;
- one fixed Spine slot order cannot reproduce arbitrary per-pixel intersections among separately
  rendered connected objects;
- real B4 parity currently targets Cycles;
- HDR, tone mapping and premultiplied-alpha variants remain separate output policy;
- representative private `.blend` parity remains required before release.

See also:

- `docs/REWRITE_B4_CONCAVE_CONTOUR.md`;
- `docs/REWRITE_B4_ALPHA_THRESHOLD.md`;
- `docs/REWRITE_STATUS.md`;
- `docs/REWRITE_CI_MANUAL_MODE.md`.
