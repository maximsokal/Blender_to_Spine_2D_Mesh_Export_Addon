# Normal / UV Segments Projection Space and Draw Order

Status: **Approved for implementation**  
Approved by the user: **2026-08-01**  
Implementation order: **documentation first, then sliced implementation**

## 1. Scope

This document defines the approved architecture for configurable 3D-to-2D projection,
per-object placement, projected depth, and deterministic setup draw order in
`Normal — UV Segments` export.

The architecture must cover:

- single-object export;
- standalone multi-object export;
- connected multi-object export;
- mixed internal composition.

Connected and mixed modes remain development/internal paths and must not be restored to
the public UI until the complete plan in this document is implemented and accepted.

The existing `Camera Projection` render/flattening pipeline is explicitly outside the
implementation scope of this task and must remain unchanged.

## 2. Existing behavior and problem statement

The current object-bake pipeline effectively uses one hard-coded global projection:

```text
Spine X = Blender world X
Spine Y = Blender world Y
Depth   = Blender world Z
```

As a result, exported multi-object placement in Spine corresponds to looking at the
Blender scene from the positive Z side toward the origin: a Top-like projection.

Current standalone and connected behavior is not normalized:

```text
Standalone:
    placement = absolute world X/Y
    setup draw order = component composition/input order

Connected:
    placement = anchor-relative world X/Y
    depth layers = Object Origin world Z
    setup draw order = Object Origin Z layers
```

The implementation must replace these implicit assumptions with one explicit projected
coordinate contract shared by all composition modes.

## 3. User-visible projection setting

When export mode is `Normal — UV Segments`, the UI must expose a dropdown named
`Projection Direction` with these values:

```text
+X
-X
+Y
-Y
+Z
-Z
Active Camera
```

The setting must not be shown for the existing `Camera Projection` export mode.

Persisted identifiers must be stable and explicit:

```text
POSITIVE_X
NEGATIVE_X
POSITIVE_Y
NEGATIVE_Y
POSITIVE_Z
NEGATIVE_Z
ACTIVE_CAMERA
```

Existing `.blend` files and scenes without this property must migrate to `POSITIVE_Z`,
which preserves the current projection behavior.

## 4. Meaning of a signed axis

A signed axis identifies the side from which the user looks toward the scene origin.

Example:

```text
+Z = observer is on the positive Z side and looks toward -Z
```

Therefore:

```text
+X = view from positive X toward -X
-X = view from negative X toward +X
+Y = view from positive Y toward -Y
-Y = view from negative Y toward +Y
+Z = view from positive Z toward -Z
-Z = view from negative Z toward +Z
```

The projected depth coordinate is defined so that larger values are closer to the
observer for every signed-axis mode.

## 5. Canonical projected space

The new architecture must define one canonical projected space:

```text
U = horizontal Spine coordinate
V = vertical Spine coordinate
D = projected depth, increasing toward the observer
```

After projection, downstream object-bake code must receive canonical geometry as:

```text
canonical X = U
canonical Y = V
canonical Z = D
```

This permits existing attachment and rig builders to continue consuming a three-axis
snapshot while removing the assumption that canonical Z always means Blender world Z.

New public/domain terminology must use `projection depth`, `canonical depth`, or
`depth group`. Existing internal `Z group` names may remain temporarily only where
renaming would unnecessarily break compatibility.

## 6. Signed-axis basis definitions

The approved world-to-canonical transforms are:

| Projection | U / Spine X | V / Spine Y | D / front depth |
|---|---:|---:|---:|
| `+X` | `+world Y` | `+world Z` | `+world X` |
| `-X` | `-world Y` | `+world Z` | `-world X` |
| `+Y` | `-world X` | `+world Z` | `+world Y` |
| `-Y` | `+world X` | `+world Z` | `-world Y` |
| `+Z` | `+world X` | `+world Y` | `+world Z` |
| `-Z` | `-world X` | `+world Y` | `-world Z` |

`+Z` must preserve the current 0.55.1 visual orientation and remains the migration
default.

The basis must be represented by typed immutable data rather than scattered sign and
axis conditionals.

## 7. Active Camera inside Normal / UV Segments

`Active Camera` in this document is not the existing `Camera Projection` mode.

It must preserve:

- one independent rig per object;
- independent object controls;
- existing UV-segment meshes;
- existing UV-baked textures;
- weighted attachment vertices;
- per-object Object Origin pivots;
- object and segment names.

It must project the complete evaluated geometry into the active camera screen space.
It must not render and flatten the scene into one combined image.

## 8. Active Camera pixel coordinate contract

The existing camera projection layout already uses export texture dimensions as the
full Spine pixel canvas. The new Normal / UV Segments camera layout must use the same
coordinate convention.

The canvas dimensions are:

```text
canvas_width  = export texture width
canvas_height = export texture height
```

Scene render resolution, viewport size, and Blender window size are not the source of
Spine coordinates for this feature.

For normalized projected camera coordinates:

```text
pixel_x = normalized_x * canvas_width
pixel_y = normalized_y * canvas_height

Spine X = pixel_x - canvas_width  / 2
Spine Y = pixel_y - canvas_height / 2
```

Camera projection mathematics must use the aspect ratio of the selected export texture.

Axis modes continue converting Blender units through the existing rig uniform scale.
Active Camera coordinates are already expressed in export pixels and must not be
multiplied by that scale a second time.

## 9. Evaluated source state

Projection and depth analysis must use one consistent evaluated dependency-graph state.

For each source object, the implementation must resolve together:

- evaluated object transform;
- evaluated mesh vertices;
- evaluated modifier result;
- active scene;
- active camera where required;
- current timeline frame;
- export texture dimensions.

The implementation must fail closed if geometry, transform, scene, or camera ownership
changes between projection analysis and final attachment construction.

Source Blender objects and source mesh data must not be mutated.

## 10. Projected Object Origin

For each source object:

```text
world_origin = evaluated_object.matrix_world.translation
projected_origin = project(world_origin)
```

The object main bone remains the Blender Object Origin pivot.

Standalone placement:

```text
<prefix>_main.x = projected_origin.u
<prefix>_main.y = projected_origin.v
```

Connected placement:

```text
<prefix>_main.x = projected_origin.u - projected_anchor_origin.u
<prefix>_main.y = projected_origin.v - projected_anchor_origin.v
```

After the connected parent hierarchy is applied, the final world setup position must
match standalone export for the same object and projection settings.

The shared Spine `root` remains common to the document. The per-object Blender pivot is
represented by `<prefix>_main`, not by moving the shared root.

## 11. Projection of complete geometry

For every evaluated mesh vertex:

```text
world_vertex = evaluated_object_matrix @ evaluated_vertex_position
projected_vertex = project(world_vertex)
projected_origin = project(world_origin)
```

The object-local canonical vertex is:

```text
local_u = projected_vertex.u - projected_origin.u
local_v = projected_vertex.v - projected_origin.v
local_d = projected_vertex.depth - projected_origin.depth
```

The canonical snapshot position becomes:

```text
position = (local_u, local_v, local_d)
```

This preserves the Object Origin as the rig pivot while allowing the whole object shape
to follow the selected signed-axis or camera projection.

For a perspective camera this projection is nonlinear. Subtracting the projected origin
preserves setup placement, but later Spine rotation remains a 2D deformation and is not
a rerendered perspective transformation. This limitation must be documented in the UI
help/release notes.

## 12. Projected depth groups

Each object uses its projected Object Origin depth as zero:

```text
projected origin depth = 0
```

Depth-group offsets may be negative, zero, or positive.

The previous Object Origin behavior from 0.55.1 remains the exact compatibility result
for `+Z`.

No artificial zero depth group may be created when no evaluated geometry exists at the
origin depth.

## 13. Per-object depth analysis

The implementation must analyse all evaluated vertices of every exported object.

A typed immutable result must retain at least:

```text
component_id
prefix
source_input_index
projection_direction
projected_origin_u
projected_origin_v
projected_origin_depth
nearest_vertex_id
nearest_vertex_world_position
nearest_vertex_depth
farthest_vertex_id
farthest_vertex_depth
projected_bounds
owned_slot_names
```

Pivot placement and draw-order depth are separate concepts:

```text
placement = projected Object Origin
draw order = nearest evaluated vertex
```

Object Origin depth must not replace nearest-vertex depth.

## 14. Nearest vertex for Active Camera

For Active Camera, each evaluated world vertex receives a positive camera-space depth
measured along the camera forward direction.

The accepted nearest-vertex rule is:

```text
nearest_depth = min(positive_camera_forward_depth for every evaluated vertex)
```

This is not Euclidean distance to the camera origin.

The same camera-space depth convention must be used for perspective and orthographic
cameras.

## 15. Nearest vertex for signed-axis modes

For signed-axis projections, canonical D grows toward the observer.

The nearest object coordinate is therefore:

```text
nearest_front_coordinate = max(D for every evaluated vertex)
```

The farthest coordinate is:

```text
farthest_front_coordinate = min(D for every evaluated vertex)
```

This rule applies consistently to `+X`, `-X`, `+Y`, `-Y`, `+Z`, and `-Z`.

## 16. Object-block setup draw order

All slots belonging to one object must remain one contiguous block.

Allowed:

```text
FarObject_Segment_0
FarObject_Segment_1
FarObject_Segment_2
NearObject_Segment_0
NearObject_Segment_1
```

Not allowed:

```text
FarObject_Segment_0
NearObject_Segment_0
FarObject_Segment_1
NearObject_Segment_1
```

The internal segment order of one object remains unchanged in this task.

A future task may add per-segment depth sorting or block splitting, but that behavior is
not part of this implementation.

## 17. Draw-order direction

Spine setup slots must be emitted from back to front so later slots draw above earlier
slots.

Active Camera:

```text
larger camera-forward distance first
smaller camera-forward distance last
```

Signed-axis modes:

```text
smaller nearest front coordinate first
larger nearest front coordinate last
```

Therefore, an object whose nearest evaluated vertex is closer to the observer receives
a later slot block and appears above farther objects.

## 18. Depth-range overlap limitation

The first implementation treats each object as indivisible for setup draw order.

If one object spans both in front of and behind another object, nearest-vertex sorting
places the entire object block according to its nearest vertex.

This cannot reproduce a per-pixel 3D z-buffer and is an explicit accepted limitation.

A future task may implement:

- per-segment depth analysis;
- segment block splitting;
- partial slot interleaving;
- more advanced overlap resolution.

## 19. Deterministic tie-breaking

When object depth keys are equal within the configured tolerance:

```text
1. preserve source input order;
2. use component_id only as the final deterministic fallback.
```

The result must not depend on Python hash order, dictionary order assumptions, temporary
Blender object names, or incidental collection traversal.

Depth tolerance must be a typed setting owned by the common projection/draw-order
contract rather than a hidden constant inside connected code.

## 20. Standalone pipeline

Standalone multi-object export must follow this order:

```text
prepare every source object
resolve one projection frame
project every Object Origin
project every evaluated vertex
calculate projected depth ranges
build every per-object Spine document
compose documents
reorder complete object slot blocks back-to-front
serialize
```

Simple component concatenation must no longer define visual depth.

## 21. Connected pipeline

Connected must consume the same projected object analysis as standalone.

Only hierarchy and local coordinate ownership differ:

```text
Standalone = absolute projected origins
Connected  = anchor-relative projected origins
```

Required invariant:

```text
standalone world setup position == connected world setup position
```

Connected layer grouping uses projected Object Origin depth.

Connected setup draw order uses nearest evaluated vertex depth.

Those values must remain separate and must not be derived from each other.

The current behavior that groups and orders only by Object Origin world Z must be
replaced by the common projected contract.

## 22. Mixed and connected public availability

Connected and mixed paths must be covered by:

- typed domain contracts;
- unit tests;
- composition tests;
- Blender headless acceptance.

They remain hidden from the public UI until all slices in this document are complete and
accepted.

The public multi-object UI remains standalone-only during implementation.

## 23. Objects outside the camera frame

Active Camera export permits geometry outside the texture canvas.

Valid projected coordinates include:

```text
Spine X < -canvas_width / 2
Spine X > +canvas_width / 2
Spine Y < -canvas_height / 2
Spine Y > +canvas_height / 2
```

An object may be partially or completely outside the camera frame without causing an
error.

## 24. Active Camera fail-closed conditions

Active Camera export must fail with actionable diagnostics when:

- `scene.camera` is missing;
- the active camera object is invalid;
- the camera type is unsupported;
- camera, object, and dependency graph do not belong to the same scene evaluation;
- camera matrices contain non-finite values;
- export texture dimensions are invalid;
- an evaluated vertex is on or behind the camera plane;
- evaluated geometry crosses the camera near plane;
- the projected Object Origin is on or behind the camera plane;
- geometry changes between depth analysis and attachment construction;
- projected data cannot preserve the Object Origin contract.

The implementation must never silently fall back to `+Z`.

## 25. Existing Camera Projection remains unchanged

This task must not modify:

- camera rendering;
- camera-render texture planning;
- alpha-union accumulation;
- crop calculation;
- camera contour generation;
- grouped-camera flattening;
- grouped-camera texture output;
- existing Camera Projection setup mesh behavior.

`Active Camera` is a Normal / UV Segments projection layout only.

## 26. UI behavior and migration

The projection dropdown is visible only when `Normal — UV Segments` is selected.

Recommended UI help for Active Camera:

```text
Projects independent UV-segment meshes into the active camera frame.
Does not flatten the scene and does not replace Camera Projection mode.
```

Existing persisted scenes migrate to `+Z`.

Unknown or malformed persisted values must fail closed or normalize explicitly at the
UI capture/migration boundary; they must not produce implicit axis behavior deep inside
the domain pipeline.

## 27. Unchanged data and behavior

The implementation must preserve unless a later approved document explicitly changes
them:

- generated UV coordinates;
- unwrap policy;
- texture packing;
- triangles;
- hull;
- edges;
- attachment paths;
- attachment names;
- control names;
- internal segment order;
- animation namespaces;
- preview animation behavior;
- source Blender objects;
- Spine version codecs;
- Spine 3.8 constraint scheduling;
- the accepted Spine 3.8 Rotation X fix;
- the accepted Spine 3.8 Scale fix;
- Camera Projection behavior.

## 28. Diagnostics

Headless and development diagnostics must report for every object:

```text
componentId
objectName
projectionDirection
worldOrigin
projectedOrigin
nearestVertexId
nearestVertexWorldPosition
nearestVertexDepth
farthestVertexId
farthestVertexDepth
projectedBounds
slotBlockIndex
slotNames
```

Active Camera diagnostics must additionally include:

```text
cameraObject
cameraType
canvasWidth
canvasHeight
cameraNearClip
cameraFarClip
```

Diagnostics must make it possible to explain both placement and slot ordering without
inspecting serialized weighted vertex streams manually.

## 29. Test matrix

### 29.1 Pure signed-axis tests

For all six directions verify:

- world point to U/V/D mapping;
- Object Origin mapping;
- sign orientation;
- near/far ordering;
- finite values;
- deterministic repeated results.

### 29.2 Single-object tests

Cover:

- custom Object Origin;
- rotated object;
- non-uniform scale;
- mirrored transform;
- evaluated modifiers;
- geometry on both sides of projected origin depth;
- geometry only in front of origin depth;
- geometry only behind origin depth;
- no source mutation.

### 29.3 Standalone multi-object tests

Cover objects separated only along:

- X;
- Y;
- Z.

Also cover:

- objects overlapping in the projection plane;
- nearest vertex differing from nearest Object Origin;
- contiguous object slot blocks;
- deterministic far-to-near order;
- equal-depth tie-breaking;
- all six signed-axis directions.

### 29.4 Active Camera tests

Cover:

- perspective camera;
- orthographic camera;
- shifted camera;
- export texture aspect ratio;
- different export width/height values;
- object partly outside the canvas;
- object completely outside the canvas;
- object behind camera plane;
- near-plane crossing;
- nearest positive camera-depth sorting;
- no double application of uniform scale.

### 29.5 Connected tests

Verify:

- standalone and connected world setup positions match;
- changing anchor changes local hierarchy but not final projected placement;
- object-block draw order matches standalone;
- projected Object Origin depth drives layers;
- nearest evaluated vertex drives slots;
- global connected controls remain valid;
- connected/mixed remain absent from the public UI.

### 29.6 Blender headless acceptance scene

Create a real Blender scene containing at least three differently sized meshes with:

- distinct X/Y/Z positions;
- custom Object Origins;
- overlapping projected bounds;
- an active perspective camera;
- at least one object whose nearest vertex order differs from Origin order.

The worker must generate a JSON diagnostics report containing all fields listed in
section 28.

## 30. Implementation slices

Implementation must proceed in order. Each slice must be reviewed and tested before the
next slice changes behavior.

### Slice 0 — Documentation only

- add this approved document;
- use a dedicated commit;
- do not change production code;
- do not change the manifest version.

### Slice 1 — Projection domain

- add `A1ProjectionDirection`;
- add immutable signed-axis basis/frame types;
- add point and vector projection;
- add depth range and nearest-vertex contracts;
- add pure unit tests;
- do not expose UI;
- do not alter exported output yet.

### Slice 2 — Single-object signed-axis projection

- implement `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`;
- project Object Origin;
- build canonical projected snapshots;
- preserve signed depth groups around the pivot;
- add Blender headless single-object acceptance.

### Slice 3 — Standalone multi-object projection and draw order

- resolve projected placement for every object;
- analyse nearest/farthest evaluated vertex depth;
- retain contiguous slot blocks;
- reorder blocks deterministically back-to-front;
- add standalone headless acceptance.

### Slice 4 — Active Camera for Normal / UV Segments

- resolve evaluated active camera;
- support perspective and orthographic projection;
- use export texture dimensions as the pixel canvas;
- project complete evaluated geometry;
- compute minimum positive camera-forward depth;
- allow geometry outside the frame;
- add camera headless acceptance.

### Slice 5 — Connected and mixed normalization

- use projected anchor-relative origins;
- use projected Origin depth for connected layers;
- use nearest-vertex depth for object slot blocks;
- share the standalone draw-order planner;
- add standalone/connected equivalence tests;
- keep connected/mixed hidden from public UI.

### Slice 6 — UI and migration

- add the Projection Direction dropdown;
- show it only for Normal / UV Segments;
- migrate old scenes to `+Z`;
- keep Camera Projection UI unchanged;
- add UI capture and migration regression tests.

### Slice 7 — Release acceptance

- run the complete Python test suite;
- run Blender headless projection matrix tests;
- run supported Spine target acceptance;
- perform manual Spine verification;
- update release documentation;
- build and validate the extension archive.

## 31. Versioning rule

The current version at document approval is `0.55.1`.

Documentation-only Slice 0 and an internal pure-contract commit that does not alter
user-visible export behavior do not increase the manifest version because no user
archive is released from them.

Every completed user-visible behavior change or correction delivered for testing must
increase the patch version.

Expected progression begins with:

```text
first user-visible implementation/fix  -> 0.55.2
next delivered correction              -> 0.55.3
next delivered correction              -> 0.55.4
```

The version must be synchronized across manifest, release contracts, release notes, and
archive naming whenever it is increased.

## 32. Acceptance criteria

The task is complete only when all of the following are true:

1. The user can select `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`, or `Active Camera` in
   Normal / UV Segments.
2. `+Z` preserves the accepted 0.55.1 layout.
3. Whole evaluated geometry, not only origins, follows the selected projection.
4. `<prefix>_main` remains the projected Blender Object Origin.
5. Standalone and connected produce the same world setup placement.
6. Every object owns one contiguous slot block.
7. Object slot blocks are ordered back-to-front from nearest evaluated vertex depth.
8. Active Camera uses minimum positive camera-forward vertex depth.
9. Camera layout coordinates use export texture dimensions.
10. Objects outside the camera canvas remain exportable.
11. Connected and mixed remain hidden from the public UI until full plan acceptance.
12. Existing Camera Projection behavior remains unchanged.
13. Source Blender geometry and transforms remain unchanged.
14. Supported Spine target and runtime regression suites pass.
15. Manual Spine verification confirms placement and draw order on the real test scene.
