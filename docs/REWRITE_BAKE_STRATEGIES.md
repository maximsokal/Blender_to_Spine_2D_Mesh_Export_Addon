# Semantic texture strategy architecture

## Purpose

The rewrite does not expose one UI switch for every Blender material combination. It inspects
the reachable connected shader graph, recursively resolves used Shader Node Groups, identifies
semantic outputs and external context, and selects one of two texture pipelines:

```text
Blender Material / Object / Scene
        |
        v
reachable recursive shader graph analysis
        |
        v
semantic channels + dependency kinds
        |
        v
ObjectBakeContext + SceneBakeContext
        |
        v
build_texture_plan()
        |
        +-- BakePlan ----------------------> object UV bake passes
        |                                     + straight-RGBA compositor
        |
        +-- CameraProjectionPlan ----------> active-camera transparent renders
                                              + sequence alpha union
                                              + stable crop and convex Spine hull
```

No strategy is selected by user-facing material checkboxes. The UI may report the chosen
pipeline and reasons, but immutable analysis and planning own the decision.

## Reachable recursive shader graph

`blender_adapter/shader_graph_analyzer.py` starts at the active
`ShaderNodeOutputMaterial` and follows only links contributing to Surface, Volume, or
Displacement. Unused editor nodes do not change the plan.

For a reachable Shader Node Group, traversal crosses the actual interface:

```text
outer group output
    -> active internal Group Output input
    -> reachable internal nodes
    -> internal Group Input output
    -> matching outer group input
```

The analyzer does not mark all nodes inside a used group as reachable. This prevents an unused
Fresnel or Layer Weight connected to an unused group input from leaking Camera/View facts.

`MaterialGraphSnapshot` records:

- reachable nodes and links;
- active output node;
- semantic channels;
- external dependencies;
- analysis issues.

Nested snapshots additionally record:

- instance-qualified node IDs;
- explicit `group_path` tuples;
- nested node-tree animation;
- recursive-cycle diagnostics.

Socket matching uses interface identifier first, name second, and interface position as a
compatibility fallback. Group expansion is limited to 64 levels and never mutates node trees.

Semantic channels:

- `SURFACE_COLOR`;
- `SURFACE_EMISSION`;
- `ALPHA`;
- `VOLUME`;
- `DISPLACEMENT`.

Dependency categories include:

- Image and Time;
- Object and Geometry;
- World and Lighting;
- Occlusion and scene objects;
- Camera and View;
- Reflection and Transmission;
- Node Group.

Stable names and paths are used instead of Python identity of Blender RNA wrappers. A `TIME`
dependency from keyframes, drivers, animated nested node trees, image sequences or movies marks
the material/object animated.

## Evaluation scopes

Every used material slot receives one primary scope:

- `LOCAL`: lighting-independent surface color and material Emission;
- `SCENE`: World, lighting, occlusion and other scene-object appearance;
- `CAMERA`: active-camera/ray appearance routed to B4;
- `AUXILIARY`: alpha and composition-only channels.

The presence of a lamp does not automatically make ordinary Principled Base Color scene-aware.
Selection depends on the reachable graph requirement.

## Object-bake strategy registry

`domain/baking/strategies.py` remains Blender-independent. Every executable object-bake
strategy declares:

- stable identifier;
- deterministic priority;
- evaluation scope;
- support predicate;
- semantic channels;
- verified Blender bake mode;
- optional copied-material preparation.

Executable strategies:

1. `SceneCombinedBakeStrategy`;
2. `SurfaceColorBakeStrategy`;
3. `EmissionBakeStrategy`;
4. `AlphaBakeStrategy`.

A camera boundary detector remains for callers explicitly requesting an object-bake plan.
Production uses `build_texture_plan()` and creates `CameraProjectionPlan` instead of inventing
a dead object-bake mode.

## Typed object-bake pass plan

`BakePlan.bake_mode` is a compatibility alias for the first pass. The source of truth is:

```text
BakePlan.passes: tuple[BakePassPlan, ...]
BakePlan.composite: BakeCompositePlan
```

Each pass records strategy ID, verified bake mode, slot coverage, semantic channels, scope and
per-slot copied-material preparation.

Verified Blender 4.4 object-bake modes used by the rewrite:

```text
DIFFUSE
COMBINED
EMIT
```

There is no `ACTIVE_CAMERA` object-bake type.

## B1: surface and Emission

Ordinary opaque Principled, Image Texture and procedural Base Color use lighting-independent
`DIFFUSE Color`. Pure material Emission uses `EMIT`.

```text
final.rgb   = clamp(surface.rgb + emission.rgb)
final.alpha = max(surface.alpha, emission.alpha)
```

Single-pass output bypasses the compositor. If a surface contribution requested as `COMBINED`
is composed with separate Emission, the registry normalizes it to `DIFFUSE` to avoid counting
Emission twice.

## B2: alpha and straight RGB

A native `DIFFUSE Color` bake of an alpha-bearing shader may return attenuated RGB. B2
evaluates straight color and opacity independently:

```text
straight surface color -> temporary Emission proxy -> EMIT
material opacity       -> grayscale Emission proxy -> EMIT
material emission      -> native EMIT when present
```

```text
final.rgb   = clamp(sum(color contribution passes))
final.alpha = alpha_pass.red
```

`blender_adapter/bake_material_preparation.py` mutates only copied materials. It stores output
links, creates temporary Math/MixRGB/Emission nodes, runs one pass, removes temporary nodes and
restores copied graphs in `finally`.

Opacity extraction supports Principled Alpha, linked/procedural scalar graphs, Transparent,
Holdout, nested Mix Shader and Add Shader trees, animated values and drivers.

Straight-color extraction supports Principled Base Color, image/procedural graphs,
Transparent/Holdout branches without premultiplying the visible branch, and nested Mix/Add
Shader trees.

## B3: scene-aware object baking

`SceneCombinedBakeStrategy` uses real `COMBINED` object baking for explicit requirements such
as Subsurface, Sheen, Toon, Translucent, Hair, Ambient Occlusion, World illumination and
scene-object occlusion.

One mesh may mix local and scene-aware slots. Each pass preserves matched copied materials and
replaces unmatched slots with black Emission proxies.

The temporary bake target occupies the source transform. The live source is temporarily
excluded from render during scene-aware object baking, then restored in `finally`.

When scene `COMBINED` and explicit alpha are composed, RGB is converted to straight color:

```text
straight.rgb = combined.rgb / alpha, alpha > epsilon
straight.rgb = 0,                    alpha == 0
final.alpha  = alpha_pass.red
```

NumPy and deterministic `array('f')` fallbacks implement the same operation.

## B4: camera-render projection

`domain/baking/camera_projection.py` selects B4 for Camera/View, Reflection/Transmission,
Volume and render-evaluated displacement.

Affected graph families include Fresnel, Layer Weight, Light Path, Glass, Refraction,
Principled Transmission, reflective appearance and Principled Volume, including dependencies
nested inside reachable Shader Node Groups.

`CameraProjectionPlan` is a frozen `BakePlan` subtype. Its synthetic camera pass preserves
frame/output metadata but is never executed with `bpy.ops.object.bake`.

### B4 render pipeline

Responsibilities are split:

- `camera_projection_state.py`: reversible Blender render/visibility/frame state;
- `camera_projection_image.py`: staged-image decode, alpha masks and crop rewrite;
- `camera_projection_executor_core.py`: render and layout orchestration;
- `camera_projection_executor.py`: public facade;
- `projection_layout.py`: pure union crop and convex hull algorithms.

For every frame:

1. validate object/Scene/World/camera/light identities;
2. preserve render/frame/visibility state;
3. keep only the source directly camera-visible;
4. retain other objects for reflection, transmission, diffuse and shadow rays;
5. render a transparent full frame to a staged path;
6. decode that staged image and extract alpha at threshold `1 / 255`.

After all frames succeed:

1. union all alpha masks;
2. compute one bounding crop expanded by existing `bake_margin`;
3. build one deterministic counter-clockwise convex hull;
4. rewrite every frame to identical cropped dimensions;
5. rebuild the camera projection attachment;
6. recompose typed single/multi/mixed Spine documents;
7. serialize JSON;
8. commit every output atomically.

### Stable sequence layout

`CameraProjectionLayout` stores full dimensions, exclusive crop bounds, union hull, threshold,
padding, frame count and visible-pixel count.

Every sequence frame shares the same crop, UVs, screen offset, hull, fan triangulation and
attachment dimensions. A later frame cannot be clipped by a crop derived from only one setup
frame.

The convex hull uses alpha pixel-boundary points. Deep concavities and holes remain represented
by texture alpha. For hull vertex count `H`:

```text
UV values            = H * 2
triangle index values = (H - 2) * 3
```

UVs address the padded crop:

```text
u = (x - crop_min_x) / crop_width
v = 1 - (y - crop_min_y) / crop_height
```

Spine positions retain full-frame placement:

```text
x = pixel_x - full_width  / 2
y = pixel_y - full_height / 2
```

Cropping therefore reduces texture area without recentering the camera result.

### Post-render document finalization

The final layout does not exist during initial preparation. Production output reserves JSON,
renders/crops textures, rebuilds B4 attachments, recomposes typed documents, writes JSON and
then commits.

This path is implemented for single, standalone multi, connected multi and mixed exports.
Serialized JSON is never patched or merged.

The reservations-only compatibility API keeps full-frame B4 output for external callers that
serialize JSON before staging. Production uses the detailed staging API that returns the exact
layout.

See `docs/REWRITE_CAMERA_PROJECTION.md` for the complete contract.

## Executor boundaries

- `bake_executor_core.py`: object-bake primitives and real object-bake operator;
- `semantic_bake_executor.py`: B1-B3 execution and composition;
- `camera_projection_state.py`: B4 reversible state;
- `camera_projection_image.py`: B4 image operations;
- `camera_projection_executor_core.py`: B4 orchestration;
- `camera_projection_executor.py`: B4 facade;
- `texture_executor.py`: typed plan dispatch without operator access;
- `bake_executor.py`: stable public facade and two failure-injection operator hooks.

The only real operator access points are:

```text
bake_executor_core._call_bake_operator()
bake_executor._call_render_operator()
```

Architecture tests also cover single/multi/mixed output and projection finalization modules.

## Automatic support matrix

Real Blender decoded-image coverage includes:

- opaque Principled, Image Texture and procedural Base Color;
- pure Emission;
- mixed surface/Emission slots and shared Principled graphs;
- constant and linked Image Alpha;
- Transparent/Mix Shader in both orders;
- nested transparency and pure Transparent;
- animated Alpha;
- scene light, World and Ambient Occlusion response;
- mixed local/scene slots;
- scene straight RGBA;
- animated lights;
- Fresnel/Layer Weight render, crop and convex hull;
- Glass and Principled Volume projection;
- animated projection with one sequence-union crop;
- nested Layer Weight and Volume groups;
- unused group input reachability precision;
- standalone, connected and mixed cropped B4 composition;
- rollback during local, alpha, scene, sequence, render and crop/finalization stages;
- source state and temporary datablock restoration.

## Explicit extension boundaries

### Convex rather than concave screen geometry

The current hull is convex. Concavities and holes remain texture-alpha regions. A concave
contour simplification policy should only be added if real production fixtures justify the
additional topology and runtime cost.

### Fixed alpha threshold

Union detection uses decoded alpha `>= 1 / 255`. Threshold configuration and antialias coverage
are output policies, not material strategy UI modes.

### Connected multi-object B4 depth

Connected and mixed documents contain correct cropped layers. Independently rendered layers
still cannot reproduce arbitrary per-pixel depth intersections using one fixed Spine slot
order. Grouped camera rendering or depth-aware composition is required for that case.

### Render engine and HDR

Real B4 parity currently targets Blender 4.4 Cycles. Eevee, custom Compositor output, tone
mapping, premultiplication variants and HDR runtime expectations require separate policies and
fixtures.

## Extension contract

A new strategy or texture policy must be implemented without special-case material UI switches.
Required sequence:

1. add immutable graph/object/scene facts;
2. select object bake or camera projection deterministically;
3. produce immutable typed plans/layouts;
4. mutate only copied materials or reversible Blender state;
5. reuse atomic transactions;
6. add pure planner/compositor/geometry tests;
7. add real Blender decoded-image tests;
8. verify state, temporary datablocks, rollback, sequence and multi-object behavior;
9. update parity documentation before claiming production support.
