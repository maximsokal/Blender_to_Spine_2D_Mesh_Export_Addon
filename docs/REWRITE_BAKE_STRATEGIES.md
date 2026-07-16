# Semantic texture strategy architecture

## Purpose

The rewrite must not expose one UI switch for every Blender material combination. It
inspects the connected shader graph, identifies semantic outputs and external context,
then selects one of two texture pipelines:

```text
Blender Material / Object / Scene
        |
        v
reachable shader graph analysis
        |
        v
semantic channels + dependencies
        |
        v
ObjectBakeContext + SceneBakeContext
        |
        v
build_texture_plan()
        |
        +-- BakePlan ----------------------> object UV bake passes
        |                                     + RGBA compositor
        |
        +-- CameraProjectionPlan ----------> active-camera transparent render
                                              + full-frame Spine quad
```

No strategy is selected by user-facing material checkboxes. The UI may report the chosen
pipeline and reasons, but analysis and planning own the decision.

## Reachable shader graph

`blender_adapter/shader_graph_analyzer.py` starts at the active
`ShaderNodeOutputMaterial` and walks only connected links contributing to that output.
Unused editor nodes do not change the plan.

`MaterialGraphSnapshot` records:

- reachable nodes and links;
- active output node;
- semantic channels;
- external dependencies;
- analysis issues.

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

Stable node names and socket links are used instead of Python identity of Blender RNA
wrappers. A `TIME` dependency from keyframes or drivers marks the material/object as
animated just like an image sequence or movie.

## Evaluation scopes

Every used material slot receives one primary evaluation scope:

- `LOCAL`: lighting-independent surface color and material Emission;
- `SCENE`: World, lighting, occlusion and other scene-object appearance;
- `CAMERA`: active-camera/ray appearance routed to B4;
- `AUXILIARY`: alpha or another future composition-only channel.

The presence of a lamp does not automatically make ordinary Principled Base Color
scene-aware. Selection depends on the reachable graph requirement.

## Object-bake strategy registry

`domain/baking/strategies.py` remains Blender-independent. Every executable object-bake
strategy declares:

- stable identifier;
- deterministic priority;
- evaluation scope;
- support predicate;
- semantic channels;
- Blender bake mode;
- optional copied-material preparation.

Executable strategies:

1. `SceneCombinedBakeStrategy`;
2. `SurfaceColorBakeStrategy`;
3. `EmissionBakeStrategy`;
4. `AlphaBakeStrategy`.

A camera boundary detector remains in the registry for callers that explicitly request an
object-bake plan. Production planning uses `build_texture_plan()` and converts that same
requirement into `CameraProjectionPlan` instead of returning a dead bake mode.

## Typed pass plan

`BakePlan.bake_mode` remains a compatibility alias for the first pass. The source of truth
is:

```text
BakePlan.passes: tuple[BakePassPlan, ...]
BakePlan.composite: BakeCompositePlan
```

Each pass records:

- strategy ID;
- verified Blender object-bake mode;
- material slot coverage;
- semantic channels;
- evaluation scope;
- per-slot copied-material preparation.

Verified Blender 4.4 object-bake modes used by the rewrite:

```text
DIFFUSE
COMBINED
EMIT
```

There is no `ACTIVE_CAMERA` object-bake type.

## B1: surface and Emission

Ordinary opaque Principled, Image Texture and procedural Base Color use
lighting-independent `DIFFUSE Color`. Pure material Emission uses `EMIT`.

```text
final.rgb   = clamp(surface.rgb + emission.rgb)
final.alpha = max(surface.alpha, emission.alpha)
```

Single-pass output bypasses the compositor so the architecture alone does not alter legacy
files.

If a surface contribution requested as `COMBINED` is also composed with a separate Emission
pass, the registry normalizes it to `DIFFUSE` to avoid counting Emission twice.

## B2: alpha and straight RGB

A native `DIFFUSE Color` bake of an alpha-bearing shader may return attenuated RGB. B2
evaluates straight color and opacity independently:

```text
straight surface color -> temporary Emission proxy -> EMIT pass
material opacity       -> grayscale Emission proxy -> EMIT pass
material emission      -> native EMIT pass when present
```

```text
final.rgb   = clamp(sum(color contribution passes))
final.alpha = alpha_pass.red
```

`blender_adapter/bake_material_preparation.py` mutates only copied materials. It stores the
copied output link, creates temporary Math/MixRGB/Emission nodes, runs one pass, removes all
temporary nodes and restores the copied graph in `finally`.

Opacity extraction supports:

- Principled Alpha, linked Image Alpha and procedural scalar graphs;
- Transparent BSDF and Holdout as zero opacity;
- ordinary surface shaders as full opacity;
- Mix Shader weighted opacity;
- Add Shader clamped opacity;
- nested Mix/Add trees;
- animated values and drivers.

Straight-color extraction supports:

- Principled Base Color;
- Image/procedural color graphs;
- Transparent/Holdout branches without premultiplying the visible branch;
- Mix Shader and Add Shader;
- nested trees.

## B3: scene-aware object baking

`SceneCombinedBakeStrategy` uses real `COMBINED` object baking for explicit requirements
such as:

- Subsurface and Sheen;
- Toon, Translucent and Hair shaders;
- Ambient Occlusion;
- World illumination;
- scene-object occlusion/environment dependencies.

One mesh may mix local and scene-aware slots. Each pass preserves matched copied material
slots and replaces unmatched slots with black Emission proxies.

The temporary bake target occupies the source transform. During scene-aware object baking
the live source is temporarily excluded from render, then restored in `finally`.

When scene `COMBINED` and explicit alpha are composed, RGB can be converted to straight
color:

```text
straight.rgb = combined.rgb / alpha, alpha > epsilon
straight.rgb = 0,                    alpha == 0
final.alpha  = alpha_pass.red
```

NumPy and deterministic `array('f')` fallbacks implement the same operation.

## B4: camera-render projection

`domain/baking/camera_projection.py` selects B4 for:

- Camera or View dependencies;
- Reflection or Transmission dependencies;
- Volume;
- render-evaluated displacement.

Affected graph families include Fresnel, Layer Weight, Light Path, Glass, Refraction,
Principled Transmission, reflective appearance and Principled Volume.

`CameraProjectionPlan` is a frozen `BakePlan` subtype. Its synthetic camera pass preserves
frame/output metadata but is never executed with `bpy.ops.object.bake`.

`blender_adapter/camera_projection_executor.py`:

1. validates immutable object/scene identities;
2. captures render/frame/visibility state;
3. makes only the source directly camera-visible;
4. keeps other objects available to reflection, transmission, diffuse and shadow rays;
5. enables transparent film;
6. renders each frame to an atomic staged path;
7. validates the output;
8. restores all state in `finally`.

`application/a1_camera_projection.py` creates one stable full-frame Spine quad with four
vertices and two triangles.

See `docs/REWRITE_CAMERA_PROJECTION.md` for the full B4 contract and boundaries.

## Executor boundaries

- `bake_executor_core.py`: object-bake resource primitives and real object-bake operator;
- `semantic_bake_executor.py`: B1-B3 execution and composition;
- `camera_projection_executor.py`: B4 render transaction;
- `texture_executor.py`: plan dispatch without operator access;
- `bake_executor.py`: stable public facade and the two failure-injection operator hooks.

The only real operator access points are:

```text
bake_executor_core._call_bake_operator()
bake_executor._call_render_operator()
```

## Automatic support matrix

Real Blender decoded-pixel coverage includes:

- opaque Principled, Image Texture and procedural Base Color;
- pure Emission;
- mixed surface/Emission slots and shared Principled graphs;
- constant Alpha and linked Image Alpha;
- Transparent/Mix Shader in both orders;
- nested transparency and pure Transparent;
- animated Alpha;
- scene light, World and Ambient Occlusion response;
- mixed local/scene slots;
- scene straight RGBA;
- animated lights;
- Fresnel/Layer Weight production camera projection;
- Glass camera projection;
- Principled Volume camera projection;
- animated camera-dependent projection frames;
- rollback during local, alpha, scene, sequence and render stages;
- source state and temporary datablock restoration.

## Explicit extension boundaries

### Recursive node groups

`NODE_GROUP` is recorded, but internal group trees are not recursively snapshotted. A future
increment must enter group node trees with stable group paths, map group sockets, prevent
recursive cycles and keep datablock identity separate from node instance identity.

### B4 crop and hull

B4 currently uses a full-frame transparent texture and full-frame quad. Stable union crop
and screen-space hull generation are future geometry policies.

### Connected multi-object B4 depth

Several source-only full-frame layers cannot reproduce arbitrary per-pixel intersections
using one fixed slot order. Grouped rendering or depth-aware composition is required before
connected B4 parity is claimed.

### HDR/output policy

Ordinary additive RGB is clamped. OPEN_EXR, tone mapping and runtime HDR expectations belong
to output/composition policy, not material strategy UI switches.

## Extension contract

A new strategy or texture pipeline must be implemented without adding special-case UI mode
switches.

Required sequence:

1. extend graph/object/scene analysis only with immutable facts;
2. add a semantic channel/dependency only when existing values cannot express the need;
3. select object bake or camera projection deterministically;
4. produce immutable typed plans;
5. mutate only copied materials or reversible scene state;
6. reuse atomic output transactions;
7. add pure planner/compositor/geometry tests;
8. add real Blender decoded-pixel tests;
9. verify source state, temporary datablocks, rollback, sequence and multi-material behavior;
10. update parity documentation before claiming production support.
