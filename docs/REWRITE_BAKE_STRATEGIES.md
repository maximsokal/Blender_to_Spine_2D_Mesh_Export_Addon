# Semantic bake strategy architecture

## Purpose

The rewrite must not expose one UI switch for every Blender material combination. It
inspects the connected material graph, identifies semantic outputs and external context,
selects deterministic strategies, executes one or more real Blender passes, and composes
one Spine straight-RGBA texture.

```text
Blender Material + Object + Scene
        |
        v
reachable shader graph analysis
        |
        v
semantic channels + dependency kinds
        |
        v
ObjectBakeContext + SceneBakeContext
        |
        v
BakeStrategyRegistry
        |
        v
BakePassPlan[] + BakeCompositePlan
        |
        v
reversible copied-material preparation
        |
        v
transactional Blender execution
        |
        v
validated final texture
```

No strategy is selected by user-facing material checkboxes. The UI may report which
pipeline was chosen and why, but material, object, and scene analysis own the decision.

## Reachable material graph

`blender_adapter/shader_graph_analyzer.py` starts at the active
`ShaderNodeOutputMaterial` and walks only links contributing to that output. Unconnected
editor nodes do not change the result.

The Blender-independent `MaterialGraphSnapshot` records:

- reachable nodes and links;
- active output node;
- semantic output channels;
- external dependency kinds;
- analysis issues.

Semantic channels:

- `SURFACE_COLOR`;
- `SURFACE_EMISSION`;
- `ALPHA`;
- `VOLUME`;
- `DISPLACEMENT`.

Dependency kinds currently include:

- `IMAGE` and `TIME`;
- `OBJECT` and `GEOMETRY`;
- `WORLD` and `LIGHTING`;
- `OCCLUSION` and `SCENE_OBJECTS`;
- `VIEW` and `CAMERA`;
- `REFLECTION` and `TRANSMISSION`;
- `NODE_GROUP`.

Stable node names and socket links are used instead of Python identity of Blender RNA
wrappers. A `TIME` dependency from a keyframe or driver marks the material as animated,
just like an image sequence or movie.

## Immutable object and scene inputs

`domain/baking/context.py` defines Blender-independent planning inputs.

### `ObjectBakeContext`

It records:

- source object identity and type;
- world matrix;
- collection membership;
- render, camera-ray, and shadow visibility;
- object/data animation presence.

### `SceneBakeContext`

It records:

- scene identity and render engine;
- analysis frame;
- World identity, node types, color, Background strength, and animation;
- visible lights with type, energy, color, transform, shadow flag, and animation;
- active camera identity, projection type, transform, lens/ortho scale, clipping, and
  animation;
- visible object identities;
- shadow-caster identities;
- view transform, look, exposure, and gamma.

Planning consumes these snapshots directly. Scene-aware execution validates stable
identities again before temporary Blender datablocks are created. Numeric values that are
expected to animate are evaluated at each frame rather than compared to the initial
snapshot.

## Evaluation scopes

Every `BakePassPlan` declares one `BakeEvaluationScope`.

### `LOCAL`

The appearance can be represented from the material/object surface without preserving
scene lighting or camera rays.

Examples:

- ordinary Principled Base Color;
- Image Texture and procedural Base Color;
- pure material Emission;
- B2 straight-color and alpha extraction.

### `SCENE`

The appearance depends on lights, World, occlusion, or other scene objects, but Blender's
ordinary Cycles object bake can evaluate it with `COMBINED`.

Examples currently detected and executed:

- Subsurface and Sheen contributions;
- Toon, Translucent, Hair, and similar lighting-dependent shaders;
- Ambient Occlusion;
- World illumination;
- scene-object occluders and shadow/environment contributors.

### `CAMERA`

The appearance depends on camera/view/ray direction and cannot be represented correctly
by Blender 4.4 object UV baking.

Examples:

- Fresnel;
- Layer Weight;
- Light Path;
- Glass and Refraction;
- Principled Transmission;
- metallic/coat reflection that must preserve camera/environment appearance;
- reflection/refraction involving other objects.

Real Blender 4.4 validation proved that `bpy.ops.object.bake` exposes no camera-ray bake
type. Therefore these graphs do **not** fall back to ordinary `COMBINED`; the camera
projection boundary detector returns an actionable `camera-render projection` error. B4
must render from the active camera and project that result onto a Spine-compatible
mesh/texture.

### `AUXILIARY`

A pass supplies a semantic channel used by composition rather than a complete appearance.
The current example is `AlphaBakeStrategy`.

## Strategy registry

`domain/baking/strategies.py` contains the Blender-independent registry.

Implemented executable strategies:

1. `SceneCombinedBakeStrategy`;
2. `SurfaceColorBakeStrategy`;
3. `EmissionBakeStrategy`;
4. `AlphaBakeStrategy`.

The registry also contains a camera projection boundary detector. It classifies
camera-dependent appearance and stops planning before a `BakePassPlan` is emitted, because
there is no valid Blender 4.4 object-bake mode for camera rays.

Every executable pass records:

- strategy identifier;
- evaluation scope;
- real Blender bake mode;
- material-slot coverage;
- semantic-channel coverage;
- per-slot copied-material preparation.

`BakeMode` intentionally contains only the object-bake types used and verified in Blender
4.4:

```text
DIFFUSE
COMBINED
EMIT
```

## B1 surface and emission

Ordinary opaque surface color uses lighting-independent `DIFFUSE Color`. Pure material
Emission uses `EMIT`.

When surface and emission are separate contributions:

```text
final.rgb   = clamp(surface.rgb + emission.rgb)
final.alpha = max(surface.alpha, emission.alpha)
```

If a caller requests `COMBINED` for a local surface portion that is also composed with a
separate emission pass, it is normalized to `DIFFUSE`; `COMBINED` already contains
emission and would count it twice.

Single-pass opaque output bypasses the compositor so previous local files do not change
merely because the architecture supports multiple strategies.

## B2 alpha and straight color

A normal `DIFFUSE Color` bake of an alpha-bearing shader can return attenuated or
premultiplied RGB. B2 evaluates color and opacity independently:

```text
straight surface color -> temporary Emission proxy -> EMIT
material opacity       -> grayscale Emission proxy -> EMIT
material emission      -> native EMIT when present
```

Final composition:

```text
final.rgb   = clamp(sum(color contribution passes))
final.alpha = alpha_pass.red
```

Opacity extraction supports:

- Principled Alpha, linked Image Alpha, and procedural/value graphs;
- Transparent BSDF and Holdout as zero opacity;
- ordinary surface shaders as full opacity;
- Mix Shader as `(1 - Fac) * opacity(A) + Fac * opacity(B)`;
- Add Shader as clamped addition;
- nested Mix/Add trees;
- animated socket values and drivers.

Straight-color extraction supports:

- Principled Base Color;
- linked Image/procedural color graphs;
- common shader `Color` and `Base Color` sockets;
- Transparent/Holdout as no color contribution;
- Mix Shader without multiplying straight RGB by final coverage;
- Add Shader color composition;
- nested Mix/Add trees.

Pure Transparent produces black RGB and alpha zero. Opaque slots sharing an object with
transparent slots contribute alpha one.

## B3 scene-aware object baking

### Strategy selection

A normal Principled material does not become scene-aware merely because a lamp exists in
the file. This preserves the historical local-albedo behavior.

`SCENE_COMBINED` is selected only when the reachable material graph explicitly declares a
scene dependency such as lighting, World, occlusion, or scene objects.

### Per-slot isolation

One mesh may contain local and scene-aware material slots. Each pass receives an explicit
per-slot preparation plan:

```text
matched slot     -> PRESERVE
unmatched slot   -> ZERO_TO_EMISSION
```

`ZERO_TO_EMISSION` temporarily replaces only the copied material's Surface output with a
black Emission proxy. This prevents a local material from appearing in the scene pass or a
scene material from being counted again in the local pass.

### Source-object exclusion

The temporary bake target occupies the source object's transform. During a scene-aware
pass the live source object is temporarily excluded from render, preventing coincident
duplicate geometry while leaving the source datablock available to Object Info and other
references.

`hide_render` is captured and restored in `finally` on success and failure.

### Scene alpha

A `COMBINED` scene pass may return RGB multiplied by coverage. If the material also has a
separate alpha pass, `BakeCompositePlan.unpremultiply_color_by_alpha` converts the scene
color back to straight RGB before writing the final texture:

```text
straight.rgb = combined.rgb / alpha, when alpha > epsilon
straight.rgb = 0,                    when alpha == 0
final.alpha  = alpha_pass.red
```

Both NumPy and deterministic `array('f')` fallback implementations use the same rule.

### Animation

World, lights, camera, and source object/data animation are captured as dependencies.
Sequence frame tasks set the Blender timeline frame before every strategy pass. Real tests
verify that a keyframed light produces different decoded output frames and that the
original frame is restored afterward.

### Transaction and cleanup

Scene-aware passes remain inside the same caller-owned atomic transaction as JSON, local
passes, alpha passes, static textures, and sequence frames.

A failure restores:

- previous output bytes;
- source `hide_render`;
- active object, selection, and mode;
- timeline frame;
- scene render/bake settings;
- copied material links and temporary nodes;
- temporary images, materials, meshes, objects, and collections.

## Executor boundaries

- `bake_executor_core.py` owns validation, temporary mesh/image primitives, reservations,
  and the sole real Blender object-bake operator implementation;
- `semantic_bake_executor.py` owns strategy execution, scene validation, copied-material
  preparation, and composition;
- `scene_bake_analyzer.py` produces immutable object/scene snapshots;
- `scene_bake_execution.py` owns reversible live-source render exclusion;
- `scene_material_preparation.py` owns scene-pass material-slot masks;
- `bake_executor.py` remains the stable public facade.

## Verified real Blender matrix

Decoded-pixel Blender 4.4 tests now cover:

- ordinary Principled/Image/procedural color;
- pure Emission;
- separate Surface and Emission material slots;
- one Principled graph with simultaneous Base Color and Emission;
- Principled constant Alpha and linked Image Alpha;
- Transparent/Mix Shader in both orders and nested mixes;
- animated Alpha;
- scene `COMBINED` responding to light energy;
- World-only illumination changes;
- Ambient Occlusion responding to another render-visible object;
- mixed local and scene-aware slots without double counting;
- scene-dependent alpha converted to straight RGBA;
- keyframed light sequence frames;
- camera-dependent graph rejection at the projection boundary;
- rollback after a scene pass begins;
- source/context/frame/material/datablock restoration.

## Explicit remaining boundaries

### Camera render projection

Fresnel, Layer Weight, Light Path, Glass, Refraction, Transmission, and reflection that
must preserve camera appearance require B4. The implementation must:

1. render a deterministic active-camera view;
2. define the object/scene visibility set;
3. retain or derive camera-space depth/coverage;
4. project the rendered result onto a Spine-compatible mesh or generated camera plane;
5. preserve sequence timing and atomic output behavior.

### Volume

Volume output cannot be represented by ordinary surface UV object baking. It already
returns a structured camera-projection planning error and belongs to the same B4 render
projection pipeline.

### Node groups

A `NODE_GROUP` dependency is recorded, but internal group trees are not recursively
snapshotted yet. Recursive support must map group inputs/outputs through a stable group
path, prevent cycles, and distinguish node instances from shared node-tree datablocks.

### HDR policy

Ordinary exported textures clamp additive RGB. OpenEXR preservation or configurable tone
mapping belongs to a later output policy, not to material strategy selection.

## Extension contract

A new strategy must not add UI material-mode switches or special-case chains to the public
executor.

Required sequence:

1. extend immutable graph/object/scene facts only when existing values are insufficient;
2. classify the required evaluation scope;
3. register one deterministic strategy or explicit projection boundary;
4. produce typed pass, material-preparation, and composition plans;
5. mutate only copied node trees;
6. add pure resolver/compositor tests;
7. add real Blender decoded-pixel tests;
8. verify source state, temporary datablocks, animation, rollback, multi-material objects,
   and common-rig export.

No strategy may mutate the user's source material or rely on undocumented global selection
or mode state.
