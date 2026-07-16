# Semantic bake strategy architecture

## Purpose

The rewrite must not expose one UI switch for every Blender material combination. It
must inspect the connected shader graph, identify the outputs and external context
required to evaluate it, choose deterministic bake strategies, execute one or more
passes, and compose a single Spine texture.

The architecture is therefore:

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
BakeStrategyRegistry
        |
        v
BakePassPlan[] + BakeCompositePlan
        |
        v
transactional Blender execution
        |
        v
validated final RGBA texture
```

No strategy is selected from user-facing checkboxes. The UI may report the selected
pipeline and reasons, but the decision belongs to analysis and planning.

## B1 implementation

### Reachable shader graph

`blender_adapter/shader_graph_analyzer.py` starts at the active
`ShaderNodeOutputMaterial` and walks only links that contribute to that output.
Unconnected editor nodes do not change the plan.

The Blender-independent result is `MaterialGraphSnapshot`:

- reachable nodes and links;
- active output node;
- semantic output channels;
- external dependencies;
- analysis issues.

Current semantic channels:

- `SURFACE_COLOR`;
- `SURFACE_EMISSION`;
- `ALPHA`;
- `VOLUME`;
- `DISPLACEMENT`.

Current dependency categories:

- image and time;
- object and geometry;
- view and camera;
- world and lighting;
- node group.

The snapshot uses stable node names and socket links. It does not rely on Python
identity of Blender RNA wrappers.

### Strategy registry

`domain/baking/strategies.py` contains a Blender-independent registry. Every strategy
has a stable identifier, priority, support predicate, semantic channels, and Blender
bake mode selection.

B1 registers:

1. `SurfaceColorBakeStrategy`;
2. `EmissionBakeStrategy`.

A surface-only object receives one pass. An emission-only object receives one `EMIT`
pass. An object containing surface and emission contributions receives separate
`DIFFUSE Color` and `EMIT` passes even when both contributions belong to the same
Principled material.

If a legacy or caller setting requests `COMBINED` for the surface portion of a
multi-pass surface-plus-emission plan, the registry normalizes that pass to `DIFFUSE`.
`COMBINED` already contains emission and would otherwise count the emission twice.
Single-pass legacy behavior is unchanged.

### Multi-pass plan

`BakePlan` retains `bake_mode` as a compatibility alias for the first pass, but the
source of truth is:

```text
BakePlan.passes: tuple[BakePassPlan, ...]
BakePlan.composite: BakeCompositePlan
```

Each pass records:

- strategy identifier;
- Blender bake mode;
- material slot coverage;
- semantic channel coverage.

All used polygon material slots must be covered by at least one pass before Blender
execution begins.

### Composition

`blender_adapter/bake_compositor.py` copies pass images into float buffers and combines
them without touching source datablocks.

B1 composition for surface plus emission is:

```text
final.rgb   = clamp(surface.rgb + emission.rgb)
final.alpha = max(surface.alpha, emission.alpha)
```

NumPy is used when available inside Blender. A deterministic `array('f')` fallback is
provided. Single-pass output bypasses the compositor so previous files do not change
only because the architecture became multi-pass capable.

### Transaction and cleanup

Every pass is executed inside the existing caller-owned atomic file transaction.
Temporary pass images, final images, copied materials, target mesh/object/collection,
selection, active object, mode, timeline frame, and scene bake properties are cleaned
or restored on success and failure.

The only Blender bake operator call remains confined to
`bake_executor._call_bake_operator()`.

## Current automatic support

B1 supports and tests:

- ordinary Principled surface color;
- Image Texture and procedural color graphs evaluated through a surface-color pass;
- pure Emission materials;
- separate surface and Emission material slots on one mesh;
- one Principled material with both Base Color and Emission Color;
- mixed static and sequence objects through the existing object-level transaction;
- real Cycles pixel composition and source-state restoration.

## Explicit current boundaries

These are architectural extension points, not reasons to add UI switches.

### Alpha

Alpha is detected as a semantic channel, but B1 keeps it with the historical surface
pass. B2 must add an `AlphaBakeStrategy` that rewrites only copied node trees to expose
the evaluated alpha expression through an emission/grayscale pass, then writes that
mask into final alpha.

Required B2 cases include:

- Principled Alpha;
- Image Texture Alpha;
- Transparent BSDF mixed with a surface shader;
- nested Mix Shader transparency;
- animated alpha.

### Node groups

B1 records a `NODE_GROUP` dependency but does not recursively snapshot internal group
trees. A following graph-analysis increment must enter group node trees with a stable
group path, map group input/output sockets, prevent recursive cycles, and preserve
datablock identity separately from node instance identity.

### Scene-dependent appearance

World, lighting, view, and camera dependencies are detected, but B1 does not yet choose
scene-aware strategies. B3 must introduce immutable object/scene context snapshots and
strategies such as:

- scene combined;
- camera combined;
- selected-to-active;
- transmission/reflection.

The strategy must declare which lights, world, camera, shadow casters, reflection
objects, visibility collections, and color-management settings are part of its input.

### Volume and camera projection

Volume currently produces a structured planning error naming the missing
camera-projection strategy. It must not be silently flattened through a surface UV
pass. B4 must render a deterministic camera view and generate a camera-projected Spine
texture/mesh when surface UV baking cannot represent the appearance.

### HDR policy

B1 clamps additive RGB when writing ordinary exported textures. A later output policy
may retain unclamped values in OpenEXR or apply a configured tone-mapping transform,
but the decision belongs to output/composition policy rather than material strategy.

## Extension contract

A new strategy should be implemented without changing operator UI or adding a chain of
special cases to the executor.

The required sequence is:

1. extend graph/object/scene analysis only with immutable facts;
2. add a new stable semantic channel or dependency only when existing values cannot
   express the requirement;
3. implement one `BakeStrategy` registered with deterministic priority;
4. produce typed pass plans;
5. implement copied-material preparation only when the Blender pass cannot evaluate the
   channel directly;
6. implement or reuse a typed compositor operation;
7. add pure registry/compositor tests;
8. add real Blender pixel tests;
9. verify source state, temporary datablocks, atomic rollback, sequences, multi-material
   objects, and common-rig export.

No strategy may mutate the user's source material or depend on undocumented global
selection/mode state.
