# Semantic bake strategy architecture

## Purpose

The rewrite must not expose one UI switch for every Blender material combination. It
inspects the connected shader graph, identifies the outputs and external context needed
to evaluate it, selects deterministic bake strategies, executes one or more passes, and
composes one Spine RGBA texture.

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
copied-material preparation
        |
        v
transactional Blender execution
        |
        v
validated final straight-RGBA texture
```

No strategy is selected by user-facing material checkboxes. The UI may report the
selected pipeline and reasons, but analysis and planning own the decision.

## Reachable shader graph

`blender_adapter/shader_graph_analyzer.py` starts at the active
`ShaderNodeOutputMaterial` and walks only links contributing to that output. Unconnected
editor nodes do not change the plan.

The Blender-independent `MaterialGraphSnapshot` records:

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

Stable node names and socket links are used instead of Python identity of Blender RNA
wrappers. A `TIME` dependency from keyframes or drivers marks the material and object as
animated just like an image sequence or movie.

## Strategy registry

`domain/baking/strategies.py` contains a Blender-independent registry. Every strategy
has a stable identifier, deterministic priority, support predicate, semantic channels,
bake mode, and optional copied-material preparation.

Registered strategies:

1. `SurfaceColorBakeStrategy`;
2. `EmissionBakeStrategy`;
3. `AlphaBakeStrategy`.

A surface-only object receives one pass. An emission-only object receives one `EMIT`
pass. Mixed contributions receive separate passes even when they belong to the same
material slot.

If a caller requests `COMBINED` for a surface portion that is also composed with a
separate emission pass, the registry normalizes that surface pass to `DIFFUSE` because
`COMBINED` already contains emission and would count it twice.

## Typed multi-pass plan

`BakePlan.bake_mode` remains a compatibility alias for the first pass. The source of
truth is:

```text
BakePlan.passes: tuple[BakePassPlan, ...]
BakePlan.composite: BakeCompositePlan
```

Each pass records:

- strategy identifier;
- Blender bake mode;
- material slot coverage;
- semantic channel coverage;
- per-slot material preparation when the native Blender pass cannot expose the channel.

All used polygon material slots must be valid before Blender execution begins.

## B1 surface and emission composition

For ordinary opaque Principled, Image Texture, and procedural Base Color graphs, the
surface pass remains lighting-independent `DIFFUSE Color`. Pure material emission uses
`EMIT`.

Mixed surface plus emission composition is:

```text
final.rgb   = clamp(surface.rgb + emission.rgb)
final.alpha = max(surface.alpha, emission.alpha)
```

Single-pass opaque output bypasses composition so previous files do not change merely
because the architecture became multi-pass capable.

## B2 alpha and transparency

### Why alpha is a separate pass

A normal `DIFFUSE Color` bake of an alpha-bearing shader can return attenuated or
premultiplied RGB. Accepting that result would produce dark fringes and lose the original
straight surface color.

B2 therefore evaluates color and opacity independently:

```text
straight surface color -> temporary Emission proxy -> EMIT pass
material opacity       -> grayscale Emission proxy -> EMIT pass
material emission      -> native EMIT pass when present
```

The final compositor writes:

```text
final.rgb   = clamp(sum(color contribution passes))
final.alpha = alpha_pass.red
```

The final Blender image uses straight alpha. The RGB values are not multiplied by final
opacity.

### Copied-material preparation

`blender_adapter/bake_material_preparation.py` modifies only material copies owned by the
bake transaction. For every prepared pass it:

1. stores the copied active Material Output Surface links;
2. builds temporary Math, MixRGB, and Emission proxy nodes;
3. exposes one semantic expression;
4. performs the bake;
5. removes every temporary node;
6. restores the copied graph before the next pass.

The user's source node tree is never modified.

For an alpha-bearing object, every surface slot is evaluated through the same
straight-color Emission proxy. Used slots that have no surface-color contribution are
forced to black during that pass, preventing pure Emission material slots from being
counted in both the color and emission passes.

### Opacity expression rules

Current recursive opacity extraction supports:

- Principled Alpha input, including linked Image Alpha and procedural/value graphs;
- Transparent BSDF and Holdout as zero opacity;
- ordinary surface shaders as full opacity;
- Mix Shader as `(1 - Fac) * opacity(A) + Fac * opacity(B)`;
- Add Shader as clamped addition;
- nested Mix/Add Shader trees;
- animated socket values and drivers through frame-by-frame evaluation.

Opaque material slots sharing the same object are written as opacity `1` in the alpha
pass. A purely transparent material produces black RGB with alpha `0`.

### Straight surface color rules

Current recursive straight-color extraction supports:

- Principled Base Color;
- linked Image Texture and procedural Base Color graphs;
- common shader `Color` or `Base Color` sockets;
- Transparent/Holdout branches as having no color contribution;
- Mix Shader between transparent and colored branches without multiplying RGB by
  coverage;
- Mix Shader between two colored branches using its factor;
- Add Shader by adding and clamping color contributions;
- nested Mix/Add Shader trees.

An alpha-bearing shader node without an identifiable color channel fails with a
structured preparation error rather than writing a black texture silently.

### Alpha composition and transactions

`ADD_RGB_REPLACE_ALPHA` routes explicit color pass indices and one explicit alpha pass.
The compositor reads the alpha mask from the alpha pass red channel, not from the bake
target image's incidental alpha channel.

Alpha passes use the same atomic transaction as JSON and every texture frame. A failure
on the alpha pass after a successful color pass restores prior output bytes and removes
staged, backup, image, material, mesh, object, and collection data.

## Executor boundaries

The executor is split by responsibility:

- `bake_executor_core.py` owns validation, temporary mesh/image primitives, reservations,
  and the real Blender bake operator call;
- `semantic_bake_executor.py` owns strategy passes, copied-material preparation, and
  composition;
- `bake_executor.py` is the stable public compatibility facade.

The only real `bpy.ops.object.bake` access remains confined to
`bake_executor_core._call_bake_operator()`.

## Current automatic support

Implemented and covered by real Blender decoded-pixel tests:

- opaque Principled surface color;
- Image Texture and procedural Base Color;
- pure Emission;
- separate surface and Emission slots on one mesh;
- one Principled material with Base Color and Emission Color;
- Principled constant Alpha;
- linked Image Alpha;
- Transparent BSDF mixed on either side of Mix Shader;
- nested transparency Mix Shader;
- pure Transparent material;
- animated Alpha sequences;
- alpha-pass rollback after an already successful color pass;
- mixed static and sequence objects in the existing object-level transaction.

## Explicit extension boundaries

These are architectural extension points, not reasons to add UI switches.

### Node groups

A `NODE_GROUP` dependency is recorded, but internal group trees are not recursively
snapshotted yet. A following graph-analysis increment must enter group node trees with a
stable group path, map group input/output sockets, prevent recursive cycles, and preserve
datablock identity separately from node instance identity.

### Scene-dependent appearance

World, lighting, view, and camera dependencies are detected, but scene-aware strategies
are not implemented yet. B3 must introduce immutable object/scene context snapshots and
strategies such as:

- scene combined;
- camera combined;
- selected-to-active;
- transmission/reflection.

The strategy must declare which lights, world, camera, shadow casters, reflection
objects, visibility collections, and color-management settings are part of its input.

B2 alpha support does not imply correct Glass, Refraction, Transmission, Fresnel, Layer
Weight, or Light Path appearance. Those require B3 context-aware evaluation.

### Volume and camera projection

Volume produces a structured planning error naming the missing camera-projection
strategy. It is not silently flattened through a surface UV pass. B4 must render a
deterministic camera view and generate a camera-projected Spine texture/mesh when surface
UV baking cannot represent the appearance.

### HDR policy

Additive RGB is clamped when writing ordinary exported textures. A later output policy
may retain unclamped values in OpenEXR or apply a configured tone-mapping transform, but
that decision belongs to output/composition policy rather than material strategy.

## Extension contract

A new strategy must be implemented without changing operator UI or adding special-case
chains to the public executor.

Required sequence:

1. extend graph/object/scene analysis only with immutable facts;
2. add a semantic channel or dependency only when existing values cannot express the
   requirement;
3. implement one registered strategy with deterministic priority;
4. produce typed pass and material-preparation plans;
5. mutate only copied node trees when Blender cannot expose the channel directly;
6. implement or reuse a typed compositor operation;
7. add pure registry/compositor tests;
8. add real Blender decoded-pixel tests;
9. verify source state, temporary datablocks, atomic rollback, sequences, multi-material
   objects, and common-rig export.

No strategy may mutate the user's source material or depend on undocumented global
selection or mode state.
