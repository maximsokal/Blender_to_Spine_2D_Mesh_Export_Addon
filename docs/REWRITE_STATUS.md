# Rewrite status

The active rewrite branch is `rewrite/a1-domain-foundation`; A1 compatibility targets
Spine 4.2.43. Deterministic geometry, loop-level UV lineage, semantic local/alpha/scene
baking, typed Spine composition, connected `all_objects`, and both production operators
are implemented.

## Production status

- `object.save_uv_as_json` keeps its public ID and uses Rewrite by default;
- single output preserves `<object>_merged.json` plus `<object>_Baked.png`;
- `Control icons` preserves the v0.23 `_rotation_X/_rotation_Z/_rotation_Y/_main` slots and
  attachments;
- `Preview animation` preserves v0.23 control-bone timelines;
- `object.spine2d_multi_export` keeps its public ID and uses Rewrite by default;
- standalone, connected, and mixed Connect-flag selections are supported;
- one checked object remains standalone;
- final JSON and every static/sequence texture share one atomic transaction;
- Legacy remains an explicit selectable backend and is never an automatic fallback;
- connected shader graphs are analyzed from active Material Output;
- unused editor nodes do not change strategy selection;
- material, object, and scene requirements are immutable domain snapshots;
- local, scene-aware, and auxiliary passes may coexist on one mesh;
- per-material mode switches were not added to the UI;
- Blender-independent domain and parity modules import without real `bpy`;
- the add-on version is unchanged.

## Semantic bake pipeline: B1, B2, and B3

```text
active connected shader graph
        -> semantic channels/dependencies
        -> ObjectBakeContext + SceneBakeContext
        -> evaluation scope per material slot
        -> BakeStrategyRegistry
        -> BakePassPlan[]
        -> reversible copied-material preparation
        -> real Blender pass images
        -> BakeCompositePlan
        -> one atomic straight-RGBA texture
```

### Evaluation scopes

- `LOCAL`: ordinary surface color, Image/procedural color, and material Emission;
- `SCENE`: World, lighting, occlusion, and other scene-object dependencies;
- `CAMERA`: view/ray/reflection/transmission requirements routed to the B4 camera-render
  projection boundary;
- `AUXILIARY`: explicit alpha and other future composition-only channels.

### Executable strategies

- `SceneCombinedBakeStrategy`;
- `SurfaceColorBakeStrategy`;
- `EmissionBakeStrategy`;
- `AlphaBakeStrategy`.

Camera-dependent graphs are detected by the registry, but Blender 4.4 exposes no
camera-ray object-bake type. They return an actionable camera-render projection error
instead of silently falling back to an incorrect UV `COMBINED` bake.

`BakeMode` contains only the verified Blender 4.4 object-bake types used by the rewrite:

```text
DIFFUSE
COMBINED
EMIT
```

## B1: local surface and emission

Ordinary opaque surface/Image/procedural color uses lighting-independent `DIFFUSE Color`.
Pure material Emission uses `EMIT`.

Mixed local contributions use explicit float-buffer composition:

```text
final.rgb   = clamp(surface.rgb + emission.rgb)
final.alpha = max(surface.alpha, emission.alpha)
```

A local surface pass requested as `COMBINED` is normalized to `DIFFUSE` when a separate
Emission pass exists, avoiding double counting.

## B2: alpha and straight color

Alpha-bearing materials evaluate surface color and opacity independently:

```text
straight surface color -> temporary Emission proxy -> EMIT
material opacity       -> grayscale Emission proxy -> EMIT
material emission      -> native EMIT when present
```

```text
final.rgb   = clamp(sum(color passes))
final.alpha = alpha_pass.red
```

Recursive copied-material extraction supports:

- Principled Base Color and Alpha;
- linked Image Color/Alpha and procedural socket graphs;
- Transparent BSDF/Holdout;
- Mix Shader in either transparent order;
- nested Mix and Add Shader trees;
- animated socket values/drivers;
- opaque and transparent slots on the same object;
- pure transparent output.

Temporary proxy nodes exist only on copied materials and are removed in `finally`. The
source material node tree is never modified.

## B3: scene-aware object baking

### Immutable context

`ObjectBakeContext` captures source identity, transform, collection membership,
visibility, and animation.

`SceneBakeContext` captures:

- render engine and analysis frame;
- World identity, nodes, Background strength, and animation;
- visible lights with energy/color/transform/shadow/animation;
- active camera identity and projection settings;
- visible objects and shadow casters;
- color-management settings.

Production `prepare_a1_object()` captures these facts before `build_bake_plan()` and stores
them in the final `BakePlan`.

### Automatic scene selection

Normal Principled Base Color remains local even when the file contains lights. This
preserves legacy albedo behavior.

`SCENE_COMBINED` is selected only for reachable graph requirements such as:

- Subsurface and Sheen;
- Toon, Translucent, Hair, and other lighting-dependent shaders;
- Ambient Occlusion;
- World illumination;
- scene-object occlusion/environment dependencies.

### Mixed local and scene slots

For each pass, matched material slots are preserved and unmatched slots are replaced on
copied materials with black Emission proxies. This allows one mesh to contain local and
scene-aware material slots without double counting either contribution.

### Live-source exclusion

The temporary bake target occupies the source transform. During a scene-aware pass the
live source object is temporarily excluded from render. Its previous `hide_render` value
is restored in `finally`, including rollback paths.

### Scene alpha

When a scene `COMBINED` pass and explicit alpha pass are composed, RGB may be converted
from coverage-multiplied values back to straight color:

```text
straight.rgb = combined.rgb / alpha, alpha > epsilon
straight.rgb = 0,                    alpha == 0
final.alpha  = alpha_pass.red
```

NumPy and deterministic `array('f')` paths implement the same operation.

### Scene animation

World, light, camera, and source animation are included in dependency snapshots. Sequence
frame tasks evaluate the Blender timeline before every pass. Keyframed-light tests verify
that decoded sequence frames differ and that the original frame is restored.

## Executor boundaries

- `bake_executor_core.py`: low-level validation, temporary mesh/image resources, atomic
  reservations, and the sole real object-bake operator implementation;
- `semantic_bake_executor.py`: strategy passes, runtime scene validation, source exclusion,
  material preparation, and composition;
- `scene_bake_analyzer.py`: immutable object/scene snapshots and runtime identity checks;
- `scene_bake_execution.py`: reversible live-source render exclusion;
- `scene_material_preparation.py`: per-pass material-slot masking;
- `bake_executor.py`: stable public facade preserving existing imports and failure
  injection.

## Real Blender compatibility matrix

Every image is decoded; a PNG signature alone is not accepted as success.

Covered scenarios include:

- multiple Principled slots with distinct colors;
- generated Image Texture and procedural Checker;
- separate Surface and Emission slots;
- one Principled graph with Base Color and Emission;
- Principled constant Alpha and linked Image Alpha;
- Transparent/Mix Shader in both orders;
- nested transparency and pure Transparent;
- animated Alpha sequence;
- scene `COMBINED` responding to light energy;
- World-only illumination changes;
- Ambient Occlusion responding to another object;
- mixed local and scene-aware material slots;
- scene-dependent alpha written as straight RGBA;
- keyframed-light sequence frames;
- camera/view graph rejection at the projection boundary;
- rollback during local, alpha, sequence, and scene passes;
- source `hide_render`, context, frame, scene settings, material graphs, and temporary
  datablock restoration;
- three connected multi-material objects in one `all_objects` rig;
- one sequence object while other connected objects remain static;
- exact JSON/static PNG/sequence PNG output sets;
- isolated Legacy/Rewrite fixture orchestration.

## Production defects found by the matrix

1. `COMBINED` could return `FINISHED` with an opaque black PNG when no useful lighting was
   present;
2. Blender clamped temporary polygon material indices to slot `0` when indices were
   assigned before material slots existed;
3. Blender RNA wrapper identity was unstable for graph traversal;
4. alpha-bearing `DIFFUSE Color` could lose straight RGB;
5. graph `TIME` dependencies were omitted from animated object analysis;
6. scene-aware baking could duplicate source and temporary geometry without explicit
   render exclusion;
7. local and scene-aware material slots required explicit black masks to prevent double
   counting;
8. Blender 4.4 object baking has no camera-ray bake type, so Fresnel/Layer Weight/Glass
   cannot be represented by an invented `ACTIVE_CAMERA` mode.

## Validation

- Python 3.10: **460 passed, 4 skipped**;
- Python 3.11: **460 passed, 4 skipped**;
- dedicated Blender 4.4 Alpha workflow passes;
- dedicated Blender 4.4 Scene workflow passes light, World, AO, mixed-scope, animated
  light, scene-alpha, camera-boundary, and rollback fixtures;
- full Blender 4.4 geometry, modifiers, UV, Cycles, semantic multi-pass, legacy-derived
  bake matrix, UV seams, parity, multi-object, rollback, operator lifecycle, fixture tools,
  and isolated Legacy/Rewrite workflows pass;
- temporary Blender datablocks and source state are checked for leaks or mutation.

## Explicit remaining boundaries

### B4 camera render projection

Fresnel, Layer Weight, Light Path, Glass, Refraction, Transmission, and camera-preserving
reflection require a deterministic active-camera render followed by texture/mesh
projection. They do not fall back to object UV baking.

Volume belongs to the same projection family because it cannot be flattened through a
surface UV pass without changing its meaning.

### Recursive node groups

Node groups are detected as dependencies, but internal group trees are not yet recursively
expanded. Arbitrary group-based transparency or scene dependency is therefore not claimed.

### HDR policy

Ordinary output clamps additive RGB. OpenEXR preservation or configurable tone mapping is
still an output-policy extension.

## Remaining production blockers

1. representative private production `.blend` fixtures with actual v0.23 JSON/images;
2. accepted JSON and decoded-image parity reports for that fixture matrix;
3. B4 camera-render projection for camera/view/reflection/transmission/volume appearance;
4. recursive node-group analysis;
5. controlled Legacy removal only after real-project parity;
6. version bump and release packaging only after the parity gate is accepted.

See `docs/REWRITE_A1_GOLDEN_PARITY.md` for the fixture and parity procedure and
`docs/REWRITE_BAKE_STRATEGIES.md` for the strategy extension contract.
