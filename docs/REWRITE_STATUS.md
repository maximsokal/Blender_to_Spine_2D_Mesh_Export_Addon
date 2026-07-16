# Rewrite status

The active rewrite branch is `rewrite/a1-domain-foundation`; A1 compatibility targets
Spine 4.2.43. Deterministic geometry, loop-level UV lineage, transactional semantic
multi-pass baking, typed Spine composition, connected `all_objects`, and both production
export operators are implemented.

## Production status

- `object.save_uv_as_json` keeps its public ID and uses Rewrite by default;
- single output preserves legacy naming: `<object>_merged.json` plus
  `<object>_Baked.png`;
- `Control icons` produces the exact v0.23 `_rotation_X/_rotation_Z/_rotation_Y/_main`
  bounding-box slots and attachments;
- `Preview animation` produces the exact v0.23 control-bone timelines;
- both visual options can be disabled independently;
- `object.spine2d_multi_export` keeps its public ID and uses Rewrite by default;
- standalone, connected, and mixed Connect-flag selections are supported;
- one checked object remains standalone;
- final JSON and every static/sequence texture share one atomic transaction;
- Legacy remains an explicit selectable backend and is never an automatic fallback;
- connected shader graphs are analyzed from active Material Output; unused editor nodes do
  not change the plan;
- semantic channels and external dependencies are immutable graph snapshots;
- a Blender-independent strategy registry creates one or more typed bake passes;
- ordinary opaque surface/image/procedural color uses lighting-independent `DIFFUSE`;
- pure Emission uses `EMIT`;
- surface and Emission contributions coexist through separate passes and composition;
- Alpha and transparency are extracted through copied-material Emission proxies;
- alpha-bearing surface RGB is evaluated independently as straight color, preventing
  opacity-premultiplied dark output;
- per-polygon material indices are restored only after temporary material slots exist;
- Blender-independent domain/parity tooling imports without real `bpy`;
- the add-on version is unchanged.

## Semantic bake B1 and B2

```text
active connected shader graph
        -> semantic channels/dependencies
        -> BakeStrategyRegistry
        -> BakePassPlan[]
        -> reversible copied-material preparation
        -> real Blender pass images
        -> BakeCompositePlan
        -> one atomic straight-RGBA texture
```

Registered strategies:

- `SurfaceColorBakeStrategy`;
- `EmissionBakeStrategy`;
- `AlphaBakeStrategy`.

### Surface and emission

Opaque surface color uses `DIFFUSE Color`; emission uses `EMIT`. Mixed surface/emission
adds float RGB contributions and clamps ordinary exported RGB. A surface pass requested
as `COMBINED` is normalized to `DIFFUSE` when a separate emission pass exists, preventing
double counting.

### Alpha and straight color

Alpha-bearing materials use independent passes:

```text
straight surface color -> temporary Emission proxy -> EMIT
material opacity       -> grayscale Emission proxy -> EMIT
material emission      -> native EMIT when present
```

Final composition uses explicit routing:

```text
final.rgb   = clamp(sum(color passes))
final.alpha = alpha_pass.red
```

Temporary proxy nodes are created only in copied materials and removed in `finally`.
Original Material Output links are restored between passes. Source materials are never
mutated.

Current recursive extraction supports:

- Principled Base Color and Alpha;
- linked Image Color/Alpha and procedural socket graphs;
- Transparent BSDF/Holdout;
- Mix Shader in either transparent order;
- nested Mix Shader;
- Add Shader opacity/color composition;
- animated sockets/drivers through per-frame evaluation;
- opaque slots sharing an object with transparent slots;
- pure transparent output.

A graph `TIME` dependency now marks material/object analysis as animated in addition to
image sequence/movie dependencies.

### Executor boundaries

- `bake_executor_core.py`: low-level validation, Blender temporary resources, atomic
  reservations, and the sole real `bpy.ops.object.bake` call;
- `semantic_bake_executor.py`: strategy execution, copied-material preparation, and
  composition;
- `bake_executor.py`: stable public facade preserving old imports and failure injection.

See `docs/REWRITE_BAKE_STRATEGIES.md` for the full extension contract.

## Legacy-derived bake compatibility matrix

Every output is decoded; a PNG signature alone is not accepted.

Covered scenarios include:

- multiple Principled material slots with distinct colors;
- generated Image Texture and procedural Checker;
- separate surface and Emission slots;
- one Principled graph with Base Color and Emission Color;
- Principled constant Alpha;
- linked Image Alpha;
- Transparent/Mix Shader in both orders;
- nested Mix Shader opacity and straight color;
- pure Transparent material;
- animated Alpha sequence with distinct decoded alpha values;
- three connected objects in one `all_objects` rig with multiple materials;
- exactly one sequence object while others remain static;
- exact JSON/static PNG/sequence PNG output sets;
- failure during sequence or alpha pass restores previous bytes with no staged leftovers;
- active object, selection, mode, frame, scene settings, source node trees, and temporary
  Blender datablocks are restored.

The matrix has found and fixed production defects:

1. `COMBINED` could return `FINISHED` with an opaque black PNG without lighting;
2. Blender clamped polygon material indices to slot `0` when assigned before slots;
3. Blender RNA wrapper identity was unstable for graph traversal;
4. alpha-bearing `DIFFUSE Color` could lose straight RGB;
5. graph `TIME` dependencies were omitted from object animated analysis.

## Validation

- Python 3.10: 449 passed, 4 skipped;
- Python 3.11: 449 passed, 4 skipped;
- dedicated Blender 4.4 Alpha suite passes static, linked-image, nested, animated sequence,
  pure-transparent, and alpha-pass rollback fixtures;
- full Blender 4.4 geometry, modifiers, UV, Cycles, semantic multi-pass, legacy-derived
  bake matrix, UV seams, parity, multi-object, rollback, operator lifecycle, fixture tools,
  and isolated Legacy/Rewrite orchestration pass;
- temporary Blender datablocks and source state are checked for leaks or mutation.

## Explicit remaining boundaries

- node groups are detected but internal group trees are not recursively expanded yet;
- view/camera/world/lighting dependencies are detected, but B3 scene-aware context and
  strategies are not implemented;
- B2 alpha support does not imply correct Glass, Refraction, Transmission, Fresnel, Layer
  Weight, Light Path, reflection, or lighting-preserving output;
- Volume returns a structured missing camera-projection strategy error and belongs to B4;
- HDR output/tone mapping remains a future output policy.

## Remaining production blockers

1. representative real project `.blend` fixtures with actual v0.23 JSON/images;
2. accepted JSON and image parity reports for that fixture matrix;
3. B3 scene/camera-aware strategies before claiming reflection, transmission, or
   lighting-preserving output;
4. recursive node-group analysis before claiming arbitrary group-based transparency;
5. controlled removal of Legacy only after real-project parity;
6. version bump and release packaging only after the parity gate is accepted.

See `docs/REWRITE_A1_GOLDEN_PARITY.md` for the fixture and parity procedure.
