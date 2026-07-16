# Rewrite status

The active rewrite branch is `rewrite/a1-domain-foundation`; A1 compatibility targets
Spine 4.2.43. Deterministic geometry, loop-level UV lineage, transactional and semantic
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
- Legacy remains an explicit selectable backend for both operators and is never an
  automatic fallback;
- library-level A1 visual options remain opt-in, while the Blender UI explicitly passes
  its Scene properties;
- connected shader graphs are analyzed from the active Material Output; unused editor
  nodes do not change the bake plan;
- semantic channels and external dependencies are stored in immutable graph snapshots;
- a Blender-independent strategy registry converts material requirements into one or
  more typed bake passes;
- ordinary surface/image/procedural color uses lighting-independent `DIFFUSE` color
  baking;
- pure Emission uses `EMIT`;
- surface and Emission materials may coexist on one object and are automatically baked
  through separate passes and composed into one texture;
- one Principled material may contribute both Base Color and Emission Color through the
  same automatic multi-pass pipeline;
- per-polygon material slot indices are restored from the immutable bake snapshot only
  after every temporary Blender material slot exists;
- Blender-independent domain and parity tooling imports without a real `bpy` runtime;
- the add-on version is unchanged.

## Semantic bake B1

The B1 pipeline is:

```text
active connected shader graph
        -> semantic channels/dependencies
        -> BakeStrategyRegistry
        -> BakePassPlan[]
        -> real Blender pass images
        -> BakeCompositePlan
        -> one atomic final texture
```

Current registered strategies:

- `SurfaceColorBakeStrategy`;
- `EmissionBakeStrategy`.

Current composition for mixed surface/emission output adds float RGB contributions,
uses the maximum pass alpha, and clamps ordinary exported RGB. Single-pass output does
not pass through the compositor, preserving prior output behavior.

When a caller requests `COMBINED` for a surface pass that will also be composed with an
`EMIT` pass, the surface pass is normalized to `DIFFUSE`. `COMBINED` already contains
emission and would otherwise count it twice.

Current explicit extension boundaries:

- Alpha is detected but still follows the historical surface pass; a dedicated alpha
  extraction strategy belongs to B2;
- node groups are marked as dependencies but are not recursively expanded yet;
- view/camera/world/lighting dependencies are detected, but scene-aware strategies and
  immutable scene context belong to B3;
- Volume produces a structured missing camera-projection strategy error and belongs to
  B4 rather than being silently flattened through UV baking.

See `docs/REWRITE_BAKE_STRATEGIES.md` for the complete extension contract.

## Legacy-derived bake compatibility matrix

The real Blender suite reconstructs failure-prone legacy inputs rather than checking
only that a PNG file exists. Every output is decoded and must contain usable alpha/RGB
pixels.

Covered scenarios:

- one mesh with two standard Principled material slots and distinct baked colors;
- generated Image Texture and procedural Checker materials;
- separate surface and Emission slots on one object, with both colors present after
  decoded-pixel composition;
- one Principled graph with simultaneous Base Color and Emission Color;
- three connected objects in one `all_objects` rig, each with multiple materials;
- exactly one object exporting a three-frame sequence while the others remain static;
- sequence frames must contain different decoded pixels;
- sequence metadata is present only on the animated object's attachments;
- one common JSON plus the exact expected static/sequence PNG set;
- failure during the second sequence frame restores the previous JSON and every static
  or sequence texture byte-for-byte, with no staged/backup leftovers;
- active object, selection, mode, frame, scene bake settings, source node trees, and
  temporary Blender datablocks are restored.

The test matrix has found and fixed these production defects:

1. `COMBINED` baking could return `FINISHED` and write a fully opaque black PNG when the
   scene had no effective lighting;
2. Blender clamped every temporary polygon material index to slot `0` because indices
   were assigned before material slots were created;
3. Blender RNA node/socket Python wrapper identity was not stable enough for graph
   traversal; reachable graph matching now uses stable node names and socket links.

## Validation

- Python 3.10: 441 passed, 4 skipped;
- Python 3.11: 441 passed, 4 skipped;
- Blender 4.4 geometry, modifiers, UV, Cycles, semantic multi-pass, decoded bake pixels,
  UV seams, parity, homogeneous multi-object, mixed multi-object, rollback, and
  registered operator lifecycle tests pass;
- real multi-pass tests verify separate Surface/Emission slots and one Principled
  surface-plus-emission graph by decoded output pixels;
- real single-operator tests verify Rewrite, explicit Legacy, no automatic fallback,
  legacy JSON/texture naming, control icons, preview animation, and both disabled states;
- the isolated fixture harness runs Legacy and Rewrite in separate Blender processes,
  then runs semantic JSON and decoded image comparison from normal Python;
- temporary Blender datablocks and source state are checked for leaks or mutation.

## Remaining production blockers

1. representative real project `.blend` fixtures with actual v0.23 JSON and image
   outputs;
2. accepted JSON and image parity reports for that fixture matrix;
3. B2 dedicated alpha/transparency strategies before claiming broad transparent-material
   support;
4. B3 scene/camera-aware strategies before claiming reflections, transmission, or
   lighting-preserving output;
5. controlled removal of legacy orchestration only after real-project parity is proven;
6. add-on version bump and release packaging only after the parity gate is accepted.

See `docs/REWRITE_A1_GOLDEN_PARITY.md` for the fixture and parity procedure.
