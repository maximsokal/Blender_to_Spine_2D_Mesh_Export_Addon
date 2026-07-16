# Rewrite status

The active rewrite branch is `rewrite/a1-domain-foundation`; A1 compatibility targets
Spine 4.2.43. Deterministic geometry, loop-level UV lineage, transactional baking,
typed Spine composition, connected `all_objects`, and both production export operators
are implemented.

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
- final JSON and every texture share one atomic transaction;
- Legacy remains an explicit selectable backend for both operators and is never an
  automatic fallback;
- library-level A1 visual options remain opt-in, while the Blender UI explicitly passes
  its Scene properties;
- A1 surface/image/procedural textures use lighting-independent `DIFFUSE` color baking;
- pure Emission material sets use `EMIT`; unsupported mixed emission/surface slot sets
  fail before producing a partially black texture;
- per-polygon material slot indices are restored from the immutable bake snapshot only
  after every temporary Blender material slot exists;
- Blender-independent domain and parity tooling imports without a real `bpy` runtime;
- the add-on version is unchanged.

## Legacy-derived bake compatibility matrix

The real Blender suite now reconstructs failure-prone legacy inputs rather than checking
only that a PNG file exists. Every output is decoded and must contain usable alpha/RGB
pixels.

Covered scenarios:

- one mesh with two standard Principled material slots and distinct baked colors;
- generated Image Texture and procedural Checker materials;
- three connected objects in one `all_objects` rig, each with multiple materials;
- exactly one object exporting a three-frame sequence while the others remain static;
- sequence frames must contain different decoded pixels;
- sequence metadata is present only on the animated object's attachments;
- one common JSON plus the exact expected static/sequence PNG set;
- failure during the second sequence frame restores the previous JSON and every static
  or sequence texture byte-for-byte, with no staged/backup leftovers;
- active object, selection, mode, frame, scene bake settings, source node trees, and
  temporary Blender datablocks are restored.

This matrix found and fixed two production defects that the previous tests missed:

1. `COMBINED` baking could return `FINISHED` and write a fully opaque black PNG when the
   scene had no effective lighting;
2. Blender clamped every temporary polygon material index to slot `0` because indices
   were assigned before material slots were created.

## Validation

- Python 3.10: 425 passed, 4 skipped;
- Python 3.11: 425 passed, 4 skipped;
- Blender 4.4 geometry, modifiers, UV, Cycles, decoded bake pixels, UV seams, parity,
  homogeneous multi-object, mixed multi-object, rollback, and registered operator
  lifecycle tests pass;
- real single-operator tests verify Rewrite, explicit Legacy, no automatic fallback,
  legacy JSON/texture naming, control icons, preview animation, and both disabled states;
- the isolated fixture harness runs Legacy and Rewrite in separate Blender processes,
  then runs semantic JSON and decoded image comparison from normal Python;
- temporary Blender datablocks and source state are checked for leaks or mutation.

## Remaining production blockers

1. representative real project `.blend` fixtures with actual v0.23 JSON and image
   outputs;
2. accepted JSON and image parity reports for that fixture matrix;
3. controlled removal of legacy orchestration only after real-project parity is proven;
4. add-on version bump and release packaging only after the parity gate is accepted.

See `docs/REWRITE_A1_GOLDEN_PARITY.md` for the fixture and parity procedure.
