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
- the add-on version is unchanged.

## Validation

- Python 3.10: 381 passed, 4 skipped;
- Python 3.11: 381 passed, 4 skipped;
- Blender 4.4 geometry, modifiers, UV, Cycles, UV seams, parity, homogeneous
  multi-object, mixed multi-object, rollback, and registered operator lifecycle tests
  pass;
- real single-operator tests verify Rewrite, explicit Legacy, no automatic fallback,
  legacy JSON/texture naming, control icons, preview animation, and both disabled states;
- temporary Blender datablocks and source state are checked for leaks or mutation.

## Remaining production blockers

1. representative real project `.blend` fixtures with actual v0.23 JSON and image
   outputs;
2. accepted JSON and image parity reports for that fixture matrix;
3. controlled removal of legacy orchestration only after real-project parity is proven;
4. add-on version bump and release packaging only after the parity gate is accepted.

See `docs/REWRITE_A1_GOLDEN_PARITY.md` for the fixture and parity procedure.
