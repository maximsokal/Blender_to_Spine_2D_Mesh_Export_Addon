# Rewrite status

The active rewrite branch is `rewrite/a1-domain-foundation`; A1 compatibility targets
Spine 4.2.43. The deterministic geometry, loop-level UV lineage, transactional bake,
typed Spine composition, connected `all_objects` rig, and multi-object UI migration are
implemented.

## Production status

- `object.spine2d_multi_export` keeps its public ID and uses the Rewrite backend by
  default;
- standalone, connected, and mixed Connect-flag selections are supported;
- one checked object remains standalone;
- the final JSON and every texture share one atomic transaction;
- Legacy remains an explicit selectable backend and is never an automatic fallback;
- the single-object `main.save_uv_as_json()` operator remains legacy;
- the add-on version is unchanged.

## Validation

- Python 3.10: 376 passed, 4 skipped;
- Python 3.11: 376 passed, 4 skipped;
- Blender 4.4 geometry, modifiers, UV, Cycles, UV seams, parity, homogeneous
  multi-object, mixed multi-object, rollback, and registered operator lifecycle tests
  pass;
- temporary Blender datablocks and source state are checked for leaks/mutation.

## Remaining blockers

1. representative real project `.blend` fixtures with actual v0.23 JSON/image outputs;
2. accepted JSON and image parity reports for that fixture matrix;
3. migration of the existing single-object operator while preserving its public ID and
   Scene properties;
4. legacy removal only after real-project parity is proven.

See `docs/REWRITE_A1_GOLDEN_PARITY.md` for the fixture and parity procedure.
