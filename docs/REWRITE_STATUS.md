# Rewrite status

## Active direction

The active rewrite branch is `rewrite/a1-domain-foundation` and the compatibility
mode is A1. The engine preserves the legacy Spine 4.2.43 rig contract while replacing
unsafe coordinate matching, intermediate JSON merging, and uncontrolled Blender state
mutation.

The older `pipeline_v2` stabilization experiment is not part of the rewrite and must not
be merged into `main`.

## Implemented

### Geometry, UV, and attachment lineage

- immutable mesh snapshots with separate local IDs and stable source lineage;
- exact `SourceLoopId` UV transfer without rounded positions or nearest-point matching;
- deterministic legacy seed-normal segmentation with disjoint face coverage;
- deterministic manifold-disk decomposition without random k-means;
- source-lineage-preserving ear-clipping triangulation;
- source-vertex Z-group assignment performed once on the original snapshot;
- one full-object texturing topology containing every segmentation/decomposition seam;
- one global unwrap with exact UV propagation to every export region;
- one Spine attachment vertex per unique `(VertexId, UV)` pair;
- external seam duplicates remain consecutive hull vertices and internal seam duplicates
  remain after the physical hull;
- triangle and edge indices resolve through exact loops rather than coordinate searches.

### Blender services

- read-only source and evaluated-mesh readers;
- explicit modifier lineage policies;
- transactional Blender context, mode, selection, frame, and render-state restoration;
- temporary Object, Mesh, Material, and Image ownership with guaranteed cleanup;
- isolated full-object UV unwrap;
- read-only material analysis;
- real Cycles baking;
- reusable in-memory `prepare_a1_object()` stage;
- full single-object service with one atomic JSON plus texture transaction;
- homogeneous multi-object service for standalone or connected selections;
- mixed multi-object service where two or more checked objects form `all_objects` and
  unchecked objects remain standalone in the same final document;
- one checked object retains the historical standalone behavior;
- rollback of the final JSON and all textures even when a later object fails after an
  earlier bake has completed.

### Spine A1 domain

- typed Spine document, serializer, and cross-reference validator;
- weighted-vertex encoder/decoder;
- centralized A1 naming profile;
- legacy-compatible rig and attachment builders;
- strict multi-document composition with immutable local-to-global bone maps;
- unknown weighted bone indices are errors and never fall back to root;
- deterministic animation namespaces and collision-free constraint order rebasing;
- typed connected `all_objects` builder with explicit anchor, offsets, and Z layers;
- no intermediate JSON merge or post-serialization weighted-index remap;
- structural fingerprint and semantic legacy/rewrite parity comparator.

### Multi-object UI migration

The existing public operator ID is unchanged:

```text
object.spine2d_multi_export
```

The registered operator uses the rewritten transactional engine by default. Existing
Scene and Object properties are translated through `a1_ui_bridge.py` into typed settings.
Supported UI cases:

- all selected objects standalone;
- all selected objects connected;
- mixed connected and standalone selection;
- per-object sequence settings;
- active object as the deterministic first/anchor candidate;
- historical `<active>_plus_<N>_objects.json` naming.

The previous exporter remains available through the explicit `LEGACY` backend. A
rewrite failure is reported and never silently starts the legacy exporter.

### Golden parity tooling

- `tools/compare_a1_exports.py` compares legacy and rewrite JSON files outside Blender;
- exit codes: `0` compatible, `1` incompatible, `2` invalid input;
- path-specific errors and warnings;
- explicit numeric tolerance policies;
- semantic weighted-stream comparison with exact bone-index checks;
- documented real-project fixture procedure in `docs/REWRITE_A1_GOLDEN_PARITY.md`.

### Architecture guards

- `domain/` and `application/` cannot import `bpy` or `bmesh`;
- geometry domain cannot depend on random-number modules;
- UV and bake operators are confined to dedicated Blender adapters;
- Blender operators are forbidden inside geometry/material traversal loops;
- GitHub Actions uploads complete pytest and Blender logs.

## Validation status

- Python 3.10: 376 passed, 4 skipped;
- Python 3.11: 376 passed, 4 skipped;
- Blender 4.4 geometry/UV, modifier lineage, Cycles bake, UV seam, and parity tests pass;
- homogeneous standalone and connected multi-object tests pass;
- mixed two-connected-plus-one-standalone test passes;
- second-bake failure restores the previous JSON and both textures;
- the registered `bpy.ops.object.spine2d_multi_export()` passes through the full add-on
  `register()`/`unregister()` lifecycle;
- Rewrite default, explicit Legacy selection, and no automatic fallback are verified;
- temporary Blender datablocks are checked for leaks.

## Production path

The multi-object operator now uses the rewrite by default, with an explicit Legacy
backend retained for controlled fallback. The single-object `main.save_uv_as_json()`
operator remains on the legacy path. The add-on version has not been bumped.

## Remaining production blockers

1. representative real project `.blend` fixtures with their actual v0.23 JSON and image
   outputs;
2. accepted JSON and image parity reports for the documented fixture matrix;
3. migration of the existing single-object operator while preserving its public ID and
   Scene properties;
4. legacy orchestration removal only after real-project parity is proven.
