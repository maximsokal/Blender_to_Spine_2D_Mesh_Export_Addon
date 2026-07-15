# Rewrite status

## Active direction

The active rewrite branch is `rewrite/a1-domain-foundation` and the compatibility
mode is A1. The new engine must preserve the legacy Spine 4.2 rig contract before
it can replace the current production exporter.

The older `pipeline_v2` stabilization experiment is not part of the rewrite and
must not be merged into `main`.

## Implemented

- immutable application request/result contracts;
- typed Spine document model;
- Spine serializer and cross-reference validator;
- weighted-vertex encoder/decoder;
- centralized legacy rig naming profile;
- structural Spine golden fingerprint;
- immutable mesh snapshots with local IDs and source lineage;
- exact `SourceLoopId` UV transfer;
- geometry fingerprinting;
- read-only Blender source-mesh adapter;
- deterministic segmentation plans with explicit boundary reasons;
- per-segment topology reports and immutable segment snapshots.

## Production path

The legacy `main.save_uv_as_json()` path remains active. No UI operator has been
switched to the rewrite and no addon version has been bumped.

## Next architectural slices

1. complex-segment decomposition without random k-means;
2. evaluated modifier lineage propagation;
3. UV unwrap transaction and snapshot write-back;
4. texture baking transaction;
5. A1 rig builder that produces `SpineDocument` directly;
6. single-object use case and golden parity fixtures;
7. in-memory multi-object composition;
8. Blender headless integration suite;
9. operator migration and legacy removal.
