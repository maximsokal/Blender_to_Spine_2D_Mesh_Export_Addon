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
- legacy seed-normal angular grouping with disjoint face coverage;
- deterministic manifold-disk decomposition without random k-means;
- reusable Euler/boundary/manifold topology analysis;
- per-segment and per-region immutable snapshots;
- evaluated modifier lineage policy and structured reports;
- transactional evaluated-mesh reader using temporary POINT/EDGE/FACE/CORNER
  attributes and guaranteed cleanup;
- static dependency tests forbidding `bpy`, `bmesh`, and `random` in geometry domain.

## Modifier policy currently supported

- `STRICT_PRESERVE`: deformation-only stacks where every source element survives
  exactly once;
- `ALLOW_SOURCE_DUPLICATION`: source vertices/faces/corners may repeat, permitting
  modifier behaviour such as Mirror or Triangulate when attributes are preserved;
- generated edges are allowed and represented with `MeshEdge.source_id = None`;
- generated vertices, faces, and corners are rejected because exact UV lineage
  cannot be proven.

## Validation status

- focused pure-Python domain and fake-Blender adapter tests pass;
- GitHub Actions passes on Python 3.10 and Python 3.11;
- the real Blender headless suite has not been added yet, so evaluated attribute
  propagation still requires Blender 4.4 fixture verification before production use.

## Production path

The legacy `main.save_uv_as_json()` path remains active. No UI operator has been
switched to the rewrite and no addon version has been bumped.

## Next architectural slices

1. UV unwrap transaction and immutable snapshot write-back;
2. texture/material bake plan and transactional execution;
3. A1 rig builder that produces `SpineDocument` directly;
4. single-object orchestration and golden parity fixtures;
5. in-memory multi-object composition;
6. Blender headless integration suite with real `.blend` fixtures;
7. operator migration and legacy removal.
