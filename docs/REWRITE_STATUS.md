# Rewrite status

## Active direction

The active rewrite branch is `rewrite/a1-domain-foundation` and the compatibility
mode is A1. The new engine must preserve the legacy Spine 4.2 rig contract before
it can replace the current production exporter.

The older `pipeline_v2` stabilization experiment is not part of the rewrite and
must not be merged into `main`.

## Implemented

### Pure application and geometry pipeline

- immutable application request/result contracts and structured issues;
- immutable mesh snapshots with separate local IDs and stable source lineage;
- exact `SourceLoopId` UV transfer without rounded positions or nearest-point
  matching;
- legacy seed-normal segmentation with complete, disjoint face coverage;
- deterministic manifold-disk decomposition without random k-means;
- Euler, boundary-component and manifold topology analysis;
- deterministic source-lineage-preserving ear-clipping triangulation;
- source-vertex Z-group assignment performed once on the original snapshot;
- pure geometry preparation pipeline:
  segmentation -> decomposition -> materialization -> triangulation;
- shared full-object texturing topology that marks segmentation and decomposition
  cuts as seams;
- exact propagation of one globally unwrapped UV layer to every triangulated
  export region;
- pure A1 document assembly from ordered UV-ready regions.

### Blender adapters

- read-only source-mesh adapter using direct RNA access;
- evaluated modifier reader using temporary POINT/EDGE/FACE/CORNER lineage
  attributes and guaranteed cleanup;
- explicit modifier lineage policies and structured diagnostics;
- transactional Blender context capture/restore;
- temporary Mesh/Object materialization using direct data APIs;
- typed UV operator plans and isolated full-object unwrap transaction;
- read-only material analysis for image, sequence, procedural, mixed, empty and
  unsupported material slots;
- transactional material copies and active bake target nodes;
- transactional scene render/bake state restoration;
- texture bake executor with one required `bpy.ops.object.bake` call per planned
  frame;
- atomic multi-file staging, commit, backup and rollback for baked textures.

### Spine A1 domain

- typed Spine document model, serializer and cross-reference validator;
- weighted-vertex encoder/decoder;
- centralized legacy rig naming profile;
- exact A1 control hierarchy and constraint builder validated against the legacy
  `Cone_merged.json` structure;
- explicit vertex-bone attachment builder with final bone indices known before
  serialization;
- in-memory multi-attachment composition with cumulative non-overlapping bone
  ranges;
- no intermediate segment JSON merge and no post-serialization weighted-index
  remap;
- structural Spine golden fingerprint.

### Architecture guards

- `domain/` and `application/` remain independent from `bpy` and `bmesh`;
- geometry domain also forbids random-number dependencies;
- UV and bake operators are confined to dedicated adapter helpers;
- Blender operators are forbidden inside geometry/material traversal loops;
- GitHub Actions stores full pytest logs as artifacts for both Python 3.10 and
  Python 3.11.

## Modifier policy currently supported

- `STRICT_PRESERVE`: deformation-only stacks where every source element survives
  exactly once;
- `ALLOW_SOURCE_DUPLICATION`: source vertices/faces/corners may repeat, permitting
  modifier behaviour such as Mirror or Triangulate when Blender preserves their
  lineage;
- generated edges are allowed and represented with `MeshEdge.source_id = None`;
- generated vertices, faces, and corners are rejected because exact source-loop UV
  correspondence cannot be proven.

## Correct shared-texture order

The A1 pipeline must not unwrap each segment independently. The required order is:

1. read/evaluate one complete source snapshot;
2. calculate segmentation and deterministic decomposition;
3. mark every internal region cut as a seam on one complete texturing snapshot;
4. unwrap and bake that complete snapshot once;
5. transfer the resulting UV layer to each triangulated region through
   `SourceLoopId`;
6. compose all region attachments in one `SpineDocument`.

This preserves the existing shared baked texture contract while removing object-name
search, coordinate tolerances and JSON merging.

## Validation boundary

- the pure domain and fake-Blender test matrix targets Python 3.10 and Python 3.11;
- real modifier attribute propagation, UV operators, Cycles baking, datablock
  cleanup and context restoration still require Blender 4.4 headless fixtures;
- the rewrite remains disabled in production until those tests and representative
  legacy golden `.blend` fixtures pass.

## Production path

The legacy `main.save_uv_as_json()` path remains active. No UI operator has been
switched to the rewrite and no addon version has been bumped.

## Next architectural slices

1. keep the expanded pure/fake test suite green and inspect saved CI artifacts;
2. add Blender 4.4 headless fixtures for source reading, modifiers, shared unwrap,
   baking and cleanup failures;
3. implement the Blender-facing single-object orchestration service using the
   existing application stages;
4. compare complete output against representative v0.23 golden exports;
5. extend the same in-memory model to connected and multi-object export;
6. migrate operators while preserving public IDs and important Scene properties;
7. remove legacy orchestration only after parity is proven.
