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
- exact propagation of one globally unwrapped UV layer to every triangulated export
  region;
- loop-level attachment projection with one Spine vertex per unique
  `(VertexId, UV)` pair;
- boundary UV seam duplicates remain consecutive hull vertices while internal seam
  duplicates remain after the physical hull;
- triangle and edge indices resolve through exact loops rather than rounded
  coordinates;
- pure A1 document assembly from ordered UV-ready regions;
- typed A1 single-object settings with explicit source geometry, modifier, UV,
  material, bake, sequence, rig and output policies;
- safe shared validation of relative image output directories for POSIX, Windows
  drive and UNC path forms.

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
- caller-owned bake staging API so PNG files can share one atomic transaction with
  the final Spine JSON;
- full single-object Blender service:
  read/evaluate -> Z groups -> geometry -> shared unwrap -> UV propagation ->
  material analysis -> bake plan -> rig -> document -> atomic PNG/JSON commit;
- structured stage-specific `ExportResult` failures without leaking Blender
  exceptions into the future operator layer;
- no original Object, Mesh, Material, selection, mode, frame or render-setting
  mutation on success or failure.

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
- structural Spine golden fingerprint;
- structured legacy/rewrite parity comparator with stable category paths;
- setup-transform and UV numeric tolerances;
- semantic weighted-stream comparison with exact bone-index checks;
- optional strict comparison for animations and nonessential mesh edges;
- machine-readable parity reports and deterministic CLI exit codes.

### Golden parity tooling

- `tools/compare_a1_exports.py` compares legacy and rewrite JSON files outside
  Blender;
- exit code `0` means compatible, `1` means incompatible, and `2` means invalid
  input;
- optional JSON report output contains every error and warning;
- volatile skeleton metadata is ignored only through explicit policy;
- `docs/REWRITE_A1_GOLDEN_PARITY.md` defines the real fixture record, procedure,
  minimum case matrix, and acceptance policy;
- malformed document sections and weighted streams become report issues instead of
  hiding later differences through fail-fast exceptions.

### Architecture guards

- `domain/` and `application/` remain independent from `bpy` and `bmesh`;
- geometry domain also forbids random-number dependencies;
- UV and bake operators are confined to dedicated adapter helpers;
- Blender operators are forbidden inside geometry/material traversal loops;
- GitHub Actions stores full pytest and Blender headless logs as artifacts.

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
6. split attachment vertices only where loop UV identity requires it;
7. compose all region attachments in one `SpineDocument`;
8. stage the JSON and every baked frame in one filesystem transaction;
9. commit all outputs together or restore every previous file.

This preserves the existing shared baked texture contract while removing object-name
search, coordinate tolerances and JSON merging.

## Validation status

- Python 3.10: 355 passed, 4 skipped;
- Python 3.11: 355 passed, 4 skipped;
- Blender 4.4 headless geometry/UV checks cover read-only mesh access, Smooth,
  Mirror, Solidify rejection, successful unwrap and forced Edit Mode failure;
- Blender 4.4 real Cycles checks cover successful EMIT baking and forced bake
  rollback;
- complete Blender 4.4 single-object checks cover valid PNG + Spine JSON output and
  joint preservation of existing PNG + JSON after forced Cycles failure;
- a folded two-face Blender fixture proves that a single export region can receive
  six loop-specific attachment vertices for four geometric vertices after Smart
  Project creates a UV seam;
- a pure center-fan fixture proves that internal UV duplicates remain after the
  physical hull;
- the parity gate is executed against a real Blender-generated JSON, accepts only
  policy-allowed metadata changes, and detects a deliberately corrupted weighted
  bone index;
- all temporary Object, Mesh, Collection, Material and Image datablocks are checked
  for leaks in the real Blender suite.

## Production path

The legacy `main.save_uv_as_json()` path remains active. No UI operator has been
switched to the rewrite and no addon version has been bumped.

## Remaining production blockers

1. representative real project `.blend` fixtures with their actual v0.23 JSON and
   image outputs;
2. accepted JSON and image parity reports for the documented fixture matrix;
3. connected and multi-object orchestration using the same in-memory document model;
4. UI/operator migration while preserving public IDs and important Scene properties;
5. legacy orchestration removal only after real-project parity is proven.
