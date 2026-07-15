# Rewrite contract: A1 legacy-rig compatibility

The rewrite is a new implementation. Legacy modules are reference material and a
behavioural oracle; they are not architectural dependencies of the new pipeline.

## Compatibility target

The first production profile is `LEGACY_ROTATABLE_MESH` and targets Spine
`4.2.43`. For equivalent input and settings, the rewritten exporter must preserve:

- ordered bone names and parent relationships;
- ordered slot names and slot-to-bone references;
- IK and transform constraint names, targets, controlled bones and order values;
- skin, slot and attachment paths;
- weighted mesh bone-index semantics;
- existing public Blender operator identifiers.

Numeric geometry, UV and animation parity is checked by dedicated golden fixtures.
Volatile skeleton metadata such as export hash and absolute image paths is excluded
from the structural compatibility fingerprint.

## Architecture boundary

Everything below `domain/` and `application/` is pure Python and must not import
`bpy` or `bmesh`. Blender data is converted to immutable snapshots by the
`blender_adapter` package. The domain produces a validated `SpineDocument`, and
only `SpineSerializer` converts that model to dictionaries and JSON.

Local mesh identifiers are snapshot-specific. Stable source-lineage identifiers
point back to the original Blender mesh and are preserved through copying,
segmentation and topology-preserving transformations. UV correspondence must use
`SourceLoopId`; rounded coordinates and nearest-point matching are not part of the
new architecture.

Segmentation is a pure deterministic operation. It returns a `SegmentationPlan`
containing source-face membership, boundary-edge reasons and topology reports.
The domain layer never creates Blender objects, writes custom properties or uses
random clustering. The current implementation matches the legacy strict threshold
rule (`angle < angle_limit` joins faces), but full parity with the legacy
seed-normal traversal and complex-segment decomposition remains a golden-test
requirement.

## Implemented foundation

The current foundation contains:

- immutable `ExportSettings`, `ExportRequest`, `ExportIssue` and `ExportResult`;
- typed Spine document entities;
- weighted vertex stream codec;
- Spine cross-reference and mesh validator;
- deterministic Spine serializer;
- centralized legacy rig naming profile;
- legacy structural fingerprinting for v0.23/new-engine comparisons;
- immutable `MeshSnapshot` with local and source IDs;
- mesh topology and lineage validation;
- exact SourceLoopId-based UV correspondence;
- deterministic face-subset extraction for segments;
- geometry fingerprinting for golden fixtures;
- a read-only Blender source-mesh adapter using direct RNA access;
- deterministic segmentation by seams, sharp edges, materials, face angle, UV
  discontinuity, mesh boundaries and non-manifold edges;
- per-segment Euler characteristic, boundary-component and manifold reports.

## Not implemented yet

- full A1 segmentation parity for legacy seed-normal grouping;
- deterministic complex-segment decomposition replacing random k-means;
- evaluated modifier lineage propagation;
- UV unwrap adapter and transactional UV write-back;
- baking transaction;
- A1 rig builder connected to `SpineDocument`;
- production operator integration.

The legacy export path remains the active production path until golden and real
Blender headless tests prove parity.
