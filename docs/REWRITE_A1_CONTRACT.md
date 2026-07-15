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

## Determinism and intentional legacy divergence

A1 preserves externally observable rig and data contracts, not legacy bugs. The
new pipeline must never reproduce behaviour that can lose or duplicate geometry.
In particular:

- seed-normal angle grouping is preserved;
- strict `angle < angle_limit` semantics are preserved;
- a face may belong to exactly one segment;
- random k-means decomposition is replaced with deterministic manifold-disk
  decomposition;
- every decomposition must prove complete and disjoint face coverage;
- non-manifold input is rejected until an explicit repair policy exists.

The public legacy `plane_cut.py` contains an unfinished random partition function,
so byte-for-byte reproduction of that path is not a valid compatibility target.
Golden fixtures will compare stable geometry, UV, rig, and attachment outcomes.

## Modifier lineage

Modifiers are evaluated only on a temporary Object and Mesh copy. Source lineage is
encoded through unique temporary INT attributes on POINT, EDGE, FACE, and CORNER
domains.

Two policies are defined:

- `STRICT_PRESERVE`: every source element survives exactly once;
- `ALLOW_SOURCE_DUPLICATION`: source vertices, faces, and corners may repeat when
  Blender propagated their lineage unambiguously.

Generated edges are permitted because edge source identity is optional. Generated
vertices, faces, or corners are rejected because exact source-loop UV transfer
cannot be proven for them.

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

The geometry domain additionally forbids random-number dependencies. Ordering must
come from local/source IDs or explicitly documented deterministic scores.

## Implemented foundation

The current rewrite branch contains:

- immutable `ExportSettings`, `ExportRequest`, `ExportIssue`, and `ExportResult`;
- typed Spine document entities;
- weighted vertex stream codec;
- Spine cross-reference and mesh validators;
- deterministic Spine serializer;
- centralized legacy rig naming profile;
- structural legacy fingerprinting;
- immutable `MeshSnapshot` with local and source IDs;
- exact SourceLoopId-based UV correspondence;
- deterministic face-subset extraction and geometry fingerprints;
- read-only source-mesh and transactional evaluated-mesh Blender adapters;
- deterministic seed-normal segmentation;
- topology analysis and manifold-disk decomposition;
- evaluated modifier lineage validation and structured diagnostics.

## Not implemented yet

- UV unwrap adapter and transactional UV write-back;
- material and texture baking transaction;
- A1 rig builder connected to `SpineDocument`;
- golden parity against representative real `.blend` fixtures;
- production operator integration.

The legacy export path remains the active production path until golden and real
Blender headless tests prove parity.
