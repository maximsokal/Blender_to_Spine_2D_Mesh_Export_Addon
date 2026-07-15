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
`bpy` or `bmesh`. Blender data is converted to immutable snapshots by a future
`blender_adapter` package. The domain produces a validated `SpineDocument`, and
only `SpineSerializer` converts that model to dictionaries and JSON.

## First foundation slice

This slice introduces:

- immutable `ExportSettings`, `ExportRequest`, `ExportIssue` and `ExportResult`;
- typed Spine document entities;
- weighted vertex stream codec;
- cross-reference and mesh validator;
- deterministic serializer;
- centralized legacy rig naming profile;
- legacy structural fingerprinting for v0.23/new-engine comparisons.

It does not register Blender classes and does not alter the legacy export path.
