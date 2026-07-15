# Rewrite A1: immutable mesh snapshots and source lineage

## Purpose

The legacy exporter correlates segment and texture-copy UV data through rounded
3D positions, face IDs, island IDs and progressively relaxed tolerances. This is
not a stable identity system: duplicated coordinates, seams, mirrored geometry
and topology changes can select the wrong loop.

The rewritten pipeline uses two independent identifier families.

## Local identifiers

`VertexId`, `EdgeId`, `LoopId` and `FaceId` identify elements inside one
`MeshSnapshot`. They are dense, ordered and unique only within that snapshot.
A segment is free to reindex its local elements without changing their origin.

## Source identifiers

`SourceVertexId`, `SourceEdgeId`, `SourceFaceId` and `SourceLoopId` describe the
lineage of an element in the original Blender mesh.

`SourceLoopId` is `(source_object_id, source_face_index, corner_index)`. The
face-local corner is the identity used for exact UV transfer. A derived mesh may
contain the same `SourceLoopId` more than once, for example when triangulation
reuses an n-gon corner in multiple triangles. The local `LoopId` remains unique.

Edges created by topology-changing operations may have no `SourceEdgeId`; this
is represented explicitly with `None` rather than a fabricated source index.

## Snapshot boundary

`MeshSnapshot` is immutable and contains no `bpy`, `bmesh`, RNA or live Blender
references. It stores:

- local and source IDs;
- vertex coordinates and normals;
- edge topology and seam/sharp state;
- face loops, material indices and normals;
- per-loop UV coordinates for named layers;
- the source object's world matrix.

`MeshSnapshotValidator` checks dense local IDs, all cross-references, face-edge
connectivity, source-object ownership, face/corner lineage and UV layer
consistency.

## Exact UV correspondence

`transfer_uv_by_source_loop()` builds a direct mapping:

```text
SourceLoopId -> baked UV
```

It does not inspect coordinates, nearest vertices, island ordering or decimal
rounding. Repeated source-loop IDs are accepted only when their UV values agree.
Conflicting values and missing source IDs are structured errors.

A non-strict transfer may preserve the target loop's existing UV while reporting
the missing source ID. It never silently treats that fallback as a successful
match.

## Blender adapter

`read_source_mesh_snapshot()` reads the original Mesh datablock through direct
RNA access. It does not call `bpy.ops`, change mode, modify selection, allocate a
BMesh or create temporary datablocks.

Evaluated modifier meshes are deliberately not handled by this adapter. A future
adapter will establish source attributes before dependency-graph evaluation and
verify whether each modifier preserved or generated lineage.
