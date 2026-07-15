# Rewrite A1: segmentation, decomposition, and modifier lineage

## Seed-normal compatibility

The legacy angular pass compares every candidate face normal against the normal of
the segment seed face. It does not accumulate pairwise normal drift. The rewrite
preserves this rule with a strict comparison:

```text
angle(seed_normal, candidate_normal) < angle_limit
```

Unlike the public legacy function, the rewrite guarantees a true partition: every
face appears exactly once and no face can be revisited by a later seed.

## Deterministic complex-region decomposition

The historical code contains multiple incompatible hole strategies and a partially
implemented random k-means path. Exact reproduction is neither deterministic nor a
stable compatibility target.

The replacement algorithm:

1. detects complex regions from Euler characteristic and boundary components;
2. starts from the lowest unassigned face ID;
3. grows a connected region only while the union remains a manifold disk;
4. prefers candidates sharing the most edges with the current region;
5. merges adjacent regions when their union remains a manifold disk;
6. verifies complete, disjoint face coverage before returning a plan.

There is no random seed, time-dependent ordering, Blender object creation, or name
search. Non-manifold input is rejected until a separate explicit repair policy is
implemented.

## Evaluated modifier lineage

The original object and Mesh datablock are never stamped with custom attributes.
The Blender adapter creates an isolated temporary object and Mesh copy and writes
unique INT attributes on these domains:

| Domain | Meaning |
| --- | --- |
| POINT | source vertex index + 1 |
| EDGE | source edge index + 1 |
| FACE | source face index + 1 |
| CORNER | source face index + 1 |
| CORNER | source corner index + 1 |

Zero is reserved for unknown/generated lineage.

The dependency graph evaluates the copied modifier stack. The adapter reads the
propagated attributes from `evaluated_object.to_mesh()` and validates them before
constructing `MeshSnapshot`.

### STRICT_PRESERVE

Every source vertex, edge, face, and corner must appear exactly once. This profile
is suitable for deformation-only modifier stacks.

### ALLOW_SOURCE_DUPLICATION

Vertices, faces, and corners may repeat their source IDs. This permits modifier
behaviour similar to mirror copies or triangulation when Blender preserves source
attributes. Generated edges are allowed because `MeshEdge.source_id` is optional.
Generated vertices, faces, or corners are rejected because they cannot participate
in exact source-loop UV correspondence.

## Transaction cleanup

The adapter guarantees cleanup in `finally`:

- `evaluated_object.to_mesh_clear()`;
- removal of the temporary object;
- removal of the temporary Mesh datablock;
- unlinking and removal of the temporary collection.

No `bpy.ops`, selection mutation, active-object mutation, or mode switching is used.
