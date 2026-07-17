# Rewrite A1: segmentation, decomposition, and modifier lineage

## Seed-normal compatibility

The legacy angular pass compares every candidate face normal against the normal of
the segment seed face. It does not accumulate pairwise normal drift. The rewrite
preserves this rule as the default `LEGACY_SEED_CONE` mode with a strict comparison:

```text
angle(seed_normal, candidate_normal) < angle_limit
```

Unlike the public legacy function, the rewrite guarantees a true partition: every
face appears exactly once and no face can be revisited by a later seed.

### Optional local-dihedral guard

`SEED_CONE_AND_LOCAL_DIHEDRAL` keeps the same seed-cone requirement and adds a
second check for every traversed adjacency edge:

```text
angle(seed_normal, candidate_normal) < angle_limit
AND
angle(current_normal, candidate_normal) < local_angle_limit
```

When no separate local limit is supplied, `angle_limit` is reused. Rejecting one
edge does not permanently reject or queue the candidate face; another already
accepted neighbour may still reach it through a locally smoother edge.

The mode is opt-in through `A1GeometryPreparationSettings`. The default remains
`LEGACY_SEED_CONE`, so existing A1 segmentation results do not change.

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

### Incremental disk topology

A complete `analyse_face_region()` scan is retained for the original segment and
for every finalized output region. It is no longer executed for every growth or
merge candidate.

One `DiskTopologyIndex` is built per immutable `MeshSnapshot` and caches:

- `edge -> linked faces`;
- `face -> ordered edges`;
- `face -> ordered vertices`;
- the immutable edge map.

Every `DiskRegionState` then maintains only its current:

- face set;
- edge incidence counts;
- vertex set;
- boundary edge set;
- boundary vertex degrees;
- Euler counts.

Adding one candidate touches only the candidate corners and affected boundary
vertices. A candidate is accepted only when:

- its shared edges form one non-empty proper cyclic interval on the candidate;
- no already-internal edge is reused;
- there is no extra vertex-only contact;
- boundary degrees remain zero or two;
- Euler remains one;
- the boundary remains non-empty.

Two disk regions merge only when their common boundary is one connected open edge
path, they have no additional shared vertex, the resulting boundary degrees remain
manifold, and Euler remains one.

The growth frontier is updated only from the accepted face. Region-to-region shared
edge counts are built once and updated locally after each merge. Candidate ordering
and merge tie-breaking remain compatible with the previous deterministic algorithm.

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
