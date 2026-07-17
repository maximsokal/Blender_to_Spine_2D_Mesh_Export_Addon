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

### Blender UI and bridge contract

The Cut panel exposes:

- `Seed angle limit`;
- `Angular mode`;
- `Local edge angle limit` only for `SEED_CONE_AND_LOCAL_DIHEDRAL`.

`CUSTOM` seam mode disables angular splitting, so the angular controls are hidden
instead of showing settings that the pipeline will ignore.

The Blender bridge converts the RNA enum to `A1AngularMode` before the application
pipeline starts. Missing properties in older `.blend` files resolve to
`LEGACY_SEED_CONE`. In legacy mode, the stored RNA local limit is normalized to
`None`, keeping the internal `A1GeometryPreparationSettings` object identical to
the pre-feature default. The same typed geometry settings are used by single,
standalone multi, connected multi, and mixed exports.

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

Every `DiskRegionState` maintains its current:

- face set and cached ordered face tuple;
- minimum and maximum face indices;
- edge incidence counts;
- vertex set;
- boundary edge set;
- boundary vertex degrees;
- count of invalid edge incidences;
- count of invalid boundary degrees;
- Euler counts and revision.

`topology` is therefore assembled in constant time from cached counts. Applying one
validated face delta updates only the candidate edges and affected boundary vertices.
It does not iterate over all edges or vertices already in the region. Counting newly
introduced vertices also checks only the candidate face instead of copying the complete
region vertex set.

A candidate is accepted only when:

- its shared edges form one non-empty proper cyclic interval on the candidate;
- no already-internal edge is reused;
- there is no extra vertex-only contact;
- boundary degrees remain zero or two;
- Euler remains one;
- the boundary remains non-empty.

Two disk regions merge only when their common boundary is one connected open edge
path, they have no additional shared vertex, the resulting boundary degrees remain
manifold, and Euler remains one. Open-path degree is counted by distinct `EdgeId`
incidence, not by unique neighbouring vertices, so two parallel edges correctly form
a two-edge cycle and cannot be mistaken for one open interface. Manually constructed
states are also rejected when they share an internal edge outside the boundary path.

The growth frontier is updated only from the accepted face. Region-to-region shared
edge counts are built once and updated locally after each merge. Candidate ordering
and merge tie-breaking remain compatible with the previous deterministic algorithm.

### Validation matrix

The focused regression set covers:

- implicit default versus explicit `LEGACY_SEED_CONE` equality;
- a `0° -> +25° -> -25°` fold where legacy stays joined and hybrid cuts the
  `50°` local transition;
- exact deterministic ring, closed cube, and periodic torus partitions;
- candidate decisions compared with complete `analyse_face_region()` results;
- complete-analysis call counts limited to input and finalized regions;
- stale incremental deltas;
- open path, disconnected path, repeated edge, and parallel-edge interfaces;
- a no-full-scan mapping fixture proving `topology`, preview, and apply use only
  local access after state construction;
- real Blender Mesh normals and real Scene RNA register/unregister behavior in the
  existing Blender 4.4 headless integration script.

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
