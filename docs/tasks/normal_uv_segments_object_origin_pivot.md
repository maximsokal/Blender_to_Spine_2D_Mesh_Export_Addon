# Normal / UV Segments: preserve Blender Object Origin as the per-object Spine pivot

## Status

Approved for implementation.

## Target release

`0.55.0`

The release archive version must not be reused after a user-visible correction. Any later correction after `0.55.0` must increment the manifest version again according to Semantic Versioning.

## Scope

Implement Object Origin based depth placement for the public object-bake pipeline used by:

- `Normal` bake mode;
- `UV Segments` geometry export;
- `TWO_AXIS_ROTATION_SCALE` rig profile;
- single-object export;
- public standalone multi-object export.

The following are explicitly outside this implementation:

- Camera Projection;
- connected and mixed composition;
- `THREE_AXIS_ROTATION` runtime behavior;
- changes to Spine target codecs or constraint schedules unrelated to pivot placement.

`THREE_AXIS_ROTATION` must remain in the internal codebase and tests, but it must be hidden from the public Blender UI until its pivot behavior is implemented and manually validated.

## User-visible requirement

For every exported Blender Mesh object, the Spine bone `<prefix>_main` is the per-object pivot and must correspond to the Blender Object Origin.

The document-level `root` bone remains shared. In standalone multi-object export it cannot represent several object pivots simultaneously. Each object therefore owns its pivot through its own `<prefix>_main` child of the shared `root`.

Expected hierarchy:

```text
root
├── Cone_main
│   └── Cone
├── Cube_main
│   └── Cube
└── Sphere_main
    └── Sphere
```

## Coordinate contract

After Blender object transform normalization:

- mesh vertex coordinates are object-local;
- local `(0, 0, 0)` is the authored Blender Object Origin;
- object rotation, scale, mirror, and shear are baked into immutable snapshot geometry;
- only world translation remains in `snapshot.world_matrix`.

### XY placement

Existing behavior is retained:

```text
main_x_pixels = object_origin_world_x * uniform_scale
main_y_pixels = object_origin_world_y * uniform_scale
```

These values are written to `<prefix>_main`.

### Depth placement

For the approved object-bake path, local Blender Z is measured relative to Object Origin, therefore:

```text
depth_reference_z = 0.0
z_group_y_pixels = (canonical_local_z - depth_reference_z) * uniform_scale
```

which simplifies to:

```text
z_group_y_pixels = canonical_local_z * uniform_scale
```

The implementation must not use the current implicit normalization:

```text
canonical_local_z - minimum_z
```

It must also not use bounding-box center normalization:

```text
canonical_local_z - ((minimum_z + maximum_z) / 2)
```

Bounding-box center is not the Blender Object Origin when the user has assigned a custom pivot.

### World Z translation

Every standalone object has its own zero depth plane at its Blender Object Origin. World Z translation is not added to the Spine depth offsets in this implementation.

## Required examples

With `S = uniform_scale`:

### Origin inside the geometry

```text
Blender local Z groups: -2, 0, +3
Spine depth offsets:     -2S, 0, +3S
```

### Origin below all geometry

```text
Blender local Z groups: +1, +2, +4
Spine depth offsets:     +1S, +2S, +4S
```

### Origin above all geometry

```text
Blender local Z groups: -4, -2, -1
Spine depth offsets:     -4S, -2S, -1S
```

### No vertices on local Z=0

No artificial Z group is created. `<prefix>_main` still represents the pivot and all real depth bones may exist only above or only below it.

## Architecture

### Typed depth-reference policy

The legacy rig request must receive an explicit typed depth-reference policy. The rig planner must not infer the reference from the minimum group.

Minimum required modes:

```python
class LegacyZGroupOriginMode(str, Enum):
    MINIMUM_Z = "MINIMUM_Z"
    OBJECT_ORIGIN = "OBJECT_ORIGIN"
```

Semantics:

- `MINIMUM_Z`: preserve the existing compatibility behavior;
- `OBJECT_ORIGIN`: use local Blender Z=0 as the depth reference.

The field must be appended to immutable dataclasses to avoid breaking historical positional construction.

### Route selection

`OBJECT_ORIGIN` is selected only when all of the following are true:

- the path is object-bake, not Camera Projection;
- the selected rig is `TWO_AXIS_ROTATION_SCALE`;
- the operation is single-object or public standalone multi-object export.

All other current paths explicitly retain `MINIMUM_Z`.

### Height overrides

An explicit `height_real_pixels` override remains an already-resolved absolute Spine offset and is not re-referenced. This preserves current override semantics and prevents double translation.

## Planned files

### `Blender_to_Spine2D_Mesh_Exporter/domain/spine/legacy_rig_contracts.py`

- Add `LegacyZGroupOriginMode`.
- Append `z_group_origin_mode` to `LegacyRigBuildRequest` with compatibility default `MINIMUM_Z`.
- Validate the exact enum type.
- Export the enum through module and package facades.

### `Blender_to_Spine2D_Mesh_Exporter/domain/spine/legacy_rig_plan.py`

- Make `build_legacy_z_group_metadata()` resolve an explicit reference Z from the request policy.
- Preserve negative, zero, and positive offsets for `OBJECT_ORIGIN`.
- Normalize negative zero to `0.0`.
- Keep deterministic ordering and existing rounding only after offset calculation.
- Keep `height_real_pixels` overrides absolute.

### `Blender_to_Spine2D_Mesh_Exporter/blender_adapter/a1_document_preparation.py`

- Resolve the selected rig profile before constructing `LegacyRigBuildRequest`.
- Select `OBJECT_ORIGIN` only for non-camera `TWO_AXIS_ROTATION_SCALE` object-bake preparation.
- Select `MINIMUM_Z` for Camera Projection and every non-approved profile/path.
- Add diagnostic statistics for the effective origin mode.

### Public UI rig selection

Locate the public rig-profile EnumProperty/item builder and remove `THREE_AXIS_ROTATION` from the visible choices without deleting its internal enum, builders, codecs, or tests.

The public default must remain `TWO_AXIS_ROTATION_SCALE`.

### `Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml`

- Set version to `0.55.0` only in the implementation commit series.
- Update every release-contract test or release document that asserts the previous version.

## Setup-pose invariants

The implementation must preserve the final visible setup pose.

When Object Origin is changed in Blender without intentionally moving the visible geometry:

- `<prefix>_main` changes to the new pivot;
- depth-bone offsets change relative to that pivot;
- generated vertex-bone local coordinates compensate through the existing projection pipeline;
- final attachment world positions remain equivalent;
- the mesh must not gain deformation, shear, or an unintended setup-pose translation.

The implementation must not change:

- generated UV coordinates;
- triangles;
- hull;
- edges;
- texture paths;
- slot draw order;
- source lineage;
- attachment names;
- constraint ordering;
- target-version serialization semantics;
- connected scheduling;
- Camera Projection output.

## Tests

### Pure rig-plan tests

Verify `OBJECT_ORIGIN`:

```text
Z = (-2, 0, +3) -> (-2S, 0, +3S)
Z = (+1, +2)    -> (+1S, +2S)
Z = (-3, -1)    -> (-3S, -1S)
```

Verify:

- negative zero becomes `0.0`;
- groups stay deterministically sorted;
- duplicate Z validation remains unchanged;
- `MINIMUM_Z` retains existing behavior;
- `height_real_pixels` remains absolute;
- non-finite inputs fail closed.

### Object preparation tests

Verify effective mode selection:

- object-bake + `TWO_AXIS_ROTATION_SCALE` -> `OBJECT_ORIGIN`;
- Camera Projection + `TWO_AXIS_ROTATION_SCALE` -> `MINIMUM_Z`;
- internal `THREE_AXIS_ROTATION` preparation -> `MINIMUM_Z`.

### Single-object Blender headless tests

Create Mesh fixtures with Object Origin:

- inside the Z range;
- below the full Z range;
- above the full Z range;
- not coincident with any vertex Z plane.

Verify `<prefix>_main`, depth-bone setup positions, generated attachment setup positions, and absence of source-object mutation.

### Standalone multi-object Blender headless tests

Export at least three objects with different Object Origins and different local Z distributions. Verify each object independently owns:

- its `<prefix>_main` pivot;
- its own local zero depth plane;
- its own negative and/or positive depth offsets.

Verify the shared `root` remains neutral and no object inherits another object's depth reference.

### UI tests

Verify:

- `TWO_AXIS_ROTATION_SCALE` is visible and remains the public default;
- `THREE_AXIS_ROTATION` is absent from visible choices;
- persisted old scene values do not crash UI drawing or request capture;
- an unavailable historical value is safely normalized or reported without silently selecting connected behavior.

### Regression tests

Run focused and full suites for:

- Spine 3.8 two-axis order/cache safety;
- Spine 4.0, 4.1, 4.2, and 4.3 two-axis export;
- single-object export;
- standalone multi-object export;
- Camera Projection unchanged;
- package/release version contract.

## Acceptance criteria

The task is complete only when all statements are true:

1. `<prefix>_main` corresponds to Blender Object Origin in public Normal / UV Segments object-bake export.
2. Local Blender Z=0 is the depth reference for `TWO_AXIS_ROTATION_SCALE`.
3. Z-group bones can be generated below and above the pivot.
4. Geometry entirely above or below the pivot remains entirely on the corresponding side in Spine.
5. No artificial zero Z group is generated.
6. Single-object export preserves the pivot.
7. Every component in standalone multi-object export preserves its own pivot independently.
8. Camera Projection output remains on the compatibility policy and is unchanged.
9. `THREE_AXIS_ROTATION` is absent from the public UI but remains in internal code.
10. Spine target codecs and constraint schedules are unchanged.
11. Focused tests, full Python tests, Blender headless acceptance, and package validation pass.
12. Manifest and release contracts report version `0.55.0`.
13. Manual validation in Blender and at least one supported Spine Editor target confirms correct rotation and scaling around the authored Object Origin.

## Implementation order

1. Commit this approved task document without production changes.
2. Add typed depth-origin contracts and pure planner tests.
3. Apply route-specific policy in object document preparation.
4. Hide Three Axis from the public UI.
5. Add single and standalone multi-object headless regressions.
6. Run focused tests.
7. Run the full suite and runtime matrix.
8. Change manifest/release contract to `0.55.0`.
9. Build and validate the extension archive.
10. Perform manual Blender/Spine pivot validation before release.
