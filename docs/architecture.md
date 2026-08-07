# Architecture

This document describes the production architecture of Blender to Spine2D Mesh Exporter
**0.129.0**.

## Package boundaries

```text
Blender_to_Spine2D_Mesh_Exporter/
  application/
  blender_adapter/
  domain/
  infrastructure/
```

### `application`

Owns export use cases, immutable settings/results, readiness, geometry-to-attachment
orchestration, composition, texture planning, and document assembly.

Application modules coordinate stages but do not own Blender RNA or low-level Blender
resource lifetime.

### `blender_adapter`

Owns every Blender-facing boundary:

- Scene and object RNA capture;
- source/evaluated Mesh access;
- temporary Mesh/Object creation;
- UV preparation;
- material graph inspection;
- semantic object baking;
- camera rendering and camera-space projection;
- render/camera/selection/frame restoration;
- UI routing and Scene migration.

Adapters may use `bpy` and `bmesh`. A BMesh created with `bmesh.new()` must be released
exactly once in `finally`. A BMesh borrowed through `bmesh.from_edit_mesh()` must never be
freed by the borrower.

### `domain`

Contains Blender-independent contracts and algorithms:

- source/local geometry identity and lineage;
- segmentation, topology, decomposition, triangulation;
- signed-axis projection bases;
- UV contracts;
- bake/camera/depth planning models;
- Spine bones, constraints, slots, attachments, skins, animation, validation, target
  adaptation, weighted streams, and serialization.

Domain modules do not import `bpy` or `bmesh`.

### `infrastructure`

Owns cross-cutting services:

- transactional registration;
- durable atomic output;
- interprocess locking;
- stale-stage/backup recovery;
- diagnostics, logging, tracing, and audit services.

## Public request flow

```text
Blender UI
-> capture immutable Scene/object settings
-> capability validation
-> readiness or export request
-> object preparation
-> Spine document assembly
-> target-specific adaptation
-> texture/JSON staging
-> staged validation
-> atomic commit
```

Later stages consume typed immutable settings rather than repeatedly reading mutable Scene
RNA.

## Ordered UI ownership

`ui.py` owns the reusable base panel controls and operators. `ui_layout.py` owns the actual
ordered production panel registered for successful extension startup and composes the
user-facing foldouts in this order:

```text
Paths and Spine 2D version
Rig
Rewrite Generated Materials
Cut
Bake
Analysis
```

Control placement is semantic rather than based on where the RNA property was originally
defined. In 0.129.0, the existing Scene property `spine2d_texture_size` is drawn by the
ordered **Bake** foldout before frame/sequence controls. It is not duplicated in **Paths
and Spine 2D version**. The RNA property, reset value, readiness dependency, and downstream
bake/render consumers remain unchanged.

## Object preparation

Each source object flows through:

```text
source geometry
-> UV preparation
-> texture planning
-> Spine document preparation
```

Stage errors preserve stage identity, object identity, warnings/statistics, and the original
exception cause.

## Normal / UV Segments

```text
source Mesh
-> source/evaluated capture and lineage
-> segmentation/decomposition
-> generated SpineBakeUV
-> projection into canonical U/V/depth
-> Z-group assignment
-> rig build
-> weighted per-region attachments
-> shared generated-vertex-bone optimization
-> material bake on unprojected source geometry
-> target-specific Spine document
```

Projection geometry and material-bake geometry are intentionally separate. Changing Normal
projection direction changes the Spine representation but does not rotate/reproject the
geometry used to evaluate the source material.

### Signed-axis projection

The six signed-axis modes use deterministic orthonormal bases from
`domain/projection.py`. U/V map to Spine X/Y and the selected depth axis owns generated
Z-group separation.

### Active Camera shared geometry stage

Both Active Camera Normal modes use the same evaluated camera frame and the same projected
snapshot. Perspective and Orthographic projection are resolved by the Blender camera
adapter before rig selection.

The projected snapshot contains:

- camera-projected U/V positions;
- camera-space per-vertex depth;
- projected Blender Object Origin;
- source identity/UV lineage.

Rig ownership is selected later during document preparation.

### Active Camera — Object Root Bone

Persisted projection ID: `ACTIVE_CAMERA`.

Setup mode: `CAMERA_VIEW_NORMAL`.

Contract:

```text
root
└── <prefix>_main                  projected Blender Object Origin
    └── <prefix>                   object-local base
        └── <prefix>_scale_rotate_X
            └── <prefix>_rotate_X
                ├── <prefix>_<z>_scale
                │   └── <prefix>_<z>
                │       └── <prefix>_<z>_camera_setup
                │           └── generated vertex bones for this depth
                └── ...
```

The camera-facing setup pose is already solved by projection, so X/Y setup rotation and
depth Transform setup values are neutral. Each `_camera_setup` child applies the inverse
of its depth-group setup Y translation. Vertex bones are parented below that child.

This produces two required properties simultaneously:

1. setup world XY equals the active-camera projection;
2. live depth separation remains available to X/Y pseudo-rotation around the object's
   projected Blender Object Origin.

This is intentionally not implemented by collapsing the depth-scale transform with a
setup `scaleX=-1`, because that mixes camera depth into setup XY and deforms camera-facing
meshes.

### Active Camera — Camera Root Bone

Persisted projection ID: `ACTIVE_CAMERA_CAMERA_ROOT`.

Application settings normalize geometry projection back to `ACTIVE_CAMERA` while selecting
setup mode `PREPROJECTED_SCREEN`.

Contract:

```text
root
└── <prefix>_main                  camera-space zero
    └── camera-relative transform/depth layer
        └── <prefix>               projected Blender Object Origin
            └── generated vertex bones
```

All attachment vertices bind through one rigid camera-depth group. Perspective and
Orthographic layer behavior is carried explicitly in the rig request. This mode reuses the
same camera-projected geometry and material-bake input as Object Root.

## Camera Projection

```text
active camera render tasks
-> alpha coverage union
-> stable crop
-> contour simplification
-> exact triangulation
-> flat screen-space attachment
-> target-specific document
```

Camera Projection is an explicit representation. It never silently replaces Normal / UV
Segments.

## Depth Camera Projection

```text
active camera visible surface
-> front-most depth sampling
-> edge-aware depth smoothing
-> bounded relief topology
-> depth-group/vertex-bone generation
-> FRONT render/crop
-> optional reserve-view renders/crops
-> weighted attachments
-> target-specific document
```

The public relief base is Farthest Visible Point. A positive Parallax Horizon Angle can
retain connected reserve surfaces by accumulated unsigned dihedral cost. FRONT and reserve
attachments use one union geometry/rig where source identity is shared.

Temporary virtual cameras and render proxies are isolated Blender resources and are removed
on success and failure.

## Attachment projection

Blender UV identity belongs to loops. The attachment projector therefore preserves
`(SourceVertexId, UV)` identity instead of assuming one UV per geometric vertex.

Final mesh attachments preserve:

- UV-specific attachment vertices;
- triangulation corner order;
- physical Spine hull semantics;
- explicit Z-group binding;
- deterministic generated vertex-bone ownership.

Setup-degenerate side geometry is retained for deformable rig modes because later control
movement can restore visible area.

## Shared generated vertex bones

Segmentation can duplicate the same source point across attachments. The optimizer shares
only generated component vertex bones whose final setup semantics are identical. Parent
identity is part of the key, so different depth groups never collapse together.

Only weighted bone indices are remapped. UVs, triangles, hull, edges, paths, local
influence positions, and weights remain unchanged.

## Target-specific Spine adaptation

Canonical rig/document construction happens before target adaptation. The target layer then
encodes the selected Spine version and any required bone-index or sequence representation
changes.

Supported standalone target versions are 3.8.99, 4.0.64, 4.1.24, 4.2.43, and 4.3.23.
Unsupported scope/profile/target combinations fail before expensive work.

## Multi-object composition

Public selected-object export creates standalone composition. Internal connected and mixed
composition routes are explicit and capability-gated.

Each object owns its preparation and generated vertex-bone optimization before outer
composition. Generated bones are never shared across unrelated object boundaries.

The outer request owns one atomic transaction for the final JSON and every texture.

## Readiness

Readiness executes production preparation without final commit and records immutable
blockers/warnings/statistics. Relevant source or settings changes stale or invalidate the
cached report.

Readiness diagnostics do not weaken export validation; production export still validates
all required contracts.

## Atomic output

Output files are reserved and staged before installation. Existing finals may be protected
with backups.

Required properties:

1. deterministic path reservation;
2. complete staged files before installation;
3. rollback/restoration on partial failure;
4. stale stage/backup recovery;
5. no deletion of work owned by another live process.

## Source-state integrity

Production export must not permanently change source topology, UVs, materials, transforms,
active object, selection, mode, renderer, frame, active camera, View Layer, or visibility
state outside the intended transaction.

Temporary Blender datablocks are removed on success and failure paths.

## Related documents

- [Usage](usage.md)
- [Settings Reference](settings-reference.md)
- [Rig Profiles](rig-profiles.md)
- [Output Format](output-format.md)
- [Testing](testing.md)
