# Rig Profiles

This document describes the current generated Spine rig behavior in Blender to Spine2D
Mesh Exporter **0.128.0**.

Historical reference skeleton dumps are not maintained here. Executable fixtures and Git
history provide regression provenance; this document describes the current production
contract.

## Profiles

### 2-Axis Rotation + Scale

Persisted ID: `TWO_AXIS_ROTATION_SCALE`.

This is the public UI profile and the fresh-Scene default.

Visible controls:

```text
<prefix>_rotation_X
<prefix>_rotation_Y
<prefix>_scale
```

Single-object constraint evaluation order:

```text
0  X rotation Transform
1  IK
2  Uniform Scale Transform
3  Depth Scale Transform
4  Y rotation Transform
```

### 3-Axis Rotation

Persisted ID: `LEGACY_ROTATABLE_MESH`.

The profile remains available where required for persisted data and explicit supported
internal composition paths. It is not the ordinary fresh-Scene public UI profile.

## Setup-pose modes

Rig profile and setup-pose mode are separate typed decisions. The current two-axis builder
supports distinct topology/constraint behavior for ordinary model-space export,
active-camera Object Root, active-camera Camera Root, and depth-surface ownership.

Important setup modes include:

```text
NORMALIZED_SINGLE
PRESERVE_COMPOSITION
CAMERA_VIEW_NORMAL
CAMERA_DEPTH_SURFACE
PREPROJECTED_SCREEN
```

The route owner selects the mode; it is not inferred from names or coordinate values.

## Ordinary model-space hierarchy

Signed-axis Normal / UV Segments uses the standard per-depth hierarchy:

```text
root
├── <prefix>_main
│   ├── <prefix>
│   │   ├── <prefix>_scale_rotate_X
│   │   │   └── <prefix>_rotate_X
│   │   │       ├── <prefix>_1_scale
│   │   │       │   └── <prefix>_1
│   │   │       │       └── generated vertex bones for depth group 1
│   │   │       ├── <prefix>_2_scale
│   │   │       │   └── <prefix>_2
│   │   │       │       └── generated vertex bones for depth group 2
│   │   │       └── ...
│   │   ├── IK/helper chain
│   │   └── object-local scale/pivot ownership
│   ├── <prefix>_rotation_X
│   └── <prefix>_rotation_Y
└── <prefix>_scale
```

Depth groups use deterministic offsets derived from the source projection and configured
Z-group origin policy.

## Active Camera — Object Root Bone

Persisted projection ID: `ACTIVE_CAMERA`.

Rig setup mode: `CAMERA_VIEW_NORMAL`.

The camera projection has already solved the visible setup orientation. The rig must keep
that exact setup projection while retaining per-depth deformation for later X/Y control
movement.

Current hierarchy:

```text
root
└── <prefix>_main
    └── <prefix>
        └── <prefix>_scale_rotate_X
            └── <prefix>_rotate_X
                ├── <prefix>_1_scale
                │   └── <prefix>_1
                │       └── <prefix>_1_camera_setup
                │           └── generated vertex bones for group 1
                ├── <prefix>_2_scale
                │   └── <prefix>_2
                │       └── <prefix>_2_camera_setup
                │           └── generated vertex bones for group 2
                └── ...
```

Properties:

- `<prefix>_main` is positioned at the projected Blender Object Origin.
- X/Y Transform setup rotations are neutral.
- Depth Transform setup translation and scale are neutral.
- Every `*_camera_setup` child has the inverse setup Y offset of its depth group.
- Vertex bones bind below the `*_camera_setup` child.
- The parent depth pair remains live and full-rank for later pseudo-rotation.

Conceptually, setup evaluation is:

```text
projected object pivot
+ depth-group setup translation
+ inverse camera-setup translation
+ projected vertex XY
= projected object pivot + projected vertex XY
```

When the animator moves X/Y controls, the live depth parent can deform the vertex because
only the setup translation was cancelled; depth information was not deleted.

This is the required distinction between setup compensation and depth flattening.

## Active Camera — Camera Root Bone

Persisted projection ID: `ACTIVE_CAMERA_CAMERA_ROOT`.

The application normalizes camera geometry to the common Active Camera projection and
selects rig setup mode `PREPROJECTED_SCREEN`.

Hierarchy:

```text
root
└── <prefix>_main                  camera-space zero
    └── camera-relative X/Y/depth ownership
        └── one rigid depth group
            └── <prefix>           projected Blender Object Origin
                └── generated vertex bones
```

Properties:

- one camera-depth group;
- `main` at camera-space zero;
- projected Object Origin below the camera-relative layer;
- vertex bones parented to the object base below that layer;
- Perspective/Orthographic camera kind carried by the rig request;
- independent object scale does not change camera-relative placement.

## Shared camera geometry contract

Object Root and Camera Root do not perform separate camera projections. Both consume the
same evaluated active-camera projection, the same projected source geometry, the same UV
lineage, and the same unprojected material-bake geometry.

Only rig hierarchy and depth ownership differ.

This contract prevents camera-root selection from changing texture evaluation or projected
mesh shape.

## Depth Camera Projection rig ownership

Depth Camera Projection uses generated depth-surface placement rather than the Normal
camera-root policies. The route uses `CAMERA_DEPTH_SURFACE` where appropriate so already
solved relief placement is not re-applied as a second setup offset.

FRONT and optional reserve attachments share the generated rig when they belong to the same
Depth Camera Projection object.

## Generated vertex bones

Every attachment vertex is represented by one full-weight generated Spine bone before
optional sharing/compaction.

The parent is resolved by setup mode:

- signed-axis/model-space: the matching depth rotation bone;
- Active Camera Object Root: the matching `*_camera_setup` inverse-setup bone;
- Active Camera Camera Root: the object base below the rigid camera layer;
- other explicitly supported routes: their declared setup owner.

Weighted attachment local coordinates remain `(0, 0)` with weight `1` because the generated
vertex bone owns the setup XY position.

## Shared vertex-bone optimization

Equivalent generated vertex bones may be shared across segmented attachments when their
final setup semantics match. Parent identity is part of the semantic key.

Therefore:

- two identical XY vertices under the same setup parent may share one generated bone;
- two identical XY vertices under different depth groups or camera-setup parents may not.

Only weighted bone indices are remapped. Attachment UVs, triangles, hull, edges, local
influence coordinates, and weights are preserved.

## Validation requirements

Rig regression coverage must prove:

- deterministic bone and constraint order;
- finite numeric payloads;
- valid Spine references;
- exact target adaptation of weighted bone indices;
- signed-axis setup behavior remains unchanged;
- Active Camera Object Root uses one inverse setup child per depth group;
- Object Root vertex bones bind below those inverse children;
- Object Root setup world XY reconstructs the camera projection;
- Object Root keeps per-depth live deformation after setup cancellation;
- Camera Root uses one rigid camera-depth layer and camera-space-zero main placement;
- Object Root and Camera Root share projected geometry and material-bake input;
- Depth Camera Projection does not receive Normal Object Root compensation accidentally.

## Related documents

- [Architecture](architecture.md)
- [Usage](usage.md)
- [Settings Reference](settings-reference.md)
- [Testing](testing.md)
