# Usage Guide

This guide describes Blender to Spine2D Mesh Exporter **0.128.0**.

## Open the exporter

1. Open Blender 5.2 or newer.
2. Save the `.blend` file.
3. Select at least one Mesh object.
4. Keep the active object in Object Mode.
5. Open **3D View > Sidebar > Blender to Spine2D Mesh Exporter**.

Run **Analyze** after changing selection, geometry, modifiers, UVs, seams, materials,
renderer, camera, frame settings, or exporter settings.

## Choose an export mode

### Normal / UV Segments

Use this mode when the exported object should remain a deformable Spine mesh.

```text
source Mesh
-> geometry capture and lineage validation
-> automatic or custom-seam segmentation
-> manifold disk decomposition
-> generated SpineBakeUV layout
-> projection into Spine XY/depth space
-> weighted mesh attachments
-> semantic texture bake
-> target-specific Spine JSON
```

The source material bake uses unprojected source geometry carrying the generated UV. The
selected projection changes the Spine representation, not the material-evaluation geometry.

### Camera Projection

Use this mode for a flat camera-facing representation.

```text
active camera render
-> sequence alpha-coverage union
-> stable crop
-> contour and triangulation
-> screen-space attachment
-> target-specific Spine JSON
```

### Depth Camera Projection

Use this mode for a camera-facing 2.5D relief representation.

```text
active camera visible surface
-> front-most depth sampling
-> edge-aware smoothing
-> bounded relief topology
-> generated weighted vertex bones
-> camera render and stable crop
-> crop-local UV remap
-> target-specific Spine JSON
```

The active camera may be Perspective or Orthographic.

## Normal / UV projection direction

Normal / UV Segments exposes eight projection choices.

### Signed world axes

```text
+X
-X
+Y
-Y
+Z
-Z
```

These use deterministic orthonormal world-axis projection bases. The selected axis defines
Spine X, Spine Y, and depth while preserving deformable per-depth rig ownership.

### Active Camera — Object Root Bone

Use this when the object must initially look exactly as seen by the active camera while its
own Blender Object Origin remains the animation pivot.

Behavior:

- Perspective and Orthographic cameras are supported.
- The projected Blender Object Origin becomes the object's Spine main-bone position.
- Camera-space depth is retained per exported depth group.
- X/Y setup rotation is neutral because the geometry is already camera-facing.
- Every depth group receives a generated `<group>_camera_setup` inverse-setup bone.
- Vertex bones are parented below the inverse-setup bone.
- The inverse setup cancels only the setup depth translation; live X/Y pseudo-rotation
  still sees the original depth separation.

This prevents camera-facing meshes from being stretched in their setup pose while keeping
the Blender Object Origin as the control pivot.

### Active Camera — Camera Root Bone

Use this when camera-space zero should own the generated Spine main bone.

Behavior:

- the same evaluated camera projection is used as Object Root;
- `main` is placed at camera-space zero;
- the projected Blender Object Origin is stored below the camera-relative hierarchy;
- all attachment vertices bind through one rigid camera-depth layer;
- Perspective and Orthographic camera-layer behavior remains explicit;
- material bake geometry is unchanged from Object Root.

The two Active Camera choices therefore differ in rig hierarchy and depth ownership, not
in the camera projection or baked material input.

## Configure cutting

### Seam Maker: Auto

Auto grows deterministic surface regions from angular limits.

- **Seed cone** compares candidate face normals with the segment seed normal.
- **Seed cone + local dihedral** also limits the angle across each shared edge.

### Seam Maker: Custom

Custom uses Blender edges marked as seams and disables angular splitting. The topology
pipeline may still decompose a seam-defined region when required to produce valid manifold
disk attachments.

Depth Camera Projection creates its own relief topology and does not use source seam
controls.

## Configure Depth Camera Projection

Public depth controls:

- **Depth smoothing** — edge-aware smoothing amount.
- **Depth edge threshold** — prevents smoothing/triangulation across large depth jumps.
- **Depth mesh error (px)** — requested screen-space relief sampling density.
- **Max depth points** — hard generated-point limit.
- **Parallax Horizon Angle** — optional reserve-surface traversal budget.

The public depth base is **Farthest Visible Point**. The farthest retained visible surface
has zero rig offset and nearer retained points extend toward the camera.

### Parallax Horizon Angle

`0°` keeps the front-only result.

A positive value can retain connected surfaces around the visible horizon using accumulated
unsigned dihedral cost. Retained reserve faces are assigned to deterministic virtual camera
directions. Each non-empty reserve view receives its own face-isolated render, crop,
texture namespace, and weighted attachment while sharing the generated rig with FRONT.

Reserve slots are emitted before the FRONT slot so FRONT remains above them in Spine draw
order.

If the union surface exceeds **Max depth points**, Analyze and Export fail instead of
silently dropping requested reserve geometry.

## Configure materials

Generated material policy:

- **Require Source** — source materials are required.
- **Generate If Missing** — generate a temporary material only when required data is absent.
- **Force Generated** — ignore source materials and use the generated pattern.

Generated material patterns are temporary and never modify the source material graph.

Normal / UV Segments exports the original Mesh datablock. Geometry created only by active
modifiers is not part of that topology. Analyze reports active ignored modifiers so the
viewport/Spine difference is visible. Apply or convert modifiers when their generated
geometry must be exported.

## Configure texture sequences

For each selected Mesh:

```text
Frames = 0  -> static output at the current frame
Frames > 0  -> Loop texture sequence for this object
```

`Start` selects the first source timeline frame. Selected objects keep independent timing.
Static siblings do not receive sequence metadata just because another selected object is
animated.

For Depth Camera Projection with reserve views, FRONT and reserve views use the same frame
tasks but keep independent stable crops and image namespaces.

## Choose the Spine target

Supported public target labels map to exact metadata versions:

```text
Spine 3.8 -> 3.8.99
Spine 4.0 -> 4.0.64
Spine 4.1 -> 4.1.24
Spine 4.2 -> 4.2.43
Spine 4.3 -> 4.3.23
```

Unsupported target/profile/composition combinations fail before expensive geometry or bake
work.

## Analyze

Analyze runs the production preparation path without final file commit. Review:

- blockers and warnings;
- source/exported geometry statistics;
- region and attachment counts;
- material and bake strategy;
- camera/depth statistics;
- sequence ownership;
- ignored modifier diagnostics.

A stale report should be regenerated after any relevant source or settings change.

## Export one object

1. Make the Mesh active.
2. Configure Export Mode and mode-specific settings.
3. Choose the Spine target.
4. Run Analyze.
5. Review diagnostics.
6. Run **Export Current Object**.

Typical output:

```text
<ObjectName>_merged.json
images/<ObjectName>_Baked.png
```

## Export selected objects

1. Select at least two Mesh objects.
2. Configure shared Scene settings.
3. Configure per-object Frames and Start values.
4. Run Analyze.
5. Run **Export Selected Objects**.

Public selected-object export is standalone composition. Connected and mixed composition
remain explicit internal/development routes and are validated only for their supported
capability combinations.

## Import into Spine

1. Keep the JSON/image relative directory relationship unchanged.
2. Open the exact selected Spine version.
3. Import the generated JSON.
4. Point Spine to the exported images directory when required.
5. Verify setup pose, slot order, UV placement, generated controls, and sequences.
6. For Active Camera Object Root, verify the setup pose matches the Blender camera view and
   rotating X/Y occurs around the projected Blender Object Origin.
7. For Active Camera Camera Root, verify the object remains correctly positioned below the
   camera-relative root.
8. For positive Depth parallax, verify reserve slots remain below FRONT.

## Reset settings

The main reset restores the current defaults, including:

- Normal / UV Segments;
- `+Z` projection;
- texture size 1024;
- Seam Maker Auto;
- current-frame static baking;
- Parallax Horizon Angle `0°`.

## Continue reading

- [Settings Reference](settings-reference.md)
- [Rig Profiles](rig-profiles.md)
- [Output Format](output-format.md)
- [Troubleshooting](troubleshooting.md)
