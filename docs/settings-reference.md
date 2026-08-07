# Settings Reference

This document describes the maintained public settings in Blender to Spine2D Mesh Exporter
**0.129.0**. Fresh-Scene defaults are listed below. The current Scene settings schema is
**8**.

## Export

### Export Mode

| Value | Default | Behavior |
| --- | --- | --- |
| Normal / UV Segments | Yes | Segments source-derived topology, creates generated bake UV, bakes the material, and exports weighted mesh attachments. |
| Camera Projection | No | Renders through the active camera and exports a flat cropped screen-space mesh. |
| Depth Camera Projection | No | Renders through the active camera and exports a bounded visible depth-relief mesh with generated vertex bones and optional reserve views. |

Changing Export Mode invalidates cached readiness.

### Projection Direction

Visible only for **Normal / UV Segments**.

| Label | Persisted ID | Default | Contract |
| --- | --- | --- | --- |
| +X | `POSITIVE_X` | No | U = world +Y, V = world +Z, depth = world +X. |
| -X | `NEGATIVE_X` | No | U = world -Y, V = world +Z, depth = world -X. |
| +Y | `POSITIVE_Y` | No | U = world -X, V = world +Z, depth = world +Y. |
| -Y | `NEGATIVE_Y` | No | U = world +X, V = world +Z, depth = world -Y. |
| +Z | `POSITIVE_Z` | Yes | U = world +X, V = world +Y, depth = world +Z. |
| -Z | `NEGATIVE_Z` | No | U = world -X, V = world +Y, depth = world -Z. |
| Active Camera — Object Root Bone | `ACTIVE_CAMERA` | No | U/V come from the active camera; each object keeps its Blender Object Origin as the Spine main-bone pivot and retains per-depth deformation. |
| Active Camera — Camera Root Bone | `ACTIVE_CAMERA_CAMERA_ROOT` | No | Uses the same camera projection but camera-space zero owns the Spine main bone and the object is placed below one rigid camera-depth layer. |

Both Active Camera modes support evaluated Perspective and Orthographic cameras. They do
not become Camera Projection or Depth Camera Projection; they remain Normal / UV Segments
representations.

#### Object Root setup contract

`ACTIVE_CAMERA` resolves to the `CAMERA_VIEW_NORMAL` rig setup mode during document
preparation.

- X/Y setup rotation is neutral.
- Per-vertex camera-depth groups are preserved.
- Each depth group receives a generated `<group>_camera_setup` child.
- The child uses the inverse setup Y offset and owns generated vertex bones.
- The depth Transform constraint uses neutral setup translation/scale values.

The inverse child cancels the camera-depth setup translation while leaving the live depth
hierarchy available to X/Y pseudo-rotation.

#### Camera Root setup contract

`ACTIVE_CAMERA_CAMERA_ROOT` is normalized to the same active-camera geometry projection
but selects `PREPROJECTED_SCREEN` rig setup.

- camera-space zero owns `<prefix>_main`;
- the projected Object Origin is stored below the camera-relative hierarchy;
- one rigid camera-depth group owns all attachment vertices;
- camera kind is carried as Perspective or Orthographic rig semantics.

### Spine Version

| UI value | Exact JSON version |
| --- | --- |
| Spine 3.8 | 3.8.99 |
| Spine 4.0 | 4.0.64 |
| Spine 4.1 | 4.1.24 |
| Spine 4.2 | 4.2.43 |
| Spine 4.3 | 4.3.23 |

Target/profile/composition compatibility is validated before expensive export work.

### JSON

Directory for the final Spine JSON. Blender-relative paths resolve through
`bpy.path.abspath`. Export requires a saved `.blend` and a writable destination.

### Images Subfolder

| Type | Default |
| --- | --- |
| Relative path | `images/` |

The path is normalized below the JSON output directory.

## Rig

### Rig Profile

The public UI uses **2-Axis Rotation + Scale**. The persisted profile property remains
available internally so saved files and explicit development composition paths can be
validated without silently rewriting data.

| Value | Persisted ID | Fresh Scene default |
| --- | --- | --- |
| 3-Axis Rotation | `LEGACY_ROTATABLE_MESH` | No |
| 2-Axis Rotation + Scale | `TWO_AXIS_ROTATION_SCALE` | Yes |

Public Normal / UV Segments Active Camera Camera Root requires the 2-Axis profile because
its rigid camera-relative setup is defined by that hierarchy.

### 2-Axis controls

```text
<prefix>_rotation_X
<prefix>_rotation_Y
<prefix>_scale
```

The single-object constraint phase order is:

```text
0  Rotation X Transform
1  IK
2  Uniform Scale Transform
3  Depth Scale Transform
4  Rotation Y Transform
```

The exact setup payload depends on signed-axis, Active Camera Object Root, Active Camera
Camera Root, or depth-surface ownership. See [Rig Profiles](rig-profiles.md).

### Control Icons

| Type | Default |
| --- | --- |
| Boolean | Disabled |

### Preview Animation

Adds the generated preview animation when enabled. It does not replace the setup-pose
contract or target-specific sequence encoding.

## Cut

### Seam Maker

| Value | Default | Behavior |
| --- | --- | --- |
| Auto | Yes | Uses deterministic angular segmentation. |
| Custom | No | Uses user-marked seams and disables angular splitting. |

Depth Camera Projection generates its own relief topology and does not use source seam
controls.

### Seed Angle Limit

| Type | Range | Default |
| --- | --- | --- |
| Integer degrees | 1 through 89 | 30 |

### Angular Mode

| Value | Default | Behavior |
| --- | --- | --- |
| Seed cone | Yes | Compares each candidate face normal with the segment seed normal. |
| Seed cone + local dihedral | No | Also limits the angle across each traversed shared edge. |

## Rendered camera settings

### Projection Alpha Threshold

Visible for Camera Projection and Depth Camera Projection.

| Type | Range | Default |
| --- | --- | --- |
| Float | 0.0 through 1.0 | `1 / 255` |

Pixels below the threshold do not contribute to usable render coverage or crop bounds.

## Depth Camera Projection

### Depth Base

The public policy is **Farthest Visible Point**.

```text
farthest retained visible surface -> rig offset 0
nearer retained points            -> non-negative offset toward the camera
```

The internal `OBJECT_ORIGIN` policy is not a public selector and fails closed unless its
geometry preconditions are satisfied.

### Depth Smoothing

| Type | Range | Default |
| --- | --- | --- |
| Float | 0.0 through 1.0 | 0.35 |

One edge-aware smoothing pass. Samples are not blended across resolved depth discontinuities.

### Depth Edge Threshold

| Type | Range | Default |
| --- | --- | --- |
| Float fraction | 0.0 through 1.0 | 0.08 |

The value is resolved against visible depth range. Candidate smoothing/triangulation does
not cross larger jumps.

### Depth Mesh Error (px)

| Type | Range | Default |
| --- | --- | --- |
| Float pixels | 0.25 through 128.0 | 4.0 |

Requested screen-space sampling spacing for the generated relief surface.

### Max Depth Points

| Type | Range | Default |
| --- | --- | --- |
| Integer | 4 through 4096 | 128 |

Hard limit for retained generated relief points. Exceeding the limit blocks export rather
than silently changing requested topology.

### Parallax Horizon Angle

Visible only for Depth Camera Projection.

| Type | Hard range | Soft range | Default |
| --- | --- | --- | --- |
| Rotation angle | 0° through 89° | 0° through 45° | 0° |

Blender displays degrees and persists radians.

- `0°` keeps FRONT-only output.
- A positive value traverses connected source faces by accumulated unsigned dihedral cost.
- Retained reserve faces are assigned to deterministic virtual directions.
- Each non-empty view owns a face-isolated render and crop.
- FRONT and reserve attachments share one generated rig.
- Reserve slots are emitted before FRONT.

## Bake

### Texture Size

| Type | Range | Default |
| --- | --- | --- |
| Even integer | 64 through 4096 | 1024 |

Controls semantic object-bake textures and rendered-camera targets. This is one Scene-level
setting shared by all objects in the current export request. It appears first in the
**Bake** foldout and no longer appears in **Paths and Spine 2D version**.

Changing Texture Size invalidates cached readiness because it changes bake/render output
resolution and camera-projection canvas dimensions.

### Frames and Start

For selected-object export each Mesh stores independent timing.

```text
Frames = 0  -> static texture at current frame
Frames > 0  -> Loop texture sequence for that object
```

Texture Size remains shared even though Frames and Start are per-object in selected-object
export.

### Sequence FPS Override

| Type | Range | Default |
| --- | --- | --- |
| Float | 0.0 through 1000.0 | 0.0 |

Zero uses Scene FPS. A positive value overrides exported playback timing.

### Rendered-camera scene influence

Camera Projection and Depth Camera Projection expose controls for supported shadow,
reflection/transmission, World, and render participation behavior. These settings affect
the camera-render transaction without allowing unrelated objects to become unintended
final attachments.

## Generated Materials

Policies:

```text
Require Source
Generate If Missing
Force Generated
```

Generated patterns are temporary. They do not modify source material graphs.

## Analysis

Analyze executes production preparation without final output commit. It reports structured
blockers/warnings and geometry, material, rig, attachment, camera, depth, sequence, and
modifier statistics.

Changing any relevant source state or exporter setting invalidates or stales the report.

## Saved Scene migration

Scene schema **8** preserves valid saved values and initializes missing Depth parallax data
with a safe `0°` default. Persisted `ACTIVE_CAMERA` continues to mean **Active Camera —
Object Root Bone**; the Camera Root mode uses the separate
`ACTIVE_CAMERA_CAMERA_ROOT` identifier.

`spine2d_texture_size` remains the same persisted Scene property in 0.129.0; only its visual
owner moved from the paths/version foldout to Bake, so no Scene migration is required for
this UI change.
