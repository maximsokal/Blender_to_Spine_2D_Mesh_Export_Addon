# Settings Reference

This document describes the public Blender 5.2 Scene and object settings in extension
version **0.90.0**. Defaults apply to a genuinely new Scene. Saved files migrate to Scene
settings schema 8 without changing their selected export mode.

## Export

### Export mode

The selector contains exactly three values.

| Value | Default | Behavior |
| --- | --- | --- |
| Normal - UV Segments | Yes | Segments geometry, creates generated bake UV, bakes source surfaces, and exports region-based weighted Spine attachments. |
| Camera Projection | No | Renders through the active camera and exports a flat cropped screen-space mesh. |
| Depth Camera Projection | No | Renders through the active camera and exports an optimized visible depth-relief mesh with generated vertex bones and optional horizon reserve attachments. |

Changing Export mode invalidates cached readiness. Run Analyze again before export.

### Projection direction

Visible only for **Normal - UV Segments**.

| Label | Persisted ID | Default | U / V / depth contract |
| --- | --- | --- | --- |
| +X | `POSITIVE_X` | No | U = world +Y, V = world +Z, depth = world +X. |
| -X | `NEGATIVE_X` | No | U = world -Y, V = world +Z, depth = world -X. |
| +Y | `POSITIVE_Y` | No | U = world -X, V = world +Z, depth = world +Y. |
| -Y | `NEGATIVE_Y` | No | U = world +X, V = world +Z, depth = world -Y. |
| +Z | `POSITIVE_Z` | Yes | U = world +X, V = world +Y, depth = world +Z. |
| -Z | `NEGATIVE_Z` | No | U = world -X, V = world +Y, depth = world -Z. |
| Active Camera | `ACTIVE_CAMERA` | No | U/V come from the active camera frame and depth is camera-local Z. |

Active Camera in this selector remains a Normal / UV Segments route. It does not become
flat Camera Projection or Depth Camera Projection.

### Projection alpha threshold

Visible for Camera Projection and Depth Camera Projection.

| Type | Range | Default |
| --- | --- | --- |
| Float | 0.0 through 1.0 | `1 / 255` |

Pixels below the threshold are excluded from rendered-camera coverage and crop bounds.

### Depth base

Depth Camera Projection uses **Farthest Visible Point** publicly in 0.90.0.

```text
farthest visible surface → rig offset 0
nearer retained points   → non-negative offset toward the camera
```

The architecture also implements `OBJECT_ORIGIN`, but the property is hidden. It fails
closed unless Object Origin lies behind every visible point. The UI does not expose a
fourth mode or a public base selector.

### Depth smoothing

Visible only for Depth Camera Projection.

| Type | Range | Default |
| --- | --- | --- |
| Float | 0.0 through 1.0 | 0.35 |

Controls one edge-aware depth smoothing pass. Neighboring samples are blended only when
their depth difference does not exceed the resolved edge threshold.

### Depth edge threshold

| Type | Range | Default |
| --- | --- | --- |
| Float fraction | 0.0 through 1.0 | 0.08 |

The value is multiplied by the visible object depth range. Candidate triangles and
smoothing neighborhoods do not cross larger depth jumps. Values that disconnect every
candidate triangle block readiness/export instead of silently flattening the object.

### Depth mesh error (px)

| Type | Range | Default |
| --- | --- | --- |
| Float pixels | 0.25 through 128.0 | 4.0 |

This is the requested screen-space sample spacing for the generated relief lattice.
Smaller values retain more points until the hard point limit is reached.

### Max depth points

| Type | Range | Default |
| --- | --- | --- |
| Integer | 4 through 4096 | 128 |

Hard limit for retained depth points and their generated vertex bones. The source Blender
vertex count is not copied directly.

When a positive Parallax Horizon Angle causes the union of FRONT and reserve surfaces to
exceed this limit, readiness and export fail closed. The exporter does not silently remove
reserve faces, lower the requested horizon angle, or create a second independent rig.

### Parallax Horizon Angle

Visible only for Depth Camera Projection.

| Type | Hard range | Soft range | Default |
| --- | --- | --- | --- |
| Rotation angle | 0° through 89° | 0° through 45° | 0° |

Blender displays this setting in degrees because the RNA property uses the `ANGLE`
subtype and `ROTATION` unit. The persisted value and the domain contract use radians.

`0°` preserves the established front-only path: one camera texture, one weighted mesh
attachment, and no reserve camera plans.

For a positive value, the exporter performs deterministic Dijkstra traversal over
face adjacency. The path cost is the accumulated unsigned dihedral angle across shared
edges. Faces outside the active-camera visible set may be retained when their minimum
accumulated cost is within the requested horizon budget.

Retained reserve faces are assigned to deterministic virtual directions:

```text
RIGHT, UP_RIGHT, UP, UP_LEFT,
LEFT, DOWN_LEFT, DOWN, DOWN_RIGHT
```

Each non-empty direction owns:

- exact evaluated source-face indices;
- one fitted Perspective lens or Orthographic scale;
- one temporary face-isolated render proxy;
- one alpha-union crop across that view's sequence frames;
- one texture namespace and one weighted reserve attachment.

Reserve attachments reuse the same generated vertex-bone rig as the FRONT attachment.
Reserve slots are emitted before the FRONT slot so the FRONT remains above them in Spine
draw order. Shared hinge vertices keep shared generated bones.

### Texture size

| Type | Range | Default |
| --- | --- | --- |
| Even integer | 64 through 4096 | 1024 |

Controls object bake textures and rendered-camera targets.

### Spine version

| UI value | Exact JSON version |
| --- | --- |
| Spine 3.8 | 3.8.99 |
| Spine 4.0 | 4.0.64 |
| Spine 4.1 | 4.1.24 |
| Spine 4.2 | 4.2.43 |
| Spine 4.3 | 4.3.23 |

Standalone single- and multi-object capability is validated by the target/profile
registry. Connected and mixed composition remain limited to supported Spine 4.2 routes.

### JSON

Directory for the final Spine JSON. Blender-relative paths resolve through
`bpy.path.abspath`. Export requires a saved `.blend` and a writable destination.

### Images Subfolder

| Type | Default |
| --- | --- |
| Relative path | `images/` |

Backslashes become forward slashes. Leading `./` and surrounding slashes are removed.

## Rig

### Rig profile

The public Rewrite UI exports the 2-Axis Rotation + Scale profile. Historical 3-Axis
values remain persisted for compatibility and explicit development/API composition.

| Value | Persisted ID | Fresh Scene default |
| --- | --- | --- |
| 3-Axis Rotation | `LEGACY_ROTATABLE_MESH` | No |
| 2-Axis Rotation + Scale | `TWO_AXIS_ROTATION_SCALE` | Yes |

Normal / UV Segments and Depth Camera Projection both use the existing generated
vertex-bone pipeline. Flat Camera Projection retains its flat rendered contour attachment.

For public Depth Camera Projection with Farthest Visible Point, the existing Normal rig
uses `MINIMUM_Z` as the Z-group origin policy. This maps the farthest visible depth to
zero and keeps every generated group offset non-negative toward the camera.

Positive parallax does not create a second reserve rig. FRONT and reserve attachments are
subsets of one union MeshSnapshot and use one shared Z-group assignment and generated
vertex-bone namespace.

### 2-Axis controls

```text
<prefix>_rotation_X
<prefix>_rotation_Y
<prefix>_scale
<prefix>_main
```

The single-object constraint phase order is:

```text
0  Rotation X Transform
1  IK
2  Uniform Scale Transform
3  X Depth Scale Transform
4  Rotation Y Transform
```

`TWO_AXIS_ROTATION_SCALE` supports connected composition through a dedicated five-phase connected constraint schedule. Connected order values are assigned by ordered Z layer.
Objects in the same layer may intentionally share one phase order.

### Control icons

| Type | Default |
| --- | --- |
| Boolean | Disabled |

### Saved Scene migration

Scene schema 8 adds `spine2d_depth_parallax_horizon_angle` with a default of `0.0`
radians. Existing valid export mode, Spine target, seam mode, rig profile, depth quality
settings, paths, and per-object sequence timing remain unchanged.

The zero default is a compatibility boundary: a migrated Scene does not gain reserve
textures or attachments until the user explicitly chooses a positive angle.

## Cut

### Seam Maker

| Value | Default | Behavior |
| --- | --- | --- |
| Auto | Yes | Uses angular segmentation controls. |
| Custom | No | Uses user-marked seams and disables angular splitting controls. |

Depth Camera Projection creates its own generated relief topology. Its discontinuities are
controlled by Depth edge threshold, while horizon reserve growth is controlled by
Parallax Horizon Angle. The Cut foldout displays an explanatory message instead of source
seam controls for this mode.

### Seed angle limit

| Type | Range | Default |
| --- | --- | --- |
| Integer degrees | 1 through 89 | 30 |

### Angular mode

| Value | Default | Behavior |
| --- | --- | --- |
| Seed cone | Yes | Compares each candidate face normal with the segment seed normal. |
| Seed cone + local dihedral | No | Also limits the angle across each traversed shared edge. |

## Bake

### Frames and Start

For one active object, Scene `Frames` and `Start` are used. For selected-object export,
every Mesh stores its own values.

```text
Frames = 0  → one static texture at current frame
Frames > 0  → Loop texture sequence for that object only
```

This applies independently to Normal / UV Segments, Camera Projection, and Depth Camera
Projection. With positive parallax, each FRONT/reserve view receives the same frame count
but owns a separate stable crop and image sequence.

### Sequence FPS override

| Type | Range | Default |
| --- | --- | --- |
| Float | 0.0 through 1000.0 | 0.0 |

Zero uses Scene FPS. A positive value overrides playback timing.

### Rendered-camera scene influence

Visible for Camera Projection and Depth Camera Projection:

- Include shadows from scene objects;
- Include reflection/transmission objects;
- World affects lighting/reflections.

These settings affect render-ray participation without allowing unrelated objects to
become direct camera-visible output.

## Analysis

Analyze runs production preparation without writing files. Depth Camera Projection reports
retained depth points, visible and reserve source triangles, FRONT/reserve attachment
counts, virtual texture-view counts, maximum relief, weighted attachment statistics, and
structured blockers. Any geometry, material, camera, selection, frame, mode, target,
depth-setting, or Parallax Horizon Angle change makes the report stale.
