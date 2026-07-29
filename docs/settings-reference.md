# Settings Reference

This document describes the user-facing settings registered by the Blender 5.2 extension. Defaults below apply to a genuinely new Scene. Migration rules for saved Scenes are described in the Rig section.

## Main panel order

Every section uses the same boxed foldout style and is rendered in this exact order:

```text
Export
Rig
Rewrite Generated Materials
Cut
Bake
Export Readiness
```

Analysis results and the guarded export action are contained inside the final **Export Readiness** foldout. Rig and generated-material settings are no longer separate child panels below the export action.

## Export

### Export mode

| Value | Default | Behavior |
| --- | --- | --- |
| Normal - UV Segments | Yes | Segments geometry, creates generated bake UVs, bakes textures, and exports region-based Spine mesh attachments. |
| Camera Projection | No | Renders through the active camera and exports one screen-space projection attachment. |

Changing Export mode invalidates the cached readiness report and schedules a new analysis.

### Projection alpha threshold

Visible only in Camera Projection mode.

| Type | Range | Default |
| --- | --- | --- |
| Float | 0.0 through 1.0 | `1 / 255` |

Pixels below the configured threshold are excluded from the initial camera-projection coverage mask. Later coverage cleanup, crop, contour, and triangulation rules still apply.

### Texture size

| Type | Range | Default |
| --- | --- | --- |
| Even integer | 64 through 4096 | 1024 |

The value controls square semantic bake textures and camera-projection render targets used by the current UI pipeline. Larger values increase render time, memory use, and output size.

### JSON

Directory for final Spine JSON output. Blender-relative paths are resolved through `bpy.path.abspath`. When empty, the exporter uses the saved `.blend` directory through its default output resolver.

Export requires a saved `.blend` and a writable destination.

### Images Subfolder

| Type | Default |
| --- | --- |
| Relative path | `images/` |

Backslashes are normalized to forward slashes. Leading `./` and surrounding slashes are removed. An empty value resolves to `images`.

The final texture directory is below the JSON output directory.

### Connect

Per-object setting shown when multiple Mesh objects are selected.

| Type | Default |
| --- | --- |
| Boolean | Disabled |

At least two selected objects must have Connect enabled to create a connected subgroup. One connected object falls back to standalone composition with a warning.

`TWO_AXIS_ROTATION_SCALE` supports connected composition through a dedicated five-phase connected constraint schedule. The connected group and every object retain independent X, Y, and Scale controls; no Rotation Z control or synthetic sixth constraint is generated. Global and per-object phases are ordered as Rotation X, IK, Uniform Scale, X Depth Scale, and Rotation Y, with unique contiguous Spine constraint orders across the complete document.

## Rig

### Rig profile

| Value | Persisted ID | Fresh Scene default | Behavior |
| --- | --- | --- | --- |
| 3-Axis Rotation | `LEGACY_ROTATABLE_MESH` | No | Existing X/Y/Z compatibility rig. |
| 2-Axis Rotation + Scale | `TWO_AXIS_ROTATION_SCALE` | Yes | Generates X/Y pseudo-rotation controls and one independent uniform Scale control. No Rotation Z control is generated. |

Changing Rig profile invalidates cached readiness and schedules a new analysis because bone names, constraint order, weighted bone indices, control attachments, and preview animation change.

The two-axis profile follows the complete Spine 4.2.43 reference stored in [Rig Profiles](rig-profiles.md). Model-specific `BOX`, `TOP`, and `BOTTOM` names are not copied. They are generalized through the object prefix, ordered Z groups, and existing per-vertex bones.

### Single-object setup pose

A single-object export uses `NORMALIZED_SINGLE` setup policy:

```text
<prefix>_main.x = 0
<prefix>_main.y = 0
<prefix>_rotation_X.rotation = 0
<prefix>_rotation_Y.rotation = 0
```

The original object placement is transferred to the internal `<prefix>` base bone and to the control layout. This keeps the exported mesh in the same world-space position while giving the animator a neutral visible setup pose.

The reference X/Y setup angles are retained as transform-constraint rotation offsets. They are not discarded.

### Multi-object setup pose

Each standalone or connected object source uses `PRESERVE_COMPOSITION`: its existing `<prefix>_main` placement and reference X/Y setup rotations remain part of the object-local rig so composition cannot flatten or overlap the scene.

Connected composition then adds a separate neutral global wrapper. The wrapper controls have zero setup rotation. For the two-axis profile, every `<prefix>_scale` control is converted from root space to `<prefix>_main` local space before composition. This keeps the object, its control icons, and its constraint targets in one transform space while global layers move the complete object rig.

The selected object policy is explicit immutable data passed through UI settings into the rig build request. It is never inferred from object names or coordinate values.

### 2-Axis controls

The generated control set is:

```text
<prefix>_rotation_X
<prefix>_rotation_Y
<prefix>_scale
<prefix>_main
```

X, Y, and Scale controls share one editor X coordinate. Their Y positions are separated by one control length:

```text
Rotation X
    one control length
Rotation Y
    one control length
Scale
```

The Scale transform affects `<prefix>_rotate_X` and every Z-group rotation bone. Constraint evaluation uses the reference order:

```text
0  Rotation X Transform
1  IK
2  Uniform Scale Transform
3  X Depth Scale Transform
4  Rotation Y Transform
```

### Reset Rig Profile

The reset button beside the profile selector restores:

```text
2-Axis Rotation + Scale (TWO_AXIS_ROTATION_SCALE)
```

It does not modify texture, cutting, baking, material, or path settings.

### Control icons

| Type | Default |
| --- | --- |
| Boolean | Enabled |

The generated control attachments match the selected profile:

- 3-Axis: X, Y, Z, Main;
- 2-Axis + Scale: X, Y, Scale, Main.

### Preview animation

| Type | Default |
| --- | --- |
| Boolean | Enabled |

The preview matches the selected profile. The two-axis preview references only X, Y, and Scale controls and contains no Z timeline.

### Saved Scene migration

Schema 5 changes the default only for genuinely fresh Scenes:

- a new Scene with no persisted Rewrite settings receives `TWO_AXIS_ROTATION_SCALE`;
- a saved pre-profile project is assigned `LEGACY_ROTATABLE_MESH` for compatibility;
- a schema-4 Scene preserves whichever rig profile the user already selected;
- current schema values are never overwritten on registration or file loading.

## Rewrite Generated Materials

### Material Source

| Value | Default | Behavior |
| --- | --- | --- |
| Require Source | Yes | Missing required source material data blocks export. |
| Generate If Missing | No | Uses generated material only when required source material data is missing. |
| Force Generated | No | Ignores source materials and always uses the generated pattern. |

### Generated Pattern

| Value | Default | Behavior |
| --- | --- | --- |
| Solid Gray | Yes | Uses one opaque RGB color. |
| One Region - One Color | No | Assigns a deterministic color to each final region. |
| One Polygon - One Color | No | Assigns a deterministic color to each final triangulated exported polygon. |

### Generated Gray

| Type | Range | Default |
| --- | --- | --- |
| RGB color | Each channel 0.0 through 1.0 | `(0.5, 0.5, 0.5)` |

Generated output is always opaque; alpha is fixed to 1.0.

## Cut

### Seam Maker

| Value | Default | Behavior |
| --- | --- | --- |
| Auto | Yes | Uses angular segmentation controls. |
| Custom | No | Uses user-marked seams and disables angular splitting controls. |

Older development scenes are migrated once to the current Scene settings schema. Deliberate choices made after migration are preserved.

### Seed angle limit

| Type | Range | Default |
| --- | --- | --- |
| Integer degrees | 1 through 89 | 30 |

In Auto mode, a candidate face must satisfy the selected angular policy relative to the region seed.

### Angular mode

| Value | Default | Behavior |
| --- | --- | --- |
| Seed cone | Yes | Compares each candidate face normal with the segment seed normal. |
| Seed cone + local dihedral | No | Also limits the angle across each traversed shared edge. |

### Local edge angle limit

Visible only for **Seed cone + local dihedral**.

| Type | Range | Default |
| --- | --- | --- |
| Float degrees | 0 through 180 | 30.0 |

The value limits local face-to-face angle changes across shared edges while the seed-cone condition remains active.
