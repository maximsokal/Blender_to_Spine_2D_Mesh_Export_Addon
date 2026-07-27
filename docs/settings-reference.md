# Settings Reference

This document describes the user-facing settings registered by the Blender 5.2 extension. Defaults are the values used by a new or reset Scene.

## Main panel: Export

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

### Control icons

| Type | Default |
| --- | --- |
| Boolean | Enabled |

Includes the compatibility control-icon structures in the final Spine document.

### Preview animation

| Type | Default |
| --- | --- |
| Boolean | Enabled |

Includes the compatibility preview animation in the final Spine document.

### Connect

Per-object setting shown when multiple Mesh objects are selected.

| Type | Default |
| --- | --- |
| Boolean | Disabled |

At least two selected objects must have Connect enabled to create a connected subgroup. One connected object falls back to standalone composition with a warning.

## Main panel: Cut

### Seam Maker

| Value | Default | Behavior |
| --- | --- | --- |
| Auto | Yes | Uses angular segmentation controls. |
| Custom | No | Uses user-marked seams and disables angular splitting controls. |

Older development scenes are migrated once to the current Scene settings schema with Auto selected. A deliberate Custom choice made after migration is preserved.

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

## Main panel: Bake

### Frames for render

| Type | Range | Default |
| --- | --- | --- |
| Integer | 0 or greater | 0 |

`0` exports the current frame only. A positive value creates one texture task per frame starting at Start frame.

### Start frame

| Type | Range | Default |
| --- | --- | --- |
| Integer | 0 or greater | 0 |

For one selected object, frame settings are stored on the Scene. For multi-object export, each selected object has its own frame settings.

### Last frame

Read-only UI value:

```text
Frames = 0: Start
Frames > 0: Start + Frames - 1
```

## Generated Materials panel

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

## Readiness analysis

### Analyze

Runs the production preparation pipeline without committing final export files. The report stores:

- overall READY, WARNING, BLOCKED, NOT_ANALYSED, or STALE state;
- object-level issues;
- geometry, topology, UV, material, texture, rig, and attachment statistics;
- blocker and warning counts.

Export requires a current report that allows export.

## Add-on Preferences: diagnostics

### Preserve failed work files

| Type | Default |
| --- | --- |
| Boolean | Disabled |

When enabled, failed `.spine2d-stage-*` files may be retained for manual inspection. Backup restoration still follows safety rules.

### Recover stale work files

| Type | Default |
| --- | --- |
| Boolean | Enabled |

Before reserving new outputs, the exporter may restore missing finals from stale backups and remove abandoned stage or backup files that are not owned by a live process.

## Add-on Preferences: logging

### Enable file logging

| Type | Default |
| --- | --- |
| Boolean | Disabled |

Enables the configured file handler in addition to normal Blender console logging.

### Log file path

File destination used when file logging is enabled.

### Filter modules

Filters the displayed per-file logger list in Add-on Preferences. It does not change active logger levels by itself.

### Per-file log level

Every discovered Python module has an independent level selector:

```text
ERROR
WARNING
INFO
DEBUG
```

### Refresh Module List

Rescans Python modules and preserves existing per-file levels where possible.

## Reset behavior

The main Reset operator restores:

```text
Export mode                  Normal - UV Segments
Texture size                 1024
Images Subfolder             images/
Control icons                enabled
Preview animation            enabled
Seed angle limit             30
Angular mode                 Seed cone
Local edge angle limit       30
Seam Maker                   Auto
Frames for render            0
Start frame                  0
```

The Generated Materials Reset operator restores:

```text
Material Source              Require Source
Generated Pattern            Solid Gray
Generated Gray               0.5, 0.5, 0.5
```

## Related documents

- [Usage](usage.md)
- [Output Format](output-format.md)
- [Troubleshooting](troubleshooting.md)