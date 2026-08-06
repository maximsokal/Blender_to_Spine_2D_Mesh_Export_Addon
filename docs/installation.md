# Installation Guide

## Requirements

- Blender 5.2 or newer.
- A matching supported Spine Editor target: 3.8.99, 4.0.64, 4.1.24, 4.2.43, or 4.3.23.
- Windows is the currently tested desktop platform.
- A saved `.blend` file and a writable output directory.
- Enough memory and disk space for the selected texture resolution, frame count, and optional parallax reserve views.

Blender 4.x and Blender 5.0/5.1 are not supported. The minimum version is declared in `Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml`.

## Install the release archive

1. Close Blender processes that use an older build.
2. Open Blender 5.2 or newer.
3. Open **Edit > Preferences > Extensions**.
4. Choose **Install from Disk**.
5. Select `blender_to_spine2d_mesh_exporter-0.125.0.zip`.
6. Enable **Blender to Spine2D Mesh Exporter**.
7. Open a 3D View, press `N`, and select the extension tab.

Do not unpack the archive. Its root must contain `blender_manifest.toml` and `__init__.py`.

## Update an existing installation

1. Disable or remove the old extension.
2. Close Blender completely.
3. Start Blender again.
4. Install `blender_to_spine2d_mesh_exporter-0.125.0.zip` through **Install from Disk**.
5. Reopen the project and run **Analyze** before export.

Closing Blender prevents loaded Python modules and cached extension metadata from keeping the previous implementation active.

## Export modes

Version 0.125.0 exposes three independent modes:

```text
Normal - UV Segments
Camera Projection
Depth Camera Projection
```

`Depth Camera Projection` requires an active Perspective or Orthographic camera. It evaluates the camera-facing surface, generates a bounded weighted relief mesh, renders the source material through the camera, and remaps generated camera UVs into the final crop. It does not export camera animation.

The public depth base is **Farthest Visible Point**. The farthest retained point receives zero relief offset and all remaining retained points extend only toward the camera.

## Normal / UV Segments and modifiers

Normal / UV Segments exports the original Mesh datablock. Geometry generated only by active Blender modifiers is not included in the serialized Spine mesh.

Version 0.125.0 adds a red warning inside the Analysis foldout whenever the current Normal / UV Segments request contains active modifiers. The warning lists:

```text
Object name
Modifier name
Modifier type
Viewport/render state
```

For example, an unapplied Bevel modifier can make a coin look rounded in Blender while Spine receives the original sharp-edged mesh. Apply or convert the modifier before export when its generated geometry must appear in Spine.

The warning does not automatically apply modifiers. Topology-changing modifier evaluation requires stable vertex, loop, UV, region, and rig lineage and is therefore kept separate from the current original-mesh Normal route.

## Normal / UV Segments material baking

Audited view-dependent materials, including supported `BSDF_GLOSSY` graphs, can use the Cycles `COMBINED` object-bake route while preserving Normal / UV Segments topology. Conservatively traversed muted shader nodes remain advisory when every input was analyzed.

The real coin acceptance path is:

```text
NORMAL_UV_SEGMENTS
ORIGINAL geometry
OBJECT_BAKE
COMBINED
```

## Parallax reserve

`Parallax Horizon Angle` is available only for Depth Camera Projection.

```text
0°
    Preserve the established front-only behavior: one texture and one attachment.

Greater than 0°
    Retain connected surfaces around the visible horizon up to the accumulated
    unsigned dihedral-angle budget. Each retained reserve direction receives its
    own fitted virtual camera, face-isolated texture, crop, and mesh attachment.
```

Blender displays the setting in degrees and stores it in radians. The supported range is `0°` through `89°`; the default is `0°` and the UI soft maximum is `45°`.

Reserve attachments reuse the same generated vertex-bone rig as the front attachment. Reserve slots are serialized before the front slot so the front remains above them in Spine draw order. The source camera, source mesh, materials, UV state, selection, frame, and render state are restored after export.

## Per-object sequence timing

For **Export Selected Objects**, every selected Mesh has independent `Frames` and `Start` values:

```text
Frames = 0
    Export one static texture evaluated at the current frame.

Frames > 0
    Export a Loop texture sequence for this object only.
    Start selects the first timeline frame.
```

When parallax reserve is enabled, FRONT and every reserve view receive the same frame-task count, but each view owns its own stable alpha-union crop and image namespace. Static siblings do not inherit sequence metadata or animation timelines.

## Scene settings migration

Version 0.125.0 uses Scene settings schema 8. Migration preserves existing valid export mode, rig profile, Spine target, projection direction, seam mode, material settings, paths, per-object sequence timing, and all established Depth Camera Projection settings.

Schema 8 initializes only the missing parallax field:

```text
Parallax Horizon Angle = 0°
```

That default keeps saved files on the established front-only path until the user explicitly enables reserve coverage.

Current values can be inspected in Blender's Python Console:

```python
scene = bpy.context.scene
print(
    scene.spine2d_settings_schema_version,
    scene.spine2d_texture_export_mode,
    scene.spine2d_target_spine_version,
    scene.spine2d_rig_profile,
    scene.spine2d_depth_base_mode,
    scene.spine2d_depth_smoothing,
    scene.spine2d_depth_edge_threshold,
    scene.spine2d_depth_mesh_error_pixels,
    scene.spine2d_depth_max_points,
    scene.spine2d_depth_parallax_horizon_angle,
)
```

## Build locally

From the repository root in PowerShell:

```powershell
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$Python = ".\.venv-tests\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Blender -PathType Leaf)) {
    throw "Blender executable not found: $Blender"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}

Remove-Item ".\dist" -Recurse -Force -ErrorAction SilentlyContinue
& $Python tools\prepare_package.py --blender $Blender
if ($LASTEXITCODE -ne 0) {
    throw "Extension package build failed"
}
```

Expected archive:

```text
dist/blender_to_spine2d_mesh_exporter-0.125.0.zip
```

Validate the built archive:

```powershell
& $Blender --command extension validate `
    dist\blender_to_spine2d_mesh_exporter-0.125.0.zip

if ($LASTEXITCODE -ne 0) {
    throw "Built extension validation failed"
}
```

## Manual validation

After installation:

1. Open a representative saved project.
2. Verify that the three export modes appear.
3. Select a Normal / UV Segments object with an active modifier and confirm the Analysis foldout shows the modifier alert.
4. Apply or convert the modifier, rerun Analyze, and confirm the alert disappears.
5. Export the real coin asset in Normal / UV Segments and confirm a non-empty PNG plus Spine JSON.
6. Select `Depth Camera Projection` and an active Perspective camera.
7. Leave `Parallax Horizon Angle` at `0°`; export and confirm one texture and one attachment.
8. Set a positive angle on a folded test mesh; confirm separate FRONT and reserve textures and reserve-before-front slot order.
9. Repeat with an Orthographic camera.
10. Import the output into the exact selected Spine version.

Continue with the [Usage Guide](usage.md), [Settings Reference](settings-reference.md), and [Testing and Release Validation](testing.md).
