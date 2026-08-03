# Installation Guide

## Requirements

- Blender 5.2 or newer.
- A matching supported Spine Editor target: 3.8.99, 4.0.64, 4.1.24, 4.2.43, or 4.3.23.
- Windows is the currently tested desktop platform.
- A saved `.blend` file and a writable output directory.
- Enough memory and disk space for the selected texture resolution and frame count.

Blender 4.x and Blender 5.0/5.1 are not supported. The minimum version is declared in `Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml`.

## Install the release archive

1. Close Blender processes that use an older build.
2. Open Blender 5.2 or newer.
3. Open **Edit > Preferences > Extensions**.
4. Choose **Install from Disk**.
5. Select `blender_to_spine2d_mesh_exporter-0.81.0.zip`.
6. Enable **Blender to Spine2D Mesh Exporter**.
7. Open a 3D View, press `N`, and select the extension tab.

Do not unpack the archive. Its root must contain `blender_manifest.toml` and `__init__.py`.

## Update an existing installation

1. Disable or remove the old extension.
2. Close Blender completely.
3. Start Blender again.
4. Install `blender_to_spine2d_mesh_exporter-0.81.0.zip` through **Install from Disk**.
5. Reopen the project and run **Analyze** before export.

Closing Blender prevents loaded Python modules and cached extension metadata from keeping the previous implementation active.

## Export modes

Version 0.81.0 exposes three independent modes:

```text
Normal - UV Segments
Camera Projection
Depth Camera Projection
```

`Depth Camera Projection` requires an active Perspective or Orthographic camera. It evaluates the visible camera-facing surface, generates a bounded weighted relief mesh, renders the source material through the camera, and remaps the generated camera UVs into the final crop. It does not export camera animation.

The public depth base is **Farthest Visible Point**. The farthest retained point receives zero relief offset and all remaining retained points extend only toward the camera.

## Per-object sequence timing

For **Export Selected Objects**, every selected Mesh has independent `Frames` and `Start` values:

```text
Frames = 0
    Export one static texture evaluated at the current frame.

Frames > 0
    Export a Loop texture sequence for this object only.
    Start selects the first timeline frame.
```

Static siblings do not inherit sequence metadata or animation timelines.

## Scene settings migration

Version 0.81.0 uses Scene settings schema 7. Migration preserves existing valid export mode, rig profile, Spine target, projection direction, seam mode, material settings, paths, and per-object sequence timing.

Schema 7 initializes only missing depth fields:

```text
Depth base = Farthest Visible Point
Depth smoothing = production default
Depth edge threshold = production default
Depth mesh error = production default
Max depth points = production default
```

The hidden Object Origin depth policy remains an internal compatibility contract and is not selectable in the production UI.

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
dist/blender_to_spine2d_mesh_exporter-0.81.0.zip
```

Validate the built archive:

```powershell
& $Blender --command extension validate `
    dist\blender_to_spine2d_mesh_exporter-0.81.0.zip

if ($LASTEXITCODE -ne 0) {
    throw "Built extension validation failed"
}
```

## Manual validation

After installation:

1. Open a representative saved project.
2. Verify that the three export modes appear.
3. Select `Depth Camera Projection` and an active camera.
4. Export a static object and a two-frame material sequence.
5. Confirm that JSON and PNG files are produced.
6. Import the output into the exact selected Spine version.
7. Move the generated X/Y/Scale controls and confirm that the relief deformation remains stable.

Continue with the [Usage Guide](usage.md), [Settings Reference](settings-reference.md), and [Testing and Release Validation](testing.md).
