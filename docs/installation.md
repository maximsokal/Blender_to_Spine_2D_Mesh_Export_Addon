# Installation Guide

This guide describes Blender to Spine2D Mesh Exporter **0.129.0**.

## Requirements

- Blender 5.2 or newer.
- Windows is the currently tested desktop platform.
- A saved `.blend` file.
- A writable export directory.
- A matching supported Spine target: 3.8.99, 4.0.64, 4.1.24, 4.2.43, or 4.3.23.

The minimum Blender version is declared in
`Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml`.

## Install

1. Close Blender processes using another build of the extension.
2. Open Blender 5.2 or newer.
3. Open **Edit > Preferences > Extensions**.
4. Choose **Install from Disk**.
5. Select `blender_to_spine2d_mesh_exporter-0.129.0.zip`.
6. Enable **Blender to Spine2D Mesh Exporter**.
7. Open a 3D View, press `N`, and select the extension tab.

Do not unpack the archive. Its root must contain `blender_manifest.toml` and `__init__.py`.

## Update an installed development build

1. Disable/remove the currently installed extension.
2. Close Blender completely.
3. Start Blender again.
4. Install the new ZIP through **Install from Disk**.
5. Reopen the project.
6. Run **Analyze** before exporting.

Closing Blender prevents loaded Python modules from making a new archive appear to behave
like an older installed build.

## Verify the installed UI

The public Export Mode selector must contain:

```text
Normal / UV Segments
Camera Projection
Depth Camera Projection
```

For Normal / UV Segments, Projection Direction must include:

```text
+X
-X
+Y
-Y
+Z
-Z
Active Camera — Object Root Bone
Active Camera — Camera Root Bone
```

`ACTIVE_CAMERA` remains the persisted ID for Object Root. Camera Root uses the separate
`ACTIVE_CAMERA_CAMERA_ROOT` ID.

The **Paths and Spine 2D version** foldout contains Spine target and output paths. The
scene-wide **Texture size** control is located in **Bake**, before the frame/sequence
controls. Moving the control does not change the persisted RNA property or saved value.

## Active Camera requirements

Both Normal / UV Active Camera modes require a valid active Perspective or Orthographic
camera.

- **Object Root Bone** keeps each Blender Object Origin as the Spine pivot and preserves
  per-depth deformation through generated inverse-setup bones.
- **Camera Root Bone** uses camera-space zero as the Spine main-bone pivot and one rigid
  camera-depth layer.

The two modes share the same projected geometry and material-bake input.

## Depth Camera Projection requirements

Depth Camera Projection requires an active Perspective or Orthographic camera and a
renderable source representation.

The public depth base is Farthest Visible Point. `Parallax Horizon Angle = 0°` keeps
front-only output. Positive angles can create reserve camera views/textures/attachments.

## Normal / UV modifiers

Normal / UV Segments exports the original Mesh datablock topology. Geometry created only by
active modifiers is not included automatically.

Analyze reports ignored active modifiers. Apply or convert a modifier when its generated
geometry must exist in the exported Spine mesh.

## Scene migration

Current Scene settings schema: **8**.

Migration preserves valid saved settings and initializes missing parallax data to `0°`.
Existing persisted `ACTIVE_CAMERA` values continue to select Active Camera Object Root;
Camera Root is opt-in through its separate persisted ID. `spine2d_texture_size` remains the
same Scene property; only its UI location changed to the Bake foldout.

## Build locally

From the repository root in PowerShell:

```powershell
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$SourceDir = ".\Blender_to_Spine2D_Mesh_Exporter"
$DistDir = ".\dist"
$Archive = Join-Path $DistDir "blender_to_spine2d_mesh_exporter-0.129.0.zip"

if (-not (Test-Path -LiteralPath $Blender -PathType Leaf)) {
    throw "Blender executable not found: $Blender"
}

New-Item -ItemType Directory -Force $DistDir | Out-Null
Remove-Item -LiteralPath $Archive -Force -ErrorAction SilentlyContinue

& $Blender `
    --command extension build `
    --source-dir $SourceDir `
    --output-filepath $Archive

if ($LASTEXITCODE -ne 0) {
    throw "Extension build failed"
}

& $Blender --command extension validate $Archive
if ($LASTEXITCODE -ne 0) {
    throw "Extension validation failed"
}

Get-FileHash -LiteralPath $Archive -Algorithm SHA256
```

Expected archive:

```text
dist/blender_to_spine2d_mesh_exporter-0.129.0.zip
```

## After installation

Use a representative project and verify:

1. `Texture size` appears in Bake and not in Paths and Spine 2D version.
2. Analyze completes without unexpected blockers.
3. Signed-axis Normal export imports correctly.
4. Active Camera Object Root matches the Blender camera setup pose without stretching.
5. Object Root X/Y controls pivot around the projected Blender Object Origin.
6. Active Camera Camera Root keeps correct camera-relative placement.
7. Camera Projection produces a flat camera-facing attachment.
8. Depth Camera Projection produces the expected FRONT relief and optional reserve views.
9. The generated JSON imports into the exact selected Spine target.

Continue with [Usage](usage.md), [Settings Reference](settings-reference.md), and
[Testing](testing.md).
