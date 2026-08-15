# Installation Guide

This guide describes Blender to Spine2D Mesh Exporter **0.151.0**.

## Requirements

- Blender 5.2 or newer.
- Windows is the currently tested desktop platform.
- A saved `.blend` file.
- A writable export directory.
- A supported Spine schema family: 3.8, 4.0, 4.1, 4.2, or 4.3.

The extension defaults to exact project versions 3.8.99, 4.0.64, 4.1.24, 4.2.43, and
4.3.23. A different canonical `major.minor.patch` value can be configured for each family
in the add-on Preferences when a Spine project uses another patch release.

The minimum Blender version is declared in
`Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml`.

## Install

1. Close Blender processes using another build of the extension.
2. Open Blender 5.2 or newer.
3. Open **Edit > Preferences > Extensions**.
4. Choose **Install from Disk**.
5. Select `blender_to_spine2d_mesh_exporter-0.151.0.zip`.
6. Enable **Blender to Spine2D Mesh Exporter**.
7. In the add-on Preferences, expand **Spine project JSON versions** and set the exact
   Editor/project version for every Spine family you use.
8. Open a 3D View, press `N`, and select the extension tab.

Do not unpack the archive. Its root must contain `blender_manifest.toml` and `__init__.py`.

## Exact Spine project versions

The 3D View **Spine version** selector chooses the JSON schema family and codec. The five
exact patch values are global `AddonPreferences`, not per-`.blend` Scene settings. For
example, selecting Spine 4.2 with a configured project version `4.2.35` keeps the 4.2 codec
and writes `4.2.35` to `skeleton.spine` and the versioned JSON filename.

Only a canonical exact value from the matching family is accepted. `4.2.35` is valid for
the 4.2 field; `4.1.24` is not. The viewport **Exact JSON version** label shows the effective
value that will be exported.

These values use Blender's normal Preferences persistence. The extension intentionally does
not force-save all Blender preferences while a version field is being edited. With Blender's
normal preference-saving behavior enabled they persist across restarts; otherwise save
Preferences through Blender before closing.

## Update an installed development build

1. Disable/remove the currently installed extension.
2. Close Blender completely.
3. Start Blender again.
4. Install the new ZIP through **Install from Disk**.
5. Reopen the project.
6. Verify the exact project versions in Add-on Preferences.
7. Run **Analyze** before exporting.

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

The **Paths and Spine 2D version** foldout contains the Spine schema-family target, effective
**Exact JSON version**, and output paths. The scene-wide **Texture size** control is located
in **Bake**, before the frame/sequence controls. Moving the control does not change the
persisted RNA property or saved value.

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
same Scene property; only its UI location changed to the Bake foldout. Exact Spine project
versions are Add-on Preferences and therefore do not change the Scene settings schema.

## Build locally

From the repository root in PowerShell:

```powershell
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$SourceDir = ".\Blender_to_Spine2D_Mesh_Exporter"
$DistDir = ".\dist"
$Archive = Join-Path $DistDir "blender_to_spine2d_mesh_exporter-0.151.0.zip"

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
dist/blender_to_spine2d_mesh_exporter-0.151.0.zip
```

## After installation

Use a representative project and verify:

1. `Texture size` appears in Bake and not in Paths and Spine 2D version.
2. Add-on Preferences show one exact project version field for each supported Spine family.
3. Changing the active family's exact patch immediately changes the viewport **Exact JSON version** label and invalidates Analyze.
4. Analyze completes without unexpected blockers after the configured exact version is valid.
5. Signed-axis Normal export imports correctly.
6. Active Camera Object Root matches the Blender camera setup pose without stretching.
7. Object Root X/Y controls pivot around the projected Blender Object Origin.
8. Active Camera Camera Root keeps correct camera-relative placement.
9. Camera Projection produces a flat camera-facing attachment.
10. Depth Camera Projection produces the expected FRONT relief and optional reserve views.
11. The JSON filename and `skeleton.spine` use the configured exact patch for the selected family.

Continue with [Usage](usage.md), [Settings Reference](settings-reference.md), and
[Testing](testing.md).