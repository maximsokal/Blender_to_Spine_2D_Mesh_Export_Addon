# Installation Guide

## Requirements

- Blender 5.2 or newer.
- A matching supported Spine Editor target: 3.8.99, 4.0.64, 4.1.24, 4.2.43, or 4.3.23.
- Windows is the currently tested desktop platform.
- A writable directory for JSON, textures, temporary stage files, backups, diagnostics, and logs.
- Enough memory and disk space for the selected texture resolution and sequence frame count.

Blender 4.x and Blender 5.0/5.1 are not supported. The minimum version is declared in `Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml` and checked before registration mutates Blender state.

Single-object and standalone multi-object output are available according to the target/profile capability matrix. Connected and mixed composition are supported only for Spine 4.2.43 and remain explicit development/API routes. Public selected-object export remains standalone-only.

## Install a release ZIP

1. Close every Blender process that is using an older build of the extension.
2. Open Blender 5.2 or newer.
3. Open **Edit > Preferences > Extensions**.
4. Open the Extensions menu and choose **Install from Disk**.
5. Select `blender_to_spine2d_mesh_exporter-0.80.0.zip`.
6. Enable **Blender to Spine2D Mesh Exporter**.
7. Open a 3D View, press `N`, and select the **Blender to Spine2D Mesh Exporter** tab.

Do not unpack the release ZIP. Its root must contain `blender_manifest.toml` and `__init__.py`.

## Update an existing installation

1. Remove or disable the old extension in **Preferences > Extensions**.
2. Close Blender completely.
3. Start Blender again.
4. Install `blender_to_spine2d_mesh_exporter-0.80.0.zip` through **Install from Disk**.
5. Reopen the project file.

Closing Blender prevents loaded Python modules and cached extension metadata from keeping the previous implementation active. Always export fresh JSON and textures after installing 0.80.0.

## Scene settings migration

Version 0.80.0 continues to use Scene settings schema 6. Raw persisted Scene ID-properties are captured before Rewrite RNA properties are registered, so newly bound defaults cannot hide values stored in an older `.blend` file.

Migration policy:

```text
Genuinely fresh Scene:
    Seam Maker = Auto
    Rig = 2-Axis Rotation + Scale
    Spine target = 4.2
    Projection direction = +Z
    Settings schema = 6

Saved Scene created before rig profiles:
    Seam Maker = Auto when required by its older schema
    Rig = 3-Axis Rotation compatibility profile
    Spine target = preserved valid value or 4.2
    Projection direction = preserved valid value or +Z
    Settings schema = 6

Saved schema-4 or newer Scene with explicit choices:
    Preserve the selected rig profile
    Preserve a valid Spine target or use 4.2
    Preserve a valid projection direction or use +Z
    Settings schema = 6
```

The current values can be inspected in Blender's Python Console:

```python
print(
    bpy.context.scene.spine2d_settings_schema_version,
    bpy.context.scene.spine2d_seam_maker_mode,
    bpy.context.scene.spine2d_rig_profile,
    bpy.context.scene.spine2d_target_spine_version,
    bpy.context.scene.spine2d_projection_direction,
)
```

Normal - UV Segments uses the selected signed axis or Active Camera projection route while retaining UV-segment meshes. The separate Camera Projection mode keeps its render, crop, contour, and flattening pipeline.

## Build locally

The repository uses Blender's official extension validator and builder through `tools/prepare_package.py`.

From the repository root on PowerShell:

```powershell
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"

if (-not (Test-Path -LiteralPath $Blender -PathType Leaf)) {
    throw "Blender executable not found: $Blender"
}

Remove-Item ".\dist" -Recurse -Force -ErrorAction SilentlyContinue

& .\.venv-tests\Scripts\python.exe `
    tools\prepare_package.py `
    --blender $Blender

if ($LASTEXITCODE -ne 0) {
    throw "Extension package build failed with exit code $LASTEXITCODE"
}
```

The expected archive is:

```text
dist/blender_to_spine2d_mesh_exporter-0.80.0.zip
```

The script:

1. resolves and validates the Blender executable;
2. checks that Blender is 5.2 or newer;
3. validates the source directory and manifest;
4. invokes Blender's official extension build command;
5. validates the physical ZIP, required root files, packaged manifest, and forbidden repository paths.

Optional arguments:

```text
--source-dir <directory-containing-__init__.py-and-blender_manifest.toml>
--output <output-archive.zip>
```

The executable may also be supplied through `BLENDER_EXECUTABLE` or found as `blender` on `PATH`.

## Validate manually

Validate the extension source directory:

```text
blender --command extension validate Blender_to_Spine2D_Mesh_Exporter
```

Validate the built ZIP:

```text
blender --command extension validate dist/blender_to_spine2d_mesh_exporter-0.80.0.zip
```

After installation, validate representative outputs in the exact selected Spine Editor version. For sequence exports, confirm frame order, Loop playback, texture paths, attachment topology, and expected object controls.

## Remove the extension

Remove it through **Preferences > Extensions** and restart Blender. The extension unregisters classes, Scene and Object RNA properties, handlers, cached readiness data, and preference classes through its normal lifecycle.

Existing custom properties stored in a `.blend` may remain serialized until the file is saved without them; they do not execute code after the extension is removed.

## Next steps

Continue with the [Usage Guide](usage.md), [Settings Reference](settings-reference.md), and [Testing and Release Validation](testing.md).
