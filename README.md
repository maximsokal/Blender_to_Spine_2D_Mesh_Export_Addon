# Blender to Spine2D Mesh Exporter

<p align="center">
  <img src="assets/cover.png" alt="Blender to Spine2D Mesh Exporter cover" width="600">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License">
  <img src="https://img.shields.io/github/v/release/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon" alt="Latest Release">
  <img src="https://img.shields.io/github/downloads/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon/total" alt="Total Downloads">
  <img src="https://img.shields.io/badge/Blender-5.2%2B-orange?logo=blender" alt="Blender 5.2 or newer">
  <a href="https://patreon.com/MaximSokolenko">
    <img src="https://img.shields.io/badge/Support-Patreon-orange.svg" alt="Support on Patreon">
  </a>
</p>

<div align="center">
  <a href="https://www.youtube.com/watch?v=f_1Zc2qCz44">
    <img src="https://img.youtube.com/vi/f_1Zc2qCz44/maxresdefault.jpg"
         alt="Blender to Spine2D Mesh Exporter demo"
         style="width:100%;max-width:600px;">
  </a>
  <p><strong>Click to watch the video</strong></p>
</div>

Blender to Spine2D Mesh Exporter converts Blender Mesh objects into Spine JSON, weighted
mesh attachments, baked or rendered textures, generated control rigs, and optional texture
sequences.

Current extension version: **0.128.0**.

## Requirements

- Blender 5.2 or newer.
- Windows is the currently tested desktop platform.
- A saved `.blend` file.
- A writable output directory.
- A matching supported Spine target:
  - Spine 3.8.99
  - Spine 4.0.64
  - Spine 4.1.24
  - Spine 4.2.43
  - Spine 4.3.23

Standalone single-object and standalone multi-object export use the target/profile
capability registry. Connected and mixed composition are supported only by the explicitly
allowed Spine 4.2 routes.

## Export modes

The public `Export Mode` selector contains three values.

### Normal / UV Segments

Preserves source-derived surface topology, splits it into valid Spine mesh regions, creates
a generated bake UV layout, bakes the source material, and exports weighted mesh
attachments.

Projection directions:

- `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`
- `Active Camera — Object Root Bone`
- `Active Camera — Camera Root Bone`

The two Active Camera choices use the same evaluated Perspective or Orthographic camera
projection and the same material-bake geometry. They differ only in generated Spine rig
ownership.

#### Active Camera — Object Root Bone

- each object keeps its own Blender Object Origin as the Spine main-bone pivot;
- camera-space depth remains per vertex/depth group;
- generated `*_camera_setup` inverse-setup bones cancel depth translation in the setup
  pose without destroying live depth deformation;
- the exported setup pose matches the active-camera projection without stretching the
  mesh.

#### Active Camera — Camera Root Bone

- camera-space zero owns the Spine main bone;
- the projected Blender Object Origin is stored below the camera-relative hierarchy;
- all attachment vertices use one rigid camera-depth layer;
- Perspective and Orthographic cameras retain their correct camera-layer behavior.

### Camera Projection

Renders through the active camera, computes stable alpha coverage and crop, creates a
screen-space contour mesh, and exports a flat camera-facing attachment.

### Depth Camera Projection

Renders through the active camera and builds a bounded visible depth-relief surface instead
of a flat contour. The generated surface uses weighted vertex bones and supports
Perspective and Orthographic cameras.

Public depth controls include:

- Depth smoothing
- Depth edge threshold
- Depth mesh error
- Max depth points
- Parallax Horizon Angle

`Parallax Horizon Angle = 0°` keeps the front-only result. A positive value may retain
connected surfaces around the visible horizon and generate reserve camera views,
face-isolated textures, and reserve attachments while sharing one generated rig.

## Rig

The public UI uses **2-Axis Rotation + Scale**. The persisted 3-Axis profile remains only
where required for compatibility and explicit internal composition paths.

The 2-Axis rig exposes:

```text
<prefix>_rotation_X
<prefix>_rotation_Y
<prefix>_scale
```

Normal / UV Segments signed-axis projection, Active Camera Object Root, Active Camera
Camera Root, and Depth Camera Projection select different setup/depth ownership policies
without silently changing the chosen export mode.

## Material and geometry behavior

- Normal / UV Segments keeps the original Mesh datablock topology.
- Active Blender modifiers that are not part of that topology are reported by Analyze.
- Apply or convert a modifier when its generated geometry must be present in Spine.
- Supported source materials use the audited object-bake path.
- Camera-dependent representations remain explicit Camera Projection or Depth Camera
  Projection choices; the exporter does not silently switch modes.
- Temporary Blender objects, meshes, images, materials, node trees, cameras, render state,
  selection, frame, and mode changes are restored or removed on every exit path.

## Texture sequences

Each selected Mesh owns independent `Frames` and `Start` settings.

```text
Frames = 0  -> one static texture at the current frame
Frames > 0  -> a Loop texture sequence for that object
```

Spine 3.8 and 4.0 use attachment-swap sequence encoding. Spine 4.1, 4.2, and 4.3 use
native sequence metadata/timelines where supported.

## Quick start

1. Install `blender_to_spine2d_mesh_exporter-0.128.0.zip` through **Edit > Preferences > Extensions > Install from Disk**.
2. Save the `.blend` file.
3. Select one or more Mesh objects in Object Mode.
4. Open **3D View > Sidebar > Blender to Spine2D Mesh Exporter**.
5. Choose an Export Mode.
6. For Normal / UV Segments, choose a Projection Direction.
7. Choose the exact Spine target.
8. Configure Cut and Bake settings.
9. Run **Analyze** and review diagnostics.
10. Export the current object or selected objects.
11. Import the generated JSON and textures into the matching Spine version.

## Interface

![Blender to Spine2D Mesh Exporter interface](assets/ui_addon.png)

## Output

Typical static single-object output:

```text
<ObjectName>_merged.json
images/<ObjectName>_Baked.png
```

Typical sequence output:

```text
images/<ObjectName>_Baked_0000.png
images/<ObjectName>_Baked_0001.png
```

Output is staged and committed atomically. Temporary `.spine2d-stage-*` and
`.spine2d-backup-*` files are transaction data, not Spine assets.

## Documentation

- [Documentation index](docs/README.md)
- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Settings reference](docs/settings-reference.md)
- [Rig profiles](docs/rig-profiles.md)
- [Architecture](docs/architecture.md)
- [Output format](docs/output-format.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Testing](docs/testing.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Examples](examples/examples.md)

The documentation describes the current product only. Historical release notes are kept
in Git history and tags rather than in the maintained documentation set.

## Build the extension

From the repository root:

```powershell
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$Source = ".\Blender_to_Spine2D_Mesh_Exporter"
$Output = ".\dist\blender_to_spine2d_mesh_exporter-0.128.0.zip"

New-Item -ItemType Directory -Force ".\dist" | Out-Null
& $Blender --command extension build --source-dir $Source --output-filepath $Output
if ($LASTEXITCODE -ne 0) { throw "Extension build failed" }

& $Blender --command extension validate $Output
if ($LASTEXITCODE -ne 0) { throw "Extension validation failed" }
```

## Project structure

```text
Blender_to_Spine2D_Mesh_Exporter/
  application/        Export orchestration and immutable use-case contracts
  blender_adapter/    Blender RNA, geometry, material, render, camera, and UI boundaries
  domain/             Blender-independent geometry, UV, baking, and Spine models
  infrastructure/     Registration, diagnostics, locking, tracing, and atomic output
  blender_manifest.toml
  __init__.py

docs/                 Maintained current documentation
examples/             Example projects and notes
tests/                Blender-independent regression suite
tests_bpy/            Real bpy regression suite
tools/                Packaging, audit, and validation tools
```

## License

Copyright (c) 2025-2026 Maxim Sokolenko.

Licensed under GNU GPL v3.0 or later. Spine Editor and Spine runtimes remain subject to
the applicable Esoteric Software licenses.
