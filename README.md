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

Blender to Spine2D Mesh Exporter converts Blender Mesh objects into Spine-ready JSON,
weighted mesh attachments, baked or camera-rendered textures, generated animation controls,
and optional texture sequences.

Current extension version: **0.152.0**.

## Requirements

- Blender 5.2 or newer.
- Windows is the currently tested desktop platform.
- A saved `.blend` file.
- A writable output directory.
- A supported Spine schema family: 3.8, 4.0, 4.1, 4.2, or 4.3.
- The exact Spine Editor/project patch version configured for that family in Add-on Preferences.

The built-in default exact versions are 3.8.99, 4.0.64, 4.1.24, 4.2.43, and 4.3.23.
They are defaults, not hard limits: the user can configure another canonical
`major.minor.patch` value inside the selected family, for example `4.2.35` while continuing
to use the Spine 4.2 codec.

Public single-object and selected-object export are standalone routes. Connected and mixed
composition remain explicit internal/development paths and are accepted only by supported
Spine 4.2 capability combinations.

## What the exporter does

| Area | Functionality |
| --- | --- |
| Geometry | Converts Blender Mesh geometry into Spine weighted mesh attachments with deterministic source lineage. |
| Projection | Supports six signed world-axis projections and two Active Camera rig-root variants for Normal / UV Segments. |
| Camera output | Supports flat Camera Projection and depth-aware Depth Camera Projection through Perspective or Orthographic cameras. |
| UV | Generates a dedicated bake UV layout while preserving source-loop correspondence. |
| Segmentation | Supports automatic angular segmentation and user-authored seam segmentation. |
| Rig | Generates a 2-Axis Rotation + Scale control rig with per-vertex/depth ownership appropriate to the selected projection mode. |
| Materials | Analyzes source shader graphs, bakes supported materials, and can generate temporary fallback materials. |
| Sequences | Exports static textures or per-object texture sequences with target-specific Spine sequence encoding. |
| Multi-object | Exports multiple selected Mesh objects in one standalone request, with optional shared assembly pivot for signed-axis Normal / UV and independent object timing. |
| Analysis | Runs production preparation without committing files and reports blockers, warnings, geometry, material, texture, rig, and depth statistics. |
| Output safety | Stages JSON and textures atomically, validates staged output, rolls back partial failures, and recovers stale work files. |
| Blender safety | Restores source Mesh, UV, material, selection, mode, frame, camera, render, and temporary datablock state after success or failure. |

## Export modes

### Normal / UV Segments

Use this mode when the Spine result must remain a deformable mesh rather than a flat camera
render.

Pipeline:

```text
Blender Mesh
-> projection into canonical Spine space
-> automatic or custom-seam segmentation
-> manifold region decomposition
-> generated SpineBakeUV layout
-> weighted Spine mesh attachments
-> semantic texture bake
-> Spine JSON + textures
```

Normal / UV Segments keeps source-derived surface topology and creates one or more weighted
attachments from the final valid regions.

#### Projection Direction

Normal / UV Segments supports:

| UI value | Persisted ID | Meaning |
| --- | --- | --- |
| `+X` | `POSITIVE_X` | World +Y -> Spine X, world +Z -> Spine Y, world +X -> depth. |
| `-X` | `NEGATIVE_X` | World -Y -> Spine X, world +Z -> Spine Y, world -X -> depth. |
| `+Y` | `POSITIVE_Y` | World -X -> Spine X, world +Z -> Spine Y, world +Y -> depth. |
| `-Y` | `NEGATIVE_Y` | World +X -> Spine X, world +Z -> Spine Y, world -Y -> depth. |
| `+Z` | `POSITIVE_Z` | World +X -> Spine X, world +Y -> Spine Y, world +Z -> depth. |
| `-Z` | `NEGATIVE_Z` | World -X -> Spine X, world +Y -> Spine Y, world -Z -> depth. |
| `Active Camera — Object Root Bone` | `ACTIVE_CAMERA` | Camera-projected geometry with each Blender Object Origin retained as the object's Spine pivot. |
| `Active Camera — Camera Root Bone` | `ACTIVE_CAMERA_CAMERA_ROOT` | Camera-projected geometry owned by a camera-relative root hierarchy. |

For two or more selected Mesh objects using one of the six signed-axis directions, **Shared
Selection Pivot** is visible and enabled by default. The exporter computes one center from
the aggregate world-space geometry bounds, uses it as the generated Spine pivot for every
part, and compensates each object's local U/V/depth coordinates so setup world geometry does
not move. Blender Object Origins and source Mesh data are never modified. Disable the toggle
to retain independent per-object pivots.

The two Active Camera choices share the same Perspective/Orthographic camera projection and
the same material-bake geometry. They differ only in rig ownership.

#### Active Camera — Object Root Bone

- keeps each Blender Object Origin as the object's Spine main-bone pivot;
- retains camera-space depth per generated depth group;
- inserts one `*_camera_setup` inverse-setup bone below each model-space depth group;
- cancels setup-only camera-depth translation without collapsing the live deformation hierarchy;
- reproduces the active-camera setup projection while preserving object-root animation controls.

Conceptually:

```text
depth scale bone
└── depth bone
    └── *_camera_setup
        └── generated vertex bone
```

#### Active Camera — Camera Root Bone

- uses camera-space zero as the Spine main-bone pivot;
- stores the projected Blender Object Origin inside the camera-relative hierarchy;
- uses one rigid camera-depth layer for the attachment geometry;
- preserves Perspective and Orthographic camera-layer behavior;
- shares projected geometry and baked appearance with Object Root mode.

### Camera Projection

Use this mode for a deliberately flat camera-facing result.

Pipeline:

```text
active camera render
-> sequence coverage union
-> alpha cleanup
-> stable crop
-> contour construction and triangulation
-> flat screen-space attachment
-> Spine JSON + cropped texture
```

Camera Projection does not reuse Normal / UV Segments region meshes. It is a separate
screen-space representation.

### Depth Camera Projection

Use this mode when a camera-rendered appearance must retain controlled depth deformation in
Spine.

Pipeline:

```text
active-camera visible surface
-> front-most depth sampling
-> edge-aware depth processing
-> bounded generated relief topology
-> weighted generated vertex bones
-> camera render and crop
-> crop-local UV remap
-> Spine JSON + depth-aware attachment
```

Supported controls:

- **Depth smoothing** — edge-aware smoothing of retained depth samples.
- **Depth edge threshold** — prevents smoothing and generated triangles from crossing large depth discontinuities.
- **Depth mesh error (px)** — requested screen-space spacing for generated relief points.
- **Max depth points** — hard upper bound for generated depth points.
- **Parallax Horizon Angle** — optionally retains connected surfaces around the visible horizon.

`Parallax Horizon Angle = 0°` produces the front-only result. A positive value can generate
reserve views for retained side surfaces. Reserve views use fitted Perspective or
Orthographic camera copies, face-isolated render proxies, independent crops and textures,
and reserve weighted attachments that reuse the same generated rig as the FRONT attachment.

Depth Camera Projection is a 2.5D representation of camera-visible geometry. Large later
rotations can reveal surfaces that were never visible or retained during export.

## Rig and animation controls

The public UI uses **2-Axis Rotation + Scale**.

Generated primary controls:

```text
<prefix>_rotation_X
<prefix>_rotation_Y
<prefix>_scale
```

The rig keeps X/Y pseudo-rotation and uniform scale as explicit Spine controls while depth
groups provide the spatial response needed by Normal / UV Segments and Depth Camera Projection.

Additional rig behavior:

- per-object Blender Object Origin pivots for ordinary Normal projection when Shared Selection Pivot is disabled, and for Active Camera Object Root mode;
- one export-only common assembly pivot for eligible multi-object signed-axis Normal / UV when Shared Selection Pivot is enabled;
- camera-relative hierarchy for Active Camera Camera Root mode;
- deterministic constraint ordering;
- optional control icons;
- optional generated preview animation;
- connected five-phase constraint scheduling on the explicitly supported internal Spine 4.2 route.

## Geometry, segmentation, and UV

### Seam Maker: Auto

Auto segmentation grows deterministic face regions from angular rules.

Available angular behavior:

- **Seed cone** — candidate faces are compared with the region seed normal.
- **Seed cone + local dihedral** — also limits the angle across each traversed shared edge.

### Seam Maker: Custom

Custom segmentation uses seams marked on the Blender Mesh. Angular splitting is disabled,
but topology validation and manifold-region decomposition still run.

### UV handling

The exporter:

- creates a generated `SpineBakeUV` layout for output baking;
- preserves source-loop lineage instead of matching geometry by rounded coordinates;
- validates required source UV dependencies used by materials;
- keeps generated Spine attachment UVs synchronized with the baked file-space texture orientation.

## Materials and baking

The exporter analyzes the effective Blender material graph before choosing an execution
path. It does not silently replace the selected export mode.

The **Bake** foldout owns the scene-wide **Texture size** setting. Texture size controls the
resolution used by generated bake textures and rendered-camera texture targets; it remains
one shared Scene setting for the complete export request rather than a per-object sequence setting.

Supported behavior includes:

- audited source-material baking for Normal / UV Segments;
- camera-rendered material evaluation for Camera Projection and Depth Camera Projection;
- temporary Cycles configuration for semantic object baking with Scene state restoration;
- diagnostics for unsupported or camera-dependent shader requirements;
- diagnostics for active Blender modifiers that are not part of the original Mesh topology;
- temporary generated-material fallback without changing the source material graph.

Generated Material policies:

- **Require Source** — block when required source material data is unavailable.
- **Generate If Missing** — generate a temporary material only when required source material data is missing.
- **Force Generated** — ignore source shading and use the selected generated pattern.

Generated patterns:

- Solid Gray
- One Region - One Color
- One Polygon - One Color

Apply or convert topology-changing Blender modifiers when their generated geometry must be
present in the exported Spine mesh.

## Camera rendering controls

Camera Projection and Depth Camera Projection support:

- active Perspective and Orthographic cameras;
- configurable alpha threshold for coverage and crop calculation;
- inclusion/exclusion of scene shadow contributors;
- inclusion/exclusion of reflection/transmission contributors;
- World participation in lighting/reflections;
- stable alpha-union crops across texture-sequence frames;
- restoration of source camera and render state after export.

The exporter does not export Blender camera animation as a Spine camera animation.

## Texture sequences

Each selected Mesh owns independent `Frames` and `Start` values. Texture resolution remains
a shared Scene-level Bake setting.

```text
Frames = 0  -> one static texture evaluated at the current frame
Frames > 0  -> Loop texture sequence for that object only
```

A sequence can use Scene FPS or an explicit sequence FPS override.

Target-specific output:

- Spine 3.8 and 4.0 use attachment-swap sequence encoding.
- Spine 4.1, 4.2, and 4.3 use native sequence metadata/timelines where supported.

In selected-object export, static and animated objects can participate in the same request
without inheriting each other's sequence metadata.

## Analyze and diagnostics

**Analyze** runs the real preparation pipeline without committing final files.

It can report:

- source and exported vertex/triangle counts;
- region and attachment counts;
- material and texture execution path;
- active modifiers ignored by the original-mesh Normal route;
- UV and topology blockers;
- camera and render blockers;
- generated bone and depth-group counts;
- retained depth points and maximum relief;
- FRONT/reserve attachment and virtual-view statistics;
- structured blockers and warnings.

Changing relevant geometry, UV, material, selection, camera, frame, renderer, projection,
Depth settings, or export settings makes the cached analysis stale. Changing any configured
exact Spine project version invalidates cached analysis for all scenes immediately.

## Output and transaction safety

Typical static single-object output:

```text
<ObjectName>_merged_spine_<exact-version>.json
images/<ObjectName>_Baked.png
```

Typical texture sequence:

```text
images/<ObjectName>_Baked_0000.png
images/<ObjectName>_Baked_0001.png
```

The exporter:

- sanitizes output stems for Windows-safe filenames;
- rejects output namespace collisions before writing;
- stages JSON and textures before installation;
- validates staged output;
- commits all files atomically;
- restores previous finals after partial installation failures when possible;
- can recover stale `.spine2d-stage-*` and `.spine2d-backup-*` work files left by an interrupted process.

Temporary stage and backup files are transaction data, not Spine assets.

## Source-scene safety

Export preparation is designed not to permanently modify the source project.

The transaction verifies or restores relevant:

- source Mesh topology;
- UV layers and coordinates;
- material graphs;
- object transforms;
- active object and selection;
- Object/Edit mode requirements;
- current frame;
- active camera;
- render engine and render settings;
- temporary objects, meshes, images, collections, materials, node trees, camera proxies, render proxies, and generated attributes.

## Supported Spine targets and exact project versions

| UI target | Default exact JSON version | Public standalone export |
| --- | --- | --- |
| Spine 3.8 | 3.8.99 | Yes |
| Spine 4.0 | 4.0.64 | Yes |
| Spine 4.1 | 4.1.24 | Yes, according to the capability registry |
| Spine 4.2 | 4.2.43 | Yes |
| Spine 4.3 | 4.3.23 | Yes, according to the capability registry |

The Scene-level **Spine version** selector chooses the schema family and therefore the
production codec. The exact Editor/project patch version is configured separately in
**Edit > Preferences > Add-ons > Blender to Spine2D Mesh Exporter > Spine project JSON
versions**. There is one persistent global setting per supported family.

Only canonical `major.minor.patch` values from the same family are accepted. For example,
`4.2.35` is valid for Spine 4.2, while `4.1.24` is rejected for that field. The effective
exact version is used consistently by `ExportSettings.spine_version`, the JSON filename,
the viewport **Exact JSON version** label, and `skeleton.spine`. Changing the patch does not
switch or emulate another schema family.

These fields are Blender Add-on Preferences rather than `.blend` Scene properties, so they
are shared by projects using the same Blender user configuration. Blender's normal
Preferences persistence owns saving them; the exporter does not force-save unrelated Blender
preferences whenever the user types in a version field.

The target/profile capability registry rejects unsupported combinations before expensive
Blender geometry or bake work begins.

## Quick start

1. Install `blender_to_spine2d_mesh_exporter-0.152.0.zip` through **Edit > Preferences > Extensions > Install from Disk**.
2. In Add-on Preferences, set the exact project patch for every Spine family you use.
3. Save the `.blend` file.
4. Select one or more Mesh objects in Object Mode.
5. Open **3D View > Sidebar > Blender to Spine2D Mesh Exporter**.
6. Choose **Normal / UV Segments**, **Camera Projection**, or **Depth Camera Projection**.
7. For Normal / UV Segments, choose the required Projection Direction. With two or more selected Mesh objects and a signed-axis direction, Shared Selection Pivot is enabled by default.
8. Choose the Spine schema family. Confirm the displayed **Exact JSON version** matches the project version configured in Preferences.
9. Configure Cut, material, camera/depth, and Bake settings as required by the selected mode. Texture resolution is configured as **Texture size** inside **Bake**.
10. Run **Analyze** and resolve blockers.
11. Export the current object or selected objects.
12. Import the generated JSON and matching texture directory into the configured Spine project version.

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

The maintained documentation describes the current product. Historical release notes live
in Git history and tags rather than in the current documentation tree.

## Build the extension

From the repository root:

```powershell
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$Source = ".\Blender_to_Spine2D_Mesh_Exporter"
$Output = ".\dist\blender_to_spine2d_mesh_exporter-0.152.0.zip"

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
