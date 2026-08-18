# Spine Mesh Exporter

<p align="center">
  <img src="assets/cover.png" alt="Spine Mesh Exporter cover" width="600">
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

**Spine Mesh Exporter 0.155.0** converts Blender Mesh objects into Spine-ready JSON,
weighted mesh attachments, baked or camera-rendered textures, generated animation controls,
and optional texture sequences.

The technical extension ID remains `blender_to_spine2d_mesh_exporter`. The public display
name is intentionally separate from that stable package identity.

## Requirements

- Blender 5.2 or newer.
- No operating-system platform restriction is declared in the extension manifest.
- A saved `.blend` file.
- A writable output directory.
- Spine schema family 3.8, 4.0, 4.1, 4.2, or 4.3.
- The exact Spine Editor/project patch version configured for that family in Add-on Preferences.

Built-in default exact versions:

```text
3.8 -> 3.8.99
4.0 -> 4.0.64
4.1 -> 4.1.24
4.2 -> 4.2.43
4.3 -> 4.3.23
```

A different canonical `major.minor.patch` value can be configured inside the selected
family. For example, `4.2.35` continues to use the Spine 4.2 codec while writing `4.2.35` to
`skeleton.spine` and the versioned JSON filename.

Public single-object and selected-object export are standalone routes. Connected and mixed
composition remain explicit internal/development paths and are accepted only by supported
capability combinations.

## What the exporter does

| Area | Functionality |
| --- | --- |
| Geometry | Converts Blender Mesh geometry into Spine weighted mesh attachments with deterministic source lineage. |
| Projection | Supports six signed world-axis projections and two Active Camera rig-root variants for Normal / UV Segments. |
| Camera output | Supports flat Camera Projection and depth-aware Depth Camera Projection through Perspective or Orthographic cameras. |
| UV | Generates dedicated bake UV data while preserving source-loop correspondence. |
| Segmentation | Supports automatic angular segmentation and user-authored seam segmentation. |
| Rig | Generates a 2-Axis Rotation + Scale control rig with projection-aware depth ownership. |
| Materials | Audits source shader graphs, bakes supported materials, and can create temporary generated fallback materials. |
| Sequences | Exports static textures or per-object texture sequences with target-specific Spine encoding. |
| Multi-object | Exports multiple selected Mesh objects with independent timing and optional Shared Selection Pivot for eligible signed-axis Normal / UV requests. |
| Analysis | Runs production preparation synchronously without final file commit and reports diagnostics/statistics. |
| Output safety | Stages JSON/textures atomically, validates staged output, rolls back partial failures, and recovers stale work files. |
| Blender safety | Restores source Mesh/UV/material/context/render state and removes temporary datablocks after success or failure. |

## Export modes

### Normal / UV Segments

Use this mode when the Spine result must remain a deformable weighted mesh.

```text
Blender Mesh
-> source geometry/UV lineage capture
-> automatic or custom-seam segmentation
-> manifold region decomposition
-> generated bake UV
-> projection into Spine U/V/depth space
-> weighted mesh attachments
-> semantic texture bake
-> target-specific Spine JSON + textures
```

Projection Direction supports:

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

For two or more selected Mesh objects on one of the six signed-axis projections, **Shared
Selection Pivot** is visible and enabled by default. It computes one export-only pivot from
the aggregate world-space geometry bounds and compensates every object snapshot so setup
world geometry does not move. Source Object Origins and Mesh data are not modified.

#### Active Camera — Object Root Bone

- uses the active Perspective or Orthographic camera for projected U/V;
- keeps each Blender Object Origin as the object's Spine main-bone pivot;
- retains camera-space depth per generated depth group;
- inserts `*_camera_setup` inverse-setup children below depth groups;
- preserves live X/Y pseudo-rotation after setup compensation.

Conceptually:

```text
depth scale bone
└── depth bone
    └── *_camera_setup
        └── generated vertex bone
```

#### Active Camera — Camera Root Bone

- uses the same evaluated camera projection as Object Root;
- places the Spine main bone at camera-space zero;
- stores the projected Blender Object Origin below the camera-relative hierarchy;
- uses one rigid camera-depth layer;
- preserves Perspective/Orthographic camera semantics.

### Camera Projection

Use this mode for a deliberately flat camera-facing representation.

```text
active camera render
-> alpha coverage union
-> stable crop
-> contour construction
-> deterministic triangulation
-> flat screen-space attachment
-> Spine JSON + cropped texture
```

Camera Projection is a separate representation; it does not silently replace Normal / UV
Segments.

### Depth Camera Projection

Use this mode for a camera-facing 2.5D relief representation.

```text
active-camera visible surface
-> front-most depth sampling
-> edge-aware depth processing
-> bounded relief topology
-> weighted generated vertex bones
-> FRONT/reserve camera render and crop
-> crop-local UV remap
-> target-specific Spine JSON
```

Public controls include **Depth smoothing**, **Depth edge threshold**, **Depth mesh error
(px)**, **Max depth points**, and **Parallax Horizon Angle**.

`Parallax Horizon Angle = 0°` keeps FRONT only. Positive values may retain connected side
surfaces and create reserve views/textures/attachments below FRONT while sharing the same
generated rig.

## Rig and animation controls

The public rig profile is **2-Axis Rotation + Scale**.

Primary generated controls:

```text
<prefix>_rotation_X
<prefix>_rotation_Y
<prefix>_scale
```

The rig keeps X/Y pseudo-rotation and uniform scale explicit while generated depth groups or
camera-relative owners provide the setup/deformation semantics required by the selected
projection mode.

Rig construction and serialization validate deterministic bone/constraint order, finite
numeric values, valid parent/reference indices, weighted streams, and target-family
compatibility.

## Geometry, segmentation, and UV

### Seam Maker: Auto

Automatic segmentation grows deterministic surface regions from angular rules. The public
angular policies include seed-normal cone behavior and optional local-dihedral restriction.

### Seam Maker: Custom

Custom mode uses Blender edges marked as seams and disables angular splitting. Topology
validation and manifold-region decomposition still run.

### UV handling

The exporter:

- creates generated bake UV data for output textures;
- preserves loop-level UV identity instead of matching vertices by rounded coordinates;
- validates source UV dependencies required by materials;
- keeps attachment UVs synchronized with file-space texture orientation.

## Materials and baking

The exporter analyzes the effective Blender material graph before selecting an execution
path. It does not silently change the requested public export mode.

The **Bake** foldout owns the scene-wide **Texture size** setting. Texture size is one Scene
setting shared by the complete export request, including selected-object exports with
independent per-object frame timing.

Generated Material policies:

```text
Require Source
Generate If Missing
Force Generated
```

Generated fallback materials are temporary and must not modify the source material graph.
Normal / UV Segments exports the original Mesh datablock topology; apply/convert modifiers
when modifier-generated topology must exist in the exported Spine mesh.

## Texture sequences

Each selected Mesh owns independent `Frames` and `Start` values:

```text
Frames = 0  -> one static texture at the current frame
Frames > 0  -> Loop texture sequence for that object
```

Spine 3.8/4.0 use the supported attachment-swap sequence representation. Spine 4.1/4.2/4.3
use native sequence metadata/timelines where supported by the family codec.

A custom exact patch inside a family does not change that family's sequence encoding policy.

## Analyze and diagnostics

**Analyze** is an explicit user action. It runs the real preparation pipeline synchronously
on Blender's main thread without committing final output files.

It can report:

- source/exported geometry counts;
- region, attachment, rig, and generated-bone statistics;
- topology/UV/material issues;
- ignored active modifiers;
- camera/render/depth diagnostics;
- sequence ownership;
- structured blockers and warnings.

The 0.155.0 runtime ships **no automatic readiness polling scheduler**: no Python worker,
`threading.Timer`, automatic depsgraph analysis callback, or load-time automatic Analyze
callback is used. Cached diagnostics can become stale after source/settings changes, but a
current Analyze report is not required to invoke production Export.

## Output and transaction safety

Typical static output:

```text
<ObjectName>_merged_spine_<exact-version>.json
images/<ObjectName>_Baked.png
```

Typical sequence output:

```text
images/<ObjectName>_Baked_0000.png
images/<ObjectName>_Baked_0001.png
```

The exporter:

- validates output namespaces before commit;
- creates transaction-owned stage/backup files;
- validates staged JSON/textures;
- commits output atomically;
- restores previous finals when a partial install fails where possible;
- recovers stale work without deleting work owned by another live process.

The installed runtime has no manifest OS restriction. Platform-specific process/path
compatibility is guarded inside the atomic layer rather than making the package Windows-only.

## Source-scene safety

Export preparation is designed not to permanently mutate the artist project. Success and
failure paths verify/restore relevant:

- source Mesh topology;
- UV layers and coordinates;
- material graphs;
- object transforms;
- active object and selection;
- mode/frame state;
- active camera and render settings;
- temporary objects, meshes, materials, images, collections, camera/render proxies, and
  generated attributes.

## Quick start

1. Install `blender_to_spine2d_mesh_exporter-0.155.0.zip` through **Edit > Preferences > Extensions > Install from Disk**.
2. Enable **Spine Mesh Exporter**.
3. Configure the exact Spine project patch for every family you use in Add-on Preferences.
4. Save the `.blend` file and select one or more Mesh objects in Object Mode.
5. Open the exporter in the 3D View Sidebar.
6. Choose Export Mode, projection/settings, Spine family, Bake Texture size, and output path.
7. Optionally run **Analyze** and review diagnostics.
8. Run **Export Current Object** or **Export Selected Objects**.
9. Import the generated JSON/textures into the configured exact Spine project version.

## Documentation

- [Documentation index](docs/README.md)
- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Settings Reference](docs/settings-reference.md)
- [Rig Profiles](docs/rig-profiles.md)
- [Output Format](docs/output-format.md)
- [Architecture](docs/architecture.md)
- [Testing](docs/testing.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Blender Extensions submission](docs/submission.md)
- [Examples](examples/examples.md)

## Development and release notes

The official extension archive is built from `Blender_to_Spine2D_Mesh_Exporter/` with
Blender's `extension build` command. Repository tests, documentation, retained legacy source,
and development-only pipeline trace implementation are excluded by manifest build rules.

Release acceptance requires the exact clean candidate commit to pass focused moderation
contracts, the full Blender-independent suite, real-bpy lifecycle tests, representative
Blender-headless exports, exact-version preference persistence, Blender extension validation,
exact-ZIP inventory/moderation scan, and clean-profile install/disable/restart/re-enable.

The 0.155.0 correction must be uploaded as a higher version to the **same existing Blender
Extensions submission** that received the moderator feedback; it must not create a duplicate
listing.
