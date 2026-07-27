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

Blender to Spine2D Mesh Exporter converts Blender mesh objects into Spine 4.2-oriented JSON, weighted mesh attachments, baked textures, generated rig data, and optional texture sequences. The current extension uses a Blender 5.2-only pipeline with deterministic geometry processing, explicit texture strategies, source-data integrity checks, and atomic output commits.

## Requirements

- Blender 5.2 or newer.
- Spine 4.2.43 is the primary compatibility target.
- Windows is the currently tested desktop platform.
- The `.blend` file must be saved before export.
- Export destinations and source image dependencies must be readable and writable as required.

Blender 4.x and Blender 5.0/5.1 are not supported by the current extension package.

## Key features

- Deterministic automatic or custom-seam mesh segmentation.
- Source-loop UV lineage instead of coordinate-based nearest-point matching.
- Normal UV Segments export with one Spine mesh attachment per final region.
- Explicit Camera Projection export for camera-dependent or screen-space output.
- Cycles semantic object baking with Blender 5.2 EEVEE scene-state restoration.
- Generated material fallback for missing or intentionally ignored source materials.
- Single-object, standalone multi-object, connected, and mixed exports.
- Static texture and frame-sequence output.
- Readiness analysis with structured blockers, warnings, and statistics.
- Atomic JSON and texture output with rollback and stale work-file recovery.
- Source mesh, UV layer, material graph, and Blender state integrity checks.

## Export modes

### Normal - UV Segments

This is the default mode. The exporter segments the source mesh, creates a generated bake UV layout, bakes the evaluated appearance to texture files, and writes region-based Spine mesh attachments.

Use this mode for geometry that should remain represented by multiple deformable Spine mesh attachments.

### Camera Projection

This mode renders through the active Blender camera, derives alpha coverage, crops the result, builds a screen-space contour, and exports one projection attachment.

Camera Projection is selected explicitly. The exporter does not silently switch from Normal mode when a material requires camera rendering.

## Seam Maker defaults

`Seam Maker` defaults to **Auto**. Existing scenes created with earlier development builds are migrated once to the current settings schema. After migration, a deliberate user choice of **Custom** is preserved.

- **Auto** uses the configured angular segmentation policy.
- **Custom** uses user-marked seams and disables angular splitting controls.

## Quick start

1. Install the release ZIP through **Edit > Preferences > Extensions > Install from Disk**.
2. Save the `.blend` file.
3. Select one or more Mesh objects in Object Mode.
4. Open **3D View > Sidebar > Blender to Spine2D Mesh Exporter**.
5. Choose the export, cut, bake, and generated-material settings.
6. Run **Analyze** and resolve every blocker.
7. Run **Export Current Object** or **Export Selected Objects**.
8. Import the generated JSON and textures into Spine 4.2.

See the [Usage Guide](docs/usage.md) and [Settings Reference](docs/settings-reference.md) for the complete workflow.

## Interface

![Blender to Spine2D Mesh Exporter interface](assets/ui_addon.png)

The panel exposes Export, Cut, and Bake sections, readiness analysis, single or selected-object export, per-object Connect flags, and a separate Generated Materials panel.

## Output overview

A single-object export uses an object-derived stem:

```text
<ObjectName>_merged.json
images/<ObjectName>_Baked.png
```

A texture sequence uses zero-padded frame numbers:

```text
images/<ObjectName>_Baked_0000.png
images/<ObjectName>_Baked_0001.png
```

Multi-object output uses the first ordered object name plus the number of additional selected objects. File stems are sanitized for ordinary Windows file APIs.

See [Output Format](docs/output-format.md) for naming, attachment, sequence, and transaction details.

## Documentation

- [Documentation index](docs/README.md)
- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Settings reference](docs/settings-reference.md)
- [Architecture](docs/architecture.md)
- [Output format](docs/output-format.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Testing and release validation](docs/testing.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Changelog](docs/CHANGELOG.md)
- [Examples](examples/examples.md)

## Local build

Use Blender's official extension validator and builder through the repository wrapper:

```text
python tools/prepare_package.py --blender <path-to-Blender-5.2-executable>
```

The resulting ZIP is written to `dist` unless an explicit output path is supplied. Its root contains `blender_manifest.toml` and `__init__.py`.

See [Installation](docs/installation.md) and [Testing](docs/testing.md) for validation commands.

## Project structure

```text
Blender_to_Spine2D_Mesh_Exporter/
  application/        Export use-case orchestration and immutable request/result contracts
  blender_adapter/    Blender RNA, geometry, material, bake, camera, and UI boundaries
  domain/             Blender-independent geometry, UV, baking, and Spine models
  infrastructure/     Registration, diagnostics, locking, tracing, and atomic output
  blender_manifest.toml
  __init__.py

docs/                 Public user and developer documentation
examples/             Example Blender projects and example notes
tests/                Blender-independent test suite
tests_bpy/            Tests executed with a real bpy runtime
tools/                Packaging, comparison, audit, and validation tools
```

## Stability and support scope

The exporter rejects inputs when it cannot prove a safe geometry, UV, material, renderer, or output contract. Complex non-manifold topology, missing required images, malformed UV layers, unsupported material graphs, invalid camera state, or Edit Mode execution can block export with structured diagnostics.

Keep backups of production scenes and validate generated assets in the target Spine version before shipping.

## Contributing

Read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) before changing production code or documentation. New behavior requires tests at the appropriate pure-Python, real-bpy, or Blender-headless boundary.

## License and credits

Copyright (c) 2025-2026 Maxim Sokolenko.

This project is licensed under GNU GPL v3.0 or later. Spine Editor and Spine runtimes remain subject to the applicable Esoteric Software licenses.