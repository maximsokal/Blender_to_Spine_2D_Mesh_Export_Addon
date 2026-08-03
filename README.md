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

Blender to Spine2D Mesh Exporter converts Blender mesh objects into Spine JSON, weighted
mesh attachments, baked textures, generated rig data, and optional texture sequences.
Version **0.81.0** uses a Blender 5.2-only Rewrite pipeline and adds a third explicit
user-facing representation: **Depth Camera Projection**.

## Requirements

- Blender 5.2 or newer.
- Spine 3.8.99, 4.0.64, 4.1.24, 4.2.43, or 4.3.23 for the matching selected JSON target.
- Windows is the currently tested desktop platform.
- The `.blend` file must be saved before export.
- Export destinations and source image dependencies must be readable and writable as required.

Blender 4.x and Blender 5.0/5.1 are not supported by the current extension package.

## Supported export scope

- Single-object and standalone multi-object export are supported for Spine 3.8 through
  Spine 4.3 according to the target/profile capability matrix.
- Spine 4.2 is the production target for connected and mixed composition.
- Spine 4.2 connected and mixed composition support both `3-Axis Rotation` and
  `2-Axis Rotation + Scale` profiles.
- Unsupported target, profile, and composition combinations fail before Blender geometry
  or bake work.
- Public selected-object export remains standalone-only. Connected and mixed services
  remain explicit development/API routes.

## Export modes

The `Export mode` selector contains exactly three values.

### Normal / UV Segments

The exporter segments the source mesh, creates a generated bake UV layout, evaluates the
source material graph, bakes textures, and writes region-based weighted Spine mesh
attachments. Projection can use a signed world axis or the active Perspective/Orthographic
camera while preserving Normal / UV Segments topology.

### Camera Projection

The exporter renders through the active Blender camera, derives alpha coverage, calculates
one sequence-safe crop, builds a screen-space contour, and exports a flat projection
attachment. Existing Camera Projection behavior is unchanged in 0.81.0.

### Depth Camera Projection

The exporter renders through the active camera like Camera Projection, but it does not
replace the object with a flat contour. It builds a bounded visible depth-relief surface
and sends its retained points through the Normal / UV Segments generated vertex-bone
pipeline.

```text
evaluated mesh
→ active-camera front-most depth sampling
→ edge-aware depth smoothing
→ bounded generated relief topology
→ one generated vertex bone per retained depth point
→ active-camera texture render and crop
→ crop-local UV remap without changing weighted vertices or triangles
```

Depth Camera Projection does not export Blender camera animation. The existing Spine
object controls remain the animation interface. Pseudo-three-dimensional form comes from
relative movement of the generated depth-point bones.

The public relief base is **Farthest Visible Point**. The farthest visible surface has zero
rig offset and every other point receives a non-negative offset toward the camera. A
hidden `Object Origin` policy is implemented for future use and fails closed unless the
origin lies behind all visible points.

Public depth controls:

- `Depth smoothing`;
- `Depth edge threshold`;
- `Depth mesh error (px)`;
- `Max depth points`.

The exporter does not copy every Blender vertex. It creates a new optimized camera-depth
surface bounded by `Max depth points`.

## Key features

- Deterministic automatic or custom-seam mesh segmentation.
- Source-loop UV lineage instead of coordinate-based nearest-point matching.
- Normal / UV Segments export with one Spine mesh attachment per final region.
- Flat Camera Projection with render, crop, contour, and screen-space attachment generation.
- Depth Camera Projection with front-most depth sampling and Normal-style weighted bones.
- Perspective and Orthographic active-camera depth projection.
- Cycles semantic object baking with Blender 5.2 state restoration.
- Animated material and object-transform sequence baking.
- Independent per-object texture timing in multi-object export.
- Texture Coordinate `Camera` and `Reflection` support in audited Normal UV object baking.
- Generated material fallback for Normal / UV Segments missing-material workflows.
- Target-specific JSON serialization for Spine 3.8, 4.0, 4.1, 4.2, and 4.3.
- Legacy attachment-swap sequences for Spine 3.8 and 4.0.
- Native sequence metadata and timelines for Spine 4.1, 4.2, and 4.3.
- Readiness analysis with structured blockers, warnings, and depth-relief statistics.
- Atomic JSON and texture output with rollback and stale work-file recovery.
- Source mesh, UV layer, material graph, transform, Camera, and Blender-state integrity checks.

## Sequence support

A sequence can use the Scene frame rate or an explicit export frame rate. Every frame is
staged and committed atomically with the final JSON.

For multi-object export, the Bake foldout stores `Frames` and `Start` separately for every
selected Mesh:

```text
Frames = 0  → static texture evaluated at the current frame
Frames > 0  → Loop texture sequence only for that object
```

Objects with different timing settings may be exported together. Static siblings do not
receive sequence metadata or animation timelines. This contract applies to all three
export modes.

## Depth Camera Projection limitations

- The generated relief represents only the surface visible from the selected camera.
- Large rotations can expose missing back or side surfaces.
- Transparent layered surfaces provide only the front-most sampled depth.
- A very small edge threshold can intentionally disconnect every candidate triangle and
  block export.
- A point limit that is too small can reduce shape fidelity; the readiness report exposes
  the retained point count and maximum relief.

These are explicit 2.5D constraints rather than silent fallbacks to a flat mesh.

## Seam Maker defaults

`Seam Maker` defaults to **Auto**. Existing scenes are migrated once to settings schema 7
without changing their selected export mode.

- **Auto** uses the configured angular segmentation policy.
- **Custom** uses user-marked seams and disables angular splitting controls.
- Depth Camera Projection generates its own relief topology; its discontinuities are
  controlled by `Depth edge threshold`.

## Quick start

1. Install `blender_to_spine2d_mesh_exporter-0.81.0.zip` through
   **Edit > Preferences > Extensions > Install from Disk**.
2. Save the `.blend` file.
3. Select one or more Mesh objects in Object Mode.
4. Open **3D View > Sidebar > Blender to Spine2D Mesh Exporter**.
5. Choose `Normal / UV Segments`, `Camera Projection`, or `Depth Camera Projection`.
6. Choose the exact Spine target and configure the mode-specific settings.
7. For selected-object export, configure static or sequence timing per object.
8. Run **Analyze** and resolve every blocker.
9. Run **Export Current Object** or **Export Selected Objects**.
10. Import the generated JSON and textures into the exact selected Spine version.

For Depth Camera Projection, assign an active Perspective or Orthographic camera and use
renderable source materials. Start with the default depth controls, then lower the mesh
error or increase the point limit only when more relief detail is required.

See the [Usage Guide](docs/usage.md) and [Settings Reference](docs/settings-reference.md)
for the complete workflow.

## Interface

![Blender to Spine2D Mesh Exporter interface](assets/ui_addon.png)

The panel uses the same paths, cut, bake, readiness, and export flow for every mode. Depth
controls are visible only while `Depth Camera Projection` is selected.

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

Depth Camera Projection keeps the same JSON/PNG naming contract. Its difference is the
weighted attachment topology and generated depth-point bones inside the Spine document.

See [Output Format](docs/output-format.md) for naming, attachment, sequence, and transaction
details.

## Validation

The 0.81.0 release gate includes:

- Blender-independent depth interpolation, smoothing, edge, point-budget, base-policy,
  UI, schema, and routing contracts;
- Perspective Depth Camera Projection for Spine 3.8 through 4.3;
- Orthographic Depth Camera Projection for Spine 4.2;
- a two-frame Spine 4.2 depth sequence;
- a public Spine 4.2 multi-object export with one depth sequence object and one static
  depth sibling;
- physical PNG, crop-local UV, weighted vertices, triangles, generated bones,
  target-specific schema, sequence ownership, and Blender-state restoration checks;
- all established Normal / UV Segments and flat Camera Projection regression matrices;
- the complete repository pytest suite and Blender extension ZIP validation.

No test result is claimed for a commit until the complete gate has run on that exact HEAD.

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
- [Release 0.81.0](docs/releases/0.81.0.md)
- [Examples](examples/examples.md)

## Local build

Use Blender's official extension validator and builder through the repository wrapper:

```text
python tools/prepare_package.py --blender <path-to-Blender-5.2-executable>
```

The resulting ZIP is written to `dist` unless an explicit output path is supplied. Its
root contains `blender_manifest.toml` and `__init__.py`.

## Project structure

```text
Blender_to_Spine2D_Mesh_Exporter/
  application/        Export use-case orchestration and immutable request/result contracts
  blender_adapter/    Blender RNA, geometry, material, render, camera, and UI boundaries
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

The exporter rejects inputs when it cannot prove a safe geometry, UV, material, renderer,
target-version, rig-profile, composition, depth-relief, or output contract. Keep backups
of production scenes and validate generated assets in the selected Spine version before
shipping.

## Contributing

Read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) before changing production code or
documentation. New behavior requires tests at the appropriate pure-Python, real-bpy, or
Blender-headless boundary.

## License and credits

Copyright (c) 2025-2026 Maxim Sokolenko.

This project is licensed under GNU GPL v3.0 or later. Spine Editor and Spine runtimes
remain subject to the applicable Esoteric Software licenses.
