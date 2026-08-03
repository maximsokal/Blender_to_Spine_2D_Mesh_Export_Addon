# Documentation

This directory contains the maintained public documentation for Blender to Spine2D Mesh
Exporter **0.81.0**.

## User documentation

- [Installation](installation.md) - requirements, installation, update, removal, and local packaging.
- [Usage](usage.md) - complete Blender-to-Spine workflow for all three export modes.
- [Settings Reference](settings-reference.md) - every public Scene and object setting.
- [Output Format](output-format.md) - JSON, texture, sequence, naming, weighted attachment, and atomic output behavior.
- [Troubleshooting](troubleshooting.md) - blockers, warnings, logs, stale work files, and bug reports.
- [Examples](../examples/examples.md) - repository example projects and validation goals.

## Developer documentation

- [Architecture](architecture.md) - package boundaries and production data flow.
- [Rig Profiles](rig-profiles.md) - selectable rig plans and generated constraint topology.
- [Testing and Release Validation](testing.md) - pure Python, real bpy, Blender headless, runtime-oracle, and packaging gates.
- [Contributing](CONTRIBUTING.md) - coding, Blender state, tests, and documentation requirements.
- [Changelog](CHANGELOG.md) - public release history.
- [0.81.0 release note](releases/0.81.0.md) - Depth Camera Projection.
- [0.80.1 release note](releases/0.80.1.md) - per-object static and sequence timing acceptance.
- [0.80.0 release note](releases/0.80.0.md) - complete all-sequence acceptance milestone.
- [Spine 4.1 release checkpoint](spine-json-versioning/RELEASE_0_47_10.md) - historical 0.47.10 evidence.

## Supported product baseline

- Extension version: 0.81.0.
- Minimum Blender version: 5.2.0.
- Standalone targets: Spine 3.8.99, 4.0.64, 4.1.24, 4.2.43, and 4.3.23 according to the target/profile capability matrix.
- Connected and mixed composition target: Spine 4.2.43.
- Currently tested desktop platform: Windows.
- Default Seam Maker mode: Auto.
- Default rig profile for genuinely fresh Scenes: 2-Axis Rotation + Scale.
- Scene settings schema: 7.

The public `Export mode` selector contains exactly:

```text
Normal / UV Segments
Camera Projection
Depth Camera Projection
```

Normal / UV Segments preserves segmented source-derived topology and generated UV.
Camera Projection renders and exports a flat cropped screen-space contour. Depth Camera
Projection renders through the active camera but builds an optimized visible depth-relief
surface with Normal-style generated vertex bones.

Depth Camera Projection uses Farthest Visible Point as the public relief base. Generated
rig offsets start at zero on the farthest visible surface and extend only toward the
camera. The hidden Object Origin policy is implemented for future use and fails closed
when the origin is not behind the complete visible surface.

## 0.81.0 acceptance baseline

The release gate adds:

- Blender-independent front-most depth interpolation, smoothing, discontinuity, point-budget, UV, base-policy, UI, routing, and schema contracts;
- Perspective Depth Camera Projection for Spine 3.8 through 4.3;
- Orthographic Depth Camera Projection for Spine 4.2;
- static and two-frame sequence Depth output;
- standalone multi-object Depth export with one sequence object and one static sibling;
- physical PNG, crop-local UV, weighted vertex, triangle, generated bone, sequence ownership, and Blender-state checks;
- every existing Normal / UV Segments and flat Camera Projection regression matrix;
- complete pytest and Blender extension ZIP validation.

Development journals and temporary Rewrite milestone documents are intentionally not part
of the public documentation set. Permanent behavior belongs in these documents and in
executable tests.
