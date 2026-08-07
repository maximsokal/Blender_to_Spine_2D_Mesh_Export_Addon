# Documentation

This directory contains the maintained documentation for Blender to Spine2D Mesh Exporter
**0.129.0**.

The documentation describes the current product only. Historical release notes, milestone
journals, and superseded implementation checkpoints belong in Git history and tags rather
than in the maintained documentation tree.

## Current product baseline

- Extension version: **0.129.0**.
- Minimum Blender version: **5.2.0**.
- Initial Blender Extensions package: **Windows x64**.
- Scene settings schema: **8**.
- Fresh Scene rig profile: **2-Axis Rotation + Scale**.
- Default export mode: **Normal / UV Segments**.
- Default Normal projection direction: **+Z**.
- `Texture size` is a Scene-level **Bake** setting shared by the complete export request.
- Supported standalone Spine targets:
  - 3.8.99
  - 4.0.64
  - 4.1.24
  - 4.2.43
  - 4.3.23
- Connected and mixed composition: supported only by explicitly allowed Spine 4.2 routes.

## Export modes

```text
Normal / UV Segments
Camera Projection
Depth Camera Projection
```

Normal / UV Segments supports six signed world-axis projections plus two active-camera
rig-root choices:

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

The two Active Camera modes share evaluated camera-projected geometry and material-bake
input. Object Root keeps each Blender Object Origin as its Spine pivot and uses per-depth
inverse setup bones. Camera Root places the main bone at camera-space zero and uses one
rigid camera-depth layer.

Depth Camera Projection builds a bounded visible relief surface and can optionally retain
parallax reserve surfaces with `Parallax Horizon Angle`.

## User documentation

- [Installation](installation.md) — installation, update, local build, and archive validation.
- [Usage](usage.md) — complete Blender-to-Spine workflow for all public export modes.
- [Settings Reference](settings-reference.md) — public Scene and object settings.
- [Output Format](output-format.md) — JSON, textures, sequences, rig ownership, naming, and atomic output.
- [Troubleshooting](troubleshooting.md) — readiness diagnostics, camera/UV/material issues, and output recovery.
- [Examples](../examples/examples.md) — repository examples and validation goals.

## Developer documentation

- [Architecture](architecture.md) — package boundaries and production data flow.
- [Rig Profiles](rig-profiles.md) — current generated rig topology and setup-pose policies.
- [Testing](testing.md) — pure Python, real bpy, Blender headless, and packaging gates.
- [Blender Extensions Submission](submission.md) — current store metadata, reviewer path, and pre-upload gate.
- [Contributing](CONTRIBUTING.md) — coding, Blender-state, testing, and documentation requirements.

## Documentation policy

Maintained documents must:

1. be written in English;
2. describe the current extension behavior;
3. avoid release-history sections and superseded implementation narratives;
4. link to executable tests when behavior needs regression evidence;
5. match the current manifest version and public UI labels;
6. avoid documenting hidden development paths as ordinary public UI features.
