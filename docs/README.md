# Documentation

This directory contains the maintained public documentation for Blender to Spine2D Mesh Exporter.

## User documentation

- [Installation](installation.md) - requirements, installation, update, removal, and local packaging.
- [Usage](usage.md) - complete Blender-to-Spine workflow.
- [Settings Reference](settings-reference.md) - every user-facing Scene, object, and add-on preference.
- [Output Format](output-format.md) - JSON, texture, sequence, naming, and atomic output behavior.
- [Troubleshooting](troubleshooting.md) - blockers, warnings, logs, stale work files, and bug reports.
- [Examples](../examples/examples.md) - repository example projects and validation goals.

## Developer documentation

- [Architecture](architecture.md) - package boundaries and production data flow.
- [Rig Profiles](rig-profiles.md) - selectable rig plan, two-axis-plus-scale design, and the complete Spine 4.2.43 reference skeleton.
- [Testing and Release Validation](testing.md) - pure Python, real bpy, Blender headless, and packaging gates.
- [Contributing](CONTRIBUTING.md) - coding, Blender state, tests, and documentation requirements.
- [Changelog](CHANGELOG.md) - public release history.

## Supported product baseline

- Extension version: 0.47.5.
- Minimum Blender version: 5.2.0.
- Primary Spine target: 4.2.43.
- Currently tested desktop platform: Windows.
- Default texture mode: Normal - UV Segments.
- Default Seam Maker mode: Auto.
- Default rig profile for genuinely fresh scenes: 2-Axis Rotation + Scale (`TWO_AXIS_ROTATION_SCALE`).
- Saved pre-profile scenes migrate to the compatibility 3-Axis Rotation profile.
- Connected 3-Axis composition reproduces the dedicated wrapper, exact constraint payloads, source-order arrays, Z-layer order sharing, and unchanged scale-compensator orders from the historical `main` implementation.
- Connected 2-Axis composition uses the same Z-layer scheduling principle with explicit X, IK, Scale, depth-scale, and Y global targets.

Development journals and temporary Rewrite milestone documents are intentionally not part of the public documentation set. Permanent behavior belongs in the documents listed above and in executable tests.
