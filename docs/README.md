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
- [Testing and Release Validation](testing.md) - pure Python, real bpy, Blender headless, runtime-oracle, and packaging gates.
- [Contributing](CONTRIBUTING.md) - coding, Blender state, tests, and documentation requirements.
- [Changelog](CHANGELOG.md) - public release history.
- [Spine 4.1 release checkpoint](spine-json-versioning/RELEASE_0_47_11.md) - exact limited target scope and accepted evidence for 0.47.11.

## Supported product baseline

- Extension version: 0.47.11.
- Minimum Blender version: 5.2.0.
- Primary full Spine target: 4.2.43.
- Limited Spine target: 4.1.24 with `2-Axis Rotation + Scale` for single-object and standalone multi-object export only.
- Production **Export Selected Objects** always creates standalone object rigs; persisted hidden Connect flags from older `.blend` files are ignored.
- Connected and mixed composition remain development-only internal capabilities and are not selected by the production UI plan.
- Spine 4.1 connected, mixed, and 3-Axis export remain blocked before geometry processing.
- Currently tested desktop platform: Windows.
- Default texture mode: Normal - UV Segments.
- Default Seam Maker mode: Auto.
- Default rig profile for genuinely fresh scenes: 2-Axis Rotation + Scale (`TWO_AXIS_ROTATION_SCALE`).
- Saved pre-profile scenes migrate to the compatibility 3-Axis Rotation profile.

Development journals and temporary Rewrite milestone documents are intentionally not part of the public documentation set. Permanent behavior belongs in the documents listed above and in executable tests.
