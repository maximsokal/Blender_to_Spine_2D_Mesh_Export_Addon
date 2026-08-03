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
- [Rig Profiles](rig-profiles.md) - selectable rig plans and generated constraint topology.
- [Testing and Release Validation](testing.md) - pure Python, real bpy, Blender headless, runtime-oracle, and packaging gates.
- [Contributing](CONTRIBUTING.md) - coding, Blender state, tests, and documentation requirements.
- [Changelog](CHANGELOG.md) - public release history.
- [0.80.1 release note](releases/0.80.1.md) - per-object static and sequence timing acceptance.
- [0.80.0 release note](releases/0.80.0.md) - complete all-sequence acceptance milestone.
- [Spine 4.1 release checkpoint](spine-json-versioning/RELEASE_0_47_10.md) - historical limited target scope and accepted evidence for 0.47.10.

## Supported product baseline

- Extension version: 0.80.1.
- Minimum Blender version: 5.2.0.
- Standalone targets: Spine 3.8.99, 4.0.64, 4.1.24, 4.2.43, and 4.3.23 according to the target/profile capability matrix.
- Connected and mixed composition target: Spine 4.2.43.
- Spine 4.2 connected and mixed composition support both `3-Axis Rotation` and `2-Axis Rotation + Scale`.
- Unsupported target, profile, and scope combinations fail before Blender geometry or bake work.
- Currently tested desktop platform: Windows.
- Default texture mode: Normal - UV Segments.
- Default Seam Maker mode: Auto.
- Default rig profile for genuinely fresh scenes: 2-Axis Rotation + Scale (`TWO_AXIS_ROTATION_SCALE`).
- Normal - UV Segments exposes persisted `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`, and `Active Camera` projection choices; new and older scenes without a stored value use `+Z`.
- The separate rendered Camera Projection mode remains an explicit render, crop, contour, and flattening pipeline.
- Public active-object and selected-object plans carry the chosen projection direction; public selected-object export remains standalone-only.
- Connected and mixed output services remain explicit development/API composition routes.
- Selected-object Bake settings remain per-object: `Frames = 0` is static current-frame output, while a positive value creates a Loop sequence only for that object.
- Scene settings schema remains version 6.

## Sequence acceptance baseline

Version 0.80.1 includes real Blender 5.2 gates at 128x128 with two sequence frames and one Cycles sample:

- all-sequence standalone multi-object export for Spine 3.8 through 4.3;
- mixed static/sequence standalone multi-object export for Spine 3.8 through 4.3;
- Normal - UV Segments and Camera Projection for every standalone target;
- all-sequence connected and mixed Spine 4.2 output in both rig profiles and both texture modes;
- one-sequence-object connected and mixed Spine 4.2 output, including sequence ownership in either mixed subgroup;
- animated location, rotation, scale, and Camera/Reflection material inputs;
- legacy attachment swaps for Spine 3.8 and 4.0;
- native sequences for Spine 4.1, 4.2, and 4.3;
- explicit proof that static siblings have one PNG, one attachment, and no sequence metadata or timeline;
- physical PNG validation, JSON schema validation, composition hierarchy checks, and Blender-state restoration.

Development journals and temporary Rewrite milestone documents are intentionally not part of the public documentation set. Permanent behavior belongs in the documents listed above and in executable tests.
