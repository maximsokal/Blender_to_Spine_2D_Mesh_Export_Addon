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
- [Spine 4.1 release checkpoint](spine-json-versioning/RELEASE_0_47_10.md) - exact limited target scope and accepted evidence for 0.47.10.

## Supported product baseline

- Extension version: 0.55.15.
- Minimum Blender version: 5.2.0.
- Primary full Spine target: 4.2.43.
- Limited Spine target: 4.1.24 with `2-Axis Rotation + Scale` for single-object and standalone multi-object export only.
- Spine 4.1 connected, mixed, and 3-Axis export remain blocked before geometry processing.
- Currently tested desktop platform: Windows.
- Default texture mode: Normal - UV Segments.
- Default Seam Maker mode: Auto.
- Default rig profile for genuinely fresh scenes: 2-Axis Rotation + Scale (`TWO_AXIS_ROTATION_SCALE`).
- Default Normal - UV Segments projection remains `+Z` until the later public UI projection slice is implemented.
- Typed Normal - UV Segments settings can select `ACTIVE_CAMERA` for single-object, standalone, connected, and mixed internal preparation. The public export route remains standalone-only until the UI slice.
- Active Camera segmentation, decomposition, and strict triangulation complete in normalized world geometry before camera projection; only already-triangulated immutable regions are transformed into camera screen/depth space.
- Source UV boundary ownership is validated immediately after world-transform normalization for both signed-axis and Active Camera routes.
- `A1ProjectedObjectAnalysis` records projected Object Origin, nearest/farthest evaluated vertex depth, projected bounds, source order, prefix, component ID, and owned slots for every Normal - UV Segments object.
- Standalone, connected, and mixed Normal - UV Segments setup slots use complete far-to-near object blocks ordered by nearest evaluated vertex; segment order inside an object remains unchanged.
- Connected object mains are anchor-relative while the global group main stores the anchor's absolute projected Object Origin, preserving the same setup world positions as standalone export.
- Connected depth layers remain grouped by projected Object Origin depth, independently from visual slot ordering.
- Mixed internal composition may interleave connected and standalone object blocks by nearest depth without changing subgroup bones, skins, constraints, attachments, or animation metadata.
- The existing rendered Camera Projection mode, crop, contour, flattening, grouped-camera behavior, and public standalone-only routing are unchanged.
- Candidate 0.55.15 implements Slice 5 connected/mixed projection normalization. Spine 4.1 connected and mixed capability restrictions remain fail-closed.
- Saved pre-profile scenes migrate to the compatibility 3-Axis Rotation profile.
- Spine 4.2 connected 3-Axis composition reproduces the dedicated wrapper, exact constraint payloads, source-order arrays, Z-layer order sharing, and unchanged scale-compensator orders from the historical `main` implementation.
- Spine 4.2 connected 2-Axis composition uses the same Z-layer scheduling principle with explicit X, IK, Scale, depth-scale, and Y global targets.

Development journals and temporary Rewrite milestone documents are intentionally not part of the public documentation set. Permanent behavior belongs in the documents listed above and in executable tests.
