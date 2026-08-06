# Changelog

This changelog records public product releases. Detailed milestone notes are preserved in `docs/releases/`.

## Current candidate status

Version **0.125.0** is the current release candidate. It adds visible diagnostics for
Blender modifiers ignored by Normal / UV Segments and extends the audited Cycles
`COMBINED` object-bake route to supported `BSDF_GLOSSY` materials.

## [0.125.0] - 2026-08-06

### Fixed

- Conservatively traversed muted shader nodes no longer become false `GRAPH_ANALYSIS_INCOMPLETE` blockers.
- Audited `BSDF_GLOSSY` materials can use the Normal / UV Segments Cycles `COMBINED` object-bake route.
- The real `coin_star.blend` asset exports through original geometry, object bake, and a non-empty PNG plus Spine JSON.

### Added

- The Analysis foldout now warns when active Blender modifiers are ignored by Normal / UV Segments.
- Modifier diagnostics include object name, modifier name, modifier type, and viewport/render state.
- Multi-object requests group modifier diagnostics by object while preserving request order.
- Modifiers disabled for both viewport and render are omitted from the alert.

### Compatibility

- Manifest version: `0.125.0`.
- Expected archive: `blender_to_spine2d_mesh_exporter-0.125.0.zip`.
- Minimum Blender version remains 5.2.0.
- Scene settings schema remains 8.
- Normal / UV Segments still exports the original Mesh datablock; apply or convert modifiers when generated geometry must appear in Spine.

See [the complete 0.125.0 release note](releases/0.125.0.md).

## [0.90.0] - 2026-08-04

### Added

- Public `Parallax Horizon Angle` for Depth Camera Projection, displayed in degrees and persisted in radians.
- Compatibility default `0°`, soft UI range through `45°`, and hard limit `89°`.
- Deterministic Dijkstra traversal over face adjacency using accumulated unsigned dihedral angle.
- Eight deterministic reserve directions: RIGHT, UP_RIGHT, UP, UP_LEFT, LEFT, DOWN_LEFT, DOWN, and DOWN_RIGHT.
- One union `MeshSnapshot` and one shared generated vertex-bone rig across FRONT and reserve attachments.
- Fitted Perspective and Orthographic virtual camera views without mutating the source camera.
- Face-isolated temporary reserve render proxies so FRONT geometry cannot occlude hidden retained surfaces.
- Independent alpha-union crop and crop-local UV layout for every FRONT/reserve view.
- Static and native/legacy sequence output for every view-owned attachment.
- Scene settings schema 8 with a safe `0.0` parallax-horizon migration default.

### Validated

- Positive Perspective parallax with a hidden forty-five-degree folded surface.
- Orthographic reserve fitting and a two-frame FRONT/reserve material sequence.
- Standalone multi-object output with two parallax objects, four textures, and one JSON document.
- Reserve slots serialized before the established FRONT slot while sharing hinge vertex bones.
- Public-path atomic rollback after FRONT and reserve files for object one have already been staged.
- Complete source Mesh, material, transform, camera, frame, selection, render-state, and temporary datablock restoration.
- Zero-angle compatibility for Spine 3.8.99, 4.0.64, 4.1.24, 4.2.43, and 4.3.23.

### Compatibility

- Manifest version: `0.90.0`.
- Expected archive: `blender_to_spine2d_mesh_exporter-0.90.0.zip`.
- Minimum Blender version remains 5.2.0.
- Scene settings schema: 8.
- Existing saved Scenes receive `Parallax Horizon Angle = 0°` and preserve established Depth output.

See [the complete 0.90.0 release note](releases/0.90.0.md).

## [0.81.0] - 2026-08-03

### Added

- Public `Depth Camera Projection` alongside `Normal / UV Segments` and `Camera Projection`.
- Evaluated-mesh front-most depth sampling through active Perspective or Orthographic cameras.
- Bounded generated relief topology controlled by Depth smoothing, Depth edge threshold, Depth mesh error, and Max depth points.
- Farthest Visible Point base policy with zero offset at the farthest visible surface and non-negative generated offsets toward the camera.
- Hidden, fail-closed Object Origin base policy for future use.
- Direct full-frame camera UV and post-render crop-local UV remapping.
- Normal-style Z groups, weighted attachments, and generated vertex bones for retained depth points.
- Scene settings schema 7 and safe defaults for new depth controls.
- Blender-independent depth-surface and public routing contracts.
- Real Blender single-object Depth acceptance for Spine 3.8 through 4.3, Perspective, Orthographic, and sequence output.
- Real Blender multi-object Depth acceptance with one two-frame sequence object and one static sibling.

### Preserved

- Existing Normal / UV Segments behavior and projection-direction choices.
- Existing flat Camera Projection render, crop, contour, and attachment behavior.
- Existing Spine target/profile/scope capability registry.
- Existing per-object static/sequence timing and atomic output transaction.
- Source Mesh, UV, material, transform, camera, render, selection, frame, and temporary datablock restoration contracts.

### Compatibility

- Manifest version: `0.81.0`.
- Expected archive: `blender_to_spine2d_mesh_exporter-0.81.0.zip`.
- Minimum Blender version remains 5.2.0.
- Scene settings schema: 7.
- Existing saved Scenes retain their selected export mode.

See [the complete 0.81.0 release note](releases/0.81.0.md).

## [0.80.1] - 2026-08-03

### Added

- Real Blender 5.2 standalone mixed static/sequence coverage for Spine 3.8, 4.0, 4.1, 4.2, and 4.3.
- Normal - UV Segments and Camera Projection coverage with one two-frame sequence object and two static siblings.
- Connected Spine 4.2 coverage with one sequence object and one static object for both rig profiles and both texture modes.
- Mixed Spine 4.2 coverage with the sequence owner inside either the connected subgroup or the standalone subgroup.
- Pure request-boundary tests proving that selected-object frame counts `(2, 0, 0)` remain independent on the final `A1MultiObjectSource` settings.

### Validated

- Static objects produce one PNG and one setup attachment.
- Static objects do not receive native sequence metadata, attachment-swap timelines, or native sequence timelines.
- The sequence object alone receives legacy attachment swaps on Spine 3.8/4.0 or native sequence metadata on Spine 4.1+.
- Connected wrapper hierarchy and mixed standalone subgroup isolation remain valid with heterogeneous object timing.
- Source transforms, current frame, active object, selection, materials, Scene bake state, render state, Camera state, and temporary datablocks are restored.

### UI and compatibility

- Existing per-object `Frames` and `Start` values in the multi-object Bake foldout are an explicitly validated public contract.
- `Frames = 0` means static current-frame output; a positive value creates a Loop sequence only for that object.
- Public selected-object export remains standalone-only.
- Scene settings schema remains version 6 for this historical release.

### Packaging

- Manifest version: `0.80.1`.
- Expected archive: `blender_to_spine2d_mesh_exporter-0.80.1.zip`.
- Minimum Blender version remains 5.2.0.

See [the complete 0.80.1 release note](releases/0.80.1.md).

## [0.80.0] - 2026-08-03

### Added

- Real Blender 5.2 standalone multi-object sequence coverage for Spine 3.8, 4.0, 4.1, 4.2, and 4.3.
- Normal - UV Segments and Camera Projection coverage for every standalone target.
- Real Blender connected and mixed Spine 4.2 sequence coverage for both `3-Axis Rotation` and `2-Axis Rotation + Scale`.
- Two-frame, 128x128, one-sample physical PNG and final JSON validation.
- Static contracts that prevent target, scope, profile, texture-mode, object-count, frame-count, or resolution coverage from being silently reduced.

### Fixed

- Animated source `matrix_world` is synchronized to the temporary Normal UV bake target on every sequence frame.
- Texture Coordinate `Camera` and `Reflection` inputs are evaluated by the audited Cycles object-bake route without changing UV-segment topology.
- Blender float32 matrix round-tripping is validated with an explicit ULP-based tolerance while real transform mismatches remain fail-closed.

### Validated

- Spine 3.8 and 4.0 legacy attachment-swap sequences.
- Spine 4.1, 4.2, and 4.3 native sequence metadata and Loop timelines.
- Connected wrapper hierarchy and mixed standalone subgroup isolation.
- Camera Projection crop, hull, triangle, UV, and attachment-dimension consistency.
- Source transforms, current frame, active object, selection, materials, Scene bake state, render state, Camera state, and temporary datablock restoration.
- Atomic JSON and texture output for public single, standalone multi-object, connected, and mixed output services.

### Packaging

- Manifest version: `0.80.0`.
- Expected archive: `blender_to_spine2d_mesh_exporter-0.80.0.zip`.
- Minimum Blender version remains 5.2.0.
- Scene settings schema remains version 6.

See [the complete 0.80.0 release note](releases/0.80.0.md).

## [0.55.18] - 2026-08-01

The 0.55 line introduced Object Origin placement, six signed-axis Normal UV projection routes, Active Camera Normal UV projection, standalone object-block draw order, connected/mixed normalization, affine setup acceptance, and the public projection-direction selector.

Detailed notes are preserved in:

- [0.55.1](releases/0.55.1.md)
- [0.55.2](releases/0.55.2.md)
- [0.55.3](releases/0.55.3.md)
- [0.55.4](releases/0.55.4.md)
- [0.55.5](releases/0.55.5.md)
- [0.55.6](releases/0.55.6.md)
- [0.55.7](releases/0.55.7.md)
- [0.55.8](releases/0.55.8.md)
- [0.55.9](releases/0.55.9.md)
- [0.55.10](releases/0.55.10.md)
- [0.55.11](releases/0.55.11.md)
- [0.55.12](releases/0.55.12.md)
- [0.55.13](releases/0.55.13.md)
- [0.55.14](releases/0.55.14.md)
- [0.55.15](releases/0.55.15.md)
- [0.55.16](releases/0.55.16.md)
- [0.55.17](releases/0.55.17.md)
- [0.55.18](releases/0.55.18.md)

## [0.47.11] - 2026-07-30

- Public selected-object export was fixed to remain standalone-only.
- Hidden legacy connected flags stopped changing ordinary Analyze/Export routing.
- Spine 4.1 standalone selected-object export no longer entered an unsupported connected scope.

## [0.47.10] - 2026-07-30

- Added the Spine 4.1.24 codec and exact target/profile/scope capability matrix.
- Added Blender-generated Spine 4.1 runtime and scale-response acceptance.
- Added target-specific weighted-attachment remapping and bridge-bone adaptation.

See [the Spine 4.1 release checkpoint](spine-json-versioning/RELEASE_0_47_10.md).

## Earlier releases

Earlier pre-Rewrite and legacy development history remains available through repository tags and Git history. Internal Rewrite milestone journals are intentionally not public product documentation.
