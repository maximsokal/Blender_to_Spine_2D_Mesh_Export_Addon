# Changelog

This changelog records public product releases. Detailed milestone notes are preserved in `docs/releases/`.

## Current candidate status

Version **0.80.0** is the current release candidate. It promotes the Blender 5.2 Rewrite exporter after completing real Blender sequence acceptance for all supported standalone Spine targets and for every production-supported connected/mixed Spine 4.2 profile and texture mode.

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
