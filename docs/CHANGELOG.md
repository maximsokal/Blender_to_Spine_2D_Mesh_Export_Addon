# Changelog

This changelog records public product releases. Internal development milestones are intentionally omitted.

## Unreleased candidate status

Version 0.41.2 is not a validated release candidate. A representative textured sword asset still imports into Spine with an incorrect setup-pose mesh layout even though material baking succeeds. The candidate must not be merged or released until the source Blender mesh, generated JSON, generated texture, and a known-good Legacy output are compared through an asset-specific regression.

## [0.41.2] - 2026-07-28

### Changed

- Source Z values are canonicalized to the historical four-decimal Legacy identity before Z-group creation and exact `SourceVertexId` binding.
- Explicit Z-group height overrides use the same canonical identity.

### Added

- Regressions preventing small evaluated-geometry floating-point differences from creating extra depth groups or inflating the base rig bone count.

### Known issue

- This change does not resolve the incorrect setup-pose layout observed for the representative textured sword asset. Version 0.41.2 remains a failed development candidate rather than a release.

## [0.41.1] - 2026-07-27

### Fixed

- Restored attachment triangle-area and physical-hull validation to the stable local projected pixel plane.
- Z-group parent translations remain available for diagnostics but no longer redefine attachment topology or hull membership.
- Correct 3D regions whose Spine setup pose is temporarily collinear, including pyramid side faces, are accepted again.
- The full Normal UV pyramid regression is protected against future setup-pose topology coupling.

## [0.41.0] - 2026-07-27

### Added

- Exact post-assembly correspondence validation for projected UVs, triangle corners, physical hull data, and weighted vertex-bone indices.
- Pure regressions for polygon-order-independent material binding and weighted-stream corruption.
- A Blender 5.2 asymmetric material fixture whose geometry order and source-material UV order intentionally differ.

### Changed

- Temporary bake material slots are assigned through exact `FaceId` to Blender polygon correspondence instead of positional polygon iteration.
- Final Spine document assembly validates serialized UV, triangle, hull, edge, and weighted-bone streams before output.

### Fixed

- Blender could materialize temporary polygons in a different order while the bake material stage still assigned source material slots positionally, placing correct source materials on the wrong baked faces.
- A shifted weighted bone index or serialized UV/triangle reorder could reach output without an explicit correspondence failure.
- The previous directional texture test used matching geometry and UV orientation and could not independently prove source-material corner identity.

## [0.40.0] - 2026-07-27

### Added

- Blender 5.2 extension packaging and runtime gate.
- Spine 4.2.43-oriented typed document pipeline.
- Normal - UV Segments export mode.
- Explicit Camera Projection export mode.
- Deterministic automatic segmentation and custom-seam workflow.
- Source geometry and `SourceLoopId` UV lineage contracts.
- Manifold disk decomposition and physical Spine hull normalization.
- Generated material policies and deterministic diagnostic color patterns.
- Single-object, standalone multi-object, connected, and mixed composition.
- Static and per-object sequence texture output.
- Readiness analysis with structured blockers, warnings, and statistics.
- Atomic JSON and texture output with backups, rollback, stale work recovery, and interprocess ownership.
- Per-file logging and failed-work-file preferences.
- Real-bpy and Blender 5.2 headless regression layers.

### Changed

- Replaced Legacy production orchestration with explicit application, domain, Blender adapter, and infrastructure boundaries.
- Camera Projection is selected explicitly and is never an automatic fallback from Normal mode.
- Semantic object baking uses a transactional Cycles path and restores Blender 5.2 EEVEE Scene state.
- Normal bake images are converted to Spine file-space row orientation before save.
- Staged texture validation samples exported Spine UVs against the saved image contract.
- Seam Maker defaults to Auto.
- Older development scenes below settings schema 3 migrate once to Auto; later deliberate Custom choices are preserved.
- Public documentation now describes only maintained product behavior and uses English-only content checks.

### Removed

- Temporary Rewrite milestone documents from the public documentation directory.
- Development-only fixture manifest examples from public user documentation.
- Blender 4.x compatibility claims from the current extension package.
- Legacy modules from the built Blender 5.2 extension archive.

### Fixed

- Registration-time Seam Maker callbacks no longer mark restored Custom values as deliberate user edits before migration.
- Source UV guards no longer intercept unrelated typed stage validation for non-Mesh test doubles.
- Original request UV bounds are validated before inset pixel sampling.
- Texture planning and object preparation orchestration remain below architecture size limits.
- Blender-loaded staged PNG sampling accounts for Spine file-space vertical orientation.
- Float32 pixel-buffer tests use precision-aware comparisons without hiding row-order errors.

## [0.23.0] - 2025-08-20

### Added

- Initial public Legacy release of the add-on.
