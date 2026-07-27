# Changelog

This changelog records public product releases. Internal development milestones are intentionally omitted.

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