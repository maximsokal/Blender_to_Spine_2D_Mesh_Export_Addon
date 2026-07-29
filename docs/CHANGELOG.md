# Changelog

This changelog records public product releases. Internal development milestones are intentionally omitted.

## Unreleased candidate status

Version 0.47.2 is a development candidate and is not validated for release. It includes the 0.47.1 projected-region fixes and adds a dedicated connected multi-object implementation for `TWO_AXIS_ROTATION_SCALE`. The candidate must not be merged or released until all automated layers and fresh manual Blender-to-Spine exports pass on the same commit.

## [0.47.2] - 2026-07-29

### Added

- Profile-aware connected constraint schedules that preserve the existing six-constraint three-axis order and add a dedicated five-phase two-axis order.
- A connected global X/Y/Scale rig built through the validated two-axis rig owner rather than through a synthetic legacy constraint.
- Pure weighted-index, placement, namespace, control, and constraint-order regressions for connected two-axis documents.
- A Blender 5.2 headless connected multi-object regression verifying global and per-object X/Y/Scale controls.

### Fixed

- Connected export no longer rejects `TWO_AXIS_ROTATION_SCALE` before document composition.
- Every connected two-axis object retains its own X, Y, and Scale controls while the subgroup receives independent global X, Y, and Scale controls.
- Global and object constraints receive unique contiguous orders across Rotation X, IK, Uniform Scale, X Depth Scale, and Rotation Y phases.
- Weighted attachment indices continue to be remapped after the global connected bones are inserted.

## [0.47.1] - 2026-07-29

### Added

- Blender-independent projected-region filtering that preserves immutable source geometry, UVs, and Source lineage while materializing only visible X/Y disk components.
- A Blender 5.2 two-axis standalone multi-object regression containing a valid 3D side wall whose two vertices collapse to the same Spine X/Y point.

### Fixed

- Normal UV multi-object export no longer fails when a legitimate three-dimensional side face is edge-on and has zero area only after projection into Spine pixel space.
- Completely edge-on regions are omitted instead of creating invalid mesh triangles; an object is blocked only when every prepared region is invisible in X/Y.
- Remaining visible regions are renumbered densely before slots, attachments, vertex bones, and weighted indices are built.

## [0.47.0] - 2026-07-29

### Added

- Shared vertex-bone optimization for coincident segment-boundary vertices belonging to the same object and Z parent.
- Typed weighted-stream remapping that preserves local influence coordinates and weights while compacting final bone indices.
- Regression coverage for the four-face pyramid: twelve segment vertex bones compact to four canonical bones without changing UVs, triangles, hulls, edges, or attachment order.

### Changed

- Single-object two-axis control bones serialize with neutral setup rotations while reference angles remain in transform-constraint offsets.
- Single-object attachments preserve Blender Object Origin as the exported rotation pivot instead of moving the pivot to the geometry bounds center.
- Extension version advanced from the 0.41 development series to 0.47.0.

### Fixed

- Segmented objects no longer export one independent Spine bone for every repeated copy of the same physical boundary vertex.
- Weighted mesh bone indices are rebuilt after vertex-bone compaction instead of relying on pre-compaction contiguous ranges.
- Older saved Scenes retain the compatibility rig profile when Blender RNA defaults are rebound during extension registration.

## [0.41.3] - 2026-07-28

### Fixed

- Semantic baking no longer replaces the source material render UV with generated `SpineBakeUV`.
- Texture Coordinate UV and Image Texture nodes now continue to sample the original render UV while Blender writes to the generated destination layout.
- `bpy.ops.object.bake` receives the destination UV layer explicitly instead of inferring it from the shader render role.
- The representative sword material graph is covered by a Blender 5.2 headless regression: Texture Coordinate UV to Mapping to Image Texture to Principled BSDF.

### Changed

- Bake UV activation validates two independent roles: active destination UV and unique source render UV.
- The previous 0.41.2 representative-asset failure is retained as historical evidence rather than treated as a release result.

## [0.41.2] - 2026-07-28

### Changed

- Source Z values are canonicalized to the historical four-decimal Legacy identity before Z-group creation and exact `SourceVertexId` binding.
- Explicit Z-group height overrides use the same canonical identity.

### Added

- Regressions preventing small evaluated-geometry floating-point differences from creating extra depth groups or inflating the base rig bone count.

### Known issue

- This change did not resolve the incorrect material placement observed for the representative textured sword asset. Version 0.41.2 remains a failed development candidate rather than a release.

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

- Blender could materialize temporary polygons in a different order from immutable snapshot faces while the bake material stage still assigned source material slots positionally.
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
