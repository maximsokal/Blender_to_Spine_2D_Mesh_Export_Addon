# Changelog

This changelog records public product releases. Internal development milestones are intentionally omitted.

## Unreleased candidate status

Version 0.47.11 is the current release candidate. It keeps the validated Spine 4.1.24 scale topology from 0.47.10 and prevents hidden persisted Connect values from routing ordinary selected-object export into development-only connected composition. Final same-commit package and isolated installation evidence are still required before release.

## [0.47.11] - 2026-07-30

### Changed

- Production `Export Selected Objects` now always builds a standalone multi-object plan.
- Connected and mixed UI planning moved behind the explicit development-only `build_development_connected_ui_export_plan()` entry.
- The extension manifest version is now `0.47.11` so Blender does not cache the routing fix as the older `0.47.10` package.

### Fixed

- Persisted hidden `Object.spine2d_connect_settings.enabled` values from older `.blend` files no longer select `CONNECTED_MULTI_OBJECT` or `MIXED_MULTI_OBJECT` during ordinary Analyze/Export.
- Spine 4.1 selected-object exports no longer fail with a connected-scope capability error when the visible production UI requested an ordinary standalone export.

### Unchanged limitations

- Spine 4.1 remains limited to `2-Axis Rotation + Scale` for single-object and standalone multi-object export.
- Spine 4.1 connected, mixed, and 3-Axis exports remain disabled.
- Connected and mixed composition remain development-only internal capabilities.

## [0.47.10] - 2026-07-30

### Added

- A registered Spine 4.1.24 JSON codec with exact legacy bone-transform field mapping and deterministic serialization.
- A fail-closed capability matrix separating target version, rig profile, and single/standalone/connected/mixed document scope.
- A Blender 5.2 standalone multi-object acceptance tool that runs the production exporter and the exact read-only Spine 4.1 runtime oracle.
- A Spine 4.1 scale-response probe that verifies Scale factors `0.5`, `1.5`, and `2.0` against object bounds around each `*_main` bone.
- Typed weighted-attachment remapping after Spine 4.1 bridge-bone insertion.

### Changed

- Spine 4.1 target adaptation now occurs after canonical attachment projection and document assembly, preserving strict deterministic rig validation.
- Spine 4.1 world-relative Scale keeps the canonical Spine 4.2 semantics and replaces only the unsafe `*_rotate_X` constrained driver with its invertible parent.
- Depth-scale constraints retain their original `*_scale` bone ownership; internal `onlyTranslation` bridge bones provide invertible parents without replacing authored zero scales.
- The extension manifest version is now `0.47.10`.

### Fixed

- Spine 4.1 exports no longer fail during projection because a target-adapted rig was incorrectly revalidated against the canonical Spine 4.2 deterministic plan.
- Standalone Spine 4.1 Scale controls no longer use the rejected `local=true` policy or allow the depth constraint to overwrite scaling on the final layer bones.
- Optional absent top-level `path` constraints are correctly treated as an empty collection by the Blender acceptance worker.

### Limitations

- Spine 4.1 connected, mixed, and 3-Axis exports remain disabled.
- Spine 4.2.43 remains the only target with the complete profile and composition matrix.

## [0.47.5] - 2026-07-29

### Added

- Exact pure regressions against the connected `_build_global_rig`, `_build_global_constraints`, `_renumber_object_constraints`, and `_apply_offsets` behavior from the historical `main` branch.
- A connected-only serialization validator that permits the historical duplicate-order diagnostic while keeping every other Spine structural, cross-reference, mesh, UV, weighted-stream, and animation error blocking.
- Explicit two-axis connected payload tests for global Rotation X, IK, Scale, depth-scale, and Rotation Y targets.

### Changed

- Connected 3-Axis composition no longer creates `all_objects` through the ordinary single-object rig builder. It uses the dedicated Legacy global wrapper with root-space controls, neutral Z layers, and the original helper-bone transforms.
- Connected constraints are no longer sorted after composition. Object constraint arrays retain source object order and global constraints are appended afterward, matching `main`.
- Object constraint orders are assigned by connected Z layer rather than by object count. Objects in one layer intentionally share an order value.
- The 3-Axis global Rotation Z constraint again targets each connected object's base bone instead of generated wrapper layers.
- Legacy object scale compensators remain present at their original standalone order `6`, exactly as in the historical merger.

### Fixed

- Global 3-Axis Rotation X, Rotation Y, Rotation Z, IK, and Scale now use the exact bone lists, targets, offsets, channel mixes, and order formula from the working Legacy connected exporter.
- Legacy connected object mains keep their complete anchor-relative Blender X/Y offsets because generated Legacy wrapper layers are setup-neutral.
- Two-axis connected order phases now use the same Z-layer grouping principle instead of assigning a unique order to every object.

## [0.47.4] - 2026-07-29

### Added

- Runtime-formula regressions that evaluated connected relative-local constraint setup deltas rather than checking only bone names and serialized control rotations.
- Blender-headless assertions for final setup world placement, wrapper rotation deltas, and three-axis global scale channel ownership.
- A dedicated immutable connected setup-correction owner executed after global constraints and object placements were composed.

### Changed

- This candidate attempted to make the connected global wrapper setup-neutral and to compensate generated layer depth through object-main local Y.

### Known issue

- Manual Spine import showed that the approach still produced broken connected rigs. It changed the Legacy hierarchy, constrained bone lists, channel payloads, constraint order phases, and compensator behavior instead of reproducing the working `main` implementation. These changes are superseded by 0.47.5.

## [0.47.3] - 2026-07-29

### Added

- Pure and Blender-headless setup-pose regressions derived from the connected pyramid exports that exposed the deformation.
- Explicit checks for neutral global two-axis controls, object-local Scale controls, disabled global Y-scale mixing, and wrapper-layer global Z targets.

### Known issue

- Manual Spine import showed that global constraint offsets and connected layer placement still produced a broken setup pose. These defects are superseded by later candidates.

## [0.47.2] - 2026-07-29

### Added

- Profile-aware connected constraint schedules and initial connected two-axis support.
- Pure weighted-index, placement, namespace, control, and constraint-order regressions for connected two-axis documents.
- A Blender 5.2 headless connected multi-object regression verifying global and per-object X/Y/Scale controls.

### Fixed

- Connected export no longer rejects `TWO_AXIS_ROTATION_SCALE` before document composition.
- Weighted attachment indices continue to be remapped after the global connected bones are inserted.

### Known issue

- The first connected implementation assigned unique orders per object and built the group wrapper through ordinary object-rig mechanics; manual Spine import later showed this was not compatible with the working Legacy connected behavior.

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
- Texture Coordinate UV and Image Texture nodes continue to sample the original render UV while Blender writes to the generated destination layout.
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
