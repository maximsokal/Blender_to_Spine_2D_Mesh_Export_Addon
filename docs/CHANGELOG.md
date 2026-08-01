# Changelog

This changelog records public product releases. Internal development milestones are intentionally omitted.

## Current candidate status

Version 0.55.16 is the current release candidate. It corrects the connected/mixed Blender acceptance oracle so valid two-axis setup hierarchies are evaluated through full Spine affine inheritance. Production Slice 5 placement, hierarchy, layers, draw order, mixed interleaving, rigs, constraints, attachments, UV topology, texture baking, rendered Camera Projection, and public UI routing are unchanged from 0.55.15.

## [0.55.16] - 2026-08-01

### Fixed

- Added a Blender-independent Spine setup-transform evaluator for connected/mixed acceptance workers.
- Setup positions now compose local translation, rotation, and scale through the complete parent chain.
- `normal` and `onlyTranslation` inheritance are supported explicitly.
- The valid generated chain containing zero helper X-scale, `onlyTranslation(+90)`, layer `-90`, and compensated object-main translation is evaluated instead of rejected.
- Signed-axis and Active Camera connected/mixed workers now share the same setup oracle and report `SPINE_AFFINE_NORMAL_ONLY_TRANSLATION`.
- Duplicate names, missing parents, cycles, conflicting inheritance aliases, unsupported modes, and non-zero shear fail closed.

### Unchanged

- Production Slice 5 code is unchanged.
- Existing rendered Camera Projection, grouped camera overlay, public standalone-only UI routing, Spine codecs, UV topology, and texture baking are unchanged.
- Spine 4.1 connected and mixed restrictions remain fail-closed.

## [0.55.15] - 2026-08-01

### Added

- Immutable `A1ProjectedObjectAnalysis` records for projected origin, bounds, nearest/farthest evaluated vertex depth, source order, component identity, prefix, and owned slots.
- Blender 5.2 headless connected/mixed acceptance for all six signed axes.
- Blender 5.2 headless connected/mixed acceptance for Perspective and Orthographic Active Camera.

### Changed

- Connected Active Camera preparation uses the same anchor-relative projected-origin policy as signed-axis Normal - UV Segments.
- The connected global main stores the selected anchor's absolute projected Object Origin, while object mains remain anchor-relative.
- Connected hierarchy layers remain grouped by projected Object Origin depth.
- Connected setup slots use complete far-to-near object blocks ordered by nearest evaluated vertex depth.
- Mixed composition applies one final slot-only nearest-depth pass across connected and standalone subgroup boundaries.

### Preserved

- Internal slot order, subgroup bones, skins, constraints, attachments, weighted indices, animation metadata, and source Blender state remain unchanged.
- Existing rendered Camera Projection, grouped camera overlay, public standalone-only routing, and Spine codecs remain unchanged.
- Spine 4.1 connected and mixed restrictions remain fail-closed.

## [0.55.14] - 2026-08-01

### Changed

- `prepare_a1_source_geometry()` delegates request resolution, world normalization, projection routing, geometry completion, statistics, and logging to explicit owners.
- Source UV boundary validation occurs immediately after world-transform normalization for signed-axis and Active Camera routes.
- Architecture and ordering contracts became route-aware.

### Fixed

- The append-only `A1DocumentAssemblySettings` suffix includes `compensate_depth_setup_y` after the UV range fields.

## [0.55.13] - 2026-08-01

### Fixed

- Active Camera expected camera-local depth uses captured tuple affine arithmetic in the Blender worker.
- The report records `depthExpectationModel = CAPTURED_TUPLE_AFFINE` and retains the `1e-8` tolerance.
- The screen oracle remains Blender `world_to_camera_view`.

## [0.55.12] - 2026-08-01

### Fixed

- Perspective Active Camera no longer sends projected non-planar n-gons into the strict source triangulator.
- Segmentation, decomposition, and triangulation complete before nonlinear camera projection.
- Already-triangulated immutable regions are projected afterward with Source lineage preserved.

## [0.55.11] - 2026-08-01

### Fixed

- Semantic release-note normalization removes Markdown backtick delimiters in addition to case, whitespace, and hyphenation differences.

## [0.55.10] - 2026-08-01

### Fixed

- The 0.55.9 release contract checks profile-owned Z-index semantics independently instead of requiring one artificial exact phrase.

## [0.55.9] - 2026-08-01

### Fixed

- The Active Camera setup-depth fixture uses profile-owned Z-group indices and resolves the zero-depth group by `z_value == 0.0`.

## [0.55.8] - 2026-08-01

### Fixed

- The Active Camera depth-compensation fixture passes typed `LegacyZGroup` values.
- Release and version contracts were synchronized after the failed 0.55.7 gate.

## [0.55.7] - 2026-08-01

### Added

- Perspective and Orthographic Active Camera projection using export texture dimensions.
- Full evaluated world-geometry and Object Origin projection into centered camera-screen pixels.
- Camera-local nearest/farthest vertex depth for standalone object-block ordering.
- Pure and Blender-headless camera regressions.

### Limitations

- The public projection dropdown was not exposed.
- Connected and mixed Active Camera remained blocked until Slice 5.

## [0.55.6] - 2026-08-01

### Added

- Deterministic projected depth bounds for every prepared Normal - UV Segments object.
- A reusable object-block draw-order planner with tolerance clustering and source-order tie breaking.
- Six-axis standalone Blender acceptance.

### Changed

- Standalone setup slots are emitted far-to-near by nearest projected vertex while preserving internal object slot order.

## [0.55.5] - 2026-08-01

### Fixed

- Positional-layout and public documentation contracts were synchronized with the appended `projection_direction` field.

## [0.55.4] - 2026-08-01

### Fixed

- Release-note semantic checks no longer depend on one exact contiguous phrase.

## [0.55.3] - 2026-08-01

### Fixed

- Signed-axis Blender acceptance uses captured tuple arithmetic instead of mixed Python-float and `mathutils` intermediates.

## [0.55.2] - 2026-08-01

### Added

- Signed-axis projection contracts and production geometry projection for `+X`, `-X`, `+Y`, `-Y`, `+Z`, and `-Z`.
- Object Origin, vertices, normals, face normals, and depth groups are projected into canonical U/V/D space.

## [0.55.1] - 2026-07-31

### Fixed

- Blender headless acceptance synchronizes dependency-graph transforms before baseline capture.
- Two-axis controls use neutral setup rotations while constraint offsets retain reference angles.

## [0.47.11] - 2026-07-30

### Changed

- Public `Export Selected Objects` always builds a standalone multi-object plan.
- Connected and mixed UI planning moved behind an explicit development-only entry.

## [0.47.10] - 2026-07-30

### Added

- Spine 4.1.24 JSON codec and fail-closed target/profile/scope capability matrix.
- Blender-to-runtime standalone acceptance and a behavioral scale-response probe.
- Weighted attachment remapping after Spine 4.1 bridge insertion.

### Fixed

- Spine 4.1 target adaptation occurs after canonical assembly.
- Scale controls preserve world-relative behavior without the rejected `local=true` policy.

## [0.47.5] - 2026-07-29

### Changed

- Connected 3-Axis composition reproduces the historical dedicated global wrapper, constraint payloads, source-order arrays, and Z-layer order sharing.
- Connected 2-Axis composition uses the same Z-layer scheduling principle.

## [0.47.4] - 2026-07-29

### Known issue

- An attempted setup-neutral wrapper correction still produced broken connected rigs and was superseded by 0.47.5.

## [0.47.3] - 2026-07-29

### Added

- Setup-pose regressions derived from connected pyramid exports.

### Known issue

- Global setup offsets remained incorrect and were superseded by later candidates.

## [0.47.2] - 2026-07-29

### Added

- Initial profile-aware connected two-axis support and constraint scheduling.

### Known issue

- Unique per-object orders and an ordinary object-rig wrapper were later shown incompatible with the historical connected behavior.

## [0.47.1] - 2026-07-29

### Fixed

- Edge-on regions that collapse only after XY projection are omitted instead of invalidating otherwise correct 3D geometry.

## [0.47.0] - 2026-07-29

### Added

- Shared vertex-bone optimization and weighted-stream remapping for coincident segment-boundary vertices.

### Changed

- Single-object attachments preserve Blender Object Origin as the exported rotation pivot.

## [0.41.3] - 2026-07-28

### Fixed

- Semantic baking preserves the source render UV while writing to generated `SpineBakeUV`.

## [0.41.2] - 2026-07-28

### Changed

- Source Z values and explicit height overrides use one canonical identity.

### Known issue

- The representative textured sword still rendered incorrectly; the candidate was not released.

## [0.41.1] - 2026-07-27

### Fixed

- Attachment topology and hull validation returned to the stable local projected pixel plane.

## [0.41.0] - 2026-07-27

### Added

- Exact post-assembly correspondence validation for UVs, triangles, hulls, edges, and weighted bone indices.

## [0.40.0] - 2026-07-27

### Added

- Blender 5.2 extension packaging and runtime gate.
- Typed Spine 4.2.43 document pipeline.
- Normal - UV Segments and rendered Camera Projection modes.
- Deterministic segmentation, custom seams, UV lineage, manifold decomposition, physical hull normalization, material policies, texture baking, single/standalone/connected/mixed composition, readiness analysis, atomic output, logging, and Blender regression layers.

### Changed

- Replaced the Legacy production orchestration with explicit application, domain, Blender adapter, and infrastructure boundaries.
