# Rewrite monolith decomposition

## Scope

This cleanup removes high-risk orchestration monoliths without changing public
Blender operators or the existing single, multi, connected and mixed output
contracts.

## 1. Object preparation

```text
prepare_a1_object
  -> a1_source_geometry_preparation.prepare_a1_source_geometry
  -> a1_uv_preparation.prepare_a1_uv
  -> a1_texture_planning.prepare_a1_texture_plan
  -> a1_document_preparation.prepare_a1_document
```

Every stage returns an immutable typed result. `A1ObjectPreparationError`
preserves stage, object ID, warnings and partial statistics. Historical imports
remain available from `a1_object_preparation`.

## 2. Multi and mixed output

Shared ownership:

```text
a1_output_staging.stage_and_finalize_a1_objects
a1_output_statistics.record_final_document_statistics
a1_output_statistics.record_grouped_camera_statistics
a1_grouped_output.apply_staged_grouped_camera_overlay
```

Named atomic lifecycles remain:

```text
a1-single-object
a1-multi-object
a1-mixed-object
```

Multi and mixed output own one JSON plus individual plus grouped texture
transaction and one final commit.

## 3. UI request capture

```text
a1_ui_selection.py
  -> object names, RNA identity, Mesh ordering, Connect flag,
     immutable _ObjectExportProfile

a1_ui_scene_capture.py
  -> Scene property reads and immutable _SceneExportProfile

a1_ui_settings.py
  -> application settings and A1MultiObjectSource construction

a1_ui_router.py
  -> single, standalone, connected and mixed routing

a1_ui_rna.py
  -> compatibility re-exports only

a1_ui_bridge.py
  -> stable production facade
```

Runtime code imports physical selection and Scene-capture owners directly.

## 4. Semantic object-bake execution

```text
bake_execution_error.py
  -> shared BakeExecutionError

semantic_bake_validation.py
  -> request, Blender context and reservation validation
  -> SemanticBakeRuntime

semantic_bake_image_io.py
  -> UV activation, image lifecycle, frame changes and staged writes

semantic_bake_execution.py
  -> reversible Scene/Mesh/material execution and composition

semantic_bake_output.py
  -> reservation, direct transaction, one commit and typed result

semantic_bake_executor.py
  -> compatibility re-exports only

bake_executor_core.py
  -> sole bpy.ops.object.bake hook and compatibility private re-exports
```

Invalid requests fail before reservation and Blender mutation. Caller-owned
staging never commits. Direct execution commits once and accepts exact path
order. The duplicate object-bake pipeline was removed from
`bake_executor_core.py`.

## 5. Single B4 camera projection

The former `camera_projection_executor_core.py` mixed validation, Scene
mutation, rendering, coverage, crop, reservation, commit and result
construction.

```text
camera_projection_validation.py
  -> complete request and reservation validation
  -> CameraProjectionRuntime

camera_projection_state.py
  -> reversible Scene/frame/visibility state

camera_projection_execution.py
  -> full-frame rendering only

camera_projection_image.py
  -> staged image decode and single/grouped crop rewrite primitives

camera_projection_postprocess.py
  -> shared ProjectionPostprocessRequest
  -> single/grouped coverage, layout and crop engine

camera_projection_output.py
  -> single caller-owned staging
  -> direct transaction, one commit and BakeExecutionResult

camera_projection_executor_core.py
  -> compatibility re-exports only

camera_projection_executor.py
  -> stable public facade
```

Single detailed flow:

```text
validate
-> reserve
-> render all frames
-> restore Blender state
-> shared coverage/layout/crop
-> return reservations + layout
```

The full-frame compatibility path performs no coverage decode or crop.
Direct execution validates before transaction creation and requires:

```text
committed paths == reservation final paths == frame-task output paths
```

## 6. Grouped B4 camera projection

The former `grouped_camera_projection_executor.py` mixed Blender RNA identity,
runtime validation, visibility, reservation, rendering, coverage and crop.

```text
grouped_camera_projection_validation.py
  -> object name and RNA identity
  -> complete grouped request validation
  -> per-source single-B4 runtime validation
  -> common Scene/renderer/output policy
  -> strict reservation-order validation
  -> GroupedCameraProjectionRuntime

grouped_camera_projection_visibility.py
  -> grouped source camera visibility
  -> direct-camera isolation of other renderables

grouped_camera_projection_execution.py
  -> reversible grouped full-frame render only

grouped_camera_projection_postprocess.py
  -> adapter to shared process_projection_outputs()
  -> grouped diagnostics

grouped_camera_projection_output.py
  -> validate before reserve
  -> caller-owned grouped reservation and staging
  -> no transaction creation and no commit

grouped_camera_projection_executor.py
  -> compatibility re-exports only
```

Grouped flow:

```text
validate all sources and output policy
-> validate caller transaction
-> reserve grouped frames
-> render all grouped frames
-> restore Blender state
-> shared coverage/layout/crop
-> return GroupedCameraProjectionStageResult
```

Multi and mixed production callers import the physical grouped output owner.
The compatibility facade retains historical private names for object identity,
runtime validation, visibility and reservation.

## 7. Shared B4 postprocess

Single and grouped B4 use one `ProjectionPostprocessRequest` and one
`process_projection_outputs()` implementation.

The shared engine owns:

- deterministic staged alpha decode;
- one `O(width * height)` sequence max-union;
- coverage cleanup;
- stable crop;
- contour and disconnected-component fallback;
- exact triangulation;
- single/grouped staged image rewrite.

It has no Scene mutation, render operator, reservation, transaction or commit.

## 8. Recursive shader-graph analysis

The former `shader_graph_analyzer.py` mixed Blender compatibility details,
recursive traversal, semantic policy and snapshot assembly.

```text
shader_graph_error.py
  -> MaterialGraphAnalysisError

shader_graph_rna.py
  -> RNA identity, names and temporary-node filtering
  -> safe node/link/socket iteration
  -> renderer-target and Material Output resolution
  -> group-interface socket compatibility

shader_graph_traversal.py
  -> RecursiveShaderGraphWalker
  -> nested groups, muted bypasses, cycles and instance-qualified IDs
  -> frozen ShaderGraphTraversalResult

shader_graph_semantics.py
  -> semantic channels and material dependencies
  -> no node/link traversal ownership

shader_graph_snapshot.py
  -> deterministic node/link ordering
  -> immutable snapshots and parallel live-node order

shader_graph_analysis.py
  -> renderer-specific orchestration
  -> MaterialGraphAnalysisResult

shader_graph_analyzer.py
  -> compatibility re-exports only
```

Physical flow:

```text
resolve material and renderer target
-> select effective Material Output
-> traverse reachable sockets/groups
-> freeze traversal result
-> derive semantic channels/dependencies
-> build deterministic snapshot + parallel live-node tuple
```

`material_analyzer.py` and the public adapter package import physical owners
directly. Historical public and private names remain available from the
compatibility facade.

## 9. Production shader capability gate

The former `production_shader_capabilities.py` mixed live graph re-analysis,
audit rebuilding, Alpha proxy policy, source-UV inspection, material-slot
orchestration and final B1-B4 routing.

```text
production_shader_capability_error.py
  -> ProductionShaderCapabilityError

production_shader_capability_merge.py
  -> audit extension through shared finding ordering

production_shader_capability_runtime.py
  -> live material graph re-analysis
  -> immutable snapshot parity
  -> live-node alignment and mute enrichment

production_shader_capability_proxy.py
  -> Alpha Group/Reroute/muted bypass findings

production_shader_capability_uv.py
  -> source UV layers, active_render and socket state
  -> Image Texture, Texture Coordinate, Normal Map, Tangent and UV Map findings

production_shader_capability_object_audit.py
  -> object/material-slot orchestration
  -> immutable audit plus proxy/UV enrichment

production_shader_capability_routing.py
  -> strongest object capability
  -> deterministic failure messages
  -> B1-B4 texture-plan selection

production_shader_capabilities.py
  -> compatibility re-exports only
```

Physical production flow:

```text
re-analyze renderer-specific live graph
-> compare output, qualified nodes, links, channels, dependencies and issues
-> validate snapshot/live-node alignment
-> enrich current mute state
-> run immutable capability audit
-> apply Alpha proxy boundary
-> apply source UV boundary
-> select object bake, B4 or explicit failure
```

Equal-name nodes in reused groups are matched by `node_id` and `group_path`, not
by name alone. Live UV or mute state therefore cannot silently move to another
group instance between analysis and planning.

`a1_texture_planning.py` imports physical object-audit and routing owners. The
routing owner performs no node-tree, socket or source-UV inspection. Audit
extension reuses `shader_capability_findings.order_unique_findings()` so one
finding key/order contract is shared by immutable and production audits.

Historical public and private names remain available from the facade.

## Single Connect fallback

Exactly one selected object with `Connect` enabled still falls back to
standalone export. The result includes warning `A1_SINGLE_CONNECT_FALLBACK` and
increments `single_connect_fallback_count`.

## Runtime trace contract

The Blender pipeline probe follows physical production ownership rather than
compatibility facades. It requires object-preparation stages, shared multi/mixed
staging, final statistics, UI router ownership and typed texture dispatch.

Grouped B4 tracing follows validation, visibility, execution, shared
postprocess and physical grouped output rather than the compatibility executor.
Shader-graph planning follows physical analysis/traversal owners rather than
`shader_graph_analyzer.py`. Capability planning follows physical production
object-audit and routing owners rather than `production_shader_capabilities.py`.

## Validation performed outside CI

No GitHub Actions workflow was triggered.

Validation for the latest decomposition includes:

- Python compilation of every new/replaced production capability module;
- source import-graph loading with domain stubs;
- acyclic physical import ownership;
- compatibility facade alias checks;
- renderer-specific graph parity across qualified nodes, links and semantics;
- equal-name reused-group instance swap rejection;
- live-node count and ordering checks before UV inspection;
- shared finding ordering and first-reason retention;
- Alpha proxy and named UV finding-code preservation;
- multiple active render UV rejection;
- physical caller imports;
- absence of `bpy.ops` access in all split modules;
- preservation of existing single/grouped B4 architecture boundaries.

The complete repository pytest suite and real Blender 4.4 integration matrices
remain separate manual release gates.
