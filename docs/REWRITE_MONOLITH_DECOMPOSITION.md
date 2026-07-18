# Rewrite monolith decomposition

## Scope

This cleanup removes high-risk orchestration monoliths without changing the public Blender operators or the existing single/multi/mixed output contracts.

## 1. Object preparation

Previous ownership:

```text
a1_object_preparation.prepare_a1_object
  validation + mesh read + lineage + geometry + UV
  + material analysis + bake planning + rig + document
```

Current ownership:

```text
prepare_a1_object
  -> a1_source_geometry_preparation.prepare_a1_source_geometry
  -> a1_uv_preparation.prepare_a1_uv
  -> a1_texture_planning.prepare_a1_texture_plan
  -> a1_document_preparation.prepare_a1_document
```

Every stage returns an immutable typed result. `A1ObjectPreparationError` carries the exact `A1SingleObjectStage`, object ID, accumulated warnings, and partial statistics.

The public compatibility surface remains in `a1_object_preparation`:

- `PreparedA1Object`;
- `A1ObjectPreparationError`;
- `StatisticsValue`;
- `prepare_a1_object`.

## 2. Multi and mixed output

The per-object loop previously existed independently in both output services:

```text
stage texture plan
-> finalize camera projection
-> merge component statistics
```

It now belongs to:

```text
a1_output_staging.stage_and_finalize_a1_objects
```

Final document and grouped-camera statistics now belong to:

```text
a1_output_statistics.record_final_document_statistics
a1_output_statistics.record_grouped_camera_statistics
```

Grouped overlay ownership validation belongs to:

```text
a1_grouped_output.apply_staged_grouped_camera_overlay
```

Atomic lifecycle operation names are distinct:

```text
a1-single-object
a1-multi-object
a1-mixed-object
```

## 3. UI request capture

The previous `a1_ui_bridge.py` mixed Blender RNA reads, settings conversion, source construction, route selection, and output invocation in one file.

The first decomposition produced:

```text
a1_ui_rna.py       -> Blender RNA identity and immutable snapshots
a1_ui_settings.py  -> application settings and source construction
a1_ui_router.py    -> single/standalone/connected/mixed routing
a1_ui_bridge.py    -> stable imports for existing callers and tests
```

`a1_ui_rna.py` still mixed object selection identity with mutable Scene-property capture. The physical ownership is now:

```text
a1_ui_selection.py      -> object names, RNA identity, active/selected Mesh order,
                           Connect flag, immutable _ObjectExportProfile
a1_ui_scene_capture.py  -> Scene property reads and immutable _SceneExportProfile
a1_ui_settings.py       -> application settings and A1MultiObjectSource construction
a1_ui_router.py         -> output route selection and invocation
a1_ui_rna.py            -> compatibility re-exports only
a1_ui_bridge.py         -> stable production facade
```

Runtime modules import `a1_ui_selection` and `a1_ui_scene_capture` directly. Existing private helper names remain available through `a1_ui_rna` and `a1_ui_bridge`, but neither compatibility facade owns implementation.

## 4. Semantic object-bake execution

Previous ownership:

```text
semantic_bake_executor.py
  input and renderer validation
  + bpy/context/Scene resolution
  + temporary Mesh/material/Scene orchestration
  + semantic pass execution and RGBA composition
  + output reservation
  + atomic commit
  + BakeExecutionResult construction
```

Current ownership:

```text
semantic_bake_validation.py
  -> validate source/snapshot/plan/settings
  -> validate source object ID, UV and material coverage
  -> resolve Cycles renderer contract, bpy, Context and Scene
  -> validate scene-aware runtime context
  -> SemanticBakeRuntime

semantic_bake_execution.py
  -> consume SemanticBakeRuntime and existing reservations
  -> reversible Scene, temporary Mesh and copied-material scope
  -> frame/pass execution and straight-RGBA composition
  -> no transaction creation, reservation, commit or rollback

semantic_bake_output.py
  -> validate before reservation
  -> reserve caller-owned staging outputs
  -> direct atomic transaction and one commit
  -> strict committed-path order validation
  -> typed BakeExecutionResult

semantic_bake_executor.py
  -> compatibility re-exports only
```

Invalid plans, snapshots, execution settings, renderer combinations, UV bindings and material coverage now fail before `AtomicFileTransaction.reserve()` and before Blender mutation. Caller-owned staging never commits. Direct execution commits once and requires committed paths to match frame-task order exactly.

The central `texture_executor.py` additionally captures a typed `TextureExecutionRequest` before dispatching object bake or B4 camera projection.

## Single Connect fallback

Exactly one selected object with `Connect` enabled cannot form a connected group. The router still falls back to standalone export, but this is no longer visible only in debug logs.

The returned `ExportResult` contains:

```text
severity: WARNING
code: A1_SINGLE_CONNECT_FALLBACK
stage: VALIDATE_REQUEST
object_id: the object with Connect enabled
context:
  selected_object_count
  connected_object_count = 1
  fallback_mode = STANDALONE
```

The result statistics include:

```text
single_connect_fallback_count = 1
```

## Runtime trace contract

The Blender pipeline probe follows physical implementation ownership rather than compatibility facades. It requires:

- all four object-preparation stages;
- shared multi/mixed staging;
- shared final-document statistics;
- `a1_ui_router` entrypoints rather than `a1_ui_bridge` implementation;
- physical UI selection and Scene-capture modules;
- semantic validation, execution and output ownership rather than the semantic facade.

## Validation performed outside CI

No GitHub Actions workflow was triggered.

Validation for the newest split included:

- Python compilation of every new or replaced production module;
- architecture tests for UI compatibility facades and physical ownership;
- architecture tests proving semantic execution cannot create or commit transactions;
- ordering checks proving validation precedes transaction creation and reservation;
- typed texture gateway checks;
- preservation of existing private UI bridge helper re-exports.

Full repository pytest and real Blender 4.4 integration remain separate manual release gates.
