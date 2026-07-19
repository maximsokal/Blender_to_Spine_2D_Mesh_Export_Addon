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

Every stage returns an immutable typed result.
`A1ObjectPreparationError` preserves the exact stage, object ID, accumulated
warnings and partial statistics.

The public compatibility surface remains in `a1_object_preparation`.

## 2. Multi and mixed output

Shared per-object staging and finalization belongs to:

```text
a1_output_staging.stage_and_finalize_a1_objects
```

Shared statistics ownership belongs to:

```text
a1_output_statistics.record_final_document_statistics
a1_output_statistics.record_grouped_camera_statistics
```

Grouped overlay validation belongs to:

```text
a1_grouped_output.apply_staged_grouped_camera_overlay
```

Named atomic lifecycle operations remain distinct:

```text
a1-single-object
a1-multi-object
a1-mixed-object
```

## 3. UI request capture

Physical ownership is:

```text
a1_ui_selection.py
  -> object names, RNA identity, active/selected Mesh ordering,
     Connect flag and immutable _ObjectExportProfile

a1_ui_scene_capture.py
  -> Scene property reads and immutable _SceneExportProfile

a1_ui_settings.py
  -> application settings and A1MultiObjectSource construction

a1_ui_router.py
  -> single, standalone, connected and mixed route selection

a1_ui_rna.py
  -> compatibility re-exports only

a1_ui_bridge.py
  -> stable production facade
```

Runtime modules import the physical selection and Scene-capture owners directly.
Historical private helper names remain available through compatibility facades.

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
  -> reversible Scene/Mesh/material execution and pass composition

semantic_bake_output.py
  -> reservation, atomic transaction, commit and typed result

semantic_bake_executor.py
  -> compatibility re-exports only

bake_executor_core.py
  -> sole bpy.ops.object.bake hook
  -> compatibility private re-exports
```

Invalid requests fail before output reservation and Blender mutation.
Caller-owned staging never commits. Direct execution commits once and accepts
only exact committed-path order.

The former duplicate object-bake pipeline in `bake_executor_core.py` has been
removed.

## 5. B4 camera projection execution

The previous `camera_projection_executor_core.py` mixed runtime validation,
Scene mutation, rendering, coverage union, contour construction, crop rewrite,
reservation, commit and result construction.

Physical ownership is now:

```text
camera_projection_error.py
  -> shared CameraProjectionExecutionError

camera_projection_validation.py
  -> source/plan/settings validation
  -> renderer and output-policy resolution
  -> bpy, Context, Scene, View Layer and Scene-context validation
  -> reservation-order validation
  -> CameraProjectionRuntime

camera_projection_state.py
  -> reversible Scene/frame/visibility state
  -> source-only camera visibility
  -> per-frame Scene configuration

camera_projection_execution.py
  -> render full-frame staged files inside the reversible state scope
  -> no coverage, crop, reservation or commit

camera_projection_image.py
  -> staged image decode and crop rewrite primitives

camera_projection_postprocess.py
  -> coverage union, cleanup, layout, contour and crop rewrite
  -> runs only after reversible render state has been restored

camera_projection_output.py
  -> caller-owned reservation
  -> detailed and compatibility staging
  -> direct atomic transaction and one commit
  -> strict committed-path order
  -> BakeExecutionResult

camera_projection_executor_core.py
  -> compatibility re-exports only

camera_projection_executor.py
  -> stable public facade
```

The detailed path is:

```text
validate
-> reserve
-> render all frames
-> restore Blender state
-> decode coverage
-> build one sequence layout
-> rewrite every staged frame
-> return reservations + layout
```

The historical reservations-only path keeps full-frame output and performs no
coverage decode or crop rewrite.

Direct execution validates before transaction creation and requires:

```text
committed paths == reservation final paths == frame-task output paths
```

The grouped B4 executor now imports the shared physical error, validation,
execution and state helpers, but its grouped render/output pipeline remains a
separate future decomposition slice.

## Single Connect fallback

Exactly one selected object with `Connect` enabled still falls back to
standalone export. The result includes warning
`A1_SINGLE_CONNECT_FALLBACK` and increments
`single_connect_fallback_count`.

## Runtime trace contract

The Blender pipeline probe follows physical production ownership rather than
compatibility facades. It requires object-preparation stages, shared multi/mixed
staging, final statistics, UI router ownership and typed texture dispatch.

## Validation performed outside CI

No GitHub Actions workflow was triggered.

Validation for the latest decomposition includes:

- Python compilation of every new or replaced production module;
- architecture tests for UI and semantic bake ownership;
- retirement checks for the duplicate object-bake core;
- B4 architecture checks for validation, state, execution, postprocess and
  output ownership;
- checks that B4 validation precedes reservation and transaction creation;
- checks that postprocessing begins only after reversible rendering returns;
- checks that direct B4 execution commits exactly once;
- compatibility checks for historical private helper aliases.

The complete repository pytest suite and real Blender 4.4 integration matrices
remain separate manual release gates.
