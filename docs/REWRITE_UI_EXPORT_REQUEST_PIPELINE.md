# Rewrite UI-to-export request pipeline

## Purpose

This document records how mutable Blender UI state becomes one deterministic Rewrite export request. It also defines the ownership boundary between selection, Scene capture, preparation and output services.

The legacy `main.py` pipeline remains available only through the explicit lazy-loaded Legacy backend. It is not the Rewrite request assembler.

## Previous legacy shape

The legacy single-object path combined the following responsibilities in `main.py`:

- reading Scene properties and global configuration values;
- mutating module-level texture dimensions;
- creating Blender working copies;
- UV preparation and segmentation;
- maintaining parallel lists of vertices, UVs, triangles and edges;
- baking;
- writing intermediate JSON files;
- merging serialized JSON;
- final output and cleanup.

This made data ownership implicit. A value could be held simultaneously by Blender datablocks, module globals, temporary objects, Python lists and intermediate JSON files.

## Current Rewrite boundary

The registered operators contain only backend selection and result reporting:

```text
object.save_uv_as_json
object.spine2d_multi_export
```

For Rewrite they call:

```text
single_object_operator / ui operator
    -> blender_adapter.a1_ui_bridge
```

`a1_ui_bridge` is a stable compatibility facade. Physical ownership is split as follows:

```text
a1_ui_selection.py
    -> object name and stable in-process Blender RNA identity
    -> active Mesh validation
    -> deterministic active-first selected Mesh ordering
    -> per-object sequence and Connect capture
    -> immutable _ObjectExportProfile

a1_ui_scene_capture.py
    -> output directory and images path
    -> texture, seam, angular and projection settings
    -> render-engine capture
    -> immutable _SceneExportProfile

a1_ui_settings.py
    -> A1SingleObjectExportSettings
    -> deterministic A1MultiObjectSource values

a1_ui_router.py
    -> single/standalone/connected/mixed route selection
    -> production output-service invocation

a1_ui_rna.py
    -> compatibility re-exports only

a1_ui_bridge.py
    -> stable imports for operators, focused tests and existing callers
```

One immutable `_SceneExportProfile` is captured for the complete transaction and one `_ObjectExportProfile` for every selected object. Stable Blender RNA identity uses `as_pointer()` when available to match transient wrappers inside the running Blender process; it is not persisted as a cross-session object ID.

The bridge does not pass raw dictionaries, parallel arrays or serialized JSON to the Rewrite pipeline.

## Single-object payload

```text
_SceneExportProfile
_ObjectExportProfile
    -> A1SingleObjectExportSettings
    -> export_a1_single_object()
    -> prepare_a1_object()
    -> PreparedA1Object
    -> typed texture dispatch
    -> optional B4 layout finalization
    -> typed SpineDocument serialization
    -> one atomic commit
```

`PreparedA1Object` owns immutable geometry, UV, material-analysis, bake-plan, rig and typed Spine values. Its only live Blender reference is `source_object`, retained at the adapter boundary for actual bake/render execution.

## Multi-object payload

Selected Mesh objects are ordered as:

1. active selected Mesh;
2. all remaining selected Mesh objects ordered by case-folded object name.

Transient Blender RNA wrappers are matched through stable RNA identity rather than Python `is`.

For every object the settings layer creates:

```text
A1MultiObjectSource(
    source_object=<live Blender handle>,
    component_id="object_<index>:<object name>",
    animation_namespace="object_<index>",
    settings=<immutable A1SingleObjectExportSettings>,
)
```

The sources are partitioned once from captured Connect flags:

```text
no connected subgroup   -> STANDALONE
all connected           -> CONNECTED
connected + standalone  -> MIXED
exactly one Connect flag -> deterministic STANDALONE fallback with ExportIssue
```

## Preparation ownership

Preparation modules own only in-memory preparation:

```text
a1_object_preparation.py
    -> PreparedA1Object

a1_multi_object_export.py
    -> prepare_a1_multi_object()
    -> PreparedA1MultiObject

a1_mixed_object_export.py
    -> prepare_a1_mixed_object()
```

They read/evaluate Blender objects, create immutable snapshots and compose typed draft documents. They must not be selected by the UI as final output services.

## Texture execution ownership

Before any object-bake or B4 dispatch, `texture_executor.py` creates:

```text
TextureExecutionRequest(
    source_object,
    target_snapshot: MeshSnapshot,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
)
```

The request validates types and requires `target_snapshot.source_object_id == plan.source_object_id` before Blender mutation.

For semantic object bake, physical responsibilities are:

```text
semantic_bake_validation.py -> complete pre-mutation runtime validation
semantic_bake_execution.py  -> reversible Blender execution into existing reservations
semantic_bake_output.py     -> reservation, atomic commit and typed result
semantic_bake_executor.py   -> compatibility facade
```

Validation completes before reservation. Caller-owned staging does not commit. Direct execution performs one atomic commit and verifies exact frame-task path order.

B4 keeps two intentional contracts:

- detailed production staging returns crop/contour layout for post-render document finalization;
- reservations-only compatibility staging keeps full-frame output for callers that serialized JSON before staging.

## Output ownership

The production output entry points are:

```text
a1_single_object_export.py::export_a1_single_object

a1_multi_object_output.py::export_a1_multi_object

a1_mixed_object_output.py::export_a1_mixed_object
```

Multi and mixed output services:

1. prepare every object;
2. reserve one final JSON path;
3. stage texture tasks inside one transaction;
4. finalize every B4 attachment after render-derived layout exists;
5. optionally build the grouped connected B4 layer;
6. recompose typed Spine documents using finalized objects;
7. serialize the final document once;
8. commit JSON and every texture atomically.

The UI router imports these output modules directly. Compatibility facades do not own output implementation.

## Remaining architecture debt

### Duplicate historical export functions

`a1_multi_object_export.py` and `a1_mixed_object_export.py` still contain historical functions named `export_a1_multi_object` and `export_a1_mixed_object`. They are no longer production UI entry points, but the names remain a future misuse risk.

Safe cleanup requires splitting preparation-only code into explicit modules or replacing those historical functions with compatibility facades that delegate to the output services.

### Draft composition before B4 finalization

`prepare_a1_multi_object()` returns a complete draft document because it is also a public preparation API. Production output must compose again after B4 finalization. This is correct but performs duplicate composition and makes `PreparedA1MultiObject.document` potentially stale for camera projection until output finalization.

A future cleanup can introduce a component-preparation result that contains objects and output validation only, with composition performed exactly once by callers that require final output.

### Generic `ExportRequest`

The generic application `ExportRequest` contract is validated and tested but the production A1 operators use A1-specific settings/source contracts instead. Its intended ownership should be clarified: either make it the actual top-level request envelope or remove it from the A1 design.

### Duplicate low-level bake orchestration

`bake_executor_core.py` still retains historical orchestration helpers in addition to the primitives used by the split semantic executor. Their usages must be audited before removal because the public failure-injection hooks and Blender fixtures depend on stable operator boundaries.

## Regression coverage

- post-render output-service module identity;
- one shared immutable Scene snapshot across multi-object settings;
- stable active-object ordering across transient RNA wrappers;
- deterministic component IDs and animation namespaces;
- registered Rewrite multi operator;
- registered connected B4 multi operator producing grouped texture and JSON metadata;
- explicit lazy-loaded Legacy backend;
- no automatic Legacy fallback after Rewrite failure;
- facade-only `a1_ui_rna` and `semantic_bake_executor`;
- physical UI selection/Scene capture ownership;
- semantic validation before transaction creation and reservation;
- no commit capability in semantic execution;
- strict committed-path order;
- typed texture dispatch before Blender mutation.
