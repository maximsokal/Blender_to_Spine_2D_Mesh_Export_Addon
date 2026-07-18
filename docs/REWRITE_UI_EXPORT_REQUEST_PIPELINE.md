# Rewrite UI-to-export request pipeline

## Purpose

This document records how mutable Blender UI state becomes one deterministic Rewrite export
request. It also defines the ownership boundary between preparation and output services.

The legacy `main.py` pipeline remains available only through the explicit Legacy backend. It is
not the Rewrite request assembler.

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

This made data ownership implicit. A value could be held simultaneously by Blender datablocks,
module globals, temporary objects, Python lists and intermediate JSON files.

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

`a1_ui_bridge` is the only production boundary that reads Rewrite Scene/Object RNA.

It captures:

- one immutable `_SceneExportProfile` for the complete transaction;
- one `_ObjectExportProfile` for every selected object;
- deterministic active-first object order;
- stable Blender RNA identity using `as_pointer()` when available;
- shared Scene geometry/bake settings;
- per-object sequence start/count and Connect flag.

The bridge does not pass raw dictionaries, parallel arrays or serialized JSON to the Rewrite
pipeline.

## Single-object payload

```text
_SceneExportProfile
_ObjectExportProfile
    -> A1SingleObjectExportSettings
    -> export_a1_single_object()
    -> prepare_a1_object()
    -> PreparedA1Object
    -> texture staging
    -> optional B4 layout finalization
    -> typed SpineDocument serialization
    -> one atomic commit
```

`PreparedA1Object` owns immutable geometry, UV, material-analysis, bake-plan, rig and typed Spine
values. Its only live Blender reference is `source_object`, retained at the adapter boundary for
actual bake/render execution.

## Multi-object payload

Selected Mesh objects are ordered as:

1. active selected Mesh;
2. all remaining selected Mesh objects ordered by case-folded object name.

Transient Blender RNA wrappers are matched through stable RNA identity rather than Python `is`.

For every object the bridge creates:

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
exactly one Connect flag -> deterministic STANDALONE fallback with warning log
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

They read/evaluate Blender objects, create immutable snapshots and compose typed draft documents.
They must not be selected by the UI as final output services.

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

The UI bridge imports these output modules directly. A regression test verifies their module
identity, and the real Blender multi-operator fixture requires grouped B4 output through the
registered button.

## Defect fixed during request-pipeline review

The bridge previously imported `export_a1_multi_object` and `export_a1_mixed_object` from the
preparation modules. Those modules still contained older output implementations that serialized
JSON before B4 render-derived finalization and used reservations-only staging.

Consequences included:

- stale full-frame B4 attachments paired with cropped textures;
- grouped connected B4 output being bypassed by the registered UI operator;
- duplicated output implementations with the same public function names;
- tests passing for local Emission materials while missing the B4 operator defect.

The bridge now imports only the post-render output services.

## Remaining architecture debt

### Duplicate historical export functions

`a1_multi_object_export.py` and `a1_mixed_object_export.py` still contain historical functions
named `export_a1_multi_object` and `export_a1_mixed_object`. They are no longer production UI
entry points, but the names are misleading and remain a future misuse risk.

Safe cleanup requires splitting preparation-only code into explicit modules or replacing those
historical functions with compatibility facades that delegate to the output services.

### Draft composition before B4 finalization

`prepare_a1_multi_object()` returns a complete draft document because it is also a public
preparation API. Production output must compose again after B4 finalization. This is correct but
performs duplicate composition and makes `PreparedA1MultiObject.document` potentially stale for
camera projection until output finalization.

A future cleanup can introduce a component-preparation result that contains objects and output
validation only, with composition performed exactly once by callers that require final output.

### Generic `ExportRequest`

The generic application `ExportRequest` contract is validated and tested but the production A1
operators use A1-specific settings/source contracts instead. Its intended ownership should be
clarified: either make it the actual top-level request envelope or remove it from the A1 design.

### Legacy eager imports and global settings

The explicit Legacy single-object path still synchronizes texture dimensions through module-level
globals in `main`, `config` and `json_export`. Legacy modules are also imported during add-on
startup. This does not contaminate the Rewrite request payload, but it remains isolated technical
debt until Legacy removal.

### One-object Connect fallback

One selected Connect flag is logged and exported as standalone because a connected group requires
at least two objects. The behavior is deterministic, but the warning is not yet returned as an
`ExportIssue` to the UI.

## Regression coverage

- post-render output-service module identity;
- one shared immutable Scene snapshot across multi-object settings;
- stable active-object ordering across transient RNA wrappers;
- deterministic component IDs and animation namespaces;
- registered Rewrite multi operator;
- registered connected B4 multi operator producing grouped texture and JSON metadata;
- explicit Legacy backend;
- no automatic Legacy fallback after Rewrite failure.
