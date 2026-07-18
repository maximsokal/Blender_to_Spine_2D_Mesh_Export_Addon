# Rewrite multi-object data flow

## Purpose

This document defines ownership between Blender UI capture, object preparation, document
composition, rendering, serialization, and atomic output. The boundary exists to prevent the
legacy pattern where one orchestrator collected mutable Blender state, parallel lists, temporary
JSON files, and output side effects at the same time.

## Production entry points

```text
object.save_uv_as_json
object.spine2d_multi_export
```

Rewrite operators route through `blender_adapter.a1_ui_bridge`. The bridge captures mutable
Scene/Object RNA once and creates immutable settings/source contracts. It must call only:

```text
a1_single_object_export.export_a1_single_object
a1_multi_object_output.export_a1_multi_object
a1_mixed_object_output.export_a1_mixed_object
```

Preparation modules are not output entry points.

## Single-object ownership

```text
Scene/Object RNA snapshot
    -> A1SingleObjectExportSettings
    -> prepare_a1_object
    -> PreparedA1Object
    -> texture/B4 staging
    -> render-derived attachment finalization
    -> Spine serialization
    -> one atomic commit
```

`PreparedA1Object` may retain the live source object only because Blender must perform the real
bake/render later. Geometry, UV, material analysis, rig, bake plan, paths, warnings and statistics
are immutable typed values.

## Multi-object source contract

Every selected object becomes one `A1MultiObjectSource`:

```text
source_object
component_id
animation_namespace
A1SingleObjectExportSettings
```

The active Mesh is first. Remaining Mesh objects are ordered deterministically. Component IDs and
animation namespaces derive from that stable order.

## Preparation-only modules

`a1_multi_object_export.py` owns:

- source validation;
- connected placement preparation settings;
- per-object `prepare_a1_object` calls;
- path ownership/collision validation;
- `PreparedA1MultiObject` construction.

`a1_mixed_object_export.py` owns:

- connected/standalone partition validation;
- connected subgroup settings;
- preparation of both subgroups;
- one shared output namespace validation.

They explicitly do **not** own:

- B4 staging;
- grouped rendering;
- document composition;
- serialization;
- filesystem transactions;
- public `export_*` functions.

`PreparedA1MultiObject` therefore contains only:

```text
settings
sources
objects
texture_output_paths
warnings
statistics
```

It has no draft `document` and no `composition` field.

## Composition ownership

`a1_multi_object_composition.py` is the only reusable composition entry point:

```text
compose_a1_multi_object_document(
    sources,
    finalized_prepared_objects,
    settings,
)
```

It accepts only already-finalized object documents. It has no Blender, render, serializer, or
filesystem dependencies.

Standalone composition namespaces animations and shares only `root`. Connected composition builds
the typed global rig, world placement, bone-index remap, constraints and layer ordering.

## Output ordering

Multi and mixed output services enforce this order:

```text
prepare objects
    -> reserve atomic outputs
    -> stage every texture
    -> finalize every render-derived B4 attachment
    -> optional grouped connected B4 render
    -> compose finalized documents exactly once
    -> apply grouped overlay
    -> serialize exactly once
    -> commit JSON and textures together
```

Serializing before B4 finalization is forbidden because cropped dimensions, contour topology,
offsets and grouped attachments do not exist before the staged render has been decoded.

## Mixed output

Mixed export partitions the finalized tuple using the original connected source count:

```text
finalized connected objects
    -> connected composition
    -> optional grouped B4 overlay

finalized standalone objects
    -> standalone composition

connected document + standalone document
    -> final outer composition
```

The connected overlay is applied before outer composition, preserving subgroup draw order.

## Structured failures

`a1_multi_object_result.build_multi_object_failure_result` is shared by multi and mixed output.
It normalizes:

- stage/error code;
- component and object identity;
- failed object substage;
- operation name;
- warnings and partial statistics.

Local duplicate `_failure_result` implementations are forbidden.

## Architecture guards

`tests/test_a1_multi_object_pipeline_boundary.py` verifies:

- preparation modules define no public `export_*` functions;
- preparation modules import no serializer, transaction, staging, or composition side effects;
- `PreparedA1MultiObject` owns no draft document;
- composition is Blender/IO independent;
- finalization occurs before composition;
- composition occurs before serialization;
- output services use the shared composition and failure-result entry points.

`tests/test_a1_multi_object_result.py` verifies the public structured failure contract.

## Removed code smells

The cleanup removed:

1. duplicate `export_a1_multi_object` and `export_a1_mixed_object` functions from preparation files;
2. JSON serialization before render-derived B4 finalization;
3. draft multi-object composition followed by a second final composition;
4. private helper imports such as `_compose_document` and `_record_object_statistics`;
5. duplicate multi/mixed failure-result implementations;
6. duplicate nested mixed statistics (`connected.component.*` / `standalone.component.*`).

## Remaining cleanup candidates

Separate future slices should address:

- split Scene/Object snapshot capture from UI routing;
- convert the one-Connect fallback into a visible `ExportIssue`;
- remove or adopt the unused generic `ExportRequest` envelope;
- lazy-load the explicit Legacy backend instead of importing all legacy modules at add-on startup;
- remove Legacy only after private production parity and release-gate approval.
