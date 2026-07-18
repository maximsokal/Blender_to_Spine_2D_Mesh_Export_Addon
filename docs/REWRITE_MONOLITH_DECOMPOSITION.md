# Rewrite monolith decomposition

## Scope

This slice removes three high-risk orchestration monoliths without changing the public Blender operators or the existing single/multi/mixed output contracts.

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

## 3. UI bridge

The previous `a1_ui_bridge.py` mixed Blender RNA reads, settings conversion, source construction, route selection, and output invocation in one file.

It is now a small compatibility facade over:

```text
a1_ui_rna.py       -> Blender RNA identity and immutable snapshots
a1_ui_settings.py  -> application settings and source construction
a1_ui_router.py    -> single/standalone/connected/mixed routing
a1_ui_bridge.py    -> stable imports for existing callers and tests
```

Existing private helper names used by focused tests remain available through the facade.

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

The Blender pipeline probe follows physical implementation ownership rather than compatibility facades. It now requires:

- all four object-preparation stages;
- shared multi/mixed staging;
- shared final-document statistics;
- `a1_ui_router` entrypoints instead of `a1_ui_bridge` facade functions.

## Validation performed outside CI

No GitHub Actions workflow was triggered.

Manual local validation included:

- Python compilation of every changed production and probe file;
- architecture tests for preparation, output, UI, and probe boundaries;
- isolated runtime contract tests for preparation orchestration;
- isolated runtime tests for output staging/statistics/grouped validation;
- UI behavior tests for RNA ordering, shared profiles, alpha and angular settings, deterministic source IDs, all routing modes, and visible Connect fallback;
- an additional execution of the exact Connect-fallback regression test in a minimal package using the addon's real package name.

Full repository pytest and real Blender 4.4 integration remain separate manual release gates.
