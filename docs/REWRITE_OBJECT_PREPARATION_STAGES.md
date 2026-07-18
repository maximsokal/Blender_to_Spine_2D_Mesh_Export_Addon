# Staged A1 object preparation

## Previous problem

`blender_adapter.a1_object_preparation.prepare_a1_object` previously owned request validation, Blender mesh reads, modifier lineage, geometry decomposition, UV construction, material analysis, bake planning, rig construction, and document assembly in one function of almost 300 executable lines.

That design had three practical failures:

1. per-file logging could not isolate an internal responsibility because every stage lived in one logger;
2. runtime tracing reported one large inclusive duration instead of meaningful source, UV, shading, and document costs;
3. changing one stage required importing and understanding every low-level dependency of the entire preparation pipeline.

## Current pipeline

```text
prepare_a1_object
  -> prepare_a1_source_geometry
  -> prepare_a1_uv
  -> prepare_a1_texture_plan
  -> prepare_a1_document
  -> PreparedA1Object
```

### `a1_source_geometry_preparation.py`

Owns:

- request and Blender Mesh-object validation;
- output-name and output-path resolution;
- render-engine contract resolution;
- original/evaluated mesh snapshot reading;
- modifier lineage warnings;
- Z-group assignment;
- segmentation and disk-region decomposition.

### `a1_uv_preparation.py`

Owns:

- texturing topology construction;
- UV unwrap execution;
- out-of-unit-square warning generation;
- UV propagation from texturing topology to prepared regions.

### `a1_texture_planning.py`

Owns:

- material graph analysis;
- material warning normalization;
- object and Scene bake-context analysis;
- render-engine consistency validation;
- shader capability audit;
- capability-checked object bake or camera-projection plan selection.

### `a1_document_preparation.py`

Owns:

- legacy-compatible rig construction;
- attachment and sequence settings;
- object-bake or camera-projection document assembly;
- final in-memory document statistics.

### `a1_preparation_contracts.py`

Owns:

- `A1ObjectPreparationError`;
- immutable preparation statistics;
- normalized warning construction.

## Error contract

Every stage keeps the exact `A1SingleObjectStage`, object ID, accumulated warnings, and accumulated statistics. A stage error is wrapped once as `A1ObjectPreparationError`; the output service remains responsible for the single user-visible stack-trace log and `ExportResult` conversion.

Unexpected orchestration failures are associated with the currently active stage rather than being incorrectly reset to `VALIDATE_REQUEST`.

## Logging and runtime probe

Each stage has its own logger and can be enabled independently in addon preferences:

```text
blender_adapter.a1_source_geometry_preparation
blender_adapter.a1_uv_preparation
blender_adapter.a1_texture_planning
blender_adapter.a1_document_preparation
```

The Blender pipeline probe treats all four functions as required calls for every Rewrite object preparation. Its report therefore exposes separate input/output shapes, call counts, durations, and exceptions for each responsibility.

## Boundaries

Preparation stages:

- do not serialize JSON;
- do not stage or commit files;
- do not own atomic transactions;
- do not render textures;
- return immutable typed products only.
