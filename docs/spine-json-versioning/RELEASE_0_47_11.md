# Release checkpoint 0.47.11

## Scope

Version `0.47.11` keeps the validated Spine 4.1.24 scale-preserving topology from `0.47.10` and fixes production selected-object routing.

Production support remains limited to:

- Spine 4.1.24;
- `2-Axis Rotation + Scale`;
- single-object export;
- standalone multi-object export.

Spine 4.1 connected, mixed, and 3-Axis requests remain blocked. Connected and mixed composition remain development-only internal capabilities.

## Fixed production-routing defect

Older `.blend` files can retain `Object.spine2d_connect_settings.enabled=True` even though the production ordered UI no longer exposes Connect controls. Version `0.47.10` still read that hidden persisted value while building the ordinary selected-object plan. Two or more stale enabled values silently selected `CONNECTED_MULTI_OBJECT`, which the Spine 4.1 capability gate correctly rejected.

Version `0.47.11` separates the two owners:

- `build_selected_ui_export_plan()` is the production entry and always creates a standalone plan;
- `build_development_connected_ui_export_plan()` is the explicit development-only entry that may read persisted Connect values.

The Object RNA property is retained so older `.blend` files load safely, but it no longer affects production Analyze or Export Selected Objects.

## Real-Blender routing gate

`tests/blender_headless/run_production_ui_standalone_policy_integration.py` registers the real extension in Blender 5.2, creates three real Mesh objects, persists `Connect=True` on two of them, and captures both routing surfaces from the same Blender state.

The gate requires:

- the ordinary production plan to be `STANDALONE` with every selected source in `standalone_sources`;
- Spine 4.1 standalone capability preflight to pass for every production source;
- the explicit development-only plan to remain `MIXED`, proving the persisted values were preserved rather than deleted;
- no operator-driven scene setup.

Required success marker:

```text
[PRODUCTION_UI_STANDALONE] PASS persisted Connect flags ignored by production
```

## Unchanged Spine 4.1 rig behavior

The accepted scale fix remains unchanged:

- world-relative uniform Scale semantics are preserved;
- the unsafe `*_rotate_X` driver is replaced only with its invertible parent;
- depth constraints remain on their original `*_scale` wrappers;
- internal `onlyTranslation` bridges provide invertible parents;
- weighted attachment indices are remapped after bridge insertion;
- no epsilon and no serialized JSON text repair are used.

## Required gates

Before distributing the ZIP:

1. Run `tests/test_a1_ui_export_plan.py`, `tests/test_production_ui_standalone_headless_contract.py`, and the focused `0.47.11` suite.
2. Run the complete pure-Python suite.
3. Run the real-`bpy` suite.
4. Run `tests/blender_headless/run_production_ui_standalone_policy_integration.py` in Blender 5.2.
5. Run the Blender-to-Spine 4.1 standalone acceptance and scale-response probe.
6. Import the fresh Blender-generated JSON into Spine Editor 4.1.24 and confirm independent controls without `all_objects_*` identifiers.
7. Build and validate `blender_to_spine2d_mesh_exporter-0.47.11.zip`.
8. Run the isolated extension install gate.

The external Spine runtime repository remains read-only input.
