# Implementation Log — Extensions Review Compliance

This file records production changes as they land on `001-extensions-review-compliance`.
It supplements `tasks.md`; nothing is considered release-ready until the exact candidate
passes the checklist and Blender package gates.

## Slice 01 — Moderator requirements normalized

- Replaced the earlier mixed/historical feedback ledger with the seven comments from the current moderation review.
- Target release is `0.155.0` on the same existing Blender Extensions submission.
- Target public name is `Spine2D Mesh Exporter`; manifest `id` remains unchanged.

## Slice 02 — Runtime threading removal

File: `Blender_to_Spine2D_Mesh_Exporter/infrastructure/exclusive_operation.py`

Changes:

- removed `threading.RLock` and `threading.get_ident`;
- exclusive export ownership is now a Blender-main-thread process-local registry;
- leases use an opaque UUID token instead of a Python thread id;
- context-manager cleanup remains deterministic through `finally`;
- no worker/background thread is introduced as a replacement.

Reason: closes the concrete runtime `threading` use discovered by the RF-002 audit.

Pending evidence: full runtime AST scan and local test suite.

## Slice 03 — Re-Polish advertisement removed

Files:

- deleted `Blender_to_Spine2D_Mesh_Exporter/repolish_ui.py`;
- removed `repolish_ui` import/registration from the root package.

Removed behavior:

- third-party Re-Polish child panel;
- external Re-Polish URL from the shipped runtime.

Reason: RF-003 requires a self-contained extension with no advertisement/third-party dependency.

Pending evidence: runtime static scan and exact built-ZIP inventory.

## Slice 04 — Root registration simplified

File: `Blender_to_Spine2D_Mesh_Exporter/__init__.py`

Removed:

- `ExtensionRegistrationState` enum;
- `_REGISTRATION_STATE`;
- degraded/reentrant registration state machine;
- generic registration step/rollback action table;
- root transactional RNA helper dependency;
- exception-throwing fallback `register()` when `bpy` is unavailable.

Current behavior:

- when Blender API is available, root calls module owners in a clear forward order and unregisters them in reverse;
- top-level Scene properties use direct `setattr` / reverse `delattr` ownership;
- when `bpy` is unavailable, `register()` and `unregister()` are no-ops;
- logging initialization remains after successful owner registration.

Reason: RF-004 explicitly rejected root registration-state/cleanup complexity.

Pending: simplify any remaining module-local registration wrappers that are not justified by actual handlers/RNA/resources.

## Slice 05 — `ui_layout.py` panel swapping removed

File: `Blender_to_Spine2D_Mesh_Exporter/ui_layout.py`

Removed:

- unregistering the canonical main panel during `register()`;
- registering a replacement panel with the same id;
- `_ORIGINAL_PANEL_REMOVED`, `_ORDERED_PANEL_REGISTERED`, `_REGISTERED_RNA` state;
- restore/rollback panel dance;
- custom foldout RNA properties owned only by the replacement UI.

Replacement:

- the canonical `ui.OBJECT_PT_Spine2DMeshPanel` remains registered continuously;
- Rig, Generated Materials, and Depth Parallax are ordinary Blender child panels;
- child panel registration is a direct declaration-order loop; unregister is reverse order.

Reason: this directly addresses the `ui_layout.py` behavior called out by RF-004.

## Slice 06 — Rig UI registration simplified

File: `Blender_to_Spine2D_Mesh_Exporter/rig_ui.py`

Removed:

- replacement of `SPINE2D_OT_ResetSettings` at registration time;
- restore-original-reset logic and global registration flags;
- transactional registration wrappers.

Replacement:

- one independent `spine2d.reset_rig_profile` operator;
- direct class registration/unregistration;
- Rig child panel only draws controls not already owned by the main panel, avoiding duplicate export-mode/projection controls;
- rig reset restores rig profile, +Z projection, shared pivot, parallax angle and preview-animation default.

## Slice 07 — Manifest moderation metadata

File: `Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml`

Changed:

```toml
version = "0.155.0"
name = "Spine2D Mesh Exporter"
tags = ["Import-Export"]
tagline = "Convert 3D objects into Spine 2D meshes"
```

Removed:

```toml
platforms = ["windows-x64"]
```

The technical `id = "blender_to_spine2d_mesh_exporter"` remains unchanged.

Platform decision: current audit found no reason to make the pure-Python installed runtime Windows-only. Windows-specific developer/test paths are not treated as installed-runtime dependencies. Final portability still requires Blender package/install validation before release.

## Slice 08 — Compliance regression tests

Added `tests/test_extensions_review_compliance.py` covering:

- exact 0.155.0 manifest metadata;
- only `Import-Export` tag;
- no platform restriction;
- no `threading` or `queue` imports anywhere in shipped production Python;
- no `PipelineTraceSession` symbol in shipped runtime;
- no Re-Polish runtime source/reference;
- no root registration state machine;
- no `ui_layout.py` main-panel unregister/restore pattern;
- required development/legacy build exclusions.

Updated `tests/test_manifest_version.py` to 0.155.0.

Updated `tests/test_texture_size_bake_ui.py` for the canonical main-panel/child-panel architecture.

## Slice 09 — Shipping-boundary threading and development-trace cleanup

Local focused gate at `eca6f72` found five source-tree `threading` imports:

- `infrastructure/atomic_work_state.py`;
- `infrastructure/export_diagnostics.py`;
- `infrastructure/export_events.py`;
- `infrastructure/pipeline_trace.py`;
- retained `legacy_loader.py`.

The first three are real Rewrite runtime modules, but their locks protected only synchronous process-local dictionaries/policy values. They now use direct main-thread state with the same reservation, listener-snapshot, and policy semantics and import no `threading`/`queue`.

`PipelineTraceSession` was located and classified as development instrumentation. Its only direct consumers found by repository call-graph search are the development probe and trace tests; the production infrastructure package does not re-export it. The manifest now excludes:

- `/infrastructure/pipeline_trace.py`;
- `/infrastructure/pipeline_trace_model.py`;
- `/infrastructure/pipeline_trace_report.py`;
- `/infrastructure/pipeline_trace_values.py`.

The retained pre-Rewrite `legacy_loader.py` is intentionally left untouched and remains excluded by `/legacy_loader.py`.

`tests/test_extensions_review_compliance.py` is now manifest-aware: source AST checks operate on Python files that are eligible to ship, instead of treating deliberately excluded repository sources as runtime. A boundary regression test verifies that the trace/legacy files are excluded while the three rewritten infrastructure modules remain inside the scanned shipping set.

This slice addresses the two failures from the first local focused gate. It does **not** claim RF-001/RF-002 closed yet: the updated focused/full/real-bpy suites still need to run locally, and the exact Blender-built ZIP remains the authoritative packaging evidence.

## Current open work after Slice 09

1. Pull the latest branch commits and rerun compile + focused tests; only then continue to the full Python and real-bpy suites.
2. If the focused gate is green, audit remaining production modules for hidden concurrency constructs (`multiprocessing`, `concurrent.futures`, direct Blender timers) and classify rather than blanket-ban Blender-managed timers.
3. Audit remaining production modules for unnecessary transactional registration helpers (`ui.py`, generated-material UI, readiness/migration owners, etc.) and simplify only where ownership does not require state.
4. Remove the old root registration-state dependency in `scene_properties.py` now that the root state machine is gone.
5. Update public README/docs/submission/testing version/name/platform/same-submission wording to 0.155.0.
6. Add exact Blender-built ZIP inventory gate and build/validate/install the final candidate.
7. Execute real bpy repeated enable/disable/restart tests and representative exports before claiming RF-004 or RF-006 fully closed.
