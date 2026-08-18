# Implementation Log — Extensions Review Compliance

This file records production changes as they land on `001-extensions-review-compliance`.
It supplements `tasks.md`; nothing is considered release-ready until the exact candidate
passes the checklist and Blender package gates.

## Slice 01 — Moderator requirements normalized

- Replaced the earlier mixed/historical feedback ledger with the seven comments from the current moderation review.
- Target release is `0.155.0` on the same existing Blender Extensions submission.
- Target public name is `Spine Mesh Exporter`; manifest `id` remains unchanged.

## Slice 02 — Runtime threading removal

File: `Blender_to_Spine2D_Mesh_Exporter/infrastructure/exclusive_operation.py`

Changes:

- removed `threading.RLock` and `threading.get_ident`;
- exclusive export ownership is now a Blender-main-thread process-local registry;
- leases use an opaque UUID token instead of a Python thread id;
- context-manager cleanup remains deterministic through `finally`;
- no worker/background thread is introduced as a replacement.

Reason: closes the concrete runtime `threading` use discovered by the RF-002 audit.

## Slice 03 — Re-Polish advertisement removed

Files:

- deleted `Blender_to_Spine2D_Mesh_Exporter/repolish_ui.py`;
- removed `repolish_ui` import/registration from the root package.

Removed behavior:

- third-party Re-Polish child panel;
- external Re-Polish URL from the shipped runtime.

Reason: RF-003 requires a self-contained extension with no advertisement/third-party dependency.

Pending evidence: exact built-ZIP inventory.

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
- partial Scene RNA registration has owner-local rollback and clears pending migration snapshots;
- when `bpy` is unavailable, `register()` and `unregister()` are no-ops;
- optional post-registration logging/preferences initialization is best-effort and cannot corrupt an otherwise successful enable.

Reason: RF-004 explicitly rejected root registration-state/cleanup complexity while still requiring deterministic ownership cleanup.

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

## Slice 06 — Class-only registration owners simplified

Files include:

- `rig_ui.py`;
- `addon_preferences.py`;
- `single_object_operator.py`;
- `ui_layout.py`.

Changes:

- class-only owners use direct `bpy.utils.register_class` loops;
- unregister runs in reverse class order;
- owner-specific resources such as the Preferences one-shot redraw timer are released explicitly by their owner;
- registration state is retained only in modules that actually replace runtime bindings or own mixed resources.

## Slice 07 — Manifest moderation metadata

File: `Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml`

Changed:

```toml
version = "0.155.0"
name = "Spine Mesh Exporter"
tags = ["Import-Export"]
tagline = "Convert 3D objects into Spine 2D meshes"
```

Removed:

```toml
platforms = ["windows-x64"]
```

The technical `id = "blender_to_spine2d_mesh_exporter"` remains unchanged.

## Slice 08 — Compliance regression tests

`tests/test_extensions_review_compliance.py` covers:

- exact 0.155.0 manifest metadata;
- only `Import-Export` tag;
- no platform restriction;
- no forbidden Python concurrency imports in manifest-eligible runtime;
- no `PipelineTraceSession` symbol in shipped runtime;
- no Re-Polish runtime source/reference;
- no root registration state machine;
- no `ui_layout.py` main-panel unregister/restore pattern;
- no shipped `*_dup` UI workaround functions;
- required development/legacy build exclusions.

## Slice 09 — Shipping-boundary concurrency and development-trace cleanup

The initial focused gate found five source-tree `threading` imports. Three were real Rewrite
runtime modules:

- `infrastructure/atomic_work_state.py`;
- `infrastructure/export_diagnostics.py`;
- `infrastructure/export_events.py`.

Their locks protected synchronous process-local dictionaries/policy values only. They now
use direct process-local state with the same reservation/listener/policy semantics and import
no `threading`/`queue`.

`PipelineTraceSession` was classified as development instrumentation. Its trace implementation
files are excluded from the extension build:

- `/infrastructure/pipeline_trace.py`;
- `/infrastructure/pipeline_trace_model.py`;
- `/infrastructure/pipeline_trace_report.py`;
- `/infrastructure/pipeline_trace_values.py`.

The retained pre-Rewrite `legacy_loader.py` remains untouched and excluded from the shipped
extension.

The compliance scanner follows manifest build exclusions rather than treating intentionally
retained repository source as installed runtime.

## Slice 10 — Root lifecycle tests migrated to the real public lifecycle

Old tests that manually iterated `REGISTRATION_STEPS`, expected the replacement panel, or
required Re-Polish were migrated to call the real public `extension.register()` and
`extension.unregister()` pair.

Covered suites include unit/source contracts, repeated real-bpy registration cycles,
handler/keymap cleanup, operator undo/redo, public blend goldens, grenade headless runners,
and memory-stress tooling.

Pre-test `extension.unregister()` calls that could hide lifecycle leaks were removed from
Scene migration real-bpy tests. Those tests now assert an unregistered baseline instead of
mutating a leaked previous state away.

## Slice 11 — Scene migration decoupled from the removed root state machine

Files:

- `blender_adapter/scene_properties.py`;
- `blender_adapter/scene_settings_migration.py`.

Changes:

- `scene_properties.py` no longer imports/queries `get_registration_state`;
- Seam Maker update logic now asks whether the specific Scene has a pending pre-registration migration snapshot;
- `scene_settings_migration.py` no longer keeps a generic `_REGISTERED` flag;
- load handler installation uses the handler collections as source of truth and local
  `added_pre`/`added_post` flags only for rollback of the current call.

This preserves saved-Scene migration semantics without recreating root registration state.

## Slice 12 — Dormant automatic readiness scheduler removed from shipped runtime

File: `Blender_to_Spine2D_Mesh_Exporter/auto_readiness.py`.

The previous intermediate implementation no longer installed its automatic scheduler, but
still shipped the complete dead mechanism. That was removed rather than merely disabled.

Removed:

- `_automatic_timer`;
- `bpy.app.timers` polling registration helpers;
- debounce deadlines/pending request state;
- automatic request-key scheduling;
- automatic depsgraph readiness callback;
- automatic load-pre/load-post readiness callbacks;
- `request_auto_analysis` and automatic-status UI text.

Current behavior:

- Analyze is explicit and synchronous on Blender's main thread;
- a small process-local re-entry flag prevents recursive Analyze calls only;
- readiness diagnostics remain advisory and never disable Export;
- Export wrappers call the production export directly and schedule no readiness work;
- register/unregister own only reversible UI method overrides.

Focused and real-bpy contracts now assert that the old scheduler symbols do not exist.

## Slice 13 — Runtime hook ownership hardened

- Preferences keeps one one-shot Blender event-loop redraw timer for exact-version edits and
  explicitly unregisters it during add-on teardown.
- The shipped hook inventory expects that Preferences timer to be the only timer surface.
- Handler append/remove ownership is scanned per shipped module.
- keymap/draw-handler/preview allocations remain prohibited unless explicitly owned.

This distinguishes a bounded Blender event-loop redraw callback from a persistent Python
background scheduler.

## Slice 14 — Platform restriction audit

Added `tests/test_runtime_portability_contract.py`.

Static manifest-eligible runtime audit now proves:

- no manifest `platforms` restriction;
- no shipped `subprocess` dependency;
- no unconditional `msvcrt`, `winreg`, or `fcntl` dependency;
- `ctypes` is confined to guarded process-identity compatibility in
  `infrastructure/atomic_work_state.py`;
- Windows process probing is guarded by `os.name == "nt"` and POSIX has an `os.kill` path;
- Linux process-start identity uses `/proc` only when available;
- unknown hosts fail closed through a process-local session marker rather than deleting
  another process's work files;
- Windows external-I/O path budgeting is disabled on non-Windows hosts unless explicitly
  requested by a test/caller;
- durable I/O uses portable `Path`, `os.replace`, and `os.fsync` primitives with a guarded
  Windows directory-fsync limitation;
- no absolute Windows drive/UNC literal is allowed in shipped Python;
- no bundled `.dll`, `.pyd`, `.so`, or `.dylib` is present in the manifest-eligible package.

This supports removing the unjustified Windows-only manifest restriction. It does not claim
that Linux/macOS real-Blender release validation has already been executed.

## Slice 15 — Same-submission documentation remediation

`docs/submission.md` describes version 0.155.0 as a correction to the **existing**
reviewed/declined Blender Extensions submission. It explicitly prohibits creating a second
listing or deleting/recreating the declined listing.

Current metadata documented there:

- Name: `Spine Mesh Exporter`;
- Version: `0.155.0`;
- technical ID unchanged;
- Tags: `Import-Export` only;
- no platform restriction declared;
- Files permission unchanged.

## Slice 16 — Targeted lifecycle gate after scheduler removal

Local evidence on exact commit `3a028d8a`:

- targeted real-bpy lifecycle: `4 passed, 49 deselected`;
- focused compliance/runtime-hook set: `27 passed`;
- both commands returned exit code `0`;
- local worktree was clean after the gate.

The real-bpy selection covered:

- twenty public registration/unregistration cycles;
- isolated module import/reload without registration side effects;
- two complete RNA ownership cycles;
- ten handler/keymap/timer cleanup cycles.

The runtime-hook test now asserts that the removed `auto_readiness._automatic_timer` symbol
stays absent rather than trying to query a callback that no longer exists.

## Slice 17 — Public title synchronized without technical-identity migration

The public display title is now **`Spine Mesh Exporter`** across:

- `blender_manifest.toml` `name`;
- root README and maintained docs/examples;
- submission metadata/workflow text;
- canonical 3D View Sidebar panel `bl_label` and `bl_category`;
- compliance/documentation/UI regression contracts;
- moderation ledger.

Stable technical identity remains unchanged:

```text
manifest id: blender_to_spine2d_mesh_exporter
package:     Blender_to_Spine2D_Mesh_Exporter
panel id:    OBJECT_PT_spine2d_mesh
RNA/operator prefixes: spine2d_*
ZIP stem:    blender_to_spine2d_mesh_exporter-0.155.0.zip
```

The documentation gate rejects both superseded public titles:

- `Spine2D Mesh Exporter`;
- `Blender to Spine2D Mesh Exporter`.

This is a branding/display correction only; it does not migrate saved Scene properties,
operators, package identity, or serialized Spine data.

## Current open work after Slice 17

1. Pull the final rename-remediation HEAD locally and run the focused public-title/documentation/UI gate.
2. If focused tests are green, run the full Blender-independent suite.
3. Run the complete real-bpy suite and representative Blender-headless exports.
4. Build the exact candidate with Blender `extension build`, run `extension validate`, and
   enumerate/scan the actual ZIP; source-tree exclusions are not final packaging evidence.
5. Install that exact ZIP in an isolated profile and run enable/disable/restart/re-enable,
   preference persistence, manual Analyze, and production export checks.
6. Only after those gates, update task/checklist completion and prepare the moderator reply
   for the same existing submission.
