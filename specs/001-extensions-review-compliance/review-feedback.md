# Blender Extensions Review Feedback Ledger

**Feature:** `001-extensions-review-compliance`  
**Base:** `main@f0a0f879d639dad860c0c8c56ddba0845aa69f17`  
**Implementation status:** active on this branch. Every production change must update this ledger/tasks/checklist.

This file contains the **seven comments from the current Blender Extensions moderation review**. Older/historical review notes are not acceptance requirements unless they are rediscovered by the current audit.

## RF-001 — `PipelineTraceSession` / development-package audit

Reviewer asks what `PipelineTraceSession` is for and whether users actually need it. If it is a development tool it must be removed from the distributed extension. Development files must be excluded through `blender_manifest.toml`, and the review archive must be built with Blender's extension command-line tooling so reviewers see only runtime files.

Current findings after implementation audit:

- `PipelineTraceSession` does exist at `infrastructure/pipeline_trace.py`;
- repository call-graph search finds it in the development Blender probe and trace tests, not in the production infrastructure package exports;
- the session installs `sys.settrace()` plus `threading.settrace()` and is therefore development instrumentation, not required by the user-facing exporter runtime;
- `pipeline_trace.py`, `pipeline_trace_model.py`, `pipeline_trace_report.py`, and `pipeline_trace_values.py` are now excluded through `blender_manifest.toml`;
- the compliance source scan follows the manifest shipping boundary instead of treating deliberately excluded repository sources as runtime;
- the built ZIP remains the authoritative final evidence.

Closure:

1. production runtime contains no `PipelineTraceSession` if it is development-only/obsolete;
2. any user-facing tracing/diagnostics that remains has a documented runtime purpose;
3. development-only sources are excluded from the Blender-built ZIP;
4. exact ZIP inventory is reviewed and `extension validate` passes.

## RF-002 — Remove `threading` and `queue`

Reviewer states that `threading` and `queue` are not allowed for this Blender extension and requests `subprocess` where process isolation is actually required.

Implementation rule:

- shipped runtime must not import `threading` or `queue`;
- do not introduce `subprocess` merely to satisfy the wording if no background work is required; prefer synchronous/main-thread execution where possible;
- if `subprocess` is required, isolate it from Blender data/API and own its lifecycle explicitly.

Current implementation findings:

- the first focused compliance run found `threading` imports in `atomic_work_state.py`, `export_diagnostics.py`, `export_events.py`, development-only `pipeline_trace.py`, and retained legacy `legacy_loader.py`;
- the three Rewrite runtime modules used only `RLock` around synchronous process-local state and now use direct main-thread state without replacement workers;
- `pipeline_trace.py` is excluded as development-only under RF-001;
- `legacy_loader.py` remains untouched and is already excluded from the extension ZIP by the manifest;
- the compliance AST scan now checks only manifest-eligible shipped Python modules.

Closure:

1. AST/static scan of every shipped Python module finds no `threading` or `queue` import;
2. no hidden current equivalent of the old background worker survives;
3. Analyze/Export/disable leave no add-on-owned Python worker alive.

Updated focused/full/real-bpy test evidence is still pending, so RF-002 is not yet marked closed.

## RF-003 — Remove Re-Polish advertisement / third-party dependency

Reviewer states that advertisements in Blender UI are not allowed, explicitly names the **Re-Polish** link, and says the hosted extension must be fully self-contained rather than depending on third-party software/services.

Current baseline findings:

- `Blender_to_Spine2D_Mesh_Exporter/repolish_ui.py` is shipped runtime code;
- it imports `webbrowser`, exposes an `Open Re-Polish` operator, and opens a third-party URL;
- root `__init__.py` imports/registers `repolish_ui`.

Closure:

1. remove the Re-Polish operator/UI/runtime registration;
2. remove runtime Re-Polish constants/modules if nothing else uses them;
3. built-package scan finds no `repolish`, Re-Polish URL, or advertisement operator;
4. extension remains fully functional without the third-party service.

## RF-004 — Simplify registration and `ui_layout.py`

Reviewer explicitly rejects the current complexity around registration: unnecessary property cleanup/state tracking, special behavior when `bpy` is absent, and `ui_layout.py` unregistering/re-registering/restoring panels. They request the normal Blender add-on registration pattern and module import/reload style.

Current baseline findings:

- root `__init__.py` has `ExtensionRegistrationState`, `_REGISTRATION_STATE`, transactional step tables, rollback wrappers and a `bpy`-absent `register()` that raises;
- root registration imports `ui` and then `ui_layout` as separate panel owners;
- `ui_layout.register()` unregisters `ui.OBJECT_PT_Spine2DMeshPanel`, registers `OBJECT_PT_Spine2DOrderedMeshPanel`, and restores the original panel during unregister/rollback;
- this is exactly the architecture the reviewer asked to remove.

Target design:

1. one canonical main panel owner — no panel replacement dance;
2. straightforward module/class registration order and reverse unregister order;
3. no root registration-state machine unless a remaining Blender resource demonstrably requires it;
4. if `bpy` is unavailable, module import may remain testable but `register()/unregister()` must be harmless no-ops rather than performing fallback lifecycle work;
5. Blender-owned resources are still unregistered normally; ordinary Python globals are not ceremonially cleared;
6. retain actionable logging without generic recovery machinery.

Closure requires real bpy repeated enable/disable tests and installed-extension restart smoke.

## RF-005 — Only `Import-Export` tag

Reviewer requirement is exact: **remove all tags except `Import-Export`.**

Current baseline:

```toml
tags = ["Import-Export", "Mesh", "UV", "Animation"]
```

Target:

```toml
tags = ["Import-Export"]
```

No interpretation or additional discoverability tags are allowed for this submission.

## RF-006 — Remove unjustified Windows-only restriction

Reviewer asks why the extension is limited to Windows because no obvious Windows-specific runtime code was visible.

Current baseline:

```toml
platforms = ["windows-x64"]
```

Implementation rule:

- audit installed runtime for OS-specific imports, locking, path semantics, native binaries and subprocess requirements;
- if no runtime Windows dependency exists, remove the platform restriction from the manifest;
- keep development/release PowerShell paths separate from installed-runtime portability;
- add portable-path/static tests where necessary.

A Windows-only test environment is not itself a runtime dependency.

## RF-007 — Remove `Blender` from extension title; same submission

Reviewer states that `Blender` is trademarked and may not be used in extension titles. `id` may stay unchanged. Reviewer explicitly says **do not make a new submission; upload a new version in the existing one.**

Current baseline:

```toml
id = "blender_to_spine2d_mesh_exporter"
name = "Blender to Spine2D Mesh Exporter"
version = "0.154.0"
```

Target for this remediation:

```toml
id = "blender_to_spine2d_mesh_exporter"   # unchanged
name = "Spine2D Mesh Exporter"
version = "0.155.0"
```

The package directory/technical extension identity stays unchanged unless Blender validation proves otherwise. Public docs/listing text must use the new display title. The corrected ZIP is uploaded as a new version of the **same existing submission**.

## Current closure status

| Reviewer item | Current status |
| --- | --- |
| RF-001 PipelineTraceSession / clean package | Implemented at source/manifest level — local updated gate + exact ZIP inventory still pending |
| RF-002 threading / queue | Runtime locks removed and scan made shipping-aware — updated focused/full/real-bpy evidence pending |
| RF-003 Re-Polish | Runtime UI/module removed — ZIP evidence pending |
| RF-004 registration / ui_layout | Root/panel architecture simplified — lifecycle evidence and remaining owner audit pending |
| RF-005 only Import-Export tag | Manifest changed — Blender validator evidence pending |
| RF-006 Windows restriction | Manifest restriction removed — portability audit/validation pending |
| RF-007 title / same submission | Manifest version/title changed — public docs and same-submission release workflow still pending |
