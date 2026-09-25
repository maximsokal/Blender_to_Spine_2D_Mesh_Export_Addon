# Research — Current Rewrite vs Blender Extensions Review

**Baseline:** `main@f0a0f879d639dad860c0c8c56ddba0845aa69f17`  
**Research type:** source/release-policy audit. No production changes are performed in this branch.

## 1. Method

The audit starts from moderator feedback, then checks the current Rewrite rather than assuming historical line numbers still exist. Each conclusion is labeled as one of:

- **Observed** — directly present in the baseline source/docs/manifest;
- **Reviewer requirement** — moderation acceptance condition;
- **Official Blender guidance** — current Blender documentation used to constrain a solution;
- **Inference / future proof required** — plausible conclusion that still needs implementation-time validation.

## 2. Current package identity and manifest

### Observed

`Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml` currently declares:

```toml
id = "blender_to_spine2d_mesh_exporter"
version = "0.154.0"
name = "Blender to Spine2D Mesh Exporter"
blender_version_min = "5.2.0"
tags = ["Import-Export", "Mesh", "UV", "Animation"]
platforms = ["windows-x64"]
```

The manifest also declares only filesystem permission and has build exclusions for caches, VCS/CI metadata, tests, docs, ZIPs, and a set of retained pre-Rewrite files.

### Consequences

- **RF-002 remains open**: the public title has not yet been changed to the reviewer-suggested `Spine Mesh Exporter`.
- **RF-005 appears already addressed** at source level: historical `backup`/`text-editing` tags are gone and the current four tags belong to Blender’s supported add-on tag taxonomy.
- **RF-006 is only partially proven**: exclusion rules exist, but only inspection of the Blender-built ZIP can prove what ships.
- **RF-007 remains open**: the package still advertises Windows x64 only.

## 3. Submission workflow conflict

### Observed

`docs/submission.md` says the current package is prepared for an **initial** Blender Extensions Platform submission and describes creating/uploading a submission candidate.

### Reviewer requirement

The moderator instructed the maintainer to upload a **new version to the same declined extension**, not delete/recreate it and not create a duplicate extension.

### Required documentation correction

The future implementation/release branch must update the publication docs so every release path says:

```text
existing declined submission
-> corrected higher extension version
-> exact validated ZIP
-> upload new version to existing submission
-> moderation re-review
```

The release procedure must never tell the maintainer to create a second listing.

## 4. Threading audit

### Reviewer historical finding

The reviewer cited a historical `spine_core/chat_persistence.py` background worker created with `threading.Thread(...)` and rejected persistent Python threading inside Blender.

### Current-main observation

The historical path is not present in the current Rewrite package at the baseline. Repository/code inspection performed for this spec also did not locate the reviewer-named `PipelineTraceSession` symbol or a current `bpy.app.timers` occurrence through the available source search.

This is useful evidence but **not enough for final closure**, because:

- search/index coverage is not the built ZIP;
- dependencies or newly added modules could reintroduce threading later;
- implicit thread producers can exist without a literal `threading.Thread` call.

### Official Blender constraint

Blender’s Python API documentation states that Python threading is not supported as a persistent background mechanism and can lead to difficult-to-diagnose crashes. The documentation specifically distinguishes a thread that completes while Blender remains synchronously blocked from a thread/timer that continues after script control returns.

### Implementation decision to validate

Preferred compliance architecture:

1. keep export/analyze orchestration on Blender’s main Python execution path;
2. make persistence synchronous at explicit lifecycle boundaries unless measurements prove that impossible;
3. if work needs chunking to keep the UI responsive, use a Blender-owned main-thread lifecycle such as a modal operator or carefully owned `bpy.app.timers` callback — not a Python thread — and only after a concrete need is demonstrated;
4. never access `bpy` from non-main-thread callbacks;
5. add a built-package static scan so historical thread constructs cannot return unnoticed.

No new scheduler should be implemented merely because the old thread is removed.

## 5. Pipeline trace / diagnostic persistence audit

### Reviewer historical finding

The reviewer cited `logging_utils.py` around a historical `PipelineTraceSession` and asked for explicit post-export/flush behavior rather than a session-owned timer arrangement. The important semantic concern is lifecycle ownership, not the obsolete line number.

### Current-main reality

The Rewrite includes structured diagnostics, atomic output, transactional staging, rollback, and logging infrastructure. The historical class name is not the correct current change target. A future code pass must trace the actual current call graph from public Analyze/Export entrypoints to any diagnostic/trace writer.

### Required lifecycle model

The implementation should document and test a state model such as:

```text
REQUEST_CREATED
  -> RUNNING
  -> SUCCEEDED | CANCELLED | FAILED
  -> TRACE_FINALIZED
  -> REQUEST_CLEANED
```

The exact enum/classes are not mandated. What is mandated is the behavior:

- one owner decides final trace contents;
- cleanup happens even when export raises;
- trace finalization does not depend on hidden background work;
- finalization is idempotent or guarded against duplicate writes;
- a trace write failure is logged without masking the original export error unless policy explicitly makes trace persistence fatal.

### Open research item

Before implementation, identify the current equivalent(s) of:

- trace session construction;
- per-stage event accumulation;
- final serialization/storage;
- deferred storage if any;
- exception/cancellation handling;
- unregister-time cleanup.

Do not patch historical filenames that no longer exist.

## 6. Root registration lifecycle audit

### Observed current design

`Blender_to_Spine2D_Mesh_Exporter/__init__.py` currently defines:

- `ExtensionRegistrationState` with `UNREGISTERED`, `REGISTERING`, `REGISTERED`, `UNREGISTERING`, `DEGRADED`;
- global `_REGISTRATION_STATE`;
- an ordered `REGISTRATION_STEPS` tuple;
- transactional Scene RNA registration;
- reverse cleanup actions;
- rollback after partial registration failure;
- a degraded state if rollback/unregister fails;
- explicit logging initialization after registration;
- guardrails around repeated or reentrant lifecycle calls.

### Reviewer concern

The reviewer considered `register()` / `unregister()` overly complex and questioned cleanup of globals.

### Engineering interpretation

The reviewer concern should **not** be translated into “delete rollback/error handling”. Blender registration is stateful, and a failure after some classes/RNA properties are registered can leave the session broken. The correct target is to remove complexity that does not correspond to a real resource or failure mode.

### Required ownership inventory

For every registration step, implementation must record:

| Owner | Resource acquired | Blender requires unregister? | Rollback needed? | Current helper |
| --- | --- | --- | --- | --- |
| Addon preferences | Blender classes/operators | Yes | Yes | module register/unregister |
| Scene settings | RNA properties | Yes | Yes | transactional RNA helper |
| UI | Panel/classes | Yes | Yes | module register/unregister |
| Readiness | handlers/timers/classes if present | audit | audit | module owner |
| Generated material UI | classes/UI state | audit | audit | module owner |
| Ordered UI | replacement panel + RNA | Yes | Yes | `ui_layout.register/unregister` |
| Re-Polish UI | module-specific resources | audit | audit | module owner |
| Single-object operator | operator class(es) | Yes | Yes | module owner |

This table must be completed from source before simplifying root lifecycle code.

### Simplification heuristic

Prefer:

- module-local resource ownership;
- one straightforward ordered register sequence;
- one reverse unregister sequence;
- rollback only for resources actually acquired;
- no resetting immutable constants or harmless Python globals;
- state flags only where Blender cannot answer whether an owned resource is registered safely.

Avoid:

- a generic framework whose complexity exceeds the resources it manages;
- suppressing cleanup errors without logging;
- broad exception handling that leaves unknown Blender state;
- re-registering classes to “recover” without knowing current RNA ownership.

## 7. `ui_layout.py` duplication audit

### Historical reviewer finding

The reviewer saw duplicated functions such as `draw_common_export_settings()` and `draw_common_export_settings_dup()` and proposed extracting shared property rendering before returning to canonical function names.

### Observed current design

The current file:

- defines one `OBJECT_PT_Spine2DOrderedMeshPanel`;
- delegates shared foldout/cut/bake/readiness behavior to `ui.OBJECT_PT_Spine2DMeshPanel` where appropriate;
- owns only ordering-specific additions and export-action placement;
- contains no reviewer-named `*_dup` functions in the inspected baseline;
- performs transactional replacement/restoration of the base panel.

### Status

The historical duplication appears resolved. The future compliance implementation should not churn this code merely to satisfy an obsolete line reference. Instead it should add/retain regression checks that prohibit a duplicate-function pattern and prove one canonical owner for each property.

### Remaining UI lifecycle concern

The ordered panel currently unregisters the base panel and registers a replacement with the same `bl_idname`, then restores the base panel during unregister. This is a legitimate lifecycle resource and must be considered in RF-008 simplification; it must not be accidentally removed from rollback ownership.

## 8. Archive hygiene audit

### Observed manifest exclusions

Current `[build].paths_exclude_pattern` covers:

- `__pycache__/` and compiled Python;
- `.git/`, `.github/`;
- `tests/`, `docs/` inside the extension build root;
- ZIP files;
- named retained pre-Rewrite modules and `Legacy/`.

### Reviewer risk model

Historical rejected archive included development files such as CI scripts and tests. Therefore the release policy must use a **post-build member inventory**.

### Proposed inventory policy

Reject any ZIP member matching categories such as:

```text
.git/**
.github/**
**/__pycache__/**
**/*.pyc
**/*.pyo
**/*.pyd unless explicitly declared runtime binary
**/tests/**
**/tests_bpy/**
**/tools/** unless a runtime tool is explicitly justified
**/docs/** unless intentionally shipped user docs
**/.venv*/**
**/.pytest_cache/**
**/.mypy_cache/**
**/.ruff_cache/**
**/dist/**
**/*.zip
CI/release-only scripts
retained non-runtime legacy implementation sources
```

The actual gate should be based on an explicit policy module/test, not a shell-only wildcard that differs between operating systems.

### Positive allow rationale

Expected runtime categories are:

- manifest;
- package `__init__.py` and production Python modules reachable by the extension;
- runtime assets/icons/resources that production code loads;
- bundled wheels only if declared and required;
- license/resource files required by Blender or licensing policy.

Final member list must be reviewed from the exact candidate ZIP.

## 9. Manifest tags audit

### Observed

Current tags: `Import-Export`, `Mesh`, `UV`, `Animation`.

### Official taxonomy

Blender’s current extension tag list includes these tags. The current exporter:

- exports a non-Blender data format → `Import-Export` is strongly relevant;
- operates on Mesh objects and creates mesh attachments → `Mesh` is relevant;
- generates/uses UV layouts → `UV` is relevant;
- generates animation controls/sequences → `Animation` is defensible.

### Recommendation

Keep the current set unless platform moderation gives different guidance. Do not add tags merely for discoverability. Add an automated manifest contract that rejects historical unrelated tags.

## 10. Platform audit

### Observed

- manifest advertises `windows-x64` only;
- README says Windows is the currently tested desktop platform;
- moderator questioned this, especially because POSIX-oriented locking had existed/been visible historically.

### Official manifest semantics

Blender documentation states that `platforms` is optional; if omitted, the extension is available on all operating systems. Therefore omission is a **support claim**, not a neutral setting.

### Required technical inventory

Before deciding, inspect all production paths for:

1. imports: `msvcrt`, `fcntl`, `winreg`, `ctypes` platform APIs, `os.name`/`sys.platform` branches;
2. file locking and stale-lock recovery;
3. atomic file replacement and directory fsync semantics;
4. path separators, drive letters, UNC assumptions, case sensitivity;
5. subprocess invocations and shell syntax;
6. binary wheels/native libraries;
7. font/image/render engine dependencies;
8. temporary directory behavior;
9. Blender executable discovery — release tools may be Windows-only without making runtime package Windows-only;
10. tests that accidentally encode Windows paths.

### Decision matrix

| Finding | Manifest outcome | Required evidence |
| --- | --- | --- |
| runtime has a hard Windows-only dependency | keep `windows-x64` | document exact dependency and Windows gates |
| runtime is portable but only Windows has been validated | either keep Windows until validation expands, or validate more OSes before broadening | explicit policy, no unsupported claim |
| runtime passes full gates on Windows/Linux/macOS advertised architectures | broaden/omit platforms as appropriate | equivalent install/export/lifecycle evidence on each platform |

A POSIX lock implementation is evidence to investigate portability, not sufficient proof of portability by itself.

## 11. Public title / technical identity audit

### Official Blender namespace guidance

Blender Extensions add repository identity to module namespaces (`bl_ext.<repository>.<extension>`). Add-on preferences should use package identity (`__package__`) rather than a hardcoded user-facing name. Subpackages needing the top-level identity should import it relatively.

### Consequence for reviewer rename

Changing the display `name` in `blender_manifest.toml` is low-risk compared with changing `id` or package directory. The compliance implementation should therefore:

1. change public title/listing wording to `Spine Mesh Exporter`;
2. preserve stable `id = "blender_to_spine2d_mesh_exporter"` unless a separate requirement exists;
3. preserve package namespace behavior;
4. run clean installed-extension AddonPreferences persistence tests after the title change.

## 12. Existing repository gates worth preserving

`docs/testing.md` already defines:

- clean exact-commit boundary;
- compileall over production/tests/tools;
- full `pytest tests` suite;
- real `tests_bpy` suite;
- Blender headless runners using `--python-exit-code 1`;
- installed extension exact-version persistence in isolated Blender configuration;
- extension build/validate, ZIP inventory, install-from-disk, lifecycle checks;
- final SHA256 capture.

The compliance implementation should extend these gates instead of creating a competing release system.

## 13. Risk register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| “Simplifying” registration removes necessary rollback | High | complete resource inventory first; fault-injection tests |
| removing a thread introduces lost diagnostics on exceptions | High | explicit finalization in `try/except/finally`; idempotency tests |
| display rename breaks preferences by changing technical ID | High | title-only change by default; installed extension persistence gate |
| platform list broadened without real validation | High | no support claim without per-platform gates |
| build exclusions look correct but ZIP still ships dev files | High | post-build ZIP inventory is authoritative |
| old reviewer line numbers cause edits to obsolete code | Medium | map intent to current call graph, not historical filenames |
| UI dedup refactor changes setting identity/defaults | Medium | RNA/operator/visibility regression tests |
| same-submission rule forgotten during release | High | hard-block checklist item and submission docs update |

## 14. Research conclusions

1. The current Rewrite already appears to have eliminated the exact historical UI duplicate functions and historical thread owner, but both require release-grade regression proof.
2. Current manifest tags are much closer to moderator expectations and use valid current Blender labels.
3. Archive exclusion configuration has improved, but archive member inspection remains mandatory.
4. Public naming, same-submission documentation, platform justification, and registration-complexity review are still open.
5. The safest implementation order is to establish static/package tests and ownership inventories **before** changing runtime lifecycle code.
6. No production code should be modified from historical reviewer line numbers until the current owner/call path has been identified.
