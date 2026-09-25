# Implementation Plan — Blender Extensions Review Compliance

**Base:** `main@f0a0f879d639dad860c0c8c56ddba0845aa69f17`  
**Documentation branch:** `001-extensions-review-compliance`  
**Important:** this document plans a later production branch. Do not implement production changes directly on this documentation branch.

## 1. Strategy

Use a staged compliance implementation with evidence gates between risky changes. The order deliberately puts **audit and tests before lifecycle refactors**, because several moderator comments point to historical code that no longer exists in the Rewrite.

Implementation principles:

- patch the current owner/call graph, never obsolete line numbers;
- preserve Blender main-thread ownership;
- prefer explicit synchronous lifecycle boundaries over hidden background work;
- simplify only after identifying resources that truly require registration/cleanup;
- treat the built ZIP and clean installed extension as final truth;
- keep technical package identity stable while changing the public title unless migration is independently required;
- preserve retained legacy repository sources but keep them outside the shipped extension;
- each production commit references the relevant `RF-*` and `FR-*` IDs.

## 2. Branching model

1. Keep this branch documentation-only.
2. When implementation starts, create a fresh implementation branch from the then-approved `main` release baseline.
3. Rebase/merge the Spec Kit documentation into that implementation branch if required, but do not use documentation commits as a substitute for source diff review.
4. Do not merge to `main` until all mandatory gates pass from one exact clean commit.
5. Upload only a version built from the merged/approved candidate and only to the same existing Blender Extensions submission.

## 3. Phase A — Freeze reviewer contracts and add non-invasive gates

### A1. Reviewer traceability contract

Files:

- `specs/001-extensions-review-compliance/*`
- future release/check tests under `tests/`

Changes:

- encode manifest title/tags/platform/archive expectations in focused tests;
- encode same-submission wording in documentation contract tests if repository policy already tests docs;
- add source scans for prohibited persistent thread constructs and duplicate `_dup` UI functions;
- make scans target the runtime package and built ZIP where appropriate.

Why first:

- creates a red/green boundary before modifying runtime code;
- prevents already-resolved historical issues from regressing.

Gate:

- focused tests fail only on genuinely open current requirements, not on obsolete historical files.

## 4. Phase B — Current trace/persistence call-graph audit and refactor

### B1. Locate current trace owners

Search from public Analyze/Export operators through:

- plan/preparation;
- progress/diagnostics accumulation;
- persistence/storage;
- exception paths;
- cancellation paths;
- final cleanup.

Produce an implementation note listing exact current functions/classes before editing.

### B2. Remove unsupported persistent thread ownership if any remains

If current code contains a long-lived Python thread/timer owner:

- remove it;
- move finalization to explicit request boundaries;
- keep Blender API access on the main Python execution path;
- do not introduce another scheduler unless a measured UX requirement proves one is necessary.

### B3. Define finalization semantics

The final implementation must explicitly state:

- what is written on success;
- what is written on cancellation;
- what is written when preparation/export fails;
- what happens when trace persistence itself fails;
- whether finalization is idempotent or guarded;
- how temporary/staging trace files are cleaned.

### B4. Tests

Add focused tests for:

- successful request finalizes once;
- failure finalizes according to policy and preserves original exception;
- cancellation finalizes according to policy;
- duplicate cleanup path does not duplicate persistence;
- unregister has no surviving worker/thread/timer;
- no Blender API call originates from a worker callback.

Gate:

- focused persistence/thread tests;
- full Blender-independent suite;
- real bpy tests touching diagnostics/export lifecycle.

## 5. Phase C — Registration ownership simplification

### C1. Build a resource table from source

For each module in current root registration order, document:

- Blender class registrations;
- RNA properties;
- handlers;
- timers;
- msgbus subscriptions;
- preview collections;
- module-local runtime caches that have Blender object references;
- migration callbacks;
- any cleanup that can fail.

### C2. Classify every root lifecycle mechanism

For current `ExtensionRegistrationState`, `REGISTRATION_STEPS`, rollback helpers, cleanup action wrappers, and global flags, classify as:

- required for a proven Blender resource failure mode;
- duplicating module-local ownership;
- diagnostic-only;
- unnecessary Python-state cleanup.

### C3. Simplify conservatively

Preferred end state:

- root `register()` reads as a short ordered set of owner registrations plus required top-level initialization;
- root `unregister()` reads as reverse owner cleanup;
- module owners clean their own Blender resources;
- transactional rollback remains where partial acquisition can corrupt Blender state;
- no generic lifecycle state remains unless fault-injection tests demonstrate that it prevents a real failure.

Do not collapse all error handling into `try: ... except: pass`.

### C4. Fault-injection tests

Test failures at representative registration steps and assert:

- only successfully acquired resources are rolled back;
- original exception remains visible;
- no duplicate panel/class/RNA remains;
- a clean subsequent enable is possible when policy says recovery is supported.

Gate:

- root lifecycle tests;
- multiple enable/disable cycles under real bpy;
- installed-extension enable → disable → restart → enable smoke.

## 6. Phase D — UI deduplication regression pass

### D1. Inventory shared drawing ownership

Inspect `ui.py`, `ui_layout.py`, `rig_ui.py`, generated-material UI, and `repolish_ui.py`.

For each user-facing property/section, record one canonical owner.

### D2. Remove only real duplication

If duplicate drawing implementations still exist outside the reviewer-named functions:

- extract the smallest shared helper that preserves Blender context semantics;
- keep ordering-specific composition in `ui_layout.py`;
- avoid an abstraction that merely forwards every `layout.prop` through another layer without reducing duplicate rules.

### D3. Preserve UI contract

Tests must prove unchanged:

- property identifiers;
- enum values/defaults;
- visibility/enabled conditions;
- operator IDs;
- foldout ordering;
- export plan/settings produced from Scene values.

Gate:

- focused UI source/behavior tests;
- real bpy register/draw smoke where feasible.

## 7. Phase E — Public identity and submission metadata

### E1. Change display title

Primary file:

- `Blender_to_Spine2D_Mesh_Exporter/blender_manifest.toml`

Related user-facing docs:

- root `README.md`;
- `docs/README.md` if it repeats the title;
- `docs/installation.md`;
- `docs/submission.md`;
- other docs found by exact-name search.

Default technical decision:

- public `name` becomes `Spine Mesh Exporter`;
- manifest `id` remains `blender_to_spine2d_mesh_exporter`;
- package directory/module identity remains unchanged.

Only deviate if Blender validation or moderator communication requires a technical-ID migration.

### E2. Correct same-submission workflow

Rewrite `docs/submission.md` sections that say “initial submission” so they describe the existing declined listing and new-version upload workflow.

Add hard stop:

> Do not delete the declined listing and do not create another extension for this add-on.

### E3. Identity regression tests

Run installed-extension tests proving:

- AddonPreferences resolve under extension namespace;
- saved exact Spine versions persist across restart;
- title change does not change technical preference identity;
- clean upgrade/install does not register duplicate classes.

Gate:

- documentation contracts;
- installed extension persistence gate;
- extension validation.

## 8. Phase F — Manifest tag contract

Current tags are already credible. This phase should be small:

1. validate current tag spelling against the Blender version used for extension validation;
2. retain only tags tied to shipped features;
3. add test rejecting historical irrelevant tags;
4. avoid expanding tags for discoverability alone.

Expected candidate set unless implementation evidence changes:

```toml
tags = ["Import-Export", "Mesh", "UV", "Animation"]
```

Gate:

- manifest tests;
- Blender `extension validate`.

## 9. Phase G — Platform decision

### G1. Static audit

Search production package for:

- `sys.platform`, `platform.system`, `os.name`;
- `msvcrt`, `fcntl`, `winreg`, OS-specific `ctypes`;
- path/drive/UNC assumptions;
- shell/subprocess commands;
- binary wheels/native extensions;
- atomic file/lock behavior;
- environment and temporary-file assumptions.

### G2. Separate runtime portability from developer tooling

A Windows-only PowerShell release command or hardcoded local Blender executable in a test runner does not necessarily make the installed extension Windows-only. Classify runtime and development tooling separately.

### G3. Choose manifest outcome

Option 1 — Keep Windows x64:

- document concrete runtime or support/test boundary;
- keep README/submission wording precise;
- do not imply other platforms are supported.

Option 2 — Broaden support:

- adapt platform-sensitive runtime code if needed;
- run equivalent real Blender gates on every advertised platform;
- only then update/omit `platforms`.

Gate:

- platform-audit test/report;
- per-advertised-platform install/export/lifecycle run.

## 10. Phase H — Release ZIP hygiene

### H1. Build policy

Build from `Blender_to_Spine2D_Mesh_Exporter` with Blender’s extension command from an exact clean commit.

### H2. Inventory policy

Create/extend a deterministic test/tool that:

- opens the exact ZIP;
- normalizes member paths;
- rejects forbidden dev/CI/test/cache/legacy categories;
- verifies required manifest and package entrypoint exist;
- optionally compares against a reviewed prefix/allowlist policy;
- reports every offending member rather than failing on the first one.

### H3. Runtime reachability review

For included assets/modules that look unusual, document why they are runtime-required. Do not solve reviewer concern by deleting a dependency that production imports.

Gate:

- ZIP inventory test;
- `extension validate` exact archive;
- clean install from exact archive.

## 11. Phase I — Full compliance release gate

Run in this order from one exact clean candidate commit:

1. clean-head/worktree check;
2. compileall;
3. focused compliance tests;
4. full `tests` suite;
5. full `tests_bpy` suite;
6. representative Blender headless exports and Analyze paths;
7. installed-extension preference persistence/restart gate;
8. repeated enable/disable lifecycle gate;
9. Blender extension build;
10. exact ZIP inventory inspection;
11. Blender extension validate;
12. clean-profile install from disk;
13. reviewer walkthrough;
14. final clean-tree/HEAD verification;
15. SHA256 of exact upload ZIP.

Any source/doc/manifest edit after step 9 invalidates the built artifact and requires rebuilding/re-running package gates.

## 12. Phase J — Upload and moderation response

1. Open the **existing declined extension submission**.
2. Upload the corrected higher-version ZIP.
3. Do not create a second extension.
4. Confirm generated listing metadata: title, compatibility, platforms, permissions, license, website, tags.
5. Add a changelog organized around reviewer issues, not internal implementation noise.
6. In moderation reply, map each former concern to concrete evidence/tests.
7. Keep artifact SHA256 and candidate Git SHA in release notes/internal record.

## 13. Rollback plan

If a runtime compliance refactor destabilizes export/lifecycle behavior:

- revert the affected implementation slice, not the Spec Kit requirements;
- retain non-invasive tests that demonstrate the unresolved issue;
- re-plan that slice from current ownership evidence;
- never bypass a failed moderator requirement by weakening/removing the test unless the requirement itself was reinterpreted with documented evidence.

If display-name change breaks technical identity:

- restore stable technical identifiers;
- keep public title change separate;
- add a migration only if technically unavoidable.

If cross-platform validation fails:

- do not advertise the failing platform;
- keep/narrow the manifest platform list until the runtime issue is fixed and validated.

## 14. Definition of done per implementation slice

A slice is complete only when:

- its `RF-*` / `FR-*` IDs are cited in the implementation record;
- source behavior is implemented without touching unrelated legacy/runtime surfaces;
- focused tests pass;
- relevant real Blender gate passes;
- docs/tasks/checklist are updated with evidence;
- no known cleanup/thread/package regression remains hidden for a later slice.
