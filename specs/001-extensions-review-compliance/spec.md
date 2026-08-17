# Feature Specification — Blender Extensions Review Compliance

**Feature ID:** `001-extensions-review-compliance`  
**Base commit:** `main@f0a0f879d639dad860c0c8c56ddba0845aa69f17`  
**Status:** Documentation approved for implementation planning; production implementation intentionally not part of this branch.

## 1. Problem statement

A previous Blender Extensions moderation pass declined the extension and identified release-policy, Blender runtime-safety, package-hygiene, platform, lifecycle, naming, and code-quality issues. Since then the repository has undergone a major Blender 5.2+ Rewrite and several historical symbols/files named by the reviewer are no longer present. The next correction must therefore do two things at the same time:

1. close every moderation requirement with evidence; and
2. avoid reintroducing obsolete historical architecture merely to make the new code resemble the old review references.

The implementation must be based on the current Rewrite package, while the reviewer feedback remains the acceptance contract.

## 2. Goals

- Produce one auditable compliance plan that maps every reviewer comment to current files, future code tasks, tests, package checks, and submission evidence.
- Eliminate unsupported persistent Python threading/thread timers from the shipped Blender extension.
- Give trace/diagnostic persistence an explicit request lifecycle with deterministic flush/cleanup ownership.
- Reduce root register/unregister complexity where it is not required by actual Blender resource ownership.
- Keep UI drawing canonical and non-duplicated.
- Ship only runtime-required files in the extension ZIP.
- Resolve the public extension title as requested by moderation without accidentally breaking stable package identity.
- Make a justified platform declaration and test every platform advertised by the manifest.
- Upload the corrected build as a **new version of the same existing submission**.
- Preserve export behavior, source-scene safety, Spine schema behavior, exact-version preferences, and retained legacy boundaries.

## 3. Non-goals

- No production code changes in this documentation branch.
- No rewrite of geometry, UV, baking, rigging, camera projection, Spine codec, exact-version, or export semantics unless a compliance change directly requires it.
- No deletion or modification of retained pre-Rewrite legacy sources merely for cleanup; they can be excluded from the shipping archive instead.
- No creation of a second Blender Extensions listing.
- No package/module/manifest-ID rename merely because the public title changes.
- No claim of macOS/Linux support without corresponding validation evidence.
- No introduction of a new background execution framework to replace one unsupported thread with another hidden scheduler.

## 4. Baseline facts

At the base commit:

- manifest version is `0.154.0`;
- public name is `Blender to Spine2D Mesh Exporter`;
- minimum Blender is 5.2.0;
- manifest tags are `Import-Export`, `Mesh`, `UV`, `Animation`;
- manifest platforms are `["windows-x64"]`;
- manifest build exclusions already cover caches, `.git`, `.github`, tests, docs, ZIPs, and named retained legacy files;
- root Rewrite registration is transactional and stateful;
- current `ui_layout.py` does not contain the reviewer-named `*_dup` drawing functions;
- historical reviewer path `spine_core/chat_persistence.py` is not part of the current Rewrite package path;
- current `docs/submission.md` still describes an initial submission, which conflicts with the moderator’s same-submission instruction.

These facts are not proof of final compliance. They are the starting state.

## 5. User/reviewer scenarios

### US-001 — Moderator inspects the corrected upload

**As a Blender Extensions moderator**, I can install the corrected ZIP, inspect its manifest and contents, enable/disable the extension, and see that the previous violations are absent without needing repository development files.

Acceptance:

- archive contains only runtime-required files/resources;
- manifest metadata is valid and relevant;
- no unsupported long-lived Python thread/timer behavior is present;
- register/unregister is stable;
- title/listing matches the accepted naming decision;
- package is uploaded as a new version under the existing submission.

### US-002 — User installs and exports after compliance refactor

**As an existing exporter user**, I can install/upgrade the compliant version and use Analyze/Export with the same supported export semantics as before.

Acceptance:

- no scene/mesh/export semantic regression is introduced by compliance work;
- exact Spine project-version preferences still resolve and persist;
- normal, camera, depth, single-object, and selected-object routes remain governed by their existing capability contracts;
- disable/re-enable does not create duplicate panels/classes/properties.

### US-003 — Maintainer produces a repeatable release candidate

**As the maintainer**, I can run documented gates from one clean commit and know whether the candidate is safe to upload.

Acceptance:

- release gate binds tests, Blender validation, ZIP inventory, install smoke, and SHA256 to one exact commit/artifact;
- failures identify which reviewer requirement remains open;
- release docs do not tell the maintainer to create a duplicate listing.

## 6. Functional requirements

### Submission and identity

**FR-001** The corrected artifact MUST be uploaded as a new version of the existing declined Blender Extensions submission.

**FR-002** Documentation and release procedure MUST explicitly prohibit deleting the declined submission and creating another extension for the same add-on.

**FR-003** The public extension title MUST be changed to the moderator-accepted title, using `Spine Mesh Exporter` as the reviewer-provided target unless later moderator communication supersedes it.

**FR-004** A public-title change MUST NOT automatically change manifest `id`, Python package path, extension namespace handling, saved preference lookup, or repository identity.

**FR-005** If a stable technical identifier is changed for an independently proven reason, the implementation MUST include explicit migration and installed-extension tests.

### Thread/runtime safety

**FR-006** Shipped production code MUST NOT own a long-lived `threading.Thread`, `threading.Timer`, or equivalent Python background thread that continues while Blender resumes normal execution.

**FR-007** Shipped production code MUST NOT call `bpy`, Blender RNA, Blender UI, `bmesh`, or Blender-owned data from a non-main Python thread.

**FR-008** Static release checks MUST scan the built runtime package, not merely selected source files, for prohibited persistent-thread constructs.

**FR-009** If a standard-library/dependency construct can spawn a thread implicitly, its use MUST be audited and either removed or proven to complete entirely while Blender is synchronously blocked.

### Pipeline trace / diagnostic persistence

**FR-010** Trace/diagnostic persistence MUST have an explicit owner in the Analyze/Export request lifecycle.

**FR-011** The lifecycle MUST define what is persisted on success, cancellation, validation failure, export failure, and cleanup failure.

**FR-012** Deferred persistence MUST NOT depend on a session-owned Python thread/timer that survives the initiating call.

**FR-013** Persistence MUST have deterministic idempotency semantics: a request cannot accidentally write the same final trace multiple times because more than one cleanup path fires.

**FR-014** Any Blender timer retained for UI/main-thread scheduling MUST be separately justified, registered/unregistered by an explicit Blender owner, and MUST NOT be confused with `threading.Timer`.

### Manifest metadata

**FR-015** Manifest tags MUST come from Blender’s current supported tag taxonomy.

**FR-016** Every selected tag MUST correspond to meaningful shipped behavior.

**FR-017** Historical irrelevant tags such as `backup` and `text-editing` MUST NOT appear in the candidate manifest.

**FR-018** Manifest version, public documentation version, built archive name, and submission changelog MUST agree for the final release candidate.

### Archive hygiene

**FR-019** The Blender-built extension ZIP MUST exclude repository-only development files, tests, CI automation, source-control metadata, local environments, caches, previous archives, temporary files, and retained non-runtime legacy implementation sources.

**FR-020** Archive hygiene MUST be checked against the built ZIP member list after build; build-exclusion configuration alone is insufficient evidence.

**FR-021** The release process MUST maintain an explicit runtime inclusion rationale for non-Python assets/resources shipped in the ZIP.

**FR-022** Blender’s `extension validate` MUST run against the exact archive that is intended for upload.

**FR-023** The exact validated archive MUST be installed from disk in a clean/isolated Blender profile before submission.

### Platform declaration

**FR-024** The implementation MUST inventory platform-sensitive code and dependencies before changing `platforms`.

**FR-025** Platform audit MUST include filesystem locking, atomic replace/fsync behavior, path semantics, subprocess/shell use, binary dependencies/wheels, environment assumptions, Blender executable invocation, and OS-specific APIs.

**FR-026** Every advertised platform MUST pass equivalent install, register, Analyze/export, output transaction, preference-persistence, disable/unregister, and archive gates.

**FR-027** If only Windows is advertised, release documentation MUST state the concrete technical/testing reason; “we only tested Windows” is a release limitation statement, not proof that the code itself requires Windows.

**FR-028** If the platform list is omitted or broadened, the candidate MUST have evidence for all resulting advertised platforms.

### Registration lifecycle

**FR-029** Every action in root `register()` and `unregister()` MUST correspond to a real Blender/runtime resource owner or a proven rollback requirement.

**FR-030** Ordinary Python constants/module globals MUST NOT be manually “cleaned” unless their retained value creates an observable Blender lifecycle problem.

**FR-031** Required Blender resources — registered classes, RNA properties, handlers, timers, message-bus subscriptions, preview collections, runtime callbacks — MUST be removed by their owner on unregister.

**FR-032** Partial registration failure MUST roll back only resources successfully acquired before the failure.

**FR-033** Repeated enable/disable and clean restart MUST leave no duplicate class, panel, RNA property, handler, timer, or subscription.

**FR-034** Simplification MUST NOT reduce diagnostic quality: registration failures still need actionable logging and the original exception chain.

### UI deduplication

**FR-035** Production UI modules MUST have one canonical implementation for each shared export property/section.

**FR-036** No production `*_dup` UI function family may be introduced to bypass refactoring.

**FR-037** Shared helpers MAY be introduced only when they reduce duplication without hiding Blender context/ownership requirements.

**FR-038** UI consolidation MUST preserve RNA property IDs, operator IDs, persisted defaults, visibility/enabled rules, section ordering, and export settings produced from the UI.

### Behavior preservation

**FR-039** Compliance work MUST NOT alter retained legacy exporter source unless a reviewer requirement specifically targets a shipped legacy artifact; exclusion from archive is preferred where legacy is intentionally retained only in the repository.

**FR-040** Compliance work MUST preserve supported Spine schema-family and exact-project-version behavior.

**FR-041** Compliance work MUST preserve Blender state restoration and resource cleanup on success and failure.

**FR-042** Existing full Python, bpy, Blender headless, installed-extension, and representative real-export gates MUST continue to pass unless a test is deliberately updated to reflect an approved compliance contract.

### Documentation and traceability

**FR-043** Every moderator feedback item MUST map to at least one task and one closure-evidence item.

**FR-044** Every future implementation commit/PR for this feature MUST identify which FR/RF IDs it closes.

**FR-045** `research.md` MUST distinguish observed current-main facts, reviewer requirements, Blender official guidance, and implementation hypotheses.

**FR-046** `tasks.md` MUST remain the fine-grained execution ledger and MUST be updated as implementation progresses rather than replaced by ad-hoc chat history.

## 7. Non-functional requirements

**NFR-001 Safety:** no compliance change may make Blender less stable merely to shorten code.

**NFR-002 Determinism:** release evidence must be reproducible from one exact commit.

**NFR-003 Maintainability:** resource ownership must be local and understandable; helpers must reduce, not redistribute, lifecycle ambiguity.

**NFR-004 Observability:** failures in registration, persistence, packaging, or platform gates must state the failing stage and preserve useful exception/log context.

**NFR-005 Compatibility:** use Blender 5.2+ supported APIs and extension packaging semantics.

**NFR-006 No test hardcoding:** tests may define fixtures/expected contracts, but production code cannot contain branches added only to satisfy test values.

## 8. Acceptance matrix

| Area | Acceptance evidence |
| --- | --- |
| Same submission | release checklist + actual upload to existing submission |
| New title | manifest/docs/listing agreement + installed preference identity test |
| Thread safety | static scan + lifecycle tests + Blender runtime smoke |
| Trace persistence | deterministic success/failure/cancel tests + no background owner |
| Tags | manifest parser/Blender validation + relevance review |
| ZIP hygiene | exact ZIP inventory allow/deny gate |
| Platforms | platform audit + per-advertised-platform Blender gates |
| Registration | repeated lifecycle + injected partial-failure rollback tests |
| UI dedup | source/static contract + UI behavior tests |
| No regressions | full tests + bpy + Blender headless/real representative exports |
| Submission | exact artifact SHA256 + same-listing upload record |

## 9. Exit criteria

This feature is complete only when:

1. every `RF-*` row in `review-feedback.md` is marked closed with evidence;
2. every mandatory `FR-*` has a passing test, documented inspection, or release artifact proof appropriate to that requirement;
3. final production changes are based on a fresh implementation branch from the intended release baseline, not committed into this documentation-only branch;
4. the exact candidate ZIP validates, installs, enables, exercises, disables, and uninstalls cleanly;
5. the new version is uploaded to the **same existing Blender Extensions submission**;
6. no unresolved moderator item is silently reclassified as “not applicable” without evidence.
