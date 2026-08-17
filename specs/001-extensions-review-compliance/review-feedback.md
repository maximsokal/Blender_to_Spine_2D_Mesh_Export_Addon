# Blender Extensions Review Feedback Ledger

**Feature:** `001-extensions-review-compliance`  
**Documentation baseline:** `main@f0a0f879d639dad860c0c8c56ddba0845aa69f17`  
**Scope of this branch:** documentation and implementation planning only. Production Python, manifest metadata, tests, and release scripts are not changed here.

This ledger is the source-of-truth transcription/paraphrase of the moderation issues that must be closed before the next Blender Extensions upload. It deliberately separates **reviewer requirements** from **current-main observations** and from **future implementation decisions**.

## RF-001 — Continue the same declined submission

Reviewer requirement:

- resolve the listed violations and upload a **new version**;
- do not delete the declined extension and submit a replacement as a different extension;
- do not create another duplicate extension for the same add-on;
- the listing may receive a new name when the corrected version is uploaded.

Current-main observation:

- `docs/submission.md` currently describes the package as an “initial Blender Extensions Platform submission”. That wording does not represent the moderator-requested resubmission workflow and must be corrected in the future implementation/release branch.

Required closure evidence:

1. release documentation says “new version of the existing submission”, not “new/initial submission”;
2. the release checklist explicitly blocks creation of a duplicate listing;
3. the upload record references the same existing extension submission.

## RF-002 — Rename the public extension title

Reviewer requirement:

- current name was considered misleading;
- reviewer suggested **`Spine Mesh Exporter`**.

Current-main observation:

- `blender_manifest.toml` still declares `name = "Blender to Spine2D Mesh Exporter"`;
- README and submission documentation use that same title.

Future implementation rule:

- treat this as a **display/listing title** change first;
- do not casually rename the manifest `id`, Python package, persisted preference identity, module namespace, or repository unless a separate migration requirement proves that necessary;
- Blender Extension package namespaces vary by repository (`bl_ext.<repository>.<extension>`), therefore identity changes must not be inferred from a title change.

Required closure evidence:

1. final manifest/listing title follows the accepted reviewer-facing name;
2. user-facing docs use the same title;
3. stable manifest/package identity is either preserved or deliberately migrated with tests;
4. clean installed-extension preference lookup still works after the display-name change.

## RF-003 — Remove unsupported persistent Python threading

Reviewer requirement:

- Blender Python integration is not thread-safe for persistent/background Python threads;
- reviewer specifically called out historical `spine_core/chat_persistence.py` and a `threading.Thread(...)` background worker;
- replace the background-thread design with Blender-safe/batch/main-thread behavior.

Current-main observation:

- the historical reviewer path `Blender_to_Spine2D_Mesh_Exporter/spine_core/chat_persistence.py` is not present in the current Rewrite package at the documentation baseline;
- the current baseline therefore must be verified by **absence/static-scan and runtime gates**, not by blindly editing a historical file that no longer exists.

Required closure evidence:

1. production extension package has no long-lived `threading.Thread`, `threading.Timer`, `multiprocessing.Queue`, or equivalent hidden background-thread owner;
2. no Blender RNA/data/UI API is accessed from non-main Python threads;
3. any retained concurrency is explicitly proven to finish while Blender is blocked, or is outside Blender and isolated from `bpy`;
4. extension disable/unregister leaves no worker/thread alive.

## RF-004 — Make `PipelineTraceSession` persistence explicit and thread-safe

Reviewer requirement:

- historical `logging_utils.py` / `PipelineTraceSession` used timer-driven deferred persistence in a way the reviewer considered thread-like/unsafe;
- make sure `bpy` is never accessed from another thread;
- prefer explicit persistence/flush points, for example a post-export flush or a direct `store_pipeline_trace_deferred()`-style operation, rather than a hidden timer owner.

Current-main observation:

- the historical reviewer symbol/path is not part of the currently located Rewrite surface at this baseline;
- current export/diagnostic persistence must still be audited so that an equivalent hidden scheduler has not replaced the old implementation.

Required closure evidence:

1. trace persistence ownership is explicit in the export/analyze lifecycle;
2. success, cancellation, and exception paths all end in a deterministic flush/cleanup decision;
3. no session-owned Python timer/thread survives the request;
4. no `bpy` call is reachable from a worker callback;
5. tests prove exactly-once/at-most-once persistence as defined by the implementation contract.

## RF-005 — Use accurate Blender Extension tags

Reviewer requirement:

- historical tags such as `backup` and `text-editing` did not describe the extension and had to be removed/replaced.

Current-main observation:

- the baseline manifest now contains `Import-Export`, `Mesh`, `UV`, and `Animation`;
- those labels are in Blender’s supported add-on tag taxonomy, and each has a plausible relation to current product behavior;
- this concern appears **materially addressed in current main**, but the release gate must prevent regression to unsupported/unrelated tags.

Required closure evidence:

1. every manifest tag is accepted by current Blender tooling/platform validation;
2. every tag is justified by shipped functionality;
3. no historical `backup` / `text-editing` tags reappear.

## RF-006 — Ship a clean extension ZIP, not development material

Reviewer requirement:

- previous package contained development files/directories such as CI scripts and `tests/`;
- remove unnecessary development content to keep the extension archive small and clean.

Current-main observation:

- the current manifest has `[build].paths_exclude_pattern` entries for caches, `.git`, `.github`, tests, docs, ZIPs, and retained pre-Rewrite source files;
- this is a positive baseline change but **the manifest pattern alone is not closure evidence**. The built ZIP inventory is the evidence.

Required closure evidence:

1. build with Blender’s extension builder from the exact candidate commit;
2. inspect every ZIP member;
3. reject `tests/`, CI scripts, repository metadata, docs not needed at runtime, temporary files, local environments, caches, prior ZIPs, and retained legacy implementation sources;
4. verify that every included Python/data file is required by the runtime or declared resource set;
5. validate the exact ZIP with Blender and install that exact ZIP in a clean profile.

## RF-007 — Audit the `windows-x64` platform restriction

Reviewer requirement:

- explain why only `windows-x64` is declared;
- reviewer noted a POSIX-lock implementation and questioned the Windows-only claim.

Current-main observation:

- `blender_manifest.toml` still declares `platforms = ["windows-x64"]`;
- README/submission docs describe Windows as the currently tested/targeted platform;
- Blender’s manifest semantics allow omitting `platforms` for all operating systems, but support must be based on evidence rather than assumption.

Required closure evidence:

1. inventory platform-sensitive imports, filesystem semantics, locking, subprocess behavior, binary wheels, shell commands, path handling, and Blender-version assumptions;
2. decide one of two explicit outcomes:
   - **portable**: remove/narrow no platform list only after required OS validation, or
   - **Windows-only**: keep `windows-x64` and document the concrete technical dependency that requires it;
3. never infer portability merely because the package is pure Python;
4. run clean install/register/export/cleanup gates on every advertised platform.

## RF-008 — Simplify and justify `register()` / `unregister()` ownership

Reviewer requirement:

- root registration/unregistration was considered overly complex;
- cleanup of globally used variables appeared unnecessary to the reviewer.

Current-main observation:

- Rewrite `__init__.py` still owns a substantial transactional lifecycle: registration state enum, ordered steps, rollback actions, RNA registration, logging initialization, and best-effort reverse cleanup;
- some complexity may protect Blender from partial registration, but it must be **earned by resource ownership**, not exist as generic defensive state machinery.

Required closure evidence:

1. inventory each registered class, RNA property, handler, timer, message-bus owner, preview collection, temporary resource, and module callback;
2. distinguish cleanup Blender actually requires from ordinary Python module globals that need no manual reset;
3. reduce the root lifecycle to the smallest design that still guarantees rollback and repeat enable/disable safety;
4. tests cover register → unregister, repeated cycles, partial failure rollback, and restart/clean-install behavior;
5. no stale panel/class/RNA registration survives disable.

## RF-009 — Remove duplicated UI drawing functions

Reviewer requirement:

- historical `ui_layout.py` contained duplicated functions such as `draw_common_export_settings()` and `draw_common_export_settings_dup()`;
- reviewer suggested extracting shared property drawing (for example `_shared_prop()`), deleting first copies, then renaming `_dup` implementations to canonical names.

Current-main observation:

- the current `ui_layout.py` inspected at the baseline no longer exposes those named duplicate functions;
- current ordered UI delegates shared behavior back to `ui.OBJECT_PT_Spine2DMeshPanel` for several drawing operations;
- this item appears **structurally addressed**, but a duplicate-code/static contract is still required so the pattern does not return.

Required closure evidence:

1. no `_dup` production drawing functions remain;
2. one canonical owner exists for each shared UI property/section;
3. refactor does not change property IDs, defaults, visibility rules, operator IDs, foldout order, or exported settings;
4. UI tests compare behavior, not only source text.

## RF-010 — Re-polish / self-contained quality pass

This is an internal quality workstream derived from the moderator feedback as a whole; it is **not treated as a verbatim reviewer quote** unless the original moderation record says otherwise.

Current-main observation:

- the Rewrite has a dedicated `repolish_ui` module and multiple transactional/resource-safety helpers;
- those abstractions must be judged against actual runtime ownership and reviewer simplicity requirements.

Required closure evidence:

1. extension runtime is self-contained inside the built package;
2. no developer-only behavior is required to run it;
3. no unrelated side effects occur on import/register;
4. UI/runtime abstractions reduce duplication instead of obscuring ownership;
5. release documentation describes only the behavior actually shipped.

## Authoritative Blender references used by this spec

- Blender Python API, **Python Threads are Not Supported**: persistent/background Python threads can cause hard-to-diagnose Blender crashes; long-lived thread/timer patterns are explicitly discouraged.
- Blender Manual, **How to Create Extensions**: manifest metadata, optional `platforms`, build/validation/install-from-disk workflow.
- Blender Manual, **Extension Tags**: current add-on tag taxonomy includes Import-Export, Mesh, UV, Animation, and other approved labels.
- Blender Manual, **Add-ons / Extensions and Namespace**: extension namespace includes repository identity; preferences use package identity and subpackages should import the top-level package identity relatively rather than hardcoding it.

## Closure status at documentation baseline

| Feedback | Baseline status | Why |
| --- | --- | --- |
| RF-001 same submission | **Open** | current submission doc still says initial submission |
| RF-002 public name | **Open** | manifest still uses old title |
| RF-003 background threading | **Needs proof** | historical path absent; static/runtime proof still required |
| RF-004 trace timer/session | **Needs proof** | historical symbol absent; current equivalent lifecycle must be audited |
| RF-005 tags | **Likely resolved / gate required** | current tags are valid/relevant candidates |
| RF-006 clean ZIP | **Partially resolved / gate required** | exclusions exist; exact built inventory still decides |
| RF-007 platform | **Open** | manifest remains Windows x64 only without documented technical decision |
| RF-008 register/unregister | **Open for design review** | current root lifecycle remains complex |
| RF-009 UI duplication | **Likely resolved / regression gate required** | named duplicate pattern is absent in current UI file |
| RF-010 quality pass | **Open continuous gate** | must be verified against final release candidate |
