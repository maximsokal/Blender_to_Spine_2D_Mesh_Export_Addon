# Checklist — Blender Extensions Review Closure

Complete this checklist against the **exact release candidate commit and ZIP**. A source-level “looks fixed” is not enough where the reviewer concern is about runtime or package contents.

## Submission continuity

- [ ] We are updating the existing declined Blender Extensions submission.
- [ ] We did not delete the declined submission.
- [ ] We did not create a duplicate extension listing.
- [ ] Candidate version is higher than the previously submitted version as required by the platform workflow.
- [ ] Submission documentation no longer describes this correction as an unrelated initial submission.

## Public title / identity

- [ ] Public extension title matches the reviewer-accepted naming decision (`Spine Mesh Exporter` unless superseded by later moderation).
- [ ] Manifest public name and user-facing documentation agree.
- [ ] Manifest `id` was not changed merely because the display title changed.
- [ ] Python package identity was not changed merely because the display title changed.
- [ ] AddonPreferences identity resolves correctly under the installed extension namespace.
- [ ] Saved exact Spine project versions persist after full Blender restart.

## Python threading and Blender main-thread safety

- [ ] No shipped module owns a persistent `threading.Thread`.
- [ ] No shipped module owns a persistent `threading.Timer`.
- [ ] No implicit background-thread construct is used without an explicit Blender-safe proof.
- [ ] No background callback accesses `bpy`, RNA, Blender UI, `bmesh`, or Blender-owned data.
- [ ] Analyze returns with no extension-owned Python background worker alive.
- [ ] Successful Export returns with no extension-owned Python background worker alive.
- [ ] Failed Export returns with no extension-owned Python background worker alive.
- [ ] Disable/unregister leaves no worker/thread owner alive.

## Pipeline trace / diagnostics persistence

- [ ] Current trace owner/call graph is documented by exact current function/class names.
- [ ] Historical reviewer line numbers were not patched blindly.
- [ ] Success finalization behavior is explicit.
- [ ] Validation/preflight failure finalization behavior is explicit.
- [ ] Export failure finalization behavior is explicit.
- [ ] Cancellation finalization behavior is explicit where applicable.
- [ ] Trace-write failure interaction with the primary error is explicit.
- [ ] Finalization is idempotent or otherwise protected from duplicate writes.
- [ ] No session-owned Python thread/timer is required for final persistence.
- [ ] Cleanup is deterministic on exception paths.

## Manifest tags

- [ ] Every tag is supported by the Blender Extension tag taxonomy/validator used for the release.
- [ ] Every tag describes real shipped behavior.
- [ ] `backup` is absent.
- [ ] `text-editing` is absent.
- [ ] No unrelated tag was added solely for visibility.

## Platform declaration

- [ ] Runtime code was audited for Windows/POSIX-specific imports.
- [ ] Runtime code was audited for file-locking semantics.
- [ ] Runtime code was audited for atomic replace/fsync semantics.
- [ ] Runtime code was audited for path/drive/UNC/case assumptions.
- [ ] Runtime code was audited for subprocess/shell assumptions.
- [ ] Runtime code was audited for native wheels/binaries.
- [ ] Development-only Windows tooling was separated from installed runtime requirements.
- [ ] Manifest `platforms` reflects the deliberate support decision.
- [ ] README/submission docs reflect exactly the same platform support decision.
- [ ] Every advertised platform passed equivalent clean install/export/lifecycle gates.
- [ ] If Windows-only remains, the concrete reason/validation boundary is documented.

## Registration / unregistration

- [ ] Every root registration step has a documented resource owner.
- [ ] Registered Blender classes are unregistered by their owner.
- [ ] RNA properties are removed by their owner.
- [ ] Handlers are removed by their owner.
- [ ] Blender timers are unregistered or naturally completed according to explicit ownership.
- [ ] Message-bus subscriptions are cleared by owner if any exist.
- [ ] Preview/icon resources are released if any exist.
- [ ] Ordinary harmless Python globals are not reset only for cosmetic cleanup.
- [ ] Partial registration failure rolls back only acquired resources.
- [ ] Root lifecycle code is no more complex than required by those resources/failure modes.
- [ ] Register → unregister passes in real bpy.
- [ ] Multiple lifecycle cycles pass.
- [ ] Clean installed extension enable → disable → restart → enable passes.
- [ ] No duplicate classes, panels, RNA properties, handlers, timers, or subscriptions remain.

## UI deduplication

- [ ] No production `*_dup` UI drawing functions exist.
- [ ] Every shared setting has one canonical drawing/policy owner.
- [ ] Any shared helper reduces actual duplication.
- [ ] Property identifiers are unchanged unless explicitly migrated.
- [ ] Defaults are unchanged unless explicitly specified.
- [ ] Operator IDs are unchanged unless explicitly specified.
- [ ] Visibility/enabled logic is behaviorally covered.
- [ ] Foldout/section order is behaviorally covered.
- [ ] UI-to-export settings mapping is unchanged.

## ZIP hygiene

- [ ] ZIP was built by Blender extension build tooling from the exact clean candidate commit.
- [ ] ZIP contains `blender_manifest.toml`.
- [ ] ZIP contains runtime entrypoint/source required by the extension.
- [ ] ZIP contains no `.git/`.
- [ ] ZIP contains no `.github/`.
- [ ] ZIP contains no CI scripts/workflow files.
- [ ] ZIP contains no `tests/`.
- [ ] ZIP contains no `tests_bpy/`.
- [ ] ZIP contains no test caches.
- [ ] ZIP contains no local virtual environments.
- [ ] ZIP contains no Python bytecode/cache directories.
- [ ] ZIP contains no previous ZIP/dist artifacts.
- [ ] ZIP contains no retained non-runtime legacy implementation files.
- [ ] Every unusual included asset/resource has a runtime reason.
- [ ] ZIP member inventory gate passes.
- [ ] Blender `extension validate` passes on this exact ZIP.

## Regression gates

- [ ] Exact expected Git HEAD is checked before gates.
- [ ] Worktree is clean before gates.
- [ ] Compileall gate passes.
- [ ] Focused compliance suite passes.
- [ ] Full Blender-independent `tests` suite passes.
- [ ] Full real-bpy `tests_bpy` suite passes.
- [ ] Representative Normal / UV export passes.
- [ ] Representative Active Camera Object Root export passes.
- [ ] Representative Active Camera Camera Root export passes.
- [ ] Representative Camera Projection export passes.
- [ ] Representative Depth Camera Projection export passes.
- [ ] Representative multi-object export passes.
- [ ] Analyze path passes.
- [ ] Exact Spine version preference persistence/restart gate passes.
- [ ] Source scene/resource restoration contracts pass.
- [ ] Worktree/HEAD are rechecked after gates.

## Exact artifact installation

- [ ] The exact validated ZIP is installed from disk in a clean/isolated Blender profile.
- [ ] Extension enables without console errors.
- [ ] Preferences are visible and writable.
- [ ] Analyze works.
- [ ] Export works.
- [ ] Disable works.
- [ ] Restart works.
- [ ] Re-enable works.
- [ ] Uninstall works.
- [ ] No development repository path is required for installed runtime.

## Final artifact identity

- [ ] Manifest version, docs version, ZIP filename, and changelog agree.
- [ ] Candidate Git SHA is recorded.
- [ ] Exact ZIP size is recorded.
- [ ] Exact ZIP SHA256 is recorded.
- [ ] No source/doc/manifest edit occurred after the final ZIP build; if one did, the ZIP was rebuilt and package gates rerun.

## Moderator response

- [ ] RF-001 same-submission instruction has evidence.
- [ ] RF-002 public title has evidence.
- [ ] RF-003 no persistent background threading has evidence.
- [ ] RF-004 explicit trace lifecycle has evidence.
- [ ] RF-005 tag cleanup has evidence.
- [ ] RF-006 clean ZIP has evidence.
- [ ] RF-007 platform decision has evidence.
- [ ] RF-008 register/unregister simplification/justification has evidence.
- [ ] RF-009 UI duplication closure has evidence.
- [ ] RF-010 self-contained quality pass has evidence.
- [ ] Corrected higher-version ZIP is uploaded to the same existing submission.
- [ ] Listing metadata is checked after upload.
- [ ] Candidate is sent for re-review.
