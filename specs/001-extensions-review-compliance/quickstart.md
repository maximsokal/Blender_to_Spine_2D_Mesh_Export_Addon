# Quickstart — Implementing the Extensions Review Fixes

This quickstart is for the future implementation branch. The current `001-extensions-review-compliance` branch contains documentation only.

## 1. Read before changing code

Read in this order:

1. `review-feedback.md` — what moderation actually required;
2. `spec.md` — mandatory functional/non-functional contracts;
3. `research.md` — what is currently observed vs inferred;
4. `plan.md` — safe implementation order;
5. `tasks.md` — fine-grained execution ledger;
6. `checklists/extensions-review.md` — final exact-artifact closure gate.

Do not start by editing the historical reviewer line numbers. Several named historical files/symbols are absent from the current Rewrite; first locate the current owner/call path.

## 2. Create the future implementation branch

Create it from the then-approved current `main`, not from an old feature code branch. Preserve this Spec Kit directory in that branch so task/evidence status travels with the implementation.

Before modifying anything, record:

```powershell
$ExpectedHead = (git rev-parse HEAD).Trim()
git status --short
```

The starting worktree must be clean.

## 3. Existing local test executables

The repository testing documentation currently uses:

```powershell
$Python = ".\.venv-tests\Scripts\python.exe"
$BpyPython = ".\.venv-bpy\Scripts\python.exe"
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
```

Use paths appropriate to the local environment if they differ. Blender headless runners must use `--python-exit-code 1` as required by the existing repository test policy.

## 4. Baseline compile and test gates

Before the first production change, establish that the implementation branch baseline is green:

```powershell
& $Python -m compileall -q `
    Blender_to_Spine2D_Mesh_Exporter `
    tests `
    tests_bpy `
    tools
if ($LASTEXITCODE -ne 0) { throw "Compilation failed" }

& $Python -m pytest tests -q --tb=short --durations=20
if ($LASTEXITCODE -ne 0) { throw "Python test suite failed" }

& $BpyPython -m pytest tests_bpy -q --tb=short
if ($LASTEXITCODE -ne 0) { throw "bpy suite failed" }
```

If the repository testing docs or test layout changes before implementation begins, use the then-current documented commands and update this quickstart.

## 5. First implementation slice: tests/audits, not runtime surgery

Add focused contracts for:

- same-submission release wording;
- reviewer-accepted public title synchronization;
- valid/relevant manifest tags;
- no historical irrelevant tags;
- prohibited persistent Python thread constructs in shipped runtime;
- classification of any `bpy.app.timers` use rather than blanket prohibition;
- no production `*_dup` UI drawing workaround;
- exact ZIP member hygiene.

Run focused tests after every addition, then the full Blender-independent suite.

## 6. Thread/persistence slice

Before editing, write down exact current function/class names for:

```text
public Analyze/Export entrypoint
-> diagnostics/trace session/collector
-> persistence function
-> exception/cancellation cleanup
-> unregister-time cleanup (if any)
```

Then implement the smallest lifecycle that satisfies `FR-006` through `FR-014`:

- no persistent Python thread/thread timer;
- no Blender API from non-main Python threads;
- one explicit finalization owner;
- deterministic success/failure/cancellation behavior;
- idempotent/guarded finalization;
- original exception preserved if trace write also fails.

Do not add a replacement scheduler unless a measured requirement needs it.

## 7. Registration slice

Build the resource-ownership inventory in `tasks.md` first. Only then modify root/module registration.

For every changed `register()` / `unregister()` function, verify:

- what Blender resource is acquired;
- what exact reverse cleanup it needs;
- what happens if the next registration step fails;
- whether a global flag is genuinely needed;
- whether the owner can be queried safely instead of maintaining duplicate Python state.

Run fault-injection tests plus real bpy repeated lifecycle tests before moving on.

## 8. UI slice

Do not refactor `ui_layout.py` just because a historical line number was mentioned. Current baseline already lacks the reviewer-named duplicate functions.

Only change UI code when current duplication is demonstrated. Preserve:

- Scene/RNA property IDs;
- defaults;
- enum values;
- operator IDs;
- foldout order;
- visibility/enabled rules;
- settings passed into export planning.

## 9. Manifest/title/submission slice

Default identity decision:

```text
public name: Spine Mesh Exporter
manifest id: blender_to_spine2d_mesh_exporter (unchanged unless separately required)
Python package identity: unchanged
submission: existing declined listing, new higher version
```

After the title change, run the installed-extension preference persistence/restart gate because a display rename must not break AddonPreferences lookup.

Current repository persistence command from `docs/testing.md`:

```powershell
& $Python tools\run_spine_version_preferences_persistence_gate.py `
    --blender $Blender `
    --source .\Blender_to_Spine2D_Mesh_Exporter `
    --output-root $PreferencePersistenceOutput
if ($LASTEXITCODE -ne 0) {
    throw "Spine exact-version preference persistence gate failed"
}
```

## 10. Platform slice

Do not remove `platforms = ["windows-x64"]` merely because pure Python often runs cross-platform.

Audit runtime separately from developer tooling. Record every platform-sensitive runtime owner. Then either:

- keep Windows x64 with a documented technical/support boundary; or
- validate each additional advertised platform with equivalent real Blender install/export/lifecycle gates before changing the manifest.

Remember: omitting `platforms` makes the extension available for all supported operating systems, so omission itself is a support claim.

## 11. Build and exact ZIP inspection

After all source/manifest/docs changes are final and tests are green, build using Blender’s extension tooling from the production extension root according to the then-current repository release instructions.

The package gate must inspect the actual ZIP member list and reject development content. Do not rely only on `paths_exclude_pattern`.

Then run Blender extension validation on that same archive and install **that exact archive** from disk in a clean/isolated profile.

Any change after ZIP creation invalidates the candidate; rebuild and rerun package/install gates.

## 12. Final exact-version persistence and lifecycle evidence

Before upload, prove at minimum:

- one custom exact patch per supported Spine family persists through a full restart;
- UI resolves the configured exact version;
- real export writes the configured exact version into filename and `skeleton.spine`;
- enable/disable/restart/re-enable has no duplicate classes/panels/RNA/handlers;
- no extension-owned persistent Python worker survives Analyze/Export/unregister;
- exact ZIP passes inventory and Blender validation.

## 13. Final Git/artifact boundary

Immediately before recording the candidate:

```powershell
$ActualHead = (git rev-parse HEAD).Trim()
if (git status --porcelain=v1) {
    throw "Working tree is not clean"
}
Write-Host "Candidate Git SHA: $ActualHead"
```

Record the exact ZIP SHA256 with the repository’s current release process. The Git SHA, manifest version, ZIP filename, and SHA256 belong together as one candidate identity.

## 14. Upload

Open the **existing declined Blender Extensions submission** and upload the corrected higher-version ZIP there.

Do not:

- delete the declined listing;
- create a second extension;
- upload a repository ZIP;
- upload an archive built before the final code/manifest/docs commit.

After upload, verify title, version, platform compatibility, permissions, license, website, and tags as rendered by the Blender Extensions platform, then submit the version for re-review.
