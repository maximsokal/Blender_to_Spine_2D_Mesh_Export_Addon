# Tasks — Blender Extensions Review Compliance

Legend:

- `[x]` completed in this documentation branch;
- `[ ]` must be completed in a future production/release branch;
- every implementation task must record evidence before being checked.

## T000 — Documentation baseline

- [x] **T001** Resolve current `main` HEAD and record `f0a0f879d639dad860c0c8c56ddba0845aa69f17`.
- [x] **T002** Create dedicated documentation branch `001-extensions-review-compliance` from that exact `main` commit.
- [x] **T003** Confirm there was no existing root `specs/` / Spec Kit feature folder on baseline main.
- [x] **T004** Capture all recovered moderator comments in `review-feedback.md`.
- [x] **T005** Separate reviewer wording from internal inference/quality workstreams.
- [x] **T006** Audit current manifest name/version/tags/platform/build exclusions.
- [x] **T007** Audit current root registration architecture.
- [x] **T008** Audit current `ui_layout.py` against historical duplicate-function feedback.
- [x] **T009** Audit current submission/testing docs relevant to release workflow.
- [x] **T010** Check current Blender official guidance for Python threading, extension manifests/platforms/tags, and extension package namespace.
- [x] **T011** Keep production Python/manifest/tests unchanged on this documentation branch.

## T100 — Convert reviewer feedback into executable contracts

- [ ] **T101** Add a focused compliance test module that parses `blender_manifest.toml`.
- [ ] **T102** Test that historical irrelevant tags `backup` and `text-editing` cannot return.
- [ ] **T103** Test that every configured tag is from the current accepted project taxonomy or Blender validator accepts it.
- [ ] **T104** Add documentation contract asserting submission workflow refers to the existing declined submission/new version.
- [ ] **T105** Add documentation contract rejecting “initial/new duplicate submission” wording where it changes the workflow meaning.
- [ ] **T106** Add title synchronization contract for manifest and required public docs.
- [ ] **T107** Add technical-identity contract that public title changes do not silently alter manifest `id`/package identity.
- [ ] **T108** Add static runtime-package scan for `threading.Thread`.
- [ ] **T109** Add static runtime-package scan for `threading.Timer` / `from threading import Timer`.
- [ ] **T110** Audit `multiprocessing.Queue` or other standard-library constructs documented by Blender as possible hidden thread users.
- [ ] **T111** Scan for direct `bpy.app.timers` ownership and classify every match rather than banning Blender timers globally.
- [ ] **T112** Add static production scan for reviewer-style UI `*_dup` functions.
- [ ] **T113** Make static scans ignore tests/fixtures only when the test itself is deliberately demonstrating prohibited source text.
- [ ] **T114** Ensure scans run against the actual extension runtime root.
- [ ] **T115** Add post-build ZIP scan so source exclusions cannot hide a packaging regression.

## T200 — Current trace/persistence call graph

- [ ] **T201** Locate the public single-object Analyze entrypoint.
- [ ] **T202** Locate the public single-object Export entrypoint.
- [ ] **T203** Locate the public selected-object Analyze/Export entrypoints.
- [ ] **T204** Trace current diagnostic event creation from entrypoint to storage.
- [ ] **T205** Locate every current trace/log file writer reachable from production export.
- [ ] **T206** Locate every deferred write/scheduler callback reachable from production export.
- [ ] **T207** Locate every exception handler that can finalize diagnostics.
- [ ] **T208** Locate every cancellation path that can finalize diagnostics.
- [ ] **T209** Locate unregister-time trace/diagnostic cleanup if it exists.
- [ ] **T210** Record exact current owner/function names in implementation notes before editing.
- [ ] **T211** Confirm historical `PipelineTraceSession` is absent or identify its current equivalent.
- [ ] **T212** Confirm historical `spine_core/chat_persistence.py` is absent or identify replacement owner.

## T300 — Remove unsupported background threading

- [ ] **T301** Inventory all production `threading` imports.
- [ ] **T302** Inventory all production queue/concurrency imports that can create hidden threads.
- [ ] **T303** Classify each concurrency use as persistent, synchronously joined, Blender-main-thread scheduler, subprocess, or test-only.
- [ ] **T304** Remove each persistent Python background-thread owner from shipped runtime.
- [ ] **T305** Remove each persistent `threading.Timer` owner from shipped runtime.
- [ ] **T306** Ensure worker-independent pure-Python concurrency, if retained, cannot reach `bpy`/RNA/Blender data.
- [ ] **T307** Ensure no worker survives return to Blender’s normal event loop.
- [ ] **T308** Add unit/static tests for the chosen no-background-thread contract.
- [ ] **T309** Add real Blender smoke proving no lingering Python worker after Analyze.
- [ ] **T310** Add real Blender smoke proving no lingering Python worker after Export.
- [ ] **T311** Add real Blender smoke proving no lingering Python worker after failed Export.
- [ ] **T312** Add disable/unregister smoke proving no lingering worker/timer owner.

## T400 — Explicit trace finalization

- [ ] **T401** Define trace state transitions for request creation/running/finalization/cleanup.
- [ ] **T402** Define success trace contents.
- [ ] **T403** Define validation/preflight failure trace contents.
- [ ] **T404** Define export failure trace contents.
- [ ] **T405** Define user cancellation trace contents where cancellation is supported.
- [ ] **T406** Define cleanup-failure trace behavior.
- [ ] **T407** Define trace-storage failure severity and interaction with primary export exceptions.
- [ ] **T408** Choose one explicit finalization owner per request.
- [ ] **T409** Make finalization idempotent or guard it against duplicate invocation.
- [ ] **T410** Remove session-owned hidden thread/timer finalization.
- [ ] **T411** Ensure finalization runs from deterministic `try/except/finally` boundaries as appropriate.
- [ ] **T412** Preserve original exception chain when persistence also fails.
- [ ] **T413** Add unit test: success finalizes exactly once.
- [ ] **T414** Add unit test: failure finalizes according to policy exactly once.
- [ ] **T415** Add unit test: cancellation finalizes according to policy exactly once.
- [ ] **T416** Add unit test: duplicate cleanup call does not duplicate final trace.
- [ ] **T417** Add unit test: trace-write failure does not silently replace primary export error.
- [ ] **T418** Add integration test covering real Blender request cleanup.

## T500 — Registration/unregistration resource inventory

- [ ] **T501** Inventory `addon_preferences.register/unregister` resources.
- [ ] **T502** Inventory Scene RNA registration resources.
- [ ] **T503** Inventory scene settings migration resources.
- [ ] **T504** Inventory `ui.register/unregister` resources.
- [ ] **T505** Inventory `rig_ui.register/unregister` resources.
- [ ] **T506** Inventory readiness invalidation resources.
- [ ] **T507** Inventory automatic readiness resources.
- [ ] **T508** Inventory generated material UI resources.
- [ ] **T509** Inventory `ui_layout.register/unregister` base-panel replacement/RNA resources.
- [ ] **T510** Inventory `repolish_ui.register/unregister` resources.
- [ ] **T511** Inventory single-object operator resources.
- [ ] **T512** Search all production modules for handlers appended to `bpy.app.handlers`.
- [ ] **T513** Search all production modules for `bpy.app.timers` registration.
- [ ] **T514** Search all production modules for `bpy.msgbus` subscriptions.
- [ ] **T515** Search all production modules for preview collections/custom icons.
- [ ] **T516** Search all production modules for module globals retaining Blender-owned objects/datablocks/context.
- [ ] **T517** Record which globals truly require reset and which are harmless Python state.

## T600 — Registration simplification

- [ ] **T601** Review whether `ExtensionRegistrationState` prevents a demonstrated lifecycle failure.
- [ ] **T602** Review whether `DEGRADED` state is necessary after owner-local cleanup is improved.
- [ ] **T603** Review whether `REGISTRATION_STEPS` adds clarity or duplicates module ownership.
- [ ] **T604** Review whether generic `RegistrationCleanupAction` is still necessary.
- [ ] **T605** Keep transactional RNA cleanup where partial RNA registration can occur.
- [ ] **T606** Keep ordered base-panel restoration in `ui_layout` if still required by panel replacement.
- [ ] **T607** Remove only redundant root-level state/cleanup after tests exist.
- [ ] **T608** Keep actionable registration logging.
- [ ] **T609** Keep original exceptions visible after rollback.
- [ ] **T610** Add fault-injection test for failure at an early registration step.
- [ ] **T611** Add fault-injection test for failure at a middle registration step.
- [ ] **T612** Add fault-injection test for failure after UI ownership is acquired.
- [ ] **T613** Test register → unregister in real bpy.
- [ ] **T614** Test multiple register/unregister cycles in real bpy.
- [ ] **T615** Test installed extension enable → disable → enable.
- [ ] **T616** Test installed extension restart after disable/enable.
- [ ] **T617** Assert no duplicate classes/panels/RNA/handlers/timers/subscriptions after cycles.

## T700 — UI duplication and behavior preservation

- [ ] **T701** Search production UI modules for functions ending `_dup`.
- [ ] **T702** Search for duplicated blocks drawing the same Scene property with separate policy logic.
- [ ] **T703** Build canonical ownership table for Paths/Spine version controls.
- [ ] **T704** Build canonical ownership table for Rig controls.
- [ ] **T705** Build canonical ownership table for Generated Materials controls.
- [ ] **T706** Build canonical ownership table for Cut controls.
- [ ] **T707** Build canonical ownership table for Bake controls.
- [ ] **T708** Build canonical ownership table for Analysis controls.
- [ ] **T709** Build canonical ownership table for Export action.
- [ ] **T710** Extract only genuinely shared helpers if duplication remains.
- [ ] **T711** Preserve property IDs and defaults.
- [ ] **T712** Preserve operator IDs.
- [ ] **T713** Preserve visibility/enabled conditions.
- [ ] **T714** Preserve foldout order.
- [ ] **T715** Preserve UI-to-ExportSettings mapping.
- [ ] **T716** Add/retain source contract prohibiting `*_dup` workaround functions.
- [ ] **T717** Add focused UI behavior regression tests.
- [ ] **T718** Run real bpy panel registration/draw smoke where supported.

## T800 — Public title and technical identity

- [ ] **T801** Change manifest public `name` to reviewer-accepted `Spine Mesh Exporter` unless later moderation says otherwise.
- [ ] **T802** Keep manifest `id = "blender_to_spine2d_mesh_exporter"` by default.
- [ ] **T803** Keep Python package directory/module identity stable by default.
- [ ] **T804** Search all user-facing docs for old public title.
- [ ] **T805** Update root README title/wording as needed.
- [ ] **T806** Update docs index/title references as needed.
- [ ] **T807** Update installation docs as needed.
- [ ] **T808** Update submission docs as needed.
- [ ] **T809** Update extension metadata tests/snapshots.
- [ ] **T810** Verify AddonPreferences `bl_idname` remains package-identity based.
- [ ] **T811** Verify subpackage preference lookup follows extension namespace-safe package identity.
- [ ] **T812** Run installed-extension preference save/restart test after title change.
- [ ] **T813** Verify exact Spine custom versions survive restart/upgrade test path.

## T900 — Same-submission release workflow

- [ ] **T901** Replace “initial submission” wording in `docs/submission.md`.
- [ ] **T902** State that the existing declined submission must be retained.
- [ ] **T903** State that correction is uploaded as a new higher version.
- [ ] **T904** Add explicit prohibition on creating another extension listing.
- [ ] **T905** Add explicit prohibition on deleting/recreating the declined listing.
- [ ] **T906** Add same-submission item to final release checklist.
- [ ] **T907** Prepare moderator response mapping every RF item to evidence.
- [ ] **T908** Record actual same-submission upload as final manual evidence.

## T1000 — Manifest tags

- [ ] **T1001** Parse current manifest tags in focused test.
- [ ] **T1002** Validate current tag spelling through Blender extension validator.
- [ ] **T1003** Keep `Import-Export` if exporter behavior remains public.
- [ ] **T1004** Keep `Mesh` if Mesh processing remains public.
- [ ] **T1005** Keep `UV` if UV generation/handling remains public.
- [ ] **T1006** Keep `Animation` only while generated animation controls/sequences remain public behavior.
- [ ] **T1007** Reject `backup`.
- [ ] **T1008** Reject `text-editing`.
- [ ] **T1009** Do not add unrelated discoverability tags.

## T1100 — Platform audit

- [ ] **T1101** Search runtime package for `sys.platform`.
- [ ] **T1102** Search runtime package for `os.name`.
- [ ] **T1103** Search runtime package for `platform.system`/architecture checks.
- [ ] **T1104** Search runtime package for `msvcrt`.
- [ ] **T1105** Search runtime package for `fcntl`.
- [ ] **T1106** Search runtime package for `winreg`.
- [ ] **T1107** Search runtime package for platform-specific `ctypes` use.
- [ ] **T1108** Audit file-lock implementation(s).
- [ ] **T1109** Audit atomic rename/replace semantics.
- [ ] **T1110** Audit file/directory fsync behavior.
- [ ] **T1111** Audit path normalization, drive-letter, UNC, and case assumptions.
- [ ] **T1112** Audit subprocess and shell invocations in runtime package.
- [ ] **T1113** Distinguish development/release subprocess tooling from installed runtime dependencies.
- [ ] **T1114** Audit wheels/native binaries declared or bundled.
- [ ] **T1115** Audit temp-directory assumptions.
- [ ] **T1116** Audit platform-specific tests and hardcoded paths.
- [ ] **T1117** Decide documented platform policy.
- [ ] **T1118** If Windows-only, document concrete runtime/support reason.
- [ ] **T1119** If broadening, run equivalent Linux validation before advertising Linux.
- [ ] **T1120** If broadening, run equivalent macOS validation before advertising macOS.
- [ ] **T1121** Update manifest `platforms` only after decision/evidence.
- [ ] **T1122** Update README/submission compatibility wording to exactly match manifest support.

## T1200 — ZIP hygiene

- [ ] **T1201** Review current `[build].paths_exclude_pattern` against current package tree.
- [ ] **T1202** Confirm `.git` is excluded.
- [ ] **T1203** Confirm `.github` is excluded.
- [ ] **T1204** Confirm Python cache/bytecode is excluded.
- [ ] **T1205** Confirm tests are excluded.
- [ ] **T1206** Confirm development docs are excluded unless intentionally shipped.
- [ ] **T1207** Confirm prior ZIPs/dist artifacts are excluded.
- [ ] **T1208** Confirm retained pre-Rewrite source is excluded from extension runtime archive.
- [ ] **T1209** Confirm CI helper scripts are excluded from archive.
- [ ] **T1210** Confirm repository `tools/` are not unintentionally inside build root/archive.
- [ ] **T1211** Build exact candidate with Blender extension build command.
- [ ] **T1212** Enumerate all ZIP members after build.
- [ ] **T1213** Fail on every forbidden member and report all offenders.
- [ ] **T1214** Assert manifest exists at expected archive root.
- [ ] **T1215** Assert package entrypoint exists.
- [ ] **T1216** Review every unusual included non-Python asset for runtime need.
- [ ] **T1217** Run Blender `extension validate` on exact ZIP.
- [ ] **T1218** Install exact ZIP from disk in isolated/clean profile.
- [ ] **T1219** Record exact ZIP byte size.
- [ ] **T1220** Record exact ZIP SHA256.

## T1300 — Full regression/release evidence

- [ ] **T1301** Verify clean expected Git HEAD before tests.
- [ ] **T1302** Verify clean worktree before tests.
- [ ] **T1303** Run compileall on production/tests/tests_bpy/tools per current testing policy.
- [ ] **T1304** Run focused compliance tests.
- [ ] **T1305** Run full `tests` suite.
- [ ] **T1306** Run full `tests_bpy` suite.
- [ ] **T1307** Run representative signed-axis Normal/UV export.
- [ ] **T1308** Run representative Active Camera Object Root export.
- [ ] **T1309** Run representative Active Camera Camera Root export.
- [ ] **T1310** Run representative Camera Projection export.
- [ ] **T1311** Run representative Depth Camera Projection export.
- [ ] **T1312** Run representative multi-object export.
- [ ] **T1313** Run Analyze path and blocker/warning reporting smoke.
- [ ] **T1314** Run exact Spine project-version persistence gate in isolated Blender config.
- [ ] **T1315** Run clean install/enable/disable/restart/re-enable/uninstall lifecycle.
- [ ] **T1316** Verify no duplicate classes/handlers/panels/properties after lifecycle test.
- [ ] **T1317** Build candidate ZIP after all source/manifest/doc changes.
- [ ] **T1318** Run ZIP inventory gate.
- [ ] **T1319** Run Blender extension validate.
- [ ] **T1320** Run install-from-disk reviewer walkthrough from exact ZIP.
- [ ] **T1321** Re-check clean HEAD/worktree after gates.
- [ ] **T1322** Rebuild/retest if anything changed after archive build.

## T1400 — Moderator-facing closure

- [ ] **T1401** Prepare changelog entry for public title change.
- [ ] **T1402** Prepare changelog entry for threading/persistence safety changes.
- [ ] **T1403** Prepare changelog entry for registration simplification.
- [ ] **T1404** Prepare changelog entry for package cleanup.
- [ ] **T1405** Prepare changelog entry for platform declaration decision.
- [ ] **T1406** Prepare concise note that current valid tags replace historical irrelevant tags.
- [ ] **T1407** Prepare note that UI duplicate drawing paths were consolidated/verified canonical.
- [ ] **T1408** Link each moderator issue to tests/evidence in internal release notes.
- [ ] **T1409** Open the existing declined Blender Extensions submission.
- [ ] **T1410** Upload the higher corrected version to that same submission.
- [ ] **T1411** Verify listing title/compatibility/platform/permissions/license/tags after upload.
- [ ] **T1412** Submit for re-review.
- [ ] **T1413** Preserve candidate Git SHA + ZIP SHA256 in release record.

## T1500 — Final Spec Kit synchronization

- [ ] **T1501** Update `review-feedback.md` baseline status column with final closure evidence.
- [ ] **T1502** Update `research.md` hypotheses with actual implementation findings.
- [ ] **T1503** Update `plan.md` if implementation order materially changes.
- [ ] **T1504** Mark completed `tasks.md` entries only after evidence exists.
- [ ] **T1505** Complete `checklists/extensions-review.md` against exact release candidate.
- [ ] **T1506** Update `quickstart.md` only if repository gate commands change.
- [ ] **T1507** Confirm no unresolved RF/FR item is hidden behind a broad “done” status.
- [ ] **T1508** Confirm all final docs describe the shipped version, not a historical internal branch.
