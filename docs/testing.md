# Testing and Release Validation

This document defines the current validation policy for Blender to Spine2D Mesh Exporter
**0.153.0**.

A focused test is not a release claim. Release evidence must be generated from one exact
clean commit and the archive built from that same commit.

## Product scope under test

- Blender 5.2.0 or newer.
- Spine schema families 3.8, 4.0, 4.1, 4.2, and 4.3 according to the capability/codec registry.
- Default exact project versions 3.8.99, 4.0.64, 4.1.24, 4.2.43, and 4.3.23.
- User-configurable canonical exact patch versions inside each supported family.
- Global Add-on Preference persistence for those five exact versions.
- Normal / UV Segments.
- Camera Projection.
- Depth Camera Projection.
- Six signed-axis Normal projection directions.
- Shared Selection Pivot for eligible multi-object signed-axis Normal / UV exports.
- Active Camera — Object Root Bone.
- Active Camera — Camera Root Bone.
- Static and per-object texture sequences.
- Scene-level Texture size owned by the Bake foldout.
- Depth parallax reserve views.
- Atomic output and Blender-state restoration.
- Scene settings schema 8.

## Executables

Example PowerShell variables:

```powershell
$Python = ".\.venv-tests\Scripts\python.exe"
$BpyPython = ".\.venv-bpy\Scripts\python.exe"
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
```

Every Blender headless runner must use `--python-exit-code 1`.

## Clean commit boundary

```powershell
$ExpectedHead = "<exact SHA under test>"
$ActualHead = (git rev-parse HEAD).Trim()
if ($ActualHead -ne $ExpectedHead) {
    throw "Unexpected HEAD: $ActualHead"
}
if (git status --porcelain=v1) {
    throw "Working tree is not clean"
}
```

Repeat the clean-tree check after tests and packaging.

## Compile check

```powershell
& $Python -m compileall -q `
    Blender_to_Spine2D_Mesh_Exporter `
    tests `
    tests_bpy `
    tools

if ($LASTEXITCODE -ne 0) { throw "Compilation failed" }
```

## Blender-independent suite

```powershell
& $Python -m pytest tests -q --tb=short --durations=20
if ($LASTEXITCODE -ne 0) { throw "Python test suite failed" }
```

Important current contracts include:

- projection enum and UI routing;
- signed-axis projection bases;
- Shared Selection Pivot capability/visibility/default behavior;
- aggregate world-space selection bounds and export-only pivot resolution;
- U/V/depth rebase preserving world-space vertices;
- legacy per-object settings identity when Shared Selection Pivot is disabled;
- Active Camera Object Root/Camera Root normalization;
- Object Root inverse-setup bone generation and vertex parenting;
- Camera Root rigid depth-layer ownership;
- material-bake geometry independence from Normal projection direction;
- Texture size Scene ownership inside Bake;
- loop-level UV identity and weighted attachment construction;
- target-specific Spine schema adaptation;
- arbitrary canonical same-family exact project patches;
- exact-version propagation into immutable ExportSettings and versioned JSON names;
- one persistent Add-on Preference field per supported Spine family;
- no production call to `wm.save_userpref` from preference update callbacks;
- sequence ownership;
- parallax reserve topology/camera planning;
- Blender-state/resource lifecycle contracts;
- manifest/documentation version synchronization.

Focused Spine exact-version coverage includes:

```text
tests/test_spine_project_exact_versions.py
tests/test_spine_version_preferences_contract.py
tests/test_spine_version_preferences_persistence_gate_standalone.py
tests/test_grenade_all_spine_targets_runner_contract.py
```

Focused Shared Pivot coverage includes:

```text
tests/test_a1_shared_pivot_contract.py
tests/test_a1_shared_pivot_rebase.py
tests/test_a1_shared_pivot_resolution.py
tests/test_a1_shared_pivot_ui.py
tests/test_a1_multi_object_preparation_settings.py
tests/test_a1_ui_export_plan.py
```

Focused UI/release coverage includes:

```text
tests/test_texture_size_bake_ui.py
tests/test_documentation_contract.py
tests/test_manifest_version.py
tests/test_extension_version_0151.py
```

## Installed extension exact-version persistence gate

Exact project versions are `AddonPreferences`, so a pure Python mock cannot prove their
real Blender persistence semantics. The release gate installs the built source as a Blender
Extension into an isolated Blender user configuration and launches two separate Blender
processes.

Process A:

1. installs/enables the extension;
2. assigns a deliberately non-default exact patch inside every supported family;
3. calls Blender's preference-save operator only inside this isolated deterministic test;
4. exits completely.

Process B starts from scratch using the same isolated Blender user configuration. It must:

1. read back all five exact values from the installed extension's real AddonPreferences;
2. create a real Mesh and material;
3. select each Spine schema family in turn;
4. build the public active-object export plan;
5. prove `ExportSettings.spine_version` equals the persisted custom exact patch;
6. run the real production export;
7. prove the versioned JSON filename and serialized `skeleton.spine` use that same patch;
8. require a real PNG output for every family.

Run it with:

```powershell
& $Python tools\run_spine_version_preferences_persistence_gate.py `
    --blender $Blender `
    --source .\Blender_to_Spine2D_Mesh_Exporter `
    --output-root $PreferencePersistenceOutput

if ($LASTEXITCODE -ne 0) {
    throw "Spine exact-version preference persistence gate failed"
}
```

The gate owns an isolated `BLENDER_USER_CONFIG`; it must never read or write the developer's
ordinary Blender Preferences. It removes the temporary installed extension/repository after
a successful run.

The current deterministic custom exact versions are one patch below each registry default:

```text
3.8.98
4.0.63
4.1.23
4.2.42
4.3.22
```

These values intentionally differ from defaults so a hidden hardcoded exact version cannot
pass the gate.

## Real Blender Shared Pivot gate

The real multi-object gate must use the artist project that exposed the assembly-pivot bug,
not a synthetic replacement:

```powershell
& $Blender `
    --factory-startup `
    --background `
    $GrenadeBlend `
    --python-exit-code 1 `
    --python tests\blender_headless\run_grenade_shared_pivot_real_export.py `
    -- `
    --expected-blend $GrenadeBlend `
    --output-directory $SharedPivotOutput

if ($LASTEXITCODE -ne 0) { throw "Shared Pivot grenade gate failed" }
```

The gate must prove that the persisted selected Mesh set enters the public selected-object
route, the aggregate pivot matches independent world-space geometry bounds, all generated
object main bones use the same projected pivot, X/Y controls remain present, production
JSON/PNG outputs are non-empty, and source object/scene/datablock state is unchanged.

## Real grenade all-target Spine matrix

The same artist-authored grenade project must also be exported through every codec that the
production Spine JSON registry declares ready. The runner obtains its target set directly
from `registered_spine_json_codecs()`; do not duplicate a target list inside the runner.

```powershell
& $Blender `
    --factory-startup `
    --background `
    $GrenadeBlend `
    --python-exit-code 1 `
    --python tests\blender_headless\run_grenade_all_spine_targets_real_export.py `
    -- `
    --expected-blend $GrenadeBlend `
    --output-directory $GrenadeAllTargetsOutput

if ($LASTEXITCODE -ne 0) { throw "Grenade all-target Spine matrix failed" }
```

This source-registration runner deliberately uses registry default exact versions because it
does not install the extension into Blender Preferences. Its job is the heavyweight real
asset regression: every codec family must export the same 14-object artist asset with Shared
Pivot and leave source object/scene/context/datablock state unchanged.

The separate installed-extension persistence gate is the authority for custom exact patch
propagation. Keeping these responsibilities separate avoids fake AddonPreferences objects in
the grenade test.

Because the grenade matrix is registry-driven, adding a future production-ready codec
automatically adds another real grenade export. A target must not be removed merely to make
a failing release green.

## Real Blender Active Camera gates

Maintain real Blender coverage for projection parity, both camera root modes, and Object
Root inverse setup. Representative runners include:

```text
tests/blender_headless/run_coin_star_normal_projection_parity_integration.py
tests/blender_headless/run_coin_star_normal_camera_root_modes_integration.py
tests/blender_headless/run_coin_star_normal_object_root_setup_compensation_integration.py
```

The gates prove shared projected geometry/material input, route-specific rig ownership,
valid inverse setup parenting, camera-projected setup XY, and source-state restoration.

## Depth Camera Projection gates

Maintain real Blender coverage for Perspective/Orthographic output, target matrix,
sequences, positive parallax, multi-object parallax/rollback, FRONT/reserve slot order, and
shared rig ownership.

Current runners include:

```text
run_depth_camera_projection_integration.py
run_depth_camera_projection_multi_object_integration.py
run_depth_parallax_integration.py
run_depth_parallax_matrix_integration.py
run_depth_parallax_multi_object_integration.py
```

## Existing multi-object/sequence gates

Maintain the established runners for standalone and supported connected/mixed composition:

```text
run_multi_object_sequence_mode_matrix_integration.py
run_multi_object_mixed_static_sequence_matrix_integration.py
run_connected_mixed_sequence_mode_matrix_integration.py
run_connected_mixed_static_sequence_matrix_integration.py
```

The capability registry, not documentation prose, is the authority for supported
scope/profile/schema-family combinations.

## Real bpy suite

```powershell
& $BpyPython scripts\run_bpy_tests.py
if ($LASTEXITCODE -ne 0) { throw "Real bpy suite failed" }
```

A missing real-bpy environment must not be reported as successful release validation.

## Build 0.153.0

```powershell
$SourceDir = ".\Blender_to_Spine2D_Mesh_Exporter"
$Archive = ".\dist\blender_to_spine2d_mesh_exporter-0.153.0.zip"

New-Item -ItemType Directory -Force ".\dist" | Out-Null
Remove-Item -LiteralPath $Archive -Force -ErrorAction SilentlyContinue

& $Blender `
    --command extension build `
    --source-dir $SourceDir `
    --output-filepath $Archive

if ($LASTEXITCODE -ne 0) { throw "Extension build failed" }

& $Blender --command extension validate $Archive
if ($LASTEXITCODE -ne 0) { throw "Extension validation failed" }

Get-FileHash -LiteralPath $Archive -Algorithm SHA256
```

The archive root must contain `blender_manifest.toml` and `__init__.py` and must not include
repository-only tests, docs, legacy runtime sources, bytecode, or nested archives excluded
by the manifest build rules.

## Manual Blender UI validation

Before packaging, verify in a saved `.blend`:

1. Add-on Preferences contain exactly one exact project-version field for each family 3.8 through 4.3.
2. Configure a non-default exact patch for the selected family and confirm the viewport **Exact JSON version** changes immediately.
3. Run Analyze, change the exact patch again, and confirm readiness becomes stale/invalidated.
4. Close and reopen Blender after saving Preferences and confirm all five values persist.
5. Expand **Paths and Spine 2D version** and confirm `Texture size` is absent.
6. Expand **Bake** and confirm `Texture size` is the first setting.
7. Select at least two Mesh objects in signed-axis Normal / UV and confirm **Shared Selection Pivot** is visible and enabled by default.
8. Reduce the selection to one Mesh or select an unsupported camera route and confirm Shared Selection Pivot is hidden.
9. Reset Scene settings and confirm Texture size returns to `1024`; global exact-version Preferences must not be reset.

## Manual Spine validation

For representative outputs in the configured exact Spine project version, verify JSON import,
texture paths, UV orientation, signed-axis controls, Shared Selection Pivot behavior, both
Active Camera root modes, Depth FRONT/reserve order, and static/sequence ownership.

## Release evidence

Record:

- exact commit SHA;
- clean worktree before and after the gate;
- Python/bpy/Blender versions;
- complete pytest result;
- installed-extension preference save/restart/custom-export gate report;
- required Blender-headless gate markers;
- real-bpy result;
- archive path, size, and SHA256;
- Blender extension validation result;
- manual Blender UI and Spine validation notes.

Never claim a test passed on a commit that was not the exact commit used to generate the
reported output.
