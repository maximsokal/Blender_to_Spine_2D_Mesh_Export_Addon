# Testing and Release Validation

## Validation layers

The repository uses four validation layers:

1. Blender-independent tests in `tests/`.
2. Real `bpy` tests in `tests_bpy/`.
3. Blender headless integration scripts in `tests/blender_headless/`.
4. Blender extension validation and ZIP packaging.

A focused test run is not a release gate. Every required layer must pass on the same candidate commit, and connected-rig changes require a fresh manual import into Spine 4.2.43.

## Compile check

```powershell
& .\.venv-tests\Scripts\python.exe `
    -m compileall `
    -q `
    Blender_to_Spine2D_Mesh_Exporter `
    tests `
    tests_bpy `
    tools

if ($LASTEXITCODE -ne 0) {
    throw "Compilation failed with exit code $LASTEXITCODE"
}
```

## Focused 0.47.4 connected setup regression

```powershell
$FocusedTests = @(
    "tests/test_connected_runtime_setup_invariants.py",
    "tests/test_connected_setup_correction_architecture.py",
    "tests/test_connected_setup_pose_regression.py",
    "tests/test_two_axis_connected_policy.py",
    "tests/test_two_axis_connected_single_layer.py",
    "tests/test_connected_group_document.py",
    "tests/test_connected_group_split_architecture.py",
    "tests/test_connected_placement_space_contract.py",
    "tests/test_spine_composition.py",
    "tests/test_manifest_version.py",
    "tests/test_documentation_contract.py"
)

& .\.venv-tests\Scripts\python.exe `
    -m pytest `
    $FocusedTests `
    -vv `
    --strict-markers `
    --durations=20

if ($LASTEXITCODE -ne 0) {
    throw "0.47.4 connected setup regressions failed"
}
```

These tests verify:

- the global X/Y wrapper constraints contribute zero setup rotation, translation, scale, and shear delta;
- global three-axis Rotation Z is relative-local and preserves the generated layer setup rotation;
- global three-axis scale owns both scale channels but no rotation, translation, or shear channel;
- each parent layer's setup Y contribution is subtracted from its object-main local Y;
- the resulting layer Y plus object-main Y equals the intended Blender visible Y;
- object Scale controls remain in their own `<prefix>_main` local space;
- connected constraint schedules remain unique and contiguous;
- weighted attachment indices and influence data remain unchanged after composition;
- the correction owner has no Blender API, mesh, UV, attachment, weighted-stream, or serialization responsibility.

## Complete Blender-independent suite

```powershell
Remove-Item ".\test-results.xml" -Force -ErrorAction SilentlyContinue

& .\.venv-tests\Scripts\python.exe `
    -m pytest `
    tests `
    -q `
    --durations=20 `
    --junitxml="test-results.xml"

if ($LASTEXITCODE -ne 0) {
    throw "Python test suite failed with exit code $LASTEXITCODE"
}
```

Do not use fail-fast for final evidence. A release run must expose every failure.

## Real bpy suite

```powershell
& .\.venv-bpy\Scripts\python.exe scripts\run_bpy_tests.py

if ($LASTEXITCODE -ne 0) {
    throw "Real bpy suite failed with exit code $LASTEXITCODE"
}
```

## Blender executable

```powershell
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"

if (-not (Test-Path -LiteralPath $Blender -PathType Leaf)) {
    throw "Blender executable not found: $Blender"
}
```

Every headless command must include `--python-exit-code 1`.

## Connected setup-pose gates

The following gates inspect the final serialized JSON using the same setup-delta assumptions as Spine 4.2. They do not pass merely because bones and constraints exist.

### Two-axis connected setup

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_two_axis_connected_multi_object_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Connected two-axis setup integration failed"
}
```

Expected marker:

```text
[TWO_AXIS_CONNECTED_MULTI] PASS test_connected_two_axis_export_builds_global_and_object_controls
```

### Three-axis connected setup

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_connected_setup_pose_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Connected three-axis setup integration failed"
}
```

Expected marker:

```text
[CONNECTED_SETUP_POSE] PASS test_connected_three_axis_setup_pose_is_not_collapsed
```

### Existing multi-object and rollback regression

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_multi_object_export_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Multi-object integration failed"
}
```

Expected marker:

```text
[MULTI] PASS 3 integration tests
```

The forced second-bake traceback in the rollback test is intentional only when the final test and Blender process both report success.

## Geometry and material gates

Run these after connected setup changes because document assembly remains shared:

```powershell
$HeadlessScripts = @(
    "tests\blender_headless\run_edge_on_multi_object_integration.py",
    "tests\blender_headless\run_vertex_bone_optimization_integration.py",
    "tests\blender_headless\run_normal_uv_pyramid_mode_integration.py",
    "tests\blender_headless\run_spine_material_correspondence_integration.py",
    "tests\blender_headless\run_uv_sampling_role_integration.py"
)

foreach ($Script in $HeadlessScripts) {
    & $Blender `
        --background `
        --factory-startup `
        --python-exit-code 1 `
        --python $Script

    if ($LASTEXITCODE -ne 0) {
        throw "Headless integration failed: $Script"
    }
}
```

## Build version 0.47.4

```powershell
Remove-Item ".\dist" -Recurse -Force -ErrorAction SilentlyContinue

& .\.venv-tests\Scripts\python.exe `
    tools\prepare_package.py `
    --blender $Blender

if ($LASTEXITCODE -ne 0) {
    throw "Extension package build failed"
}

$Zip = ".\dist\blender_to_spine2d_mesh_exporter-0.47.4.zip"

if (-not (Test-Path -LiteralPath $Zip -PathType Leaf)) {
    throw "Expected ZIP was not created: $Zip"
}

& $Blender --command extension validate $Zip

if ($LASTEXITCODE -ne 0) {
    throw "Built extension validation failed"
}
```

## Manual Spine gate

Install the new ZIP, restart Blender, and produce fresh connected JSON for both profiles. In Spine 4.2.43 verify:

- setup pose matches Blender XY composition before touching a control;
- object size and area are non-zero;
- each object's local controls move only that object;
- global controls move the entire group;
- global X/Y/Z or X/Y/Scale controls do not deform the group merely by importing the JSON;
- control icons follow their owning object and do not collapse at the origin;
- meshes, textures, UVs, pivots, and draw order remain correct.

## Release evidence

A release claim must record:

- exact candidate commit SHA;
- clean working tree or exact local modifications;
- Python and Blender versions;
- complete pytest summary;
- real-bpy summary;
- required Blender headless markers;
- package build and archive validation results;
- final ZIP path, size, timestamp, and SHA-256;
- fresh manual Blender-to-Spine import results for representative assets.

Do not state that tests passed without the corresponding logs. Do not reuse results from an older commit.

## Failure handling

When a test fails:

1. preserve the complete traceback;
2. identify the production owner or stale contract;
3. fix the owning implementation rather than weakening an unrelated test;
4. rerun the focused failure;
5. rerun the complete layer;
6. rerun downstream Blender and packaging gates when production behavior changed.

## Related documents

- [Architecture](architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Troubleshooting](troubleshooting.md)
