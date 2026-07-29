# Testing and Release Validation

## Validation layers

The repository uses four validation layers:

1. Blender-independent tests in `tests/`.
2. Real `bpy` tests in `tests_bpy/`.
3. Blender headless integration scripts in `tests/blender_headless/`.
4. Blender extension validation and ZIP packaging.

A focused test run is not a release gate. Every required layer must pass on the same candidate commit.

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

## Focused 0.47.2 geometry, rig, pivot, and material regression

```powershell
$FocusedTests = @(
    "tests/test_a1_projected_region_filter.py",
    "tests/test_a1_projected_region_filter_architecture.py",
    "tests/test_vertex_bone_optimizer.py",
    "tests/test_a1_document_assembly.py",
    "tests/test_a1_attachment_hull_normalization.py",
    "tests/test_a1_material_correspondence.py",
    "tests/test_two_axis_scale_rig_builder.py",
    "tests/test_two_axis_multi_placement.py",
    "tests/test_two_axis_connected_policy.py",
    "tests/test_connected_group_document.py",
    "tests/test_connected_group_split_architecture.py",
    "tests/test_a1_object_origin_offset.py",
    "tests/test_scene_settings_migration_contract.py",
    "tests/test_semantic_bake_image_io.py",
    "tests/test_semantic_bake_execution_uv_roles.py",
    "tests/test_uv_sampling_roles.py",
    "tests/test_a1_bake_material_bindings.py",
    "tests/test_a1_z_groups.py",
    "tests/test_normal_uv_pyramid_regression.py",
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
    throw "0.47.2 focused regressions failed"
}
```

These tests verify:

- valid three-dimensional faces that are edge-on in Spine X/Y are omitted from the 2D triangle stream;
- visible faces remain immutable and retain exact source vertex, loop, face, UV, and material lineage;
- disconnected visible components are materialized as deterministic manifold disks;
- remaining segments receive dense slot and attachment indices;
- an object is rejected only when every prepared face is invisible in X/Y;
- coincident segment-boundary points share one canonical vertex bone when parent and setup position match;
- weighted bone indices are compacted while local influence coordinates and weights remain unchanged;
- UV, triangle, hull, edge, attachment path, and mesh vertex order survive vertex-bone optimization;
- same-XY vertices in different Z parents remain independent;
- single-object two-axis controls serialize with neutral setup rotation;
- connected two-axis documents retain global and per-object X/Y/Scale controls;
- connected two-axis constraints use one unique contiguous five-phase schedule;
- connected weighted attachment indices remain valid after global rig insertion;
- Blender Object Origin remains the exported rotation pivot;
- old saved Scenes retain the compatibility rig while genuinely fresh Scenes use the two-axis default;
- the generated `SpineBakeUV` layer is the bake destination;
- the original source render UV remains the shader-sampling layer;
- `bpy.ops.object.bake` receives the destination UV layer explicitly;
- temporary bake material indices follow exact snapshot face identity;
- source Z values retain the Legacy four-decimal identity.

## Real bpy suite

Run the configured real-bpy suite through the repository runner:

```powershell
& .\.venv-bpy\Scripts\python.exe scripts\run_bpy_tests.py

if ($LASTEXITCODE -ne 0) {
    throw "Real bpy suite failed with exit code $LASTEXITCODE"
}
```

For the material-bake boundary:

```powershell
& .\.venv-bpy\Scripts\python.exe `
    -m pytest `
    tests_bpy/test_semantic_bake_real_bpy.py `
    -vv `
    -s `
    --strict-markers `
    --durations=20
```

## Blender executable

```powershell
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"

if (-not (Test-Path -LiteralPath $Blender -PathType Leaf)) {
    throw "Blender executable not found: $Blender"
}
```

Every headless command must include `--python-exit-code 1`.

## Edge-on two-axis multi-object integration

This is the direct regression for a multi-object asset containing a valid three-dimensional side wall that collapses only after projection into Spine X/Y.

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_edge_on_multi_object_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Edge-on multi-object integration failed"
}
```

Expected marker:

```text
[EDGE_ON_MULTI] PASS two-axis standalone edge-on regression
```

## Normal UV pyramid integration

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_normal_uv_pyramid_mode_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Normal UV pyramid integration failed"
}
```

Expected markers:

```text
[NORMAL_UV_PYRAMID_AUTO_ROUNDTRIP] PASS
[NORMAL_UV_PYRAMID_CUSTOM] PASS
[MISSING_IMAGE_PREFLIGHT] PASS
[EDIT_MODE_CONTRACT] PASS
[NORMAL_UV_PYRAMID] PASS
```

## Shared vertex-bone optimization integration

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_vertex_bone_optimization_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Vertex-bone optimization integration failed"
}
```

Expected marker:

```text
[VERTEX_BONE_OPTIMIZATION] PASS pyramid shared-bone regression
```

The Blender pyramid still contains twelve weighted attachment vertices across four segment meshes, but those vertices reference four canonical generated bones instead of twelve duplicated bones.

## Existing multi-object integration

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

## Connected two-axis multi-object integration

This regression executes the production connected path with two live Blender objects using `TWO_AXIS_ROTATION_SCALE`. It verifies global controls, independent per-object controls, connected layer placement, unique constraint orders, texture output, and Blender state restoration.

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_two_axis_connected_multi_object_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Connected two-axis multi-object integration failed"
}
```

Expected marker:

```text
[TWO_AXIS_CONNECTED_MULTI] PASS test_connected_two_axis_export_builds_global_and_object_controls
```

## Directional Spine UV integration

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_spine_uv_file_space_integration.py"
```

Expected marker:

```text
[SPINE_UV_FILE_SPACE_DIRECTIONAL] PASS
```

## Asymmetric material correspondence integration

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_spine_material_correspondence_integration.py"
```

Expected marker:

```text
[SPINE_MATERIAL_CORRESPONDENCE] PASS
```

This fixture intentionally uses different geometry-corner and source-material-UV orders. Its expected colors are derived from the source geometry and source UV contract rather than from another exported stream.

## Source render UV role integration

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_uv_sampling_role_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Source render UV role integration failed"
}
```

Expected marker:

```text
[PASS] test_source_render_uv_is_not_replaced_by_spine_bake_uv
```

This fixture reproduces the representative sword material graph:

```text
Texture Coordinate UV
-> Mapping
-> Image Texture
-> Principled BSDF
```

It verifies that the original source render UV samples the material while the independently active `SpineBakeUV` receives the bake output.

## Build version 0.47.2

```powershell
Remove-Item ".\dist" -Recurse -Force -ErrorAction SilentlyContinue

& .\.venv-tests\Scripts\python.exe `
    tools\prepare_package.py `
    --blender $Blender

if ($LASTEXITCODE -ne 0) {
    throw "Extension package build failed"
}

Get-Item `
    ".\dist\blender_to_spine2d_mesh_exporter-0.47.2.zip" |
    Select-Object FullName, Length, LastWriteTime
```

## Validate the archive

```powershell
& $Blender `
    --command extension validate `
    ".\dist\blender_to_spine2d_mesh_exporter-0.47.2.zip"

if ($LASTEXITCODE -ne 0) {
    throw "Built extension validation failed"
}
```

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
- manual re-export and Spine import of representative assets.

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
