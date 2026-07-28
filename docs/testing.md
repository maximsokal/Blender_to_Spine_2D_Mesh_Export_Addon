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

## Focused 0.41.3 UV sampling and material regression

```powershell
& .\.venv-tests\Scripts\python.exe `
    -m pytest `
    tests/test_semantic_bake_image_io.py `
    tests/test_semantic_bake_execution_uv_roles.py `
    tests/test_uv_sampling_roles.py `
    tests/test_a1_material_correspondence.py `
    tests/test_a1_bake_material_bindings.py `
    tests/test_a1_z_groups.py `
    tests/test_normal_uv_pyramid_regression.py `
    tests/test_manifest_version.py `
    tests/test_documentation_contract.py `
    -vv `
    --durations=20

if ($LASTEXITCODE -ne 0) {
    throw "0.41.3 focused regressions failed"
}
```

These tests verify:

- the generated `SpineBakeUV` layer is the bake destination;
- the original source render UV remains the shader-sampling layer;
- `bpy.ops.object.bake` receives the destination UV layer explicitly;
- a Texture Coordinate UV to Mapping to Image Texture graph does not sample through `SpineBakeUV`;
- serialized UV, triangle, hull, edge, and weighted-bone streams preserve projection order;
- temporary bake material indices follow exact snapshot face identity rather than Blender polygon collection order;
- source Z values retain the Legacy four-decimal identity;
- the four-face pyramid remains exportable.

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

## Build version 0.41.3

```powershell
Remove-Item ".\dist" -Recurse -Force -ErrorAction SilentlyContinue

& .\.venv-tests\Scripts\python.exe `
    tools\prepare_package.py `
    --blender $Blender

if ($LASTEXITCODE -ne 0) {
    throw "Extension package build failed"
}

Get-Item `
    ".\dist\blender_to_spine2d_mesh_exporter-0.41.3.zip" |
    Select-Object FullName, Length, LastWriteTime
```

## Validate the archive

```powershell
& $Blender `
    --command extension validate `
    ".\dist\blender_to_spine2d_mesh_exporter-0.41.3.zip"

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
- manual re-export and Spine import of the representative sword asset.

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
