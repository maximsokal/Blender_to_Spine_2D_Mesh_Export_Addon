# Testing and Release Validation

## Validation layers

The repository uses four distinct validation layers:

1. Blender-independent Python tests in `tests/`.
2. Real `bpy` tests in `tests_bpy/`.
3. Blender headless integration scripts in `tests/blender_headless/`.
4. Blender extension validation and ZIP packaging.

A passing focused test set is not a release gate. A release candidate must pass the applicable complete layers on the same candidate commit.

## Test environments

Typical Windows virtual environments:

```text
.venv-tests   Pure Python and source-contract tests
.venv-bpy     Tests executed with an installed real bpy runtime
```

Use the repository requirements and scripts appropriate to the local environment. Do not install fake Blender modules into the production package.

## Compile check

From PowerShell:

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

Do not use `--maxfail=1` for the final run. Fail-fast is useful while debugging but hides later regressions.

## Focused material correspondence tests

```powershell
& .\.venv-tests\Scripts\python.exe `
    -m pytest `
    tests/test_a1_material_correspondence.py `
    tests/test_a1_attachment_projection.py `
    tests/test_physical_hull_promotion.py `
    tests/test_legacy_attachment_builder.py `
    -vv
```

These tests validate setup-pose Z-group translation, physical hull promotion, exact UV and triangle order, and compact weighted vertex-bone indices.

## Focused documentation contract

```powershell
& .\.venv-tests\Scripts\python.exe `
    -m pytest `
    tests/test_documentation_contract.py `
    -vv
```

The documentation contract checks:

- no Cyrillic characters in maintained public documentation;
- no temporary `docs/REWRITE_*.md` files;
- required README cover, counters, badges, video, and UI image references;
- valid relative Markdown links and local image sources;
- documented Blender and extension versions;
- Auto as the documented Seam Maker default.

## Real bpy lifecycle and adapter tests

Run the full configured real-bpy suite through the repository runner when possible:

```powershell
& .\.venv-bpy\Scripts\python.exe scripts\run_bpy_tests.py

if ($LASTEXITCODE -ne 0) {
    throw "Real bpy suite failed with exit code $LASTEXITCODE"
}
```

For a focused lifecycle check:

```powershell
& .\.venv-bpy\Scripts\python.exe `
    -m pytest `
    tests_bpy/test_scene_settings_migration_real_bpy.py `
    tests_bpy/test_root_registration_headless_real_bpy.py `
    tests_bpy/test_registration_real_bpy.py `
    tests_bpy/test_runtime_hook_cleanup_real_bpy.py `
    -vv `
    -s `
    --strict-markers `
    --durations=0
```

These tests use a real Blender Python API surface. They cover registration, RNA, handlers, migration, resource ownership, and selected adapter behavior.

## Blender executable

```powershell
$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"

if (-not (Test-Path -LiteralPath $Blender -PathType Leaf)) {
    throw "Blender executable not found: $Blender"
}
```

Every headless command must use:

```text
--python-exit-code 1
```

Without it, an uncaught Python exception may still produce a successful Blender process exit code.

## Blender 5.2 API contract

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_blender_52_api_contract.py"

if ($LASTEXITCODE -ne 0) {
    throw "Blender 5.2 API contract failed with exit code $LASTEXITCODE"
}
```

## Normal UV pyramid regression

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_normal_uv_pyramid_mode_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Normal UV pyramid integration failed with exit code $LASTEXITCODE"
}
```

Expected markers include:

```text
[NORMAL_UV_PYRAMID_AUTO_ROUNDTRIP] PASS
[NORMAL_UV_PYRAMID_CUSTOM] PASS
[MISSING_IMAGE_PREFLIGHT] PASS
[EDIT_MODE_CONTRACT] PASS
[NORMAL_UV_PYRAMID] PASS
```

This is a smoke and color-region test. It is not sufficient by itself to prove directional UV-to-image correspondence.

## Directional Spine UV file-space regression

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_spine_uv_file_space_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Spine UV file-space integration failed with exit code $LASTEXITCODE"
}
```

Expected marker:

```text
[SPINE_UV_FILE_SPACE_DIRECTIONAL] PASS
```

This test detects missing or double vertical conversion, swapped axes, and basic corner reordering. Its geometry and source UV orientation are intentionally simple.

## Asymmetric source-material correspondence regression

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_spine_material_correspondence_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Spine material correspondence integration failed with exit code $LASTEXITCODE"
}
```

Expected marker:

```text
[SPINE_MATERIAL_CORRESPONDENCE] PASS
```

This fixture intentionally assigns source-material UV corners in a different order from geometry corners. It independently verifies:

```text
source geometry corner
-> source material UV
-> semantic bake
-> generated Spine UV
-> final attachment vertex
-> sampled PNG color
```

A geometry-derived expected color is used, so the test cannot pass by comparing one exported stream with another exported stream.

## Source render UV role regression

```powershell
& $Blender `
    --background `
    --factory-startup `
    --python-exit-code 1 `
    --python "tests\blender_headless\run_uv_sampling_role_integration.py"

if ($LASTEXITCODE -ne 0) {
    throw "Source render UV role integration failed with exit code $LASTEXITCODE"
}
```

This test verifies that an unlinked Image Texture Vector continues to sample the source `active_render` UV while the bake operator writes into the independently active `SpineBakeUV` layer.

## Other Blender headless coverage

The repository contains dedicated integration scripts for:

- semantic and alpha baking;
- sequence rollback;
- generated materials;
- renderer-specific material output;
- camera projection;
- grouped camera projection;
- multi-object and mixed composition;
- UV seam export;
- world-transform normalization;
- logging and diagnostics;
- projection output policy;
- resource and state restoration.

Run the scripts relevant to changed production ownership. Architecture-only changes still require the complete Python suite.

## Build the extension ZIP

```powershell
Remove-Item ".\dist" -Recurse -Force -ErrorAction SilentlyContinue

& .\.venv-tests\Scripts\python.exe `
    tools\prepare_package.py `
    --blender $Blender

if ($LASTEXITCODE -ne 0) {
    throw "Extension package build failed with exit code $LASTEXITCODE"
}
```

The wrapper validates the source and manifest before calling Blender's official extension build command.

For version 0.41.0, verify:

```powershell
Get-Item `
    ".\dist\blender_to_spine2d_mesh_exporter-0.41.0.zip" |
    Select-Object FullName, Length, LastWriteTime
```

## Validate the archive

```powershell
& $Blender `
    --command extension validate `
    ".\dist\blender_to_spine2d_mesh_exporter-0.41.0.zip"

if ($LASTEXITCODE -ne 0) {
    throw "Built extension validation failed with exit code $LASTEXITCODE"
}
```

## Release evidence

A release claim must record:

- exact candidate commit SHA;
- clean working tree or exact local modifications;
- Python executable versions;
- Blender executable path and version;
- complete pytest summary;
- real-bpy summary;
- every required Blender headless marker;
- package build result;
- archive validation result;
- final ZIP path, size, and modification time.

Do not state that tests passed without the corresponding logs. Do not substitute an earlier commit's results for the current candidate.

## Failure handling

When a test fails:

1. identify whether the failure is production behavior, test precision, stale expectation, environment, or missing dependency;
2. preserve the complete traceback;
3. fix the owning implementation or contract rather than weakening an unrelated test;
4. rerun the focused failure;
5. rerun the full layer without fail-fast;
6. rerun downstream Blender and packaging gates when production behavior changed.

## Related documents

- [Architecture](architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Troubleshooting](troubleshooting.md)
