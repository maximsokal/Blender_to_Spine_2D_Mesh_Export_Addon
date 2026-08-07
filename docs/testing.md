# Testing and Release Validation

This document defines the current validation policy for Blender to Spine2D Mesh Exporter
**0.128.0**.

A focused test is not a release claim. Release evidence must be generated from one exact
clean commit and the archive built from that same commit.

## Product scope under test

- Blender 5.2.0 or newer.
- Spine targets 3.8.99, 4.0.64, 4.1.24, 4.2.43, and 4.3.23 according to the capability registry.
- Normal / UV Segments.
- Camera Projection.
- Depth Camera Projection.
- Six signed-axis Normal projection directions.
- Active Camera — Object Root Bone.
- Active Camera — Camera Root Bone.
- Static and per-object texture sequences.
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
- Active Camera Object Root/Camera Root normalization;
- camera-projected Object Origin placement;
- Object Root inverse-setup bone generation;
- Object Root vertex parenting below inverse setup bones;
- neutral camera-facing setup constraints;
- Camera Root single rigid depth-layer ownership;
- material-bake geometry independence from Normal projection direction;
- loop-level UV identity and weighted attachment construction;
- target-specific Spine version adaptation;
- sequence ownership;
- parallax reserve topology/camera planning;
- Blender-state/resource lifecycle contracts;
- manifest/documentation version synchronization.

Focused Object Root tests include:

```text
tests/test_active_camera_inverse_setup_parenting.py
tests/test_active_camera_normal_setup_pose.py
tests/test_active_camera_root_modes.py
tests/test_active_camera_normal_object_pivot.py
tests/test_normal_projection_parity_contract.py
```

## Real Blender Active Camera gates

Use a representative 3D asset with visible depth. The repository's coin integration asset
is used by the dedicated runners.

### Projection parity

```powershell
& $Blender `
    --factory-startup `
    --background `
    $CoinBlend `
    --python-exit-code 1 `
    --python tests\blender_headless\run_coin_star_normal_projection_parity_integration.py `
    -- `
    --expected-blend $CoinBlend

if ($LASTEXITCODE -ne 0) { throw "Projection parity gate failed" }
```

The gate validates retained side geometry, complete depth-group assignment, neutral
active-camera setup, and projection-independent material-bake geometry.

### Camera root modes

```powershell
& $Blender `
    --factory-startup `
    --background `
    $CoinBlend `
    --python-exit-code 1 `
    --python tests\blender_headless\run_coin_star_normal_camera_root_modes_integration.py `
    -- `
    --expected-blend $CoinBlend

if ($LASTEXITCODE -ne 0) { throw "Camera root mode gate failed" }
```

The gate must prove that Object Root and Camera Root share projected geometry/material
input while using different depth ownership:

```text
Object Root -> per-depth groups, CAMERA_VIEW_NORMAL
Camera Root -> one rigid depth group, PREPROJECTED_SCREEN
```

### Object Root inverse setup

```powershell
& $Blender `
    --factory-startup `
    --background `
    $CoinBlend `
    --python-exit-code 1 `
    --python tests\blender_headless\run_coin_star_normal_object_root_setup_compensation_integration.py `
    -- `
    --expected-blend $CoinBlend

if ($LASTEXITCODE -ne 0) { throw "Object Root inverse setup gate failed" }
```

The current gate verifies the complete setup chain rather than individual JSON fields. It
must prove:

- one inverse `*_camera_setup` bone for every Object Root depth group;
- every Object Root generated vertex bone is parented below the correct inverse setup bone;
- inverse setup Y cancels the matching depth setup Y;
- X/Y camera-facing setup rotation is neutral;
- depth Transform setup translation/scale are neutral;
- weighted vertices resolve to the expected camera-projected setup XY;
- the source Blender object and temporary datablock inventory remain unchanged after export.

This gate exists specifically to prevent a projected mesh from passing texture/geometry
checks while still being stretched by the generated Spine setup hierarchy.

## Depth Camera Projection gates

Maintain real Blender coverage for:

- Perspective front-only output;
- Orthographic output;
- supported target matrix;
- two-frame sequence output;
- positive Perspective parallax;
- Orthographic/sequence parallax;
- multi-object parallax and rollback;
- FRONT/reserve slot order and shared rig ownership.

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
scope/profile/target combinations.

## Real bpy suite

```powershell
& $BpyPython scripts\run_bpy_tests.py
if ($LASTEXITCODE -ne 0) { throw "Real bpy suite failed" }
```

A missing real-bpy environment must not be reported as successful release validation.

## Build 0.128.0

```powershell
$SourceDir = ".\Blender_to_Spine2D_Mesh_Exporter"
$Archive = ".\dist\blender_to_spine2d_mesh_exporter-0.128.0.zip"

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

## Manual Spine validation

For representative outputs in the exact selected Spine version, verify:

- JSON imports without schema/reference errors;
- texture paths resolve;
- UVs match texture orientation;
- signed-axis controls behave as expected;
- Active Camera Object Root setup matches the Blender camera view without stretching;
- Object Root X/Y controls rotate around the projected Blender Object Origin;
- Active Camera Camera Root keeps correct camera-relative placement;
- Depth FRONT/reserve slot order is correct;
- static/sequence ownership is correct for the selected target.

## Release evidence

Record:

- exact commit SHA;
- clean worktree before and after the gate;
- Python/bpy/Blender versions;
- complete pytest result;
- required Blender-headless gate markers;
- real-bpy result;
- archive path, size, and SHA256;
- Blender extension validation result;
- manual Spine validation notes.

Never claim a test passed on a commit that was not the exact commit used to generate the
reported output.
