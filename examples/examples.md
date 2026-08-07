# Example Projects

The `examples` directory contains Blender scenes for learning and manual validation of the
current exporter. Examples do not replace automated regression tests.

Use Blender 5.2 or newer with Blender to Spine2D Mesh Exporter **0.128.0**.

## General workflow

1. Copy the example `.blend` before experimenting.
2. Open and save the copy in Blender 5.2+.
3. Select the intended Mesh object(s) in Object Mode.
4. Reset exporter settings when a clean current-default baseline is needed.
5. Choose a writable JSON directory and `images/` subfolder.
6. Choose Export Mode and, for Normal / UV Segments, Projection Direction.
7. Choose the exact Spine target.
8. Run Analyze and review diagnostics.
9. Export.
10. Import the JSON and matching textures into the selected Spine version.
11. Verify setup pose, UV alignment, generated controls, slot order, and sequence timing.

Never combine JSON from one export run with textures from another.

## Basic pyramid

Typical path:

```text
examples/01_pyramid_basic/cone_test.blend
```

Good for:

- single-object Normal / UV Segments;
- Auto segmentation;
- signed-axis projection;
- generated UV and semantic bake checks;
- basic Spine setup-pose inspection.

Suggested baseline:

```text
Export Mode:       Normal / UV Segments
Projection:        +Z
Texture Size:      128 or 256
Seam Maker:        Auto
Seed Angle Limit:  30
Frames:            0
Material Source:   Require Source
```

## Active Camera Object Root validation

Use any example with visible 3D depth and a valid active Perspective or Orthographic camera.

Select:

```text
Export Mode:       Normal / UV Segments
Projection:        Active Camera — Object Root Bone
```

Verify in Spine:

- the setup shape matches the Blender camera view;
- the mesh is not stretched or depth-flattened;
- `<prefix>_main` is the projected Blender Object Origin;
- X/Y controls pivot around that point;
- exported depth groups contain generated `*_camera_setup` inverse children.

## Active Camera Camera Root validation

With the same source/camera, select:

```text
Projection:        Active Camera — Camera Root Bone
```

Verify:

- initial projected geometry matches Object Root;
- the main bone is camera-relative rather than object-origin-relative;
- one rigid camera-depth layer owns the object placement;
- material appearance matches the Object Root camera projection for the same source state.

## Procedural crystal

Typical path:

```text
examples/02_crystal_procedural/cristall.blend
```

Good for:

- material graph analysis;
- semantic object baking;
- static versus sequence output;
- deciding between Normal and rendered-camera representations.

Start with a static low-resolution export before increasing sequence frame count.

## Multi-object text

Typical path:

```text
examples/03_spine_text_multi/text_spine.blend
```

Good for:

- public selected-object standalone export;
- per-object Frames/Start settings;
- output namespace and atomic transaction checks.

Public selected-object export is standalone composition. Connected and mixed composition
are explicit internal/development routes and are not configured as ordinary public UI
workflow here.

## Custom seams

1. Enter Edit Mode.
2. Mark intended seam edges.
3. Return to Object Mode.
4. Select `Seam Maker = Custom`.
5. Run Analyze and export.

Custom mode disables angular splitting, but topology validation/decomposition remains
active.

## Generated material validation

For geometry-focused testing:

```text
Material Source:   Force Generated
Generated Pattern: One Region - One Color
```

`One Polygon - One Color` is useful for triangulation inspection. Generated materials are
temporary and must not remain in the source file after export.

## Camera Projection

Use a valid active camera and verify:

- non-empty rendered alpha coverage;
- stable crop;
- flat screen-space attachment output;
- restored camera/frame/renderer/visibility state.

## Depth Camera Projection

Use a source with visible depth variation.

At `Parallax Horizon Angle = 0°`, verify FRONT-only relief output. With a positive angle,
verify reserve textures/attachments are generated only for retained non-empty directions
and are serialized below FRONT.

## Reporting example problems

Include extension version, commit SHA for development builds, Blender version, selected
objects, exact settings, Analyze diagnostics, traceback, generated filenames, and a Spine
screenshot when the mismatch is visual.

See [Troubleshooting](../docs/troubleshooting.md) and [Testing](../docs/testing.md).
