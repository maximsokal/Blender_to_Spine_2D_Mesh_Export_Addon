# Example Projects

The `examples` directory contains Blender scenes for learning and manual validation of the
current exporter. Examples do not replace automated regression tests.

Use Blender 5.2 or newer with **Spine Mesh Exporter 0.155.0**.

## General workflow

1. Copy the example `.blend` before experimenting.
2. Open and save the copy in Blender 5.2+.
3. Select the intended Mesh object(s) in Object Mode.
4. Reset exporter Scene settings when a clean current-default baseline is needed.
5. In Add-on Preferences, configure the exact Spine project patch for the schema family you use.
6. Choose a writable JSON directory and `images/` subfolder.
7. Choose Export Mode and, for Normal / UV Segments, Projection Direction.
8. Choose the Spine schema family and confirm **Exact JSON version**.
9. Open **Bake** and configure shared Texture size plus frame/sequence settings.
10. Run Analyze and review diagnostics.
11. Export.
12. Import the JSON and matching textures into the configured exact Spine project version.
13. Verify setup pose, UV alignment, generated controls, slot order, and sequence timing.

Never combine JSON from one export run with textures from another.

## Basic pyramid

Typical path:

```text
examples/01_pyramid_basic/cone_test.blend
```

Good for single-object Normal / UV Segments, Auto segmentation, signed-axis projection,
generated UV/semantic bake checks, and basic Spine setup-pose inspection.

Suggested baseline:

```text
Export Mode:       Normal / UV Segments
Projection:        +Z
Bake / Texture size: 128 or 256
Seam Maker:        Auto
Seed Angle Limit:  30
Frames:            0
Material Source:   Require Source
```

## Active Camera Object Root validation

Use any example with visible 3D depth and a valid active Perspective or Orthographic camera.
Select **Normal / UV Segments** and **Active Camera — Object Root Bone**. Verify in Spine that
the setup shape matches the Blender camera view, the mesh is not stretched or flattened,
`<prefix>_main` is the projected Blender Object Origin, X/Y controls pivot around it, and
depth groups contain generated `*_camera_setup` children.

## Active Camera Camera Root validation

With the same source/camera select **Active Camera — Camera Root Bone**. Verify initial
projected geometry matches Object Root, the main bone is camera-relative, one rigid
camera-depth layer owns placement, and material appearance matches Object Root.

## Procedural crystal

Typical path:

```text
examples/02_crystal_procedural/cristall.blend
```

Good for material graph analysis, semantic baking, static/sequence output, and choosing
between Normal and rendered-camera representations. Start with a low-resolution static
export before increasing frame count.

## Multi-object text

Typical path:

```text
examples/03_spine_text_multi/text_spine.blend
```

Good for public selected-object standalone export, Shared Selection Pivot, one shared Scene
Bake Texture size, per-object Frames/Start, and atomic output checks. With signed-axis
Normal / UV, Shared Selection Pivot is enabled by default; matching X/Y controls from
different parts should rotate around the same assembly pivot.

## Custom seams

1. Enter Edit Mode.
2. Mark intended seam edges.
3. Return to Object Mode.
4. Select `Seam Maker = Custom`.
5. Run Analyze and export.

Custom mode disables angular splitting, but topology validation/decomposition remains active.

## Generated material validation

For geometry-focused testing use `Force Generated` with `One Region - One Color`.
`One Polygon - One Color` is useful for triangulation inspection. Generated materials are
temporary and must not remain in the source file after export.

## Camera Projection

Use a valid active camera and verify non-empty alpha coverage, stable crop, flat screen-space
attachment output, and restored camera/frame/renderer/visibility state.

## Depth Camera Projection

Use a source with visible depth variation. At `Parallax Horizon Angle = 0°`, verify
FRONT-only relief output. With a positive angle, verify reserve textures/attachments are
generated only for retained non-empty directions and are serialized below FRONT.

## Exact project-version validation

The schema family and exact patch are separate. For example, configure a non-default 4.2
patch in Add-on Preferences, select Spine 4.2 in the scene, and confirm that the versioned
JSON filename plus `skeleton.spine` use the configured patch while rig/schema behavior
remains Spine 4.2. Repeat for each family used by the project.

## Reporting example problems

Include extension version, commit SHA for development builds, Blender version, selected
objects, exact settings, configured Spine project version, Analyze diagnostics, traceback,
generated filenames, and a Spine screenshot when the mismatch is visual.

See [Troubleshooting](../docs/troubleshooting.md) and [Testing](../docs/testing.md).
