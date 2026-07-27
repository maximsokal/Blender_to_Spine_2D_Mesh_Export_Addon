# Example Projects

The `examples` directory contains Blender scenes that can be used to learn the exporter workflow and perform manual checks. Example files are not substitutes for the automated Blender headless regression suite.

Use Blender 5.2 or newer and extension version 0.40.0 or newer.

## General workflow

For every example:

1. Make a copy of the `.blend` file.
2. Open the copy in Blender 5.2 or newer.
3. Save it before export.
4. Select the intended Mesh object or objects in Object Mode.
5. Press Reset in the exporter to establish current defaults.
6. Confirm `Seam Maker = Auto` unless the example explicitly tests marked seams.
7. Set a writable JSON directory and `images/` subfolder.
8. Run Analyze and resolve blockers.
9. Export.
10. Import the generated JSON and matching texture files into Spine 4.2.
11. Verify setup pose, attachment placement, UV alignment, rig behavior, and sequence timing when applicable.

Do not reuse JSON from one run with texture files from another run.

## Basic pyramid

Typical repository location:

```text
examples/01_pyramid_basic/cone_test.blend
```

Purpose:

- single-object workflow;
- Auto segmentation;
- low-resolution smoke export;
- material and image path verification;
- Spine setup-pose inspection.

Suggested initial settings:

```text
Export mode:         Normal - UV Segments
Texture size:        128 or 256
Seam Maker:          Auto
Seed angle limit:    30
Angular mode:        Seed cone
Frames for render:   0
Material Source:     Require Source
```

Expected output pattern:

```text
Cone_merged.json
images/Cone_Baked.png
```

The exact object-derived stem follows the active object name after Windows-safe sanitization.

Verify in Spine:

- every expected region exists;
- the texture orientation matches the mesh;
- no region samples another region's texture island;
- the setup pose matches the Blender projection contract;
- control icons and preview animation follow the selected settings.

## Procedural crystal

Typical repository location:

```text
examples/02_crystal_procedural/cristall.blend
```

Purpose:

- material graph analysis;
- semantic baking;
- optional frame-sequence output;
- comparison of Normal and explicit Camera Projection suitability.

Start with a low texture size and one frame. Increase frame count only after a static export passes readiness and imports correctly.

For a sequence:

```text
Frames for render:   positive integer
Start frame:         first desired frame
```

Expected texture pattern:

```text
<stem>_Baked_0000.png
<stem>_Baked_0001.png
...
```

The actual starting number follows Start frame.

Verify:

- the effective Material Output is supported by the selected export mode;
- external images are available or packed;
- sequence filenames and attachment metadata agree;
- Blender frame and renderer state are restored after export.

## Multi-object text

Typical repository location:

```text
examples/03_spine_text_multi/text_spine.blend
```

Purpose:

- selected-object export;
- standalone, connected, or mixed composition;
- per-object sequence settings;
- output naming and namespace preflight.

Suggested procedure:

1. Select at least two Mesh objects.
2. Decide which objects belong to a connected subgroup.
3. Enable Connect for either zero objects or at least two objects.
4. Configure per-object Frames and Start values.
5. Run Analyze.
6. Export Selected Objects.

Exactly one connected object falls back to standalone composition with a warning.

Verify in Spine:

- connected objects share the intended connected contract;
- standalone objects remain independent components;
- slots and attachments are not lost or duplicated;
- all texture paths resolve;
- the output JSON and every texture were produced by the same export transaction.

## Custom seam validation

Any suitable example Mesh can be used for Custom mode:

1. Enter Edit Mode.
2. Mark seam edges.
3. Return to Object Mode.
4. Select `Seam Maker = Custom`.
5. Run Analyze and export.

Custom mode disables angular splitting. Topology validation and manifold disk decomposition remain active.

Compare the result with Auto mode using separate output directories.

## Generated material validation

To validate geometry without relying on source shading:

```text
Material Source:     Force Generated
Generated Pattern:   One Region - One Color
```

This makes region ownership visible in the baked texture and Spine attachments. `One Polygon - One Color` is useful for triangulation and per-face inspection.

Generated materials are temporary and must not remain in the source `.blend` after export.

## Camera Projection validation

Use Camera Projection only with a valid active camera.

Verify:

- the selected objects are visible to the active camera;
- the render produces non-empty alpha coverage;
- crop and contour follow the visible render;
- the output is one screen-space projection attachment;
- the Scene camera, frame, renderer, compositor, sequencer, and visibility state are restored.

## Reporting example problems

Include:

- example path;
- extension version and commit SHA;
- Blender version;
- exact selected objects;
- all exporter settings;
- readiness issues;
- console traceback;
- generated filenames;
- a Spine screenshot when the mismatch is visual.

See [Troubleshooting](../docs/troubleshooting.md) and [Testing](../docs/testing.md).