# Usage Guide

## Open the exporter

1. Open Blender 5.2 or newer.
2. Save the `.blend` file.
3. Select at least one Mesh object.
4. Switch the active object to Object Mode.
5. In a 3D View, press `N` and open the **Blender to Spine2D Mesh Exporter** tab.

The main panel contains Export, Cut, and Bake sections, readiness analysis, and the final export button. A separate child panel contains Generated Materials settings.

## Prepare the source scene

Before analysis or export:

- save the `.blend` file;
- keep source Mesh objects in Object Mode;
- ensure required source images exist or are packed/generated inside Blender;
- select a supported renderer and a valid active camera when Camera Projection is used;
- choose a writable output directory;
- confirm that the intended UV layer and material graph are valid;
- avoid changing geometry, UVs, materials, selection, or export settings after analysis without running analysis again.

The exporter reads evaluated geometry through isolated temporary objects and verifies that source mesh, UV, material, and Blender state remain unchanged.

## Choose an export mode

### Normal - UV Segments

Normal mode is the default. It is intended for deformable Spine mesh attachments.

Pipeline summary:

```text
source Mesh
  -> evaluated geometry capture
  -> automatic or custom-seam segmentation
  -> manifold disk decomposition
  -> generated SpineBakeUV layout
  -> per-region Spine attachments
  -> semantic texture bake
  -> JSON and texture commit
```

The source Scene may use Blender 5.2 EEVEE. Semantic object baking temporarily uses the validated Cycles path and restores the original render engine and related Scene state.

Materials that require camera-space evaluation do not trigger an automatic mode switch. Select Camera Projection explicitly when the readiness report requires it.

### Camera Projection

Camera Projection renders the selected source through the active camera and exports one screen-space projection attachment.

Pipeline summary:

```text
active camera render
  -> sequence coverage union
  -> alpha cleanup and stable crop
  -> contour and triangulation
  -> projection attachment
  -> JSON and texture commit
```

Use it for camera-dependent, volume, screen-space, or other supported render-dependent appearances that cannot be represented by the Normal object-bake contract.

Camera Projection requires a valid active camera and a supported render context.

## Configure cutting

### Seam Maker: Auto

Auto is the default. The exporter uses the Seed angle limit and Angular mode to grow deterministic face regions.

- **Seed cone** compares candidates with the segment seed normal.
- **Seed cone + local dihedral** also limits the angle across each traversed shared edge.

Lower angle values usually create more regions. Higher values allow broader normal variation inside a region.

### Seam Maker: Custom

Custom uses seams marked by the user on the source Mesh. Angular splitting is disabled in this mode.

Typical workflow:

1. Enter Edit Mode.
2. Select the intended boundary edges.
3. Use **Edge > Mark Seam**.
4. Return to Object Mode.
5. Select Custom in the exporter.
6. Run analysis again.

The topology pipeline may still decompose a seam-defined region when required to produce valid manifold disk attachments. It does not enable the Auto angular split policy.

## Configure textures and output paths

Set:

- Texture size;
- JSON output directory;
- Images Subfolder;
- Control icons;
- Preview animation;
- Projection alpha threshold when Camera Projection is selected.

Texture size must be an even integer from 64 through 4096. The default is 1024.

The JSON path is a directory. The Images Subfolder is normalized as a relative path below that directory; the default is `images/`.

## Configure generated materials

The Generated Materials panel controls what happens when source materials are missing or intentionally ignored.

- **Require Source** blocks export when required source material data is unavailable.
- **Generate If Missing** creates a temporary generated material only when required material data is missing.
- **Force Generated** ignores source materials and always uses the selected generated pattern.

Patterns:

- **Solid Gray**;
- **One Region - One Color**;
- **One Polygon - One Color**.

Generated materials are temporary. The exporter removes generated materials, node trees, images, meshes, objects, and color attributes on success and failure paths.

## Configure frame output

For one selected object, Bake settings are stored on the Scene:

- Frames for render;
- Start frame;
- calculated last frame.

`Frames for render = 0` exports the current frame only.

For multiple selected objects, each object has independent Frames and Start values. This allows static and sequence objects to participate in the same multi-object request.

## Configure multi-object composition

When more than one Mesh is selected, each object receives a Connect checkbox.

- No connected objects: standalone multi-object composition.
- At least two connected objects and no standalone objects: connected composition.
- Connected and standalone objects together: mixed composition.
- Exactly one Connect checkbox: the request falls back to standalone export with a warning.

Connected objects share the connected rig contract. Standalone objects retain independent component rigs inside the final document.

## Run readiness analysis

Press **Analyze** before export.

The readiness report can include:

- source and exported vertex/triangle counts;
- region, attachment, and bone counts;
- texture pipeline and frame count;
- topology statistics;
- structured blockers and warnings.

The cached report becomes stale when relevant selection, geometry, UV, material, Scene, renderer, camera, or export settings change. Run analysis again after any such change.

The export button is enabled only when the current report allows export.

## Export one object

1. Make the Mesh active.
2. Configure settings.
3. Run Analyze.
4. Resolve every blocker.
5. Press **Export Current Object**.

The JSON stem is derived from the object name and ends with `_merged.json`. Texture paths are written below the configured Images Subfolder.

## Export multiple objects

1. Select at least two Mesh objects.
2. Choose the active object and Connect flags.
3. Configure Scene and per-object Bake settings.
4. Run Analyze.
5. Resolve every blocker.
6. Press **Export Selected Objects**.

The output stem uses the first ordered selected object name plus the number of additional selected objects.

## Import into Spine

1. Keep the JSON and image directory relationship unchanged.
2. Open or create a Spine 4.2 project.
3. Import the generated Spine JSON.
4. Point Spine to the exported images directory when needed.
5. Inspect setup pose, attachments, UV placement, constraints, preview animation, and texture sequences.

Normal-mode textures are saved with the file-space orientation expected by the exported Spine UV coordinates.

## Reset settings

The main Reset button restores:

- Normal - UV Segments;
- texture size 1024;
- default output paths;
- control icons enabled;
- preview animation enabled;
- angle limit 30;
- Seed cone angular mode;
- local angle limit 30;
- Seam Maker Auto;
- current-frame baking.

The Generated Materials panel has its own Reset button and restores Require Source, Solid Gray, and gray RGB `(0.5, 0.5, 0.5)`.

## Continue reading

- [Settings Reference](settings-reference.md)
- [Output Format](output-format.md)
- [Troubleshooting](troubleshooting.md)
- [Examples](../examples/examples.md)