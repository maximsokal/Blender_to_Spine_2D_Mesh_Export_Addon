# Usage Guide

## Open the exporter

1. Open Blender 5.2 or newer.
2. Save the `.blend` file.
3. Select at least one Mesh object.
4. Switch the active object to Object Mode.
5. In a 3D View, press `N` and open the **Blender to Spine2D Mesh Exporter** tab.

The main panel contains Paths and Spine 2D version, Rig, Rewrite Generated Materials, Cut, Bake, and Analysis sections, followed by the final export button. A separate re-polish.com child panel provides the animation-optimization link.

## Prepare the source scene

Before analysis or export:

- save the `.blend` file;
- keep source Mesh objects in Object Mode;
- ensure required source images exist or are packed/generated inside Blender;
- select a supported renderer and a valid active camera when Camera Projection or Depth Camera Projection is used;
- choose a writable output directory;
- confirm that the intended UV layer and material graph are valid;
- avoid changing geometry, UVs, materials, selection, camera, frame, or export settings after analysis without running analysis again.

The exporter reads evaluated geometry through isolated temporary objects and verifies that source mesh, UV, material, camera, selection, timeline, and Blender render state remain unchanged.

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

### Depth Camera Projection

Depth Camera Projection combines a camera render with a weighted relief mesh. It is intended for camera-dependent materials that must retain controlled X/Y/Scale deformation in Spine.

Front-only pipeline at `Parallax Horizon Angle = 0°`:

```text
active camera visible faces
  -> bounded depth lattice
  -> farthest visible point becomes zero relief
  -> generated shared vertex-bone rig
  -> FRONT camera render and alpha-union crop
  -> one weighted FRONT attachment
  -> JSON and texture commit
```

Positive parallax pipeline:

```text
active camera visible faces
  -> accumulated unsigned dihedral traversal
  -> one union MeshSnapshot for FRONT and reserve faces
  -> deterministic virtual camera assignment
  -> face-isolated FRONT and reserve camera renders
  -> one stable crop per view across all sequence frames
  -> reserve attachments followed by FRONT attachment
  -> one shared atomic JSON/texture commit
```

The active camera may be Perspective or Orthographic. Virtual reserve cameras are temporary copies. Perspective views fit the copied lens; Orthographic views fit the copied `ortho_scale`. The source camera is restored and never receives the virtual transform.

## Configure parallax reserve

`Parallax Horizon Angle` is visible only in the **Cut** foldout when **Depth Camera Projection** is selected. It is intentionally not a Rig control because it changes retained surface topology and reserve-view generation.

- `0°` exports the established FRONT-only result.
- A positive value retains connected surfaces whose minimum accumulated dihedral cost is within the selected angle.
- The default is `0°`.
- The hard maximum is `89°`; the UI soft maximum is `45°`.
- Blender displays degrees but stores radians.

The exporter may assign retained faces to up to eight virtual directions. Empty directions create no texture or attachment.

Every reserve view owns exact source-face indices. Its temporary evaluated mesh contains only those faces, preventing the FRONT surface from occluding hidden reserve texture coverage. Each reserve texture therefore contains its own surface rather than a duplicate render of the whole object.

Reserve slots are serialized before the FRONT slot. In Spine this keeps reserve surfaces below the FRONT surface. FRONT and reserve attachments share generated hinge bones where their union topology shares source vertices.

A positive horizon can increase the union point count. When the requested union exceeds **Max depth points**, Analyze and Export fail instead of silently reducing the angle or dropping faces.

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

Depth Camera Projection does not use source seam controls. Its Cut foldout instead exposes Parallax Horizon Angle beside the generated-depth explanation. Depth discontinuities are controlled by Depth edge threshold and horizon growth is controlled by Parallax Horizon Angle.

## Configure textures and output paths

Set:

- Texture size;
- JSON output directory;
- Images Subfolder;
- Control icons;
- Preview animation;
- Projection alpha threshold for rendered-camera modes;
- Depth quality and Parallax Horizon Angle for Depth Camera Projection.

Texture size must be an even integer from 64 through 4096. The default is 1024.

The JSON path is a directory. The Images Subfolder is normalized as a relative path below that directory; the default is `images/`.

FRONT and every reserve view use separate deterministic image stems. A two-frame positive-parallax export therefore creates two FRONT PNG files and two PNG files for each non-empty reserve view.

## Configure generated materials

The Generated Materials panel controls what happens when source materials are missing or intentionally ignored.

- **Require Source** blocks export when required source material data is unavailable.
- **Generate If Missing** creates a temporary generated material only when required material data is missing.
- **Force Generated** ignores source materials and always uses the selected generated pattern.

Patterns:

- **Solid Gray**;
- **One Region - One Color**;
- **One Polygon - One Color**.

Generated materials are temporary. The exporter removes generated materials, node trees, images, meshes, objects, color attributes, camera proxies, and render proxies on success and failure paths.

## Configure frame output

For one selected object, Bake settings are stored on the Scene:

- Frames for render;
- Start frame;
- calculated last frame.

`Frames for render = 0` exports the current frame only.

For multiple selected objects, each object has independent Frames and Start values. This allows static and sequence objects to participate in the same multi-object request.

For positive Depth parallax, FRONT and reserve views have matching frame counts. Crop is stable across frames inside one view, while FRONT and reserve views keep independent crop rectangles.

## Configure multi-object composition

Connected-object controls are currently retained as development-only functionality and are not shown in the production UI.

- No connected objects: standalone multi-object composition.
- At least two connected objects and no standalone objects: connected composition.
- Connected and standalone objects together: mixed composition.
- The internal connected-composition contract remains available to development integrations.

Connected objects share the connected rig contract. Standalone objects retain independent component rigs inside the final document.

Standalone multi-object Depth parallax places every object's FRONT and reserve outputs in one outer atomic transaction. A failure while staging a later object rolls back the JSON and every staged texture from earlier objects.

## Run readiness analysis

Press **Analyze** before export.

The readiness report can include:

- source and exported vertex/triangle counts;
- visible and reserve source-face counts;
- region, attachment, virtual-view, and bone counts;
- texture pipeline and frame count;
- topology and crop statistics;
- structured blockers and warnings.

The cached report becomes stale when relevant selection, geometry, UV, material, Scene, renderer, camera, frame, or export settings change. This includes Depth quality values and Parallax Horizon Angle. Run analysis again after any such change.

The report is diagnostic-only. Export remains available even when the report is missing, stale, or contains blockers; production validation still fails closed during export when the request is invalid.

## Export one object

1. Make the Mesh active.
2. Configure settings.
3. Run Analyze.
4. Review any reported blockers and warnings.
5. Press **Export Current Object**.

The JSON stem is derived from the object name and ends with `_merged.json`. Texture paths are written below the configured Images Subfolder.

For positive parallax, inspect the result statistics for FRONT/reserve view count and verify that output files contain one JSON, all FRONT frames, and all reserve frames.

## Export multiple objects

1. Select at least two Mesh objects.
2. Configure the Rig settings if needed.
3. Configure Cut and per-object Bake settings.
4. Run Analyze.
5. Review any reported blockers and warnings.
6. Press **Export Selected Objects**.

The output stem uses the first ordered selected object name plus the number of additional selected objects. All object textures and the final JSON are committed together.

## Import into Spine

1. Keep the JSON and image directory relationship unchanged.
2. Open or create a project in the exact selected Spine version.
3. Import the generated Spine JSON.
4. Point Spine to the exported images directory when needed.
5. Inspect setup pose, attachment order, UV placement, constraints, preview animation, and texture sequences.
6. For positive parallax, confirm that reserve slots occur immediately below their object's FRONT slot and that shared hinge deformation remains coherent.

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
- current-frame baking;
- Parallax Horizon Angle `0°`.

The Generated Materials panel has its own Reset button and restores Require Source, Solid Gray, and gray RGB `(0.5, 0.5, 0.5)`.

## Continue reading

- [Settings Reference](settings-reference.md)
- [Output Format](output-format.md)
- [Troubleshooting](troubleshooting.md)
- [Examples](../examples/examples.md)
