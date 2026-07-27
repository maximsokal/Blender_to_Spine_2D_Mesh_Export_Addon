# Troubleshooting

## Start with the readiness report

Run **Analyze** before export. The report is the primary source for geometry, UV, material, camera, renderer, and output blockers.

After changing selection, geometry, modifiers, UVs, seams, materials, images, renderer, camera, frame settings, or exporter settings, run Analyze again. A stale report cannot authorize export.

## Seam Maker shows Custom instead of Auto

For version 0.40.0, older scenes below settings schema 3 are migrated once to Auto.

1. Remove the previous extension build.
2. Close Blender completely.
3. Install version 0.40.0 or newer.
4. Reopen the `.blend` file.
5. Check the values in Blender's Python Console:

```python
print(
    bpy.context.scene.spine2d_settings_schema_version,
    bpy.context.scene.spine2d_seam_maker_mode,
)
```

Expected migrated state:

```text
3 AUTO
```

A deliberate Custom choice made after schema 3 is preserved. Use the main Reset button to return it to Auto manually.

## Export is unavailable or blocked

Common causes:

- the `.blend` file is not saved;
- no Mesh object is active or selected;
- the object is in Edit Mode;
- readiness was not run or is stale;
- the current readiness report contains blockers;
- output paths are empty, invalid, colliding, or not writable;
- source geometry, UV, material, image, renderer, camera, or View Layer contracts are invalid.

Read every issue code and message shown in the readiness panel. The UI displays the first issues globally and per object.

## Edit Mode rejection

The production preparation pipeline requires Object Mode and does not silently switch modes.

Return the source object to Object Mode, run Analyze again, and retry export.

This protects edit-session data and avoids ambiguous ownership of `bmesh.from_edit_mesh()` state.

## Missing or unreadable source image

When source materials matter, image dependency preflight resolves FILE, SEQUENCE, and TILED image paths before baking.

Fix by doing one of the following:

- restore the missing file;
- correct the Blender image filepath;
- pack the image into the `.blend`;
- replace a corrupt image;
- select Generate If Missing when the missing material data qualifies for generated fallback;
- select Force Generated when source shading should be ignored intentionally.

Packed, generated, and supported viewer images do not require an external filepath.

## Missing or malformed UV layer

The exporter distinguishes required UV dependencies from unused UV data.

A required UV layer can come from:

- an explicit UV Map node;
- Texture Coordinate UV output;
- an Attribute node that names an existing real UV layer;
- the active or render-active source role required by the material graph.

Fix missing names, malformed coordinate buffers, invalid active/render-active roles, or material nodes that reference the wrong layer. Then rerun Analyze.

## Material requires Camera Projection

Normal mode supports validated object-bake material capabilities. Camera-dependent, volume, render-displacement, or other render-space requirements may need Camera Projection.

The exporter does not switch modes automatically. Select **Camera Projection**, verify the active camera and render context, and run Analyze again.

## Unsupported shader graph

Material analysis follows the effective renderer Material Output and reachable graph, including nested node groups and muted bypass behavior.

A blocker can indicate:

- missing usable Material Output;
- unsupported renderer-specific output;
- graph analysis failure;
- unsupported group or dependency behavior;
- material capability incompatible with the selected export mode.

Simplify the material, repair the effective output path, use generated materials, or select a compatible explicit export mode.

## Camera Projection blocker

Confirm that:

- Camera Projection is selected;
- the Scene has a valid active camera;
- the camera object and data are available;
- the renderer is supported;
- the View Layer and visibility state are valid;
- World and lighting requirements are satisfied for the material;
- projection alpha threshold is finite in `[0, 1]`;
- the render produces usable alpha coverage.

The projection transaction temporarily changes render state and restores it afterward. A restoration failure is reported and must not be ignored.

## Non-manifold or invalid topology

Final Normal attachments must be manifold disks with complete, disjoint face coverage.

Repair:

- non-manifold edges;
- loose vertices or loose edges when they violate the input contract;
- degenerate faces;
- invalid face-edge connectivity;
- unsupported modifier-generated vertices, faces, or corners without source lineage.

Custom seams do not bypass topology validation.

## Transform or world-space blocker

The evaluated world transform must be finite and non-singular. Supported non-identity transforms are normalized through the temporary evaluation path without modifying the source object.

Apply or repair transforms when the matrix is singular, non-finite, or cannot satisfy the geometry contract.

## Texture and Spine mesh do not align

Use version 0.39.0 or newer. The Normal semantic bake path converts Blender pixel rows to the file-space orientation expected by the exported Spine UVs.

To rule out stale files:

1. close Spine;
2. remove or rename the previous JSON and texture output;
3. export again;
4. confirm file modification times changed;
5. import the new JSON and matching new texture set.

Do not mix a new JSON with a texture generated by an older build.

The Blender headless test `run_spine_uv_file_space_integration.py` verifies directional correspondence with an asymmetric texture.

## GPU texture warning

A console message such as:

```text
Failed to create GPU texture from Blender image
```

can originate from Blender's GPU display path rather than the exporter output transaction. Treat it separately from final export success.

Check:

- whether the exporter reported success or a structured failure;
- whether the final JSON and textures were committed;
- whether the staged texture validation passed;
- whether Blender can open the saved image from disk;
- GPU driver and Blender system-console details.

Do not assume the warning proves that JSON-to-texture UV mapping is correct or incorrect.

## Output path collision or Windows filename error

The exporter sanitizes invalid filename characters and reserved DOS device names. It also rejects case-insensitive output collisions before writing.

Choose distinct object names and output paths. Avoid manually targeting the same final path from multiple concurrent requests.

## Failed stage or backup files remain

Temporary files use names similar to:

```text
.spine2d-stage-*
.spine2d-backup-*
```

Normally they are removed or recovered automatically.

In Add-on Preferences, check:

- Preserve failed work files;
- Recover stale work files.

When preservation is enabled, failed stage files can remain intentionally. Backups are safety data and may be restored or removed during the next reservation.

Never delete work files owned by another currently running Blender process.

## Export replaced an existing file and failed

The atomic transaction attempts to restore previous final files from backups. Inspect:

- the original structured export error;
- any `CLEANUP_FAILED` issue;
- `.spine2d-backup-*` files;
- the configured log file or Blender system console.

Keep the original exception as the primary failure. Cleanup failures are additional diagnostics.

## Enable useful logs

Open **Edit > Preferences > Extensions > Blender to Spine2D Mesh Exporter**.

You can:

- enable file logging;
- choose the log file path;
- filter module names;
- set individual files to ERROR, WARNING, INFO, or DEBUG;
- refresh the discovered module list.

For output transaction problems, enable DEBUG for modules under `infrastructure.atomic_`, `infrastructure.export_`, and the relevant output adapter.

For material or bake problems, enable DEBUG for the relevant `blender_adapter.material_`, `shader_`, `semantic_bake_`, or `camera_projection_` modules.

## Prepare a bug report

Include:

- extension version;
- exact commit SHA when using a development build;
- Blender version and operating system;
- Spine version;
- selected export mode;
- Seam Maker and angle settings;
- material source policy;
- renderer;
- complete readiness issue codes and messages;
- full Blender system-console traceback;
- whether the source image is external, packed, generated, tiled, or a sequence;
- whether the failure reproduces in a copied minimal `.blend`;
- whether source geometry, UVs, materials, or Blender state changed unexpectedly;
- output filenames and modification times.

Do not publish private customer `.blend` files or textures without authorization.

## Related documents

- [Installation](installation.md)
- [Usage](usage.md)
- [Settings Reference](settings-reference.md)
- [Testing](testing.md)