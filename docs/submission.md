# Blender Extensions Platform Submission

This document describes the current publication candidate for Blender to Spine2D Mesh
Exporter **0.129.0**.

The extension is prepared for an initial Blender Extensions Platform submission on
**Windows x64**. Other platforms must be added to `platforms` only after the same install,
registration, export, cleanup, and package gates have been run on those platforms.

## Upload artifact

Build and upload only this extension archive:

```text
dist/blender_to_spine2d_mesh_exporter-0.129.0.zip
```

Do not upload the repository ZIP. The extension archive must be produced by Blender's
`extension build` command and must validate with `extension validate`.

## Current public metadata

```text
Name: Blender to Spine2D Mesh Exporter
Version: 0.129.0
Minimum Blender: 5.2.0
Platform: Windows x64
License: GPL-3.0-or-later
Website: https://github.com/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon
Tags: Import-Export, Mesh, UV, Animation
Permission: Files
```

The Files permission is required because the extension writes Spine JSON, textures,
diagnostics, transaction staging files, backups, and optional logs to user-selected or
user-owned filesystem locations.

## Store description

Suggested short description:

> Convert Blender Mesh objects into Spine 2D JSON, weighted mesh attachments, baked or
> camera-rendered textures, generated animation controls, and optional texture sequences.

Suggested reviewer-oriented description:

> The extension converts Blender Mesh objects into Spine-ready JSON and textures. It
> supports Normal / UV Segments with six signed-axis projections and two Active Camera rig
> roots, flat Camera Projection, depth-aware Camera Projection, 2-Axis Rotation + Scale
> controls, generated bake UVs, automatic/custom seam segmentation, texture sequences,
> multi-object standalone export, readiness analysis, and atomic output rollback. The
> current public package targets Windows x64 and Blender 5.2 or newer. It performs no
> network requests and declares only filesystem access.

## Reviewer test path

1. Install the ZIP through **Edit > Preferences > Extensions > Install from Disk**.
2. Enable **Blender to Spine2D Mesh Exporter**.
3. Save a `.blend` file containing one Mesh object with a material.
4. Select the Mesh in Object Mode.
5. Open **3D View > Sidebar > Blender to Spine2D Mesh Exporter**.
6. Keep **Normal / UV Segments**, `+Z`, Spine 4.2, and a small Texture size such as 256.
7. Choose a writable JSON output directory.
8. Run **Analyze** and resolve any blocker reported by the extension.
9. Run **Export Current Object**.
10. Confirm that the JSON and `images/` texture output are created.
11. Disable the extension, restart Blender, enable it again, and verify that no duplicate
    classes, handlers, panels, or properties are registered.

Camera and Depth Camera Projection require an active Perspective or Orthographic camera.
Spine itself is not required to verify installation, registration, Analyze, baking, JSON
creation, transaction cleanup, or extension lifecycle behavior.

## Pre-submission gate

Before every upload candidate:

- require the exact expected Git commit and a clean worktree;
- compile the production package and submission tests;
- run the focused submission/UI/documentation tests;
- run the full Blender-independent test suite;
- run the real bpy suite;
- run representative Blender-headless export gates;
- build the ZIP with Blender 5.2;
- validate the ZIP with Blender;
- inspect the ZIP inventory and confirm excluded legacy/tests/docs are absent;
- install the exact ZIP from disk in a clean Blender profile;
- enable, exercise, disable, restart, re-enable, and uninstall it;
- record the final ZIP SHA256.

Do not upload an archive that was built before the final manifest/test/documentation commit.

## Submission workflow

1. Sign in to the Blender Extensions website with Blender ID.
2. Open the extension submission page.
3. Upload the validated extension ZIP.
4. Complete any listing fields requested by the website using the current metadata above.
5. Add a concise changelog describing the current candidate, not historical internal
   development milestones.
6. Save the draft and verify the generated compatibility, permissions, license, website,
   and tags.
7. Mark the extension ready for review.
8. Monitor the review activity page and respond to moderator feedback on the submitted
   version.

The extension is published only after moderation approval.
