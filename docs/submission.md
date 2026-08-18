# Blender Extensions Platform Submission

This document describes the moderation-remediation candidate for **Spine Mesh Exporter
0.155.0**.

This release updates the **existing Blender Extensions submission** that received moderator
feedback. Do not create another extension listing, do not delete/recreate the declined
listing, and do not submit 0.155.0 as a new initial extension. Upload the corrected archive
as a new higher version of the same existing submission.

## Upload artifact

Build and upload only this extension archive:

```text
dist/blender_to_spine2d_mesh_exporter-0.155.0.zip
```

Do not upload the repository ZIP. The extension archive must be produced by Blender's
`extension build` command and must validate with `extension validate`.

The repository website declared by the manifest must expose the same candidate as the
uploaded archive. Before uploading the corrected version, fast-forward the repository
default branch to the exact validated candidate commit. Do not leave the public website
pointing at superseded source or documentation.

## Current public metadata

```text
Name: Spine Mesh Exporter
Version: 0.155.0
Minimum Blender: 5.2.0
Platform restriction: none declared
License: GPL-3.0-or-later
Website: https://github.com/maximsokal/Blender_to_Spine_2D_Mesh_Export_Addon
Tags: Import-Export
Permission: Files
```

The technical manifest ID remains:

```text
blender_to_spine2d_mesh_exporter
```

Changing the public display title must not create a new technical extension identity.

The Files permission is required because the extension writes Spine JSON, textures,
diagnostics, transaction staging files, backups, and optional logs to user-selected or
user-owned filesystem locations.

The installed runtime is pure Python and the manifest no longer declares a Windows-only
platform restriction. Platform-specific code in the atomic-output layer is guarded by host
OS checks and provides compatibility behavior rather than a Windows-only dependency. Exact
package validation remains a release gate.

## Store description

Suggested short description:

> Convert Blender Mesh objects into Spine 2D JSON, weighted mesh attachments, baked or
> camera-rendered textures, generated animation controls, and optional texture sequences.

Suggested reviewer-oriented description:

> Spine Mesh Exporter converts Blender Mesh objects into Spine-ready JSON and textures.
> It supports Normal / UV Segments with six signed-axis projections and two Active Camera
> rig roots, flat Camera Projection, Depth Camera Projection, 2-Axis Rotation + Scale
> controls, generated bake UVs, automatic/custom seam segmentation, texture sequences,
> multi-object standalone export, manual readiness diagnostics, atomic output rollback, and
> persistent per-family exact Spine project-version preferences. It requires Blender 5.2 or
> newer, performs no network requests, declares only filesystem access, and does not declare
> an operating-system platform restriction.

## Moderator-remediation summary

The 0.155.0 archive must provide evidence for the seven current review items:

1. development-only `PipelineTraceSession` sources are excluded from the distributed ZIP;
2. shipped runtime imports no `threading`, `queue`, `multiprocessing`, or `concurrent`
   background-concurrency surface, and the old automatic readiness scheduler is removed;
3. the Re-Polish advertisement/operator/runtime module is removed;
4. root/UI registration uses normal Blender ownership without the rejected root state
   machine or main-panel replacement dance;
5. the manifest contains only the `Import-Export` tag;
6. the unjustified Windows-only manifest restriction is removed after runtime portability
   audit;
7. the public title is `Spine Mesh Exporter` while the technical ID stays unchanged, and
   the corrected archive is uploaded to the same existing submission.

## Reviewer test path

1. Install the ZIP through **Edit > Preferences > Extensions > Install from Disk**.
2. Enable **Spine Mesh Exporter**.
3. In the add-on Preferences, confirm one exact project-version field exists for each Spine
   family 3.8, 4.0, 4.1, 4.2, and 4.3.
4. Save a `.blend` file containing one Mesh object with a material.
5. Select the Mesh in Object Mode.
6. Open the 3D View Sidebar and select the extension tab.
7. Keep **Normal / UV Segments**, `+Z`, Spine 4.2, and a small Texture size such as 256.
8. Confirm **Exact JSON version** matches the configured Spine 4.2 preference.
9. Choose a writable JSON output directory.
10. Optionally run **Analyze** to obtain diagnostics. Analyze is manual and synchronous;
    Export remains available even when no current analysis exists.
11. Run **Export Current Object**.
12. Confirm that the versioned JSON filename, serialized `skeleton.spine`, and `images/`
    texture output use the configured exact project patch.
13. Disable the extension, restart Blender, enable it again, and verify that no duplicate
    classes, handlers, panels, properties, timers, or method overrides remain and that saved
    Add-on Preferences persist.

Camera and Depth Camera Projection require an active Perspective or Orthographic camera.
Spine itself is not required to verify installation, registration, Analyze, baking, JSON
creation, transaction cleanup, preference persistence, or extension lifecycle behavior.

## Pre-upload gate

Before uploading every corrected candidate:

- require the exact expected Git commit and a clean worktree;
- compile the production package and submission tests;
- run the focused moderation-compliance, lifecycle, UI, documentation, and exact-version
  tests;
- run the full Blender-independent test suite;
- run the real bpy suite;
- run representative Blender-headless export gates;
- run the isolated installed-extension preference save/restart gate and five real custom
  exact-version exports;
- build the ZIP with Blender 5.2 using `extension build`;
- validate the exact ZIP with Blender using `extension validate`;
- inspect the ZIP inventory and confirm excluded legacy/tests/docs/development trace sources
  are absent;
- scan the exact ZIP for forbidden concurrency and Re-Polish references;
- install the exact ZIP from disk in a clean Blender profile;
- enable, exercise, disable, restart, re-enable, and uninstall it;
- verify no stale handlers/timers/RNA/classes/method overrides remain after disable;
- fast-forward the public default branch to the exact validated commit;
- confirm the manifest website shows the same 0.155.0 source/documentation;
- record the final ZIP byte size and SHA256.

Do not upload an archive that was built before the final manifest/test/documentation commit.

## Same-submission update workflow

1. Sign in to the Blender Extensions website with the Blender ID that owns the existing
   submission.
2. Open the **existing declined/reviewed Spine Mesh Exporter submission**.
3. Use that listing's version/update workflow to upload
   `blender_to_spine2d_mesh_exporter-0.155.0.zip`.
4. Do **not** create a second listing or a new initial submission.
5. Do **not** delete and recreate the existing listing to bypass the review state.
6. Verify the generated metadata shows `Spine Mesh Exporter`, version `0.155.0`, technical
   ID `blender_to_spine2d_mesh_exporter`, only `Import-Export`, and no unintended platform
   restriction.
7. Add a concise changelog mapping 0.155.0 to the moderator feedback rather than unrelated
   historical development milestones.
8. Save the updated version and reply to the existing moderation thread with the exact
   commit/ZIP evidence requested by the reviewer.
9. Monitor the same submission's review activity and respond there if another correction is
   requested.

The corrected version is published only after moderation approval of that same submission.
