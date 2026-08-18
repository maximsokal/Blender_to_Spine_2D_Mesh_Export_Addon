# Architecture

This document describes the production architecture of **Spine Mesh Exporter 0.155.0**.

## Package boundaries

```text
Blender_to_Spine2D_Mesh_Exporter/
  application/
  blender_adapter/
  domain/
  infrastructure/
```

### `application`

Owns export use cases, immutable settings/results, readiness, geometry-to-attachment
orchestration, composition, texture planning, and document assembly.

Application modules coordinate stages but do not own Blender RNA or low-level Blender
resource lifetime.

### `blender_adapter`

Owns every Blender-facing boundary:

- Scene, object, and Add-on Preferences RNA capture;
- source/evaluated Mesh access;
- temporary Mesh/Object creation;
- UV preparation;
- material graph inspection;
- semantic object baking;
- camera rendering and camera-space projection;
- render/camera/selection/frame restoration;
- UI routing and Scene migration.

Adapters may use `bpy` and `bmesh`. A BMesh created with `bmesh.new()` must be released
exactly once in `finally`. A BMesh borrowed through `bmesh.from_edit_mesh()` must never be
freed by the borrower.

### `domain`

Contains Blender-independent contracts and algorithms:

- source/local geometry identity and lineage;
- segmentation, topology, decomposition, triangulation;
- signed-axis projection bases;
- UV contracts;
- bake/camera/depth planning models;
- Spine bones, constraints, slots, attachments, skins, animation, validation, target
  adaptation, weighted streams, serialization, and exact-version family validation.

Domain modules do not import `bpy` or `bmesh`.

### `infrastructure`

Owns cross-cutting services such as durable atomic output, process-local export ownership,
stale-stage/backup recovery, diagnostics, logging, audit services, and the mixed-resource
registration helpers still required by owners that can partially acquire RNA/handler
resources.

Development-only pipeline-trace implementation files are retained in the repository for
tests/probes but excluded from the extension build and are not part of installed runtime.

## Public request flow

```text
Blender UI
-> capture immutable Scene/object settings
-> resolve schema family + persistent exact project version
-> capability validation
-> optional manual readiness diagnostics OR direct export request
-> object preparation
-> Spine document assembly
-> target-family adaptation
-> texture/JSON staging
-> staged validation
-> atomic commit
```

Later stages consume typed immutable settings rather than repeatedly reading mutable Blender
RNA.

## Spine family versus exact project version

`domain/spine/version_target.py` owns the five supported schema families and their immutable
default exact versions. The family is the codec/capability identity. A canonical exact patch
inside the same family is metadata, not another codec.

`blender_adapter/spine_version_preferences.py` is the only boundary that connects that pure
domain registry to Blender `AddonPreferences`. It owns one persistent exact-version field
for each family and validates that values remain canonical `major.minor.patch` strings inside
the selected family.

`a1_ui_settings.py` resolves the effective exact version once while building immutable
`ExportSettings`. Downstream code receives `ExportSettings.spine_version`; its
`spine_target` property resolves the schema family. Consequently the same effective value
feeds the viewport label, versioned JSON filename and serialized `skeleton.spine`, while the
family continues to choose the serializer codec.

Preference update callbacks invalidate readiness and redraw the UI but never call
`wm.save_userpref`; global Blender preference persistence remains owned by Blender.

## Blender registration ownership

The package root intentionally uses a direct Blender add-on lifecycle instead of a generic
root registration state machine.

Registration order is explicit and unregistration is the reverse order. Individual modules
own their own Blender resources:

- class-only modules use direct `bpy.utils.register_class` / reverse unregister loops;
- Scene RNA registration owns narrow rollback for properties acquired by the current call;
- migration owns its `load_pre` / `load_post` handlers;
- readiness invalidation owns its depsgraph handler and reversible function/method bindings;
- the manual readiness bridge owns only reversible UI method overrides;
- Add-on Preferences owns its one-shot redraw timer and explicitly releases it on teardown;
- mixed class/RNA owners retain local transactional cleanup where partial acquisition can
  otherwise leak Blender resources.

No `ExtensionRegistrationState`, degraded root mode, or `REGISTRATION_STEPS` table is part of
the 0.155.0 runtime.

## UI ownership

`ui.py` owns the canonical main panel and core export/readiness operators. `ui_layout.py`
registers ordinary child panels instead of unregistering/replacing the main panel.

Current visible ownership is:

```text
main panel
├── Paths and Spine 2D version
├── Cut
├── Bake
├── Analysis / Export action
├── Rig child panel
├── Generated Materials child panel
└── Depth Parallax child panel when applicable
```

Control placement is semantic. The existing Scene property `spine2d_texture_size` is drawn
by **Bake** and is not duplicated in **Paths and Spine 2D version**. Exact Spine project
versions are global Add-on Preferences and therefore are not Scene migration data.

## Object preparation

Each source object flows through:

```text
source geometry
-> UV preparation
-> texture planning
-> Spine document preparation
```

Stage errors preserve stage identity, object identity, warnings/statistics, and the original
exception cause.

## Normal / UV Segments

```text
source Mesh
-> source/evaluated capture and lineage
-> segmentation/decomposition
-> generated SpineBakeUV
-> projection into canonical U/V/depth
-> Z-group assignment
-> rig build
-> weighted per-region attachments
-> shared generated-vertex-bone optimization
-> material bake on unprojected source geometry
-> target-specific Spine document
```

Projection geometry and material-bake geometry are intentionally separate. Changing Normal
projection direction changes the Spine representation but does not rotate/reproject the
geometry used to evaluate the source material.

### Signed-axis projection

The six signed-axis modes use deterministic orthonormal bases from
`domain/projection.py`. U/V map to Spine X/Y and the selected depth axis owns generated
Z-group separation.

### Active Camera shared geometry stage

Both Active Camera Normal modes use the same evaluated camera frame and projected snapshot.
Perspective and Orthographic projection are resolved by the Blender camera adapter before
rig selection. The snapshot carries camera-projected U/V, camera-space depth, projected
Blender Object Origin, and source identity/UV lineage.

### Active Camera — Object Root Bone

Persisted projection ID: `ACTIVE_CAMERA`. Setup mode: `CAMERA_VIEW_NORMAL`.

```text
root
└── <prefix>_main                  projected Blender Object Origin
    └── <prefix>
        └── <prefix>_scale_rotate_X
            └── <prefix>_rotate_X
                ├── <prefix>_<z>_scale
                │   └── <prefix>_<z>
                │       └── <prefix>_<z>_camera_setup
                │           └── generated vertex bones
                └── ...
```

The camera-facing setup pose is already solved by projection, so X/Y setup rotation and
depth Transform setup values are neutral. Each `_camera_setup` child applies the inverse of
its depth-group setup translation while live depth remains available to X/Y pseudo-rotation.

### Active Camera — Camera Root Bone

Persisted projection ID: `ACTIVE_CAMERA_CAMERA_ROOT`. The application normalizes geometry
projection back to `ACTIVE_CAMERA` while selecting setup mode `PREPROJECTED_SCREEN`.

```text
root
└── <prefix>_main                  camera-space zero
    └── camera-relative transform/depth layer
        └── <prefix>               projected Blender Object Origin
            └── generated vertex bones
```

All attachment vertices bind through one rigid camera-depth group. This mode reuses the same
camera-projected geometry and material-bake input as Object Root.

## Camera Projection

```text
active camera render tasks
-> alpha coverage union
-> stable crop
-> contour simplification
-> exact triangulation
-> flat screen-space attachment
-> target-specific document
```

Camera Projection is an explicit representation and never silently replaces Normal / UV
Segments.

## Depth Camera Projection

```text
active camera visible surface
-> front-most depth sampling
-> edge-aware depth smoothing
-> bounded relief topology
-> depth-group/vertex-bone generation
-> FRONT render/crop
-> optional reserve-view renders/crops
-> weighted attachments
-> target-specific document
```

The public relief base is Farthest Visible Point. Positive Parallax Horizon Angle can retain
connected reserve surfaces. Temporary virtual cameras/render proxies are isolated Blender
resources and are removed on success and failure.

## Attachment projection

Blender UV identity belongs to loops. The attachment projector therefore preserves
`(SourceVertexId, UV)` identity instead of assuming one UV per geometric vertex. Final mesh
attachments preserve UV-specific vertices, triangulation corner order, physical Spine hull
semantics, explicit depth binding, and deterministic generated vertex-bone ownership.

## Shared generated vertex bones

Segmentation can duplicate the same source point across attachments. The optimizer shares
only generated component vertex bones whose final setup semantics are identical. Parent
identity is part of the key, so different depth groups never collapse together. Only
weighted bone indices are remapped.

## Target-specific Spine adaptation

Canonical rig/document construction happens before target adaptation. The selected schema
family then chooses its production codec. Supported standalone families are 3.8, 4.0, 4.1,
4.2, and 4.3; their built-in exact defaults are 3.8.99, 4.0.64, 4.1.24, 4.2.43, and 4.3.23.
A user-selected same-family exact patch changes emitted project metadata, not codec topology.
Unsupported scope/profile/family combinations fail before expensive work.

## Multi-object composition

Public selected-object export creates standalone composition. Internal connected and mixed
composition routes are explicit and capability-gated. Each object owns preparation and
vertex-bone optimization before outer composition; generated bones are never shared across
unrelated object boundaries. The outer request owns one atomic transaction for final JSON
and every texture.

## Readiness

Readiness is an explicit synchronous diagnostic route. **Analyze** executes production
preparation without final commit and records immutable blockers/warnings/statistics. Relevant
source/settings changes stale or invalidate the cached report. Exact-version preference edits
explicitly invalidate all open Scene readiness caches.

The installed runtime does not schedule automatic readiness analysis from a Python worker,
Blender polling timer, depsgraph callback, or load callback. A current report is not required
to invoke production Export.

## Atomic output and portability

Output files are reserved and staged before installation. Existing finals may be protected
with backups. Required properties are deterministic path reservation, complete staged files,
rollback/restoration, stale work recovery, and no deletion of work owned by another live
process.

The runtime has no declared OS restriction. Windows-specific process/path compatibility is
guarded by host checks; POSIX process liveness uses the corresponding POSIX path. Durable
filesystem behavior uses portable Python primitives with documented host limitations handled
inside the atomic layer rather than by excluding other platforms in the manifest.

## Source-state integrity

Production export must not permanently change source topology, UVs, materials, transforms,
active object, selection, mode, renderer, frame, active camera, View Layer, or visibility
state outside the intended transaction. Temporary Blender datablocks are removed on success
and failure paths.

## Related documents

- [Usage](usage.md)
- [Settings Reference](settings-reference.md)
- [Rig Profiles](rig-profiles.md)
- [Output Format](output-format.md)
- [Testing](testing.md)
