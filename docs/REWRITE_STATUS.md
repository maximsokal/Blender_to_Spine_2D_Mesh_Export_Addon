# Rewrite status

The active branch is `rewrite/a1-domain-foundation`; A1 targets Spine 4.2.43.
Rewrite remains the default backend. Legacy remains explicitly selectable and
is never an automatic fallback. The add-on version is unchanged and no release
package has been produced.

## Production operators

```text
object.save_uv_as_json
object.spine2d_multi_export
```

Single, standalone multi, connected multi and mixed exports use one atomic JSON
plus texture transaction.

## Geometry and topology

Implemented:

- deterministic geometry and loop-level UV lineage;
- `LEGACY_SEED_CONE` compatibility behavior;
- `SEED_CONE_AND_LOCAL_DIHEDRAL` shared-edge guard;
- one immutable `DiskTopologyIndex` per mesh snapshot;
- incremental `DiskRegionState`, frontier and merge adjacency;
- complete topology analysis only as input/final invariants.

## Automatic material pipeline

```text
renderer-effective shader graph
    -> recursive Shader Node Group expansion
    -> semantic channels and dependencies
    -> ObjectBakeContext + SceneBakeContext
    -> automatic strategy selection
    |
    +-- LOCAL / AUXILIARY / SCENE
    |     -> DIFFUSE / EMIT / COMBINED object bake
    |
    +-- CAMERA / VOLUME / render displacement
          -> B4 camera projection
```

Recursive analysis supports renderer-specific outputs, muted bypasses, nested
groups, instance-qualified IDs, cycles, bounded depth and no source-node
mutation.

## Semantic object-bake ownership

```text
semantic_bake_validation.py
  -> request, Blender context and reservation validation

semantic_bake_image_io.py
  -> UV, image datablock, frame and staged-image primitives

semantic_bake_execution.py
  -> reversible Scene/Mesh/material execution and composition

semantic_bake_output.py
  -> atomic reservation, commit, rollback and typed results
```

`bake_executor_core.py` owns only the direct `bpy.ops.object.bake` hook and
compatibility private re-exports.

## Single B4 ownership

```text
camera_projection_validation.py
  -> complete request and reservation validation
  -> CameraProjectionRuntime

camera_projection_state.py
  -> reversible Scene, frame and camera-visibility mutation

camera_projection_execution.py
  -> full-frame rendering only

camera_projection_postprocess.py
  -> shared single/grouped coverage, layout and crop engine

camera_projection_output.py
  -> single reservation, direct commit and typed results
```

`camera_projection_executor_core.py` is compatibility-only.
`camera_projection_executor.py` remains the stable public facade.

Single B4 validation completes before reservation. Direct execution validates
before transaction creation, commits exactly once and requires exact
reservation/frame-task order.

## Grouped B4 ownership

Grouped connected B4 is now physically decomposed:

```text
grouped_camera_projection_validation.py
  -> GroupedCameraProjectionRuntime
  -> source/RNA identity, common Scene/renderer/output policy
  -> strict grouped reservation validation

grouped_camera_projection_visibility.py
  -> grouped source visibility and direct-camera isolation

grouped_camera_projection_execution.py
  -> grouped full-frame rendering inside reversible state only

grouped_camera_projection_postprocess.py
  -> grouped adapter and diagnostics for the shared postprocess engine

grouped_camera_projection_output.py
  -> validation-before-reservation and caller-owned staging
  -> no transaction creation and no commit

grouped_camera_projection_executor.py
  -> compatibility re-exports only
```

Multi and mixed output import the physical grouped output owner. Their outer
transaction remains the sole owner of JSON, individual texture and grouped
texture commit order.

## B4 production ordering

Every detailed single or grouped B4 static/sequence export:

1. validates object(s), Scene, World, camera, lights and View Layer;
2. resolves renderer and output policy before reservation;
3. reserves outputs in immutable frame-task order;
4. captures frame, render and camera-visibility state;
5. isolates only direct camera visibility while preserving dependency rays;
6. disables Compositor and Sequencer execution without mutating data;
7. renders every transparent full-frame staged image;
8. restores Blender state;
9. decodes deterministic 8-bit alpha coverage;
10. max-unions coverage across the sequence;
11. applies hysteresis and conservative morphology;
12. derives one stable padded crop;
13. traces a simplified concave contour or safe convex fallback;
14. triangulates exactly;
15. applies HDR/tone-mapping/alpha policy during crop rewrite;
16. rebuilds typed Spine attachments/documents;
17. commits JSON and textures through the owning output transaction.

Coverage, contour and crop failures occur after original Blender state has been
restored.

## Shared B4 postprocess

`ProjectionPostprocessRequest` and `process_projection_outputs()` are shared by
single and grouped B4.

The shared engine owns one `O(width * height)` sequence accumulator and one
implementation of:

- alpha decode;
- coverage cleanup;
- stable crop;
- simplified concave contour;
- disconnected-component convex fallback;
- exact triangulation;
- straight/premultiplied and SDR/HDR image rewrite.

The crop writer explicitly accepts both `CameraProjectionPlan` and
`GroupedCameraProjectionPlan`.

## Simplified concave screen-space contour

Production defaults to `ProjectionContourMode.SIMPLIFIED_CONCAVE`.

- one outer component becomes a simple concave contour;
- holes remain texture alpha;
- disconnected components use deterministic convex fallback;
- exact collinear vertices are removed;
- only shallow reflex notches may be filled;
- convex corners are never removed;
- concave contours use deterministic ear clipping;
- triangle count, orientation and exact total area are validated.

## Coverage-weighted antialias and morphology

Production uses `HYSTERESIS_MORPHOLOGY`:

- weak threshold defaults to `1 / 255`;
- strong threshold defaults to `0.5`;
- weak coverage is retained only when connected to a strong core;
- translucent-only objects use explicit weak-only fallback;
- foreground components use 8-connectivity;
- tiny detached components are removed while the largest remains;
- only bounded enclosed pinholes are filled;
- no generic closing can bridge separate objects.

## Grouped connected B4 policy

`ConnectedB4RenderPolicy` supports:

- `INDIVIDUAL_LAYERS`;
- `AUTO_GROUPED_CAMERA`;
- `GROUPED_CAMERA_REQUIRED`.

A compatible connected set may be rendered together for real per-pixel depth.
Grouped source slots become transparent and one root-bound grouped attachment
becomes visible. Mixed local/B4 connected sets remain individual layers unless a
future policy solves depth across both coordinate spaces.

Grouped staging never creates or commits a transaction. Multi/mixed output
commits JSON, individual textures and grouped textures once and verifies exact
combined reservation order.

## HDR, tone mapping and alpha

```text
PNG / WEBP -> display-referred SDR -> Scene view transform
            -> straight alpha -> 8-bit

OPEN_EXR   -> scene-linear HDR -> no tone mapping
            -> premultiplied alpha -> 32-bit float
```

Invalid combinations fail before reservation and render. Crop rewrite performs
explicit alpha conversion, normalizes zero-alpha RGB and does not clamp finite
HDR RGB.

## Private production parity and release gate

Implemented:

- typed private manifest and capability coverage;
- protected Blender 4.4 self-hosted workflow;
- exact candidate SHA and Blender version checks;
- production operator invocation;
- source-file and in-memory mutation detection;
- temporary datablock leak detection;
- semantic Legacy/rewrite JSON comparison;
- warning/suppression validation;
- Blender-decoded PNG/WEBP/OPEN_EXR pixel parity;
- private report retention without public fixture upload.

## Validation state

The last complete automatic matrix before workflows became manual-only passed:

- Python 3.10: **484 passed, 4 skipped**;
- Python 3.11: **484 passed, 4 skipped**;
- Blender 4.4 Alpha Bake: success;
- Blender 4.4 Scene Bake: success;
- Blender 4.4 Camera Projection: success;
- full Blender 4.4 Headless: success.

For the newest grouped B4 decomposition:

- all new/replaced production modules compile;
- grouped source ownership/order tests pass;
- import graph loads with Blender/domain stubs;
- compatibility facade aliases resolve to physical owners;
- validation and output policy precede reservation;
- reversible render completes before postprocessing;
- grouped output contains no transaction creation or commit;
- production callers use the physical grouped output owner;
- single-B4 threshold, coverage and contour inspection contracts remain;
- GitHub Actions remain disabled/manual-only.

The complete pytest suite and real Blender matrices have not been rerun on the
current HEAD.

## Remaining release blockers

Release remains blocked until the same candidate SHA passes:

1. complete public Python tests;
2. all manual Blender 4.4 matrices;
3. protected private production `.blend` gate;
4. review of retained private reports and warnings;
5. restoration of intended final CI triggers;
6. explicit approval for Legacy removal, version bump and packaging.

The branch and PR remain draft until those gates pass.
