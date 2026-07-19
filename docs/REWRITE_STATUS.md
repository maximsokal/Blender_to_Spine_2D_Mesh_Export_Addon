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
- incremental `DiskRegionState` growth;
- incremental frontier and merge adjacency updates;
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
    |     -> straight-RGBA composition
    |
    +-- CAMERA / VOLUME / render displacement
          -> B4 camera projection
```

Recursive group analysis supports renderer-specific outputs, muted bypasses,
nested groups, instance-qualified IDs, cycles, bounded depth and no source-node
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

`bake_executor_core.py` no longer contains a duplicate transaction or frame/pass
pipeline. It owns only the direct `bpy.ops.object.bake` hook and compatibility
private re-exports.

## B4 execution ownership

The single-object B4 runtime now has one physical pipeline:

```text
camera_projection_validation.py
  -> complete request and reservation validation
  -> CameraProjectionRuntime

camera_projection_state.py
  -> reversible Scene, frame and camera-visibility mutation

camera_projection_execution.py
  -> full-frame rendering only

camera_projection_postprocess.py
  -> coverage union, cleanup, layout and crop rewrite

camera_projection_output.py
  -> reservation, atomic commit and typed results
```

`camera_projection_executor_core.py` is compatibility-only.
`camera_projection_executor.py` remains the stable public facade.

B4 request validation completes before reservation. Direct execution validates
before transaction creation, commits exactly once and requires exact
reservation/frame-task path order.

Coverage decode, contour construction and crop rewrite begin only after the
reversible Blender render scope has restored Scene, frame and visibility state.

The historical compatibility staging API still writes full-frame textures and
does not decode coverage or crop images.

## B4 production pipeline

Every detailed B4 static or sequence export:

1. validates object, Scene, World, camera, light, renderer and output policy;
2. reserves outputs in immutable frame-task order;
3. captures frame, render and camera-visibility state;
4. isolates only direct camera visibility while preserving dependency rays;
5. disables Scene Compositor and Sequencer execution without mutating data;
6. renders every transparent full-frame staged image;
7. restores Blender state;
8. decodes deterministic 8-bit alpha coverage;
9. max-unions coverage across the sequence;
10. applies hysteresis and conservative morphology;
11. derives one stable padded crop;
12. traces a simplified concave contour or safe convex fallback;
13. triangulates the simple contour exactly;
14. applies HDR/tone-mapping/alpha policy during crop rewrite;
15. rebuilds typed Spine attachments/documents;
16. commits JSON and textures together.

## Simplified concave screen-space contour

Production defaults to `ProjectionContourMode.SIMPLIFIED_CONCAVE`.

- one outer component becomes a simple concave contour;
- internal holes remain texture alpha;
- disconnected outer components use deterministic convex fallback;
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

## Grouped connected B4

`ConnectedB4RenderPolicy` supports:

- `INDIVIDUAL_LAYERS`;
- `AUTO_GROUPED_CAMERA`;
- `GROUPED_CAMERA_REQUIRED`.

A compatible connected set may be rendered together for real per-pixel depth.
The grouped executor now imports shared B4 error, validation, render-hook and
state helpers from their physical modules. Its grouped visibility, coverage and
output orchestration remains a separate future decomposition slice.

## HDR, tone mapping and alpha

```text
PNG / WEBP -> display-referred SDR -> Scene view transform
            -> straight alpha -> 8-bit

OPEN_EXR   -> scene-linear HDR -> no tone mapping
            -> premultiplied alpha -> 32-bit float
```

Invalid combinations fail before render. Crop rewrite performs explicit
straight/premultiplied conversion, normalizes zero-alpha RGB and does not clamp
finite HDR RGB.

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

For the newest B4 decomposition:

- ten new or replaced production modules compile;
- focused source architecture tests pass;
- validation/execution/postprocess/output boundaries are checked;
- render state restoration precedes postprocessing;
- validation precedes reservation and transaction creation;
- direct B4 execution contains exactly one commit;
- compatibility aliases remain;
- GitHub Actions remain disabled/manual-only.

The complete pytest suite and Blender matrices have not been rerun on the
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
