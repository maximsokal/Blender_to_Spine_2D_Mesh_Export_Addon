# Rewrite status

The active rewrite branch is `rewrite/a1-domain-foundation`; A1 compatibility targets
Spine 4.2.43.

Implemented production areas now include:

- deterministic geometry and loop-level UV lineage;
- legacy-compatible and local-dihedral A1 segmentation modes;
- incremental disk-region topology growth and merging;
- semantic local, alpha, scene and camera texture strategies;
- recursive reachable Shader Node Group analysis;
- stable sequence-union crop;
- configurable B4 alpha threshold;
- simplified concave screen-space contours with deterministic triangulation;
- typed Spine composition for single, standalone multi, connected multi and mixed export;
- one atomic JSON plus texture transaction;
- both public production operators.

## Production operators

```text
object.save_uv_as_json
object.spine2d_multi_export
```

Rewrite remains the default backend. Legacy remains explicitly selectable and is never an
automatic fallback. The add-on version is unchanged and no release package has been
produced from this branch.

Output naming remains compatible:

```text
<object>_merged.json
<object>_Baked.png
<object>_Baked_0000.png ...
```

## Geometry and topology

`LEGACY_SEED_CONE` preserves the previous deterministic grouping behavior.
`SEED_CONE_AND_LOCAL_DIHEDRAL` adds an explicit local shared-edge angle guard.

Disk decomposition uses:

- one immutable `DiskTopologyIndex` per mesh snapshot;
- incremental `DiskRegionState` updates;
- an incremental growth frontier;
- locally updated merge adjacency counts;
- complete `analyse_face_region()` scans only as independent input/final invariants.

## Automatic texture pipeline

```text
active connected shader graph
        -> recursive reachable group expansion
        -> semantic channels/dependencies
        -> ObjectBakeContext + SceneBakeContext
        -> build_texture_plan()
        |
        +-- LOCAL / SCENE / AUXILIARY
        |       -> BakePlan
        |       -> DIFFUSE / EMIT / COMBINED object-bake passes
        |       -> straight-RGBA composition
        |
        +-- CAMERA / VOLUME / render displacement
                -> CameraProjectionPlan
                -> active-camera transparent renders
                -> sequence alpha union
                -> stable crop
                -> simplified concave contour or safe convex fallback
                -> exact triangulation
                -> post-render typed document finalization
```

Planner selection remains automatic. No per-material mode switch was added to the UI.

## B1: local surface and Emission

Implemented:

- ordinary Principled, Image Texture and procedural surface color through
  lighting-independent `DIFFUSE Color`;
- material Emission through `EMIT`;
- independent surface and Emission slots;
- Base Color and Emission inside one Principled material;
- float-buffer composition;
- protection against counting Emission twice.

```text
final.rgb   = clamp(surface.rgb + emission.rgb)
final.alpha = max(surface.alpha, emission.alpha)
```

## B2: alpha and straight color

Alpha-bearing materials evaluate color and opacity independently:

```text
straight surface color -> temporary Emission proxy -> EMIT
material opacity       -> grayscale Emission proxy -> EMIT
material emission      -> native EMIT when present
```

Temporary proxy nodes exist only on copied materials and are removed in `finally`.
Source material node trees are not mutated.

## B3: scene-aware object baking

Evaluation scopes:

- `LOCAL`: albedo, image/procedural color and material Emission;
- `SCENE`: World, lighting, occlusion and scene-object dependencies;
- `AUXILIARY`: alpha and composition-only channels;
- `CAMERA`: B4 camera projection.

Mixed local and scene-aware material slots use reversible black masks during each pass to
prevent double counting.

## Recursive Shader Node Groups

The graph analyzer follows only sockets that contribute to the renderer-effective Material
Output. It supports:

- Cycles, Eevee and generic output targets;
- muted-node and muted-group bypass mappings;
- stable group-instance-qualified IDs;
- nested Image, Time, Camera, View, Reflection, Transmission and Volume requirements;
- nested animation detection;
- identifier/name/index socket matching;
- recursive-cycle detection;
- a maximum traversal depth of 64;
- no mutation of source node groups.

Unused group inputs do not leak camera dependencies.

## B4 camera projection

B4 is selected for Camera/View, Reflection/Transmission, Volume and render-evaluated
displacement. This includes Fresnel, Layer Weight, Light Path, Glass, Refraction,
Principled Transmission, reflective appearance and Principled Volume.

For every static or sequence frame B4:

1. validates immutable object, Scene, World, camera and light identities;
2. captures render, frame and visibility state;
3. keeps only the source directly camera-visible while retaining other objects for
   reflection, transmission, diffuse, occlusion and shadow rays;
4. renders a transparent full frame to an atomic staged path;
5. decodes the actual staged image;
6. extracts alpha using `BakeExecutionSettings.projection_alpha_threshold`;
7. merges the frame into one fixed-size sequence-union mask;
8. expands the crop using the existing `bake_margin`;
9. builds the configured screen-space contour;
10. triangulates it exactly;
11. rewrites every staged frame to the same crop dimensions;
12. rebuilds typed attachments/documents and commits all files together;
13. restores Blender state in `finally`.

### Configurable alpha threshold

The compatibility default remains exactly `1 / 255`. Finite values in `[0, 1]` are
accepted. Booleans, non-numeric values, NaN, infinities and out-of-range values are
rejected before rendering.

### Simplified concave contour

The production default is:

```python
ProjectionContourMode.SIMPLIFIED_CONCAVE
```

The pipeline traces oriented pixel-boundary loops, removes exact collinear points and
conservatively fills only shallow reflex notches. Convex corners are never removed, so
visible alpha coverage is not clipped.

One connected outer component receives a simple concave contour. Internal holes remain
texture alpha. Several disconnected outer components use a deterministic convex fallback
instead of an artificial bridge.

Concave contours use deterministic ear clipping. Validation requires:

```text
triangle count = contour vertex count - 2
all triangle signed areas > 0
sum(triangle signed areas) = contour signed area
```

`CameraProjectionLayout.hull` remains the compatibility field name; new code may use
`layout.contour`.

See `docs/REWRITE_B4_CONCAVE_CONTOUR.md`.

## Executor boundaries

- `bake_executor_core.py`: real object-bake implementation;
- `semantic_bake_executor.py`: B1-B3 execution and composition;
- `camera_projection_state.py`: reversible render/visibility/frame state;
- `camera_projection_image.py`: staged-image decode, alpha mask and crop rewrite;
- `camera_projection_executor_core.py`: B4 orchestration;
- `projection_contour.py`: pure contour extraction, simplification and triangulation;
- `projection_layout.py`: crop, sequence union and immutable layout contract;
- `texture_executor.py`: typed plan dispatch;
- `bake_executor.py`: stable operator-hook facade.

## Validation state

The last complete automatic matrix before CI was switched to manual-only passed:

- Python 3.10: **484 passed, 4 skipped**;
- Python 3.11: **484 passed, 4 skipped**;
- Blender 4.4 Alpha Bake: success;
- Blender 4.4 Scene Bake: success;
- Blender 4.4 Camera Projection: success;
- full Blender 4.4 Headless: success.

The concave-contour slice adds focused coverage for:

- exact L-shaped concavity;
- shallow-notch simplification;
- deep-concavity preservation;
- holes represented by alpha;
- diagonal contacts and disconnected fallback;
- explicit convex compatibility mode;
- exact-area ear clipping;
- arbitrary triangulation edge topology;
- complete concave `MeshSnapshot` construction;
- 250 deterministic randomized binary masks.

The changed pure-domain source was syntax-checked and the contour algorithms were exercised
locally. The full repository pytest and Blender headless matrices have not been rerun for this
new HEAD.

Automatic workflow triggers remain disabled on this branch. Before merge, restore the
original triggers and run the complete matrix once on the final candidate head.

## Ordered remaining work

1. coverage-weighted antialias and morphology cleanup around partially transparent edges;
2. depth-aware or grouped rendering for intersecting connected B4 objects;
3. a separate real Eevee and custom Compositor matrix;
4. HDR, tone-mapping and premultiplied-alpha output policy;
5. private production `.blend` parity and the release gate.

## Release blockers

- representative private production `.blend` fixtures with accepted v0.23 JSON/images;
- accepted JSON and decoded-image parity reports;
- a connected multi-object depth policy where production fixtures require one;
- Eevee/Compositor support only where production files depend on it;
- controlled Legacy removal after private parity acceptance;
- version bump and release packaging only after the parity gate is accepted.

See also:

- `docs/REWRITE_CAMERA_PROJECTION.md`;
- `docs/REWRITE_B4_ALPHA_THRESHOLD.md`;
- `docs/REWRITE_B4_CONCAVE_CONTOUR.md`;
- `docs/REWRITE_BAKE_STRATEGIES.md`;
- `docs/REWRITE_A1_GOLDEN_PARITY.md`;
- `docs/REWRITE_CI_MANUAL_MODE.md`.
