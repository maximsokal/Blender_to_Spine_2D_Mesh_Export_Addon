# Rewrite status

The active rewrite branch is `rewrite/a1-domain-foundation`; A1 compatibility targets
Spine 4.2.43. Deterministic geometry, loop-level UV lineage, semantic local/alpha/scene
baking, camera-render projection, typed Spine composition, connected `all_objects`, and both
production operators are implemented.

## Production operators

- `object.save_uv_as_json` keeps its public ID and uses Rewrite by default;
- `object.spine2d_multi_export` keeps its public ID and uses Rewrite by default;
- Legacy remains explicitly selectable and is never an automatic fallback;
- the add-on version is unchanged;
- no release package has been produced from the rewrite branch.

Single-object output preserves existing naming:

```text
<object>_merged.json
<object>_Baked.png
<object>_Baked_0000.png ...
```

Standalone, connected, and mixed Connect-flag selection flows remain available for the
existing UV-bake pipeline. JSON and every static/sequence texture share one atomic
transaction.

## Automatic texture pipeline

```text
active connected shader graph
        -> semantic channels/dependencies
        -> ObjectBakeContext + SceneBakeContext
        -> evaluation scope per material slot
        -> build_texture_plan()
        |
        +-- LOCAL / SCENE / AUXILIARY
        |       -> BakePlan
        |       -> DIFFUSE / EMIT / COMBINED object bake passes
        |       -> straight-RGBA composition
        |
        +-- CAMERA / VOLUME / render displacement
                -> CameraProjectionPlan
                -> active-camera transparent render
                -> full-frame Spine quad
```

Planner selection is automatic. Per-material mode switches were not added to the UI.

## B1: local surface and emission

Implemented:

- ordinary opaque Principled, Image Texture and procedural color through lighting-independent
  `DIFFUSE Color`;
- material Emission through `EMIT`;
- separate surface and Emission slots on one mesh;
- one Principled material containing Base Color and Emission;
- float-buffer composition;
- normalization of a surface `COMBINED` request to `DIFFUSE` when a separate Emission pass
  would otherwise be counted twice.

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

```text
final.rgb   = clamp(sum(color passes))
final.alpha = alpha_pass.red
```

Recursive copied-material extraction supports:

- Principled Base Color and Alpha;
- linked Image Color/Alpha and procedural socket graphs;
- Transparent BSDF and Holdout;
- Mix Shader in both transparent orders;
- nested Mix and Add Shader trees;
- animated socket values and drivers;
- opaque plus transparent slots;
- pure transparent output.

Temporary proxy nodes exist only on copied materials and are removed in `finally`. Source
material node trees are not mutated.

## B3: scene-aware object baking

Evaluation scopes:

- `LOCAL`: albedo, Image/procedural surface color and material Emission;
- `SCENE`: World, lighting, occlusion and other scene-object dependencies;
- `AUXILIARY`: explicit alpha and future composition-only channels;
- `CAMERA`: routed to B4 instead of object baking.

`ObjectBakeContext` records source identity, transform, collections, visibility and
animation. `SceneBakeContext` records:

- Scene and render engine;
- World identity/nodes/background strength;
- lights, energy, color, transform, shadows and animation;
- active camera identity and projection values;
- visible objects and shadow casters;
- color-management settings;
- analysis frame.

`SceneCombinedBakeStrategy` automatically selects `COMBINED` for explicit scene-dependent
appearance such as Subsurface, Sheen, Toon, Translucent, Hair and Ambient Occlusion.
Ordinary Principled Base Color remains local even when a file happens to contain lights.

One mesh may contain local and scene-aware slots. During each pass, unmatched copied
material slots are replaced by black Emission proxies to prevent double counting.

The live source is temporarily excluded while a same-position bake target is evaluated.
Original visibility, context, frame, scene settings and temporary datablocks are restored on
success and failure.

## B4: camera-render projection

Implemented files:

```text
domain/baking/camera_projection.py
application/a1_camera_projection.py
blender_adapter/camera_projection_executor.py
blender_adapter/texture_executor.py
tests/blender_headless/run_camera_projection_integration.py
.github/workflows/blender-camera-projection.yml
```

B4 is selected automatically when a used material requires:

- Camera or View dependency;
- Reflection or Transmission dependency;
- Volume output;
- render-evaluated displacement.

Covered graph families include:

- Fresnel and Layer Weight;
- Light Path;
- Glass and Refraction;
- Principled Transmission;
- camera-preserving reflection;
- Principled Volume.

`CameraProjectionPlan` is a frozen `BakePlan` subtype so existing frame, naming, attachment
path, sequence and transaction consumers remain compatible. Its synthetic
`CAMERA_COMBINED` pass is metadata only and is never sent to `bpy.ops.object.bake`.

Runtime behavior:

1. validate source, Scene, World, active camera and light identities;
2. capture render settings, timeline frame, `hide_render` and `visible_camera`;
3. show the exported source to direct camera rays;
4. hide other renderable objects only from direct camera rays;
5. keep their reflection, refraction, diffuse and shadow participation;
6. render with transparent film to a staged transaction path;
7. validate the staged image;
8. restore every captured value in `finally`;
9. commit JSON plus all frames together.

The Spine document receives one deterministic full-frame mesh:

```text
4 vertices
5 topology edges
2 triangles
4 hull vertices
UV 0..1
attachment width/height = rendered image width/height
```

The full-frame geometry is stable across camera/object animation and avoids clipping later
sequence frames.

See `docs/REWRITE_CAMERA_PROJECTION.md` for the complete B4 contract.

## Executor boundaries

- `bake_executor_core.py`: object-bake validation, temporary resources and the sole real
  `bpy.ops.object.bake` implementation;
- `semantic_bake_executor.py`: B1-B3 passes, material preparation and composition;
- `camera_projection_executor.py`: B4 render transaction and reversible visibility state;
- `texture_executor.py`: plan-type dispatch with no operator access;
- `bake_executor.py`: stable public facade containing only object-bake and render
  failure-injection hooks.

Architecture tests verify that helper modules contain no direct `bpy.ops` access.

## Real Blender compatibility matrix

Every output image is decoded; a PNG signature alone is not accepted.

Existing B1-B3 coverage includes:

- multiple Principled slots with distinct colors;
- generated Image Texture and procedural Checker;
- surface plus Emission in separate and shared materials;
- constant and linked-image Alpha;
- Transparent/Mix Shader in both orders;
- nested transparency and pure Transparent;
- animated Alpha;
- scene `COMBINED` responding to light energy;
- World illumination changes;
- Ambient Occlusion responding to another object;
- mixed local and scene-aware slots;
- scene-dependent straight RGBA;
- animated lights;
- sequence/alpha/scene rollback;
- source state and temporary datablock restoration;
- connected multi-object JSON and texture transactions;
- registered operator and isolated Legacy/Rewrite workflows.

Dedicated B4 coverage includes:

- production Layer Weight/Fresnel planning and export;
- transparent background plus visible source pixels;
- one full-frame Spine quad with 8 UV values, 6 triangle indices and hull 4;
- Glass render projection;
- Principled Volume render projection;
- animated camera-dependent sequence frames;
- restoration of the original timeline frame;
- forced render failure after previous JSON/PNG bytes exist;
- atomic rollback of JSON and texture;
- restoration of render settings, context, material graphs, `hide_render`, and
  `visible_camera`;
- absence of staged files and temporary Blender datablocks.

## Validation

- Python 3.10: **471 passed, 4 skipped**;
- Python 3.11: **471 passed, 4 skipped**;
- `Blender 4.4 Alpha Bake`: success;
- `Blender 4.4 Scene Bake`: success;
- `Blender 4.4 Camera Projection`: success;
- full `Blender 4.4 Headless`: success.

## Production defects found by the matrix

1. `COMBINED` could report success while writing an opaque black texture without useful
   lighting;
2. polygon material indices were clamped when assigned before temporary material slots;
3. Blender RNA wrapper identity was unstable for graph traversal;
4. alpha-bearing `DIFFUSE Color` could lose straight RGB;
5. graph `TIME` dependencies were omitted from animated analysis;
6. scene baking could duplicate source and temporary geometry;
7. mixed local and scene slots required explicit black masks;
8. Blender 4.4 object baking has no camera-ray bake type;
9. zero-argument `super()` is unsafe in this frozen `dataclass(slots=True)` plan subclass;
10. synthetic projection lineage must preserve the source object ID.

## Explicit remaining boundaries

### Full-frame B4 output

Transparent borders are not cropped and a screen-space alpha hull is not generated yet.
This is intentional for stable animation geometry.

### Connected multi-object B4 depth

Each B4 texture is a source-only camera layer. A fixed Spine slot order cannot reproduce
arbitrary per-pixel depth intersections among several full-frame projection layers.
Connected multi-object B4 production parity is therefore not claimed until grouped rendering
or a depth-aware composition policy is implemented.

### Recursive node groups

Blender can render group-based appearance, but planner classification does not yet recurse
into group node trees. Camera/Volume requirements hidden entirely in a group may therefore
not automatically select B4.

### Eevee and compositor pipelines

Real B4 validation currently targets Blender 4.4 Cycles. Eevee-specific parity and custom
Compositor output graphs require dedicated fixtures.

### HDR/output policy

OpenEXR can retain higher precision. Configurable tone mapping, HDR runtime expectations and
additional premultiplication policies remain output-policy work.

## Remaining release blockers

1. representative private production `.blend` fixtures with accepted v0.23 JSON/images;
2. accepted JSON and decoded-image parity reports;
3. recursive node-group dependency discovery;
4. a deliberate connected multi-object B4 depth policy;
5. controlled Legacy removal only after private parity acceptance;
6. version bump and release packaging only after the parity gate is accepted.

See also:

- `docs/REWRITE_CAMERA_PROJECTION.md`;
- `docs/REWRITE_BAKE_STRATEGIES.md`;
- `docs/REWRITE_A1_GOLDEN_PARITY.md`.
