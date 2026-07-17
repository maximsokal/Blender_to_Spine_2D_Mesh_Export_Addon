# Rewrite status

The active rewrite branch is `rewrite/a1-domain-foundation`; A1 compatibility targets
Spine 4.2.43. Deterministic geometry, loop-level UV lineage, semantic local/alpha/scene
baking, recursive Shader Node Group analysis, camera-render projection, stable sequence-union
crop, screen-space convex hulls, typed Spine composition, connected `all_objects`, and both
production operators are implemented.

## Production operators

- `object.save_uv_as_json` keeps its public ID and uses Rewrite by default;
- `object.spine2d_multi_export` keeps its public ID and uses Rewrite by default;
- Legacy remains explicitly selectable and is never an automatic fallback;
- the add-on version is unchanged;
- no release package has been produced from the rewrite branch.

Output naming remains compatible:

```text
<object>_merged.json
<object>_Baked.png
<object>_Baked_0000.png ...
```

Single, standalone multi, connected multi, and mixed Connect-flag flows use one atomic JSON
plus texture transaction.

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
                -> one stable crop + convex hull
                -> post-render typed document finalization
```

Planner selection is automatic. Per-material mode switches were not added to the UI.

## B1: local surface and Emission

Implemented:

- ordinary opaque Principled, Image Texture and procedural color through lighting-independent
  `DIFFUSE Color`;
- material Emission through `EMIT`;
- separate surface and Emission slots on one mesh;
- one Principled material containing Base Color and Emission;
- float-buffer composition;
- normalization of a surface `COMBINED` request to `DIFFUSE` when separate Emission would
  otherwise be counted twice.

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

Recursive copied-material extraction supports Principled Base Color/Alpha, linked image and
procedural sockets, Transparent, Holdout, nested Mix/Add Shader trees, animated values,
drivers, opaque plus transparent slots, and pure Transparent output.

Temporary proxy nodes exist only on copied materials and are removed in `finally`. Source
material node trees are not mutated.

## B3: scene-aware object baking

Evaluation scopes:

- `LOCAL`: albedo, Image/procedural surface color and material Emission;
- `SCENE`: World, lighting, occlusion and other scene-object dependencies;
- `AUXILIARY`: explicit alpha and composition-only channels;
- `CAMERA`: routed to B4 instead of object baking.

`ObjectBakeContext` records source identity, transform, collections, visibility and animation.
`SceneBakeContext` records Scene/render engine, World, lights, active camera, visible objects,
shadow casters, color management and analysis frame.

`SceneCombinedBakeStrategy` selects real `COMBINED` for explicit scene-dependent appearance
such as Subsurface, Sheen, Toon, Translucent, Hair and Ambient Occlusion. Ordinary Principled
Base Color remains local merely because a file contains lights.

One mesh may contain local and scene-aware slots. Unmatched copied slots become black Emission
proxies during each pass to prevent double counting. Source/context/frame/render state and all
temporary datablocks are restored on success and failure.

## Recursive Shader Node Group analysis

`blender_adapter/shader_graph_analyzer.py` recursively traverses reachable Shader Node Groups
through their actual interfaces:

```text
Group output -> internal Group Output input
Group Input output -> matching outer Group input
```

Implemented guarantees:

- only sockets contributing to the renderer-effective Material Output are expanded;
- Cycles, Eevee and generic Material Output targets are resolved independently;
- muted nodes and muted groups follow Blender `internal_links` bypass mappings;
- unused group inputs do not leak Camera/View dependencies;
- stable instance-qualified nested node IDs;
- explicit `group_path` snapshots;
- nested Image, Time, Camera, View, Reflection, Transmission and Volume discovery;
- nested node-tree animation detection;
- reachable-only material kind and image dependency classification;
- socket matching by identifier/name/index across Blender API variants;
- recursive-cycle detection;
- maximum traversal depth 64;
- no mutation of node groups.

Real Blender fixtures cover nested Layer Weight, nested Principled Volume, nested Image Texture,
renderer-specific outputs, muted camera branches, repeated group instances, and an unused
camera-bearing parent input that correctly remains outside the reachable graph.

## B4: camera-render projection

B4 is selected automatically for Camera/View, Reflection/Transmission, Volume and
render-evaluated displacement. Covered families include Fresnel, Layer Weight, Light Path,
Glass, Refraction, Principled Transmission, reflective appearance and Principled Volume,
including requirements nested in reachable Shader Node Groups.

`CameraProjectionPlan` is a frozen `BakePlan` subtype. Its synthetic `CAMERA_COMBINED` pass is
metadata only and is never sent to `bpy.ops.object.bake`.

### Render and sequence-union layout

For every static or sequence frame B4:

1. validates immutable object/Scene/World/camera/light identities;
2. captures render, frame, `hide_render` and `visible_camera` state;
3. keeps only the source directly camera-visible while retaining other objects for reflection,
   transmission, diffuse, occlusion and shadow rays;
4. renders a transparent full frame to the transaction's staged path;
5. decodes the actual staged image and extracts alpha using threshold `1 / 255`;
6. immediately ORs that frame into one fixed-size alpha union buffer;
7. releases the per-frame mask before rendering the next frame;
8. expands the union bounds by existing `bake_margin`;
9. builds one counter-clockwise convex screen-space hull;
10. rewrites every staged frame using the same crop dimensions;
11. restores all Blender state in `finally`.

The render executor therefore retains `O(width * height)` alpha-mask memory regardless of
sequence length. The compatibility tuple API remains available for existing pure-domain callers.

Blender 4.4 headless exposes a zero-sized `Render Result` after completed renders. The staged
file is therefore the source of truth for alpha analysis.

### Stable cropped Spine attachment

`domain/baking/projection_layout.py` provides immutable crop/hull contracts, a fixed-size
incremental alpha-union accumulator, and deterministic monotonic-chain convex hull generation.

Every frame in a sequence shares crop bounds, texture dimensions, full-frame screen offset,
UVs, hull vertices, triangle fan and attachment dimensions. The hull follows the union alpha
silhouette while UVs address the padded crop. Vertex positions preserve original full-frame
camera placement, so cropping does not recenter the object.

For hull size `H`:

```text
UV values              = H * 2
triangle index values  = (H - 2) * 3
```

An all-transparent render or sequence fails before commit.

### Post-render typed recomposition

A crop/hull layout is known only after all renders succeed. Production order is:

```text
prepare -> reserve JSON -> render/crop textures -> finalize B4 attachments
        -> recompose typed documents -> serialize JSON -> atomic commit
```

This path is implemented for single, standalone multi, connected multi, and mixed exports.
Multi/mixed output recomposes typed `SpineDocument` values; serialized JSON is never patched or
merged. Each attachment width/height is validated against its decoded cropped image.

## Executor boundaries

- `bake_executor_core.py`: object-bake validation/resources and sole real
  `bpy.ops.object.bake` implementation;
- `semantic_bake_executor.py`: B1-B3 passes, material preparation and RGBA composition;
- `camera_projection_state.py`: reversible Scene/frame/visibility state;
- `camera_projection_image.py`: staged-image decode, alpha mask and crop rewrite;
- `camera_projection_executor_core.py`: B4 render/incremental-union/crop orchestration;
- `camera_projection_executor.py`: stable B4 facade;
- `texture_executor.py`: plan dispatch and detailed layout result without operator access;
- `bake_executor.py`: stable public facade containing only object-bake and render hooks.

Architecture tests verify that helper, finalization and output modules contain no direct
`bpy.ops` access, and that B4 uses one streaming union accumulator rather than retaining a list
of full-frame masks.

## Real Blender compatibility matrix

Every output image is decoded; a PNG signature alone is not accepted.

B1-B3 coverage includes opaque/procedural/image color, Emission, straight Alpha, nested
transparency, animated Alpha, scene light/World/AO response, mixed local/scene slots, animated
lights, rollback, state restoration, registered operators and isolated Legacy/Rewrite flows.

Dedicated B4 coverage includes:

- Fresnel/Layer Weight planning, render, crop and convex hull;
- Glass and Principled Volume;
- animated camera-dependent frames with one sequence-union crop;
- attachment dimensions equal to decoded cropped images;
- nested Layer Weight and nested Volume groups;
- unused group input reachability precision;
- standalone, connected and mixed multi-object cropped recomposition;
- forced render failure and atomic JSON/texture rollback;
- render/context/material/visibility restoration;
- absence of staged files and temporary Blender datablocks.

## Validation

- Python 3.10: **484 passed, 4 skipped** on the last full automatic matrix before manual-only CI;
- Python 3.11: **484 passed, 4 skipped** on the last full automatic matrix before manual-only CI;
- `Blender 4.4 Alpha Bake`: success on the last full matrix;
- `Blender 4.4 Scene Bake`: success on the last full matrix;
- `Blender 4.4 Camera Projection`: success on the last full matrix;
- full `Blender 4.4 Headless`: success on the last full matrix;
- current recursive hardening focused tests: **21 passed**;
- current incremental union focused tests: **14 passed**;
- 1000 randomized old/new union-layout differential cases: identical.

Automatic workflow triggers remain disabled on the active rewrite branch, so the latest focused
hardening commits have not consumed GitHub Actions minutes. Real Blender fixtures added after the
last complete matrix remain pending a deliberate manual validation run.

## Production defects found by the matrix

1. `COMBINED` could report success while producing opaque black output without useful lighting;
2. polygon material indices were clamped before temporary slots existed;
3. Blender RNA wrapper identity was unstable for graph traversal;
4. alpha-bearing `DIFFUSE Color` could lose straight RGB;
5. graph `TIME` dependencies were omitted from animated analysis;
6. scene baking could duplicate source and temporary geometry;
7. mixed local and scene slots required explicit black masks;
8. Blender 4.4 object baking has no camera-ray bake type;
9. zero-argument `super()` is unsafe in a frozen `dataclass(slots=True)` plan subclass;
10. synthetic projection lineage must preserve source object ID;
11. scanning all group nodes would make unused inputs leak camera dependencies;
12. Blender 4.4 headless `Render Result` can remain zero-sized after success;
13. JSON must be serialized after render-derived crop/hull finalization;
14. multi/mixed output must recompose typed documents after component layouts are known;
15. sequence B4 retained every full-frame alpha mask before allocating the union mask;
16. renderer-specific Material Outputs and muted-node bypasses were not respected.

## Explicit remaining boundaries

### Convex rather than concave hull

The attachment uses a convex hull. Deep concavities and internal transparent holes remain
inside the mesh and are represented by texture alpha.

### Fixed alpha threshold

Layout detection uses decoded alpha `>= 1 / 255`. A configurable threshold or
coverage-weighted antialias policy remains output-policy work.

### Connected multi-object B4 depth

Connected and mixed outputs now contain correct cropped layers and attachments. A fixed Spine
slot order still cannot reproduce arbitrary per-pixel depth intersections among independently
rendered source-only camera layers. Grouped rendering or depth-aware composition is required
before arbitrary connected B4 visual parity is claimed.

### Eevee and compositor pipelines

Real B4 validation targets Blender 4.4 Cycles. Eevee-specific parity and custom Compositor
output graphs require dedicated fixtures.

### HDR/output policy

OPENEXR can retain higher precision. Configurable tone mapping, HDR runtime expectations and
additional premultiplication policies remain output-policy work.

## Remaining release blockers

1. representative private production `.blend` fixtures with accepted v0.23 JSON/images;
2. accepted JSON and decoded-image parity reports;
3. a deliberate connected multi-object B4 depth policy where real fixtures require it;
4. Eevee/Compositor support only if production files depend on them;
5. controlled Legacy removal only after private parity acceptance;
6. version bump and release packaging only after the parity gate is accepted.

See also `docs/REWRITE_CAMERA_PROJECTION.md`, `docs/REWRITE_BAKE_STRATEGIES.md`, and
`docs/REWRITE_A1_GOLDEN_PARITY.md`.
