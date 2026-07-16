# B4 camera-render projection

## Problem

Blender 4.4 object baking has no camera-ray bake type. A UV `COMBINED` pass therefore
cannot preserve effects whose value depends on the active camera or on ray type:

- Fresnel and Layer Weight;
- Light Path;
- Glass and Refraction;
- Principled Transmission;
- camera-preserving reflections;
- Volume;
- material displacement whose final appearance must be evaluated by render.

The rewrite does not invent an `ACTIVE_CAMERA` bake mode and does not silently flatten
these materials through an incorrect UV bake. B4 introduces a second texture execution
pipeline based on a real still render from the active Blender camera.

## Automatic routing

`domain/baking/camera_projection.py` exposes:

```text
requires_camera_projection()
build_camera_projection_plan()
build_texture_plan()
CameraProjectionPlan
```

`build_texture_plan()` is the production planner used by `prepare_a1_object()`.

```text
reachable material graph
        |
        +-- LOCAL / SCENE / AUXILIARY --> BakePlan --> object UV bake
        |
        +-- CAMERA / VOLUME / render displacement
                                      --> CameraProjectionPlan
                                      --> active-camera render
```

The complete object is routed to camera projection when any used material slot requires
camera projection. Mixing UV-baked and screen-rendered pixels inside one object would mix
two coordinate spaces and is intentionally not attempted.

A camera projection plan requires:

- immutable `ObjectBakeContext`;
- immutable `SceneBakeContext`;
- an active camera snapshot;
- PNG, WEBP, or OPEN_EXR output so transparent background is representable.

JPEG is rejected during planning.

## Compatibility contract

`CameraProjectionPlan` is a frozen `BakePlan` subtype. Existing consumers continue to use:

- `source_object_id`;
- `settings`;
- `frame_tasks`;
- `representative_task`;
- attachment path and sequence helpers;
- atomic output reservations;
- `BakeExecutionResult`.

Its one synthetic `CAMERA_COMBINED` pass is metadata only. It is never passed to
`bpy.ops.object.bake`. Runtime dispatch checks the concrete plan type and uses
`bpy.ops.render.render(write_still=True)`.

## Render execution

`blender_adapter/camera_projection_executor.py` owns the B4 transaction.

For every static or sequence frame it:

1. validates source, Scene, World, camera and light identities against the immutable plan;
2. captures render settings, timeline frame, `hide_render`, and `visible_camera` values;
3. makes the source object directly visible to the active camera;
4. disables only direct camera visibility for other renderable objects;
5. keeps their diffuse, glossy, transmission and shadow participation intact;
6. enables transparent film while keeping World lighting/reflection contribution;
7. renders directly to the transaction's staged path;
8. validates that Blender created a non-empty file;
9. restores every captured property in `finally`;
10. commits the JSON and all texture frames together.

The only real render operator access is the failure-injection hook:

```text
blender_adapter/bake_executor.py::_call_render_operator
```

`texture_executor.py` dispatches plans but contains no `bpy.ops` access.

## Source-only camera layer

Other meshes must remain available to reflection, refraction, occlusion and shadows, but
they must not be drawn directly into every exported object's texture. B4 therefore changes
only `visible_camera` for other renderable objects during a projection render.

It deliberately does **not** set `hide_render=True` on dependencies. This distinction is
required for Glass and reflective materials.

The source object's previous `hide_render` and `visible_camera` values are restored on
success and failure.

## Spine geometry

`application/a1_camera_projection.py` builds a deterministic full-frame mesh:

```text
4 vertices
5 topology edges including one diagonal
2 triangles
4 hull vertices
UV range 0..1
attachment size = render width x render height
```

The quad is centered on the legacy main bone. Its local extent is derived from texture
pixels and the rig's uniform scale so serialized Spine coordinates are exactly:

```text
(-width / 2, +height / 2)
(+width / 2, +height / 2)
(+width / 2, -height / 2)
(-width / 2, -height / 2)
```

A full-frame quad is the stable first implementation because a static or animated camera
can move the projected object anywhere in the frame. Per-frame cropping would require a
second stable geometry/offset contract and could clip later sequence frames.

## Production integration

`prepare_a1_object()` now calls `build_texture_plan()` after material and scene analysis.
When B4 is selected it:

- records `texture_pipeline=CAMERA_RENDER_PROJECTION`;
- keeps existing texture names and sequence frame names;
- builds the legacy rig with its main bone at the screen origin;
- replaces region UV attachments with one full-frame projection quad;
- keeps control icons and preview animation options compatible;
- uses the same JSON plus texture atomic transaction as object baking.

The stable public functions remain:

```text
execute_bake_plan()
stage_bake_plan_outputs()
```

They accept either `BakePlan` or `CameraProjectionPlan` and dispatch internally.

## Blender 4.4 validation

The dedicated workflow `Blender 4.4 Camera Projection` runs real Cycles renders and decoded
pixel checks for:

1. production Layer Weight/Fresnel export;
2. transparent background plus visible source coverage;
3. full-frame Spine JSON mesh topology;
4. Glass render projection;
5. Principled Volume render projection;
6. animated camera-dependent sequence frames;
7. restoration of the original timeline frame;
8. forced render failure after existing output files are present;
9. atomic restoration of previous JSON and PNG bytes;
10. restoration of context, render settings, material graphs, `hide_render`, and
    `visible_camera`;
11. absence of staged files and temporary Blender datablocks.

Pure tests additionally cover reflection, Volume, displacement, local-plan preservation,
missing camera, JPEG rejection, sequence naming, quad topology and execution-result
compatibility.

## Current boundaries

### Full-frame output

The first B4 version does not crop transparent borders and does not generate a screen-space
convex hull. This trades texture area for deterministic sequence geometry and stable
attachment offsets.

### Connected multi-object depth

Each B4 texture is a source-only camera layer. Several full-frame layers can be composed for
standalone objects, but a connected rig cannot reproduce arbitrary per-pixel depth
intersections using one fixed slot order. Connected multi-object camera projection therefore
requires a future grouped render or depth-aware layer policy before production parity is
claimed.

### Camera movement versus Spine rig movement

B4 captures the rendered camera result. Camera motion is baked into texture frames, not
converted into Spine bone motion. This is intentional for appearance parity.

### Render engine scope

The current production execution uses the configured render engine, with real Blender 4.4
validation on Cycles. Eevee-specific parity and compositor-dependent render pipelines need
separate fixtures before being claimed.

### Color and HDR

PNG/WEBP use Blender's configured color-management transform. OPEN_EXR preserves a higher
precision render target. Automatic tone mapping, premultiplied-alpha policy variants and HDR
Spine runtime behavior remain output-policy work, not material strategy switches.

### Recursive node groups

B4 can render group-based materials because Blender evaluates the live graph, but planner
classification still needs recursive node-group analysis to discover camera/volume
requirements hidden entirely inside a group.

## Extension direction

The next safe increments are:

1. recursive node-group dependency discovery;
2. optional stable union crop for complete animation sequences;
3. screen-space hull generation from a union alpha mask;
4. grouped multi-object camera layers;
5. depth-aware connected composition;
6. representative private `.blend` parity against accepted v0.23 JSON and images.

No increment should add per-material UI mode switches. Planner facts and registered pipeline
selection remain authoritative.
