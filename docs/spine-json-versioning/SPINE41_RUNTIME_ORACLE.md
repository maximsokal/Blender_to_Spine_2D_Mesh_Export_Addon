# Spine 4.1 runtime oracle

`tools/spine41_runtime_oracle.mjs` validates generated JSON with the exact vendored
Spine 4.1 WebGL runtime from `Spine2D_curve_optimization`.

This is an external acceptance gate. It does not make Spine 4.1 production-ready by
itself, and it is not executed by CI/CD.

## Requirements

- Node.js with ESM support;
- a local checkout of `maximsokal/Spine2D_curve_optimization`;
- its self-contained `vendor/spine-webgl-41/index.js` runtime bundle;
- a generated JSON whose `skeleton.spine` belongs to family 4.1.

## Windows command

From the add-on repository root:

```bat
node tools\spine41_runtime_oracle.mjs ^
  "E:\exports\Cone_plus_2_objects_spine_4.1.24.json" ^
  "E:\Spine2D_curve_optimization\vendor\spine-webgl-41\index.js"
```

The runtime entry may instead be supplied through the environment:

```bat
set SPINE41_RUNTIME_ENTRY=E:\Spine2D_curve_optimization\vendor\spine-webgl-41\index.js
node tools\spine41_runtime_oracle.mjs "E:\exports\project_spine_4.1.24.json"
```

## What is checked

Before runtime loading:

- exact 4.1 family metadata;
- non-empty globally unique constraint names;
- non-negative integer orders;
- one globally unique order for every IK/transform/path constraint;
- exact contiguous order range `0..N-1`.

With the official 4.1 runtime:

- `SkeletonJson.readSkeletonData()` succeeds;
- `Skeleton` construction succeeds;
- the default skin is activated when present;
- setup pose and `updateCache()` succeed;
- every parsed constraint appears in `_updateCache` exactly once;
- `updateWorldTransform()` succeeds;
- every bone has finite world, matrix, and applied-transform values;
- setup bounds are finite when region or mesh attachments are visible.

## Failure meaning

A failed oracle means the JSON is not eligible for production output. Do not repair the
file by changing order, scale, hierarchy, or transform fields after serialization. The
failure must be traced back to the target-aware rig builder, connected schedule, or a
verified schema mapping.

A successful oracle still requires Blender regression tests and manual import into
Spine Editor 4.1.24 before `serializer_ready` can be enabled.
