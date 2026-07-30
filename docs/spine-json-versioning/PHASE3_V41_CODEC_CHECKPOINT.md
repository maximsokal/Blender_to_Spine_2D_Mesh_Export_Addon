# Phase 3 checkpoint: Spine 4.1 limited production scope

## Status

`SPINE_4_1` has a registered JSON codec and `serializer_ready=True`, but this does not
enable every rig topology.

The production capability matrix permits only:

- exact target `4.1.24`;
- rig profile `2-Axis Rotation + Scale`;
- one object in one skeleton;
- several independent objects composed in `STANDALONE` mode under the shared `root`.

The following Spine 4.1 combinations remain fail-closed before Blender geometry work:

- `3-Axis Rotation`;
- `CONNECTED` multi-object mode;
- `MIXED` multi-object mode;
- any future topology that is not explicitly present in the capability registry.

Spine 4.2 remains unchanged and retains both rig profiles and all existing composition
modes.

## Accepted evidence

The accepted standalone multi-object path uses the real production export pipeline:

```text
Blender 5.2
-> export_a1_multi_object()
-> Spine 4.1.24 standalone JSON
-> exact vendored spine-webgl-41 runtime
-> Spine Editor 4.1.24
```

The generated document contains independent object rigs only. It has no `all_objects_*`
wrapper, no shared object-control constraints, and no cross-object constraint references.

The exact Spine 4.1 runtime accepted the Blender-generated fixture with:

- exact version `4.1.24`;
- finite setup matrices for every generated bone;
- complete constraint scheduling with every constraint included exactly once;
- globally unique contiguous constraint order;
- renderable setup attachments;
- finite positive setup bounds.

Manual Spine Editor 4.1.24 testing confirmed that the corrected Scale control now scales
the exported object correctly. This closes the earlier scale-control defect for the accepted
single-object and standalone multi-object scope.

The earlier connected candidate is not evidence for normal multi-object export. Connected
composition remains development-only for this target and is blocked by capability policy.

## Rejected scale policy

The superseded adapter changed both scale semantics:

1. it changed the uniform scale constraint from relative-world to relative-local by adding
   `local=true`;
2. it moved the depth constraint from the canonical `*_scale` wrapper bones to the final
   layer bones.

The second change made the depth constraint operate on the same layer bones that the
uniform scale constraint had already scaled, allowing the following constraint to overwrite
or distort the scale result. Runtime finiteness was necessary but not sufficient.

This policy must not be reintroduced.

## Accepted scale-preserving Spine 4.1 topology

The target-aware document adapter keeps the canonical Spine 4.2 scale roles:

- the uniform constraint remains `relative=true` and world-space;
- the depth constraint remains on the original `*_scale` wrapper bones;
- the original front-to-back constrained-bone order is preserved;
- no epsilon replaces an authored zero scale;
- no serialized JSON text repair is used.

Two hierarchy changes make those world constraints evaluable by Spine 4.1:

1. The uniform scale constraint replaces only the unsafe constrained driver
   `<prefix>_rotate_X` with its direct parent `<prefix>_scale_rotate_X`. Uniform world
   scaling propagates through the child rotation while the constrained bone itself has an
   invertible parent.
2. Every depth wrapper receives an internal `onlyTranslation` bridge. The bridge carries
   the wrapper's original setup X/Y offset. The original wrapper keeps its stable name,
   rotation, inherit mode, and depth-constraint ownership but is reparented at local zero.
   The wrapper parent is therefore invertible when Spine 4.1 updates applied transforms.

Bridge insertion occurs on the typed immutable `SpineDocument` after canonical attachment
projection. Because inserted parents shift bone-array indices, every weighted mesh stream is
decoded, remapped by old/new bone-name index, and re-encoded. Builder component metadata,
skins, and attachment objects are synchronized with the same remapped document.

## Capability ownership

`domain/spine/export_capabilities.py` owns the accepted matrix:

- target JSON family;
- rig profile;
- document composition scope.

A registered codec proves only that a typed document can be represented in the target JSON
schema. It does not prove that every generated hierarchy or constraint graph is safe or
behaviorally equivalent in that runtime.

Single-object preparation checks `SINGLE_OBJECT` before reading Blender geometry.
Multi-object preparation resolves either `STANDALONE_MULTI_OBJECT` or
`CONNECTED_MULTI_OBJECT` before geometry. A Spine 4.1 mixed request enters its connected
subgroup preflight and is rejected before any object preparation starts.

## Codec ownership

`Spine41JsonCodec` owns only verified schema representation:

- exact `skeleton.spine = 4.1.24`;
- `inherit` to legacy `transform` bone-field mapping;
- Spine 4.1 skin IK/transform membership arrays;
- removal of unsupported physics data;
- deterministic JSON encoding.

The codec preserves builder-authored constraint order. It does not repair hierarchy, setup
transforms, scale values, constraint targets, or dependency phases.

## Runtime gates

`tools/spine41_runtime_oracle.mjs` imports the vendored runtime as read-only input and
checks parser construction, update-cache coverage, finite matrices, attachment agreement,
and finite positive setup bounds.

`tools/spine41_scale_response_probe.mjs` is the automated behavior gate for the two-axis
scale profile. For every generated `*_scale` control it:

1. rejects the superseded `local=true` scale policy;
2. requires `<prefix>_scale_rotate_X` rather than the unsafe `<prefix>_rotate_X` driver;
3. isolates the object's setup slots;
4. applies scale factors `0.5`, `1.5`, and `2.0` in the exact Spine 4.1 runtime;
5. requires all matrices to remain finite;
6. requires object bounds to scale uniformly around `<prefix>_main`.

The external runtime repository remains read-only. Atlas and texture stand-ins are created
only in memory by the oracle tools.

## Release scope

Extension version `0.47.10` exposes the accepted limited Spine 4.1 scope:

- `2-Axis Rotation + Scale`;
- single-object export;
- standalone multi-object export.

It does not expose Spine 4.1 connected, mixed, or 3-Axis output.

Before publishing the package, the release commit must pass:

- focused Spine 4.1 pure-Python tests;
- complete pure-Python suite;
- complete real-`bpy` suite;
- Blender-headless standalone acceptance;
- exact Spine 4.1 runtime oracle;
- automated scale-response probe;
- official Blender extension validation/build;
- isolated install/enable/export/disable/uninstall smoke gate.

No CI/CD workflow is added or changed by this checkpoint.
