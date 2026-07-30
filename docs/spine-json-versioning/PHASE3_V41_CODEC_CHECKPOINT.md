# Phase 3 checkpoint: Spine 4.1 limited production scope

## Status

`SPINE_4_1` has a registered JSON codec and `serializer_ready=True`, but this does not
enable every rig topology and it does not by itself prove control behavior.

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

## Evidence and rejected scale policy

The standalone multi-object fixture contains three independent object rigs in one
skeleton. It has no `all_objects_*` wrapper, no shared object-control constraints, and no
cross-object constraint references.

The exact vendored Spine 4.1 runtime previously accepted the Blender-generated fixture
with finite setup matrices, complete constraint scheduling, renderable attachments, and
positive setup bounds. Manual Spine Editor testing then exposed a real scale-control defect:
the objects imported, but the generated scale constraints did not reproduce the 4.2
control response.

The rejected adapter changed both scale semantics:

1. it changed the uniform scale constraint from relative-world to relative-local by adding
   `local=true`;
2. it moved the depth constraint from the canonical `*_scale` wrapper bones to the final
   layer bones.

The second change made the depth constraint operate on the same layer bones that the
uniform scale constraint had just scaled, so the following constraint could overwrite or
distort the scale result. Runtime finiteness was therefore necessary but not sufficient.

The earlier connected candidate is not evidence for normal multi-object export. Connected
composition is development-only for this target and remains blocked by capability policy.

## Current scale-preserving Spine 4.1 candidate

The current target-aware document adapter keeps the canonical 4.2 scale roles:

- the uniform constraint remains `relative=true` and world-space;
- the depth constraint remains on the original `*_scale` wrapper bones;
- no epsilon replaces an authored zero scale;
- no serialized JSON text repair is used.

Two hierarchy changes make those world constraints evaluable by Spine 4.1:

1. The uniform scale constraint replaces only the unsafe constrained driver
   `<prefix>_rotate_X` with its direct parent `<prefix>_scale_rotate_X`. Uniform world
   scaling then propagates through the child rotation while the constrained bone itself has
   an invertible parent.
2. Every depth wrapper receives an internal `onlyTranslation` bridge. The bridge carries
   the wrapper's original setup X/Y offset; the original wrapper keeps its stable name,
   rotation, inherit mode, and depth-constraint ownership but is reparented at local zero.
   The wrapper parent is therefore invertible when Spine 4.1 updates applied transforms.

Bridge insertion occurs on the typed immutable `SpineDocument` after canonical attachment
projection. Because the inserted parents shift bone-array indices, every weighted mesh
stream is decoded, remapped by old/new bone name index, and re-encoded. Builder component
metadata and skin attachment objects are synchronized with the same remapped document.

This topology is a candidate until the new runtime scale-response probe and Spine Editor
control test pass on a fresh Blender-generated JSON.

## Capability ownership

`domain/spine/export_capabilities.py` owns the accepted matrix:

- target JSON family;
- rig profile;
- document composition scope;
- known limitations attached to the accepted pair.

A registered codec proves only that a typed document can be represented in the target
JSON schema. It does not prove that every generated hierarchy or constraint graph is safe
or behaviorally equivalent in that runtime.

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

The codec deliberately preserves builder-authored constraint order. It never repairs
hierarchy, setup transforms, scale values, constraint targets, or dependency phases.

## Runtime gates

`tools/spine41_runtime_oracle.mjs` imports the vendored runtime as read-only input and
checks parser construction, update-cache coverage, finite matrices, attachment agreement,
and finite positive setup bounds.

`tools/spine41_scale_response_probe.mjs` is the behavior gate for the two-axis scale
profile. For every generated `*_scale` control it:

1. rejects the superseded `local=true` scale policy;
2. requires `<prefix>_scale_rotate_X` rather than the unsafe `<prefix>_rotate_X` driver;
3. isolates the object's setup slots;
4. applies scale factors `0.5`, `1.5`, and `2.0` in the exact Spine 4.1 runtime;
5. requires all matrices to remain finite;
6. requires the object bounds to scale uniformly around `<prefix>_main`.

Neither runtime gate proves editor interaction by itself. The exact generated JSON must
still be imported into Spine Editor 4.1.24 and the Scale control tested directly.

## Next gates

Before describing the scale defect as fixed or building a release package:

- run the focused Spine 4.1 pure-Python tests;
- run the complete pure-Python suite;
- run the complete real-`bpy` suite;
- export a fresh single and standalone multi-object JSON directly from Blender;
- run `spine41_runtime_oracle.mjs` on those exact files;
- run `spine41_scale_response_probe.mjs` on those exact files;
- import those same files into Spine Editor 4.1.24;
- verify `0.5`, `1.5`, and `2.0` scale behavior visually;
- keep connected/mixed disabled until their own Editor fixtures and acceptance gates pass;
- keep 3-Axis disabled until its setup and compensator phases are proven;
- verify Spine 4.2 output remains byte-identical.

No CI/CD workflow is added or changed by this checkpoint.
