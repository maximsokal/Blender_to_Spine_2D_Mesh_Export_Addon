# Phase 3 checkpoint: Spine 4.1 limited production scope

## Status

`SPINE_4_1` now has a registered JSON codec and `serializer_ready=True`, but this does
not enable every rig topology.

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

The standalone multi-object fixture contains three independent object rigs in one
skeleton. It has no `all_objects_*` wrapper, no shared object-control constraints, and no
cross-object constraint references.

The exact vendored Spine 4.1 runtime accepted the fixture with:

- version `4.1.24`;
- 58 finite bones;
- 24 slots;
- 3 IK constraints;
- 12 transform constraints;
- 15 globally unique contiguous orders (`0..14`);
- all 15 constraints scheduled exactly once;
- 12 setup renderable attachments;
- finite positive setup bounds.

Manual Spine Editor 4.1.24 import displayed the three standalone objects in a usable
state. Scale-constraint behavior is not yet fully equivalent to Spine 4.2 and is tracked
as a separate compatibility defect.

The earlier connected candidate is not evidence for normal multi-object export. Connected
composition is development-only for this target and remains blocked by capability policy.

## Capability ownership

`domain/spine/export_capabilities.py` owns the accepted matrix:

- target JSON family;
- rig profile;
- document composition scope;
- known limitations attached to the accepted pair.

A registered codec proves only that a typed document can be represented in the target
JSON schema. It does not prove that every generated hierarchy or constraint graph is safe
for that runtime.

Single-object preparation checks `SINGLE_OBJECT` before reading Blender geometry.
Multi-object preparation resolves either `STANDALONE_MULTI_OBJECT` or
`CONNECTED_MULTI_OBJECT` before geometry. A Spine 4.1 mixed request enters its connected
subgroup preflight and is rejected before any object preparation starts.

## Target-aware two-axis implementation

The Spine 4.1 two-axis builder uses no epsilon and does not change setup bone scales.

Two constraints have target-aware representations:

1. The uniform scale constraint remains relative but evaluates in local applied space.
   This avoids world-space decomposition below the generated axis-collapse parent.
2. The depth scale constraint targets final layer bones instead of their
   `onlyTranslation` wrappers, keeping the constrained parent matrices invertible.

A pure setup-matrix validator rejects remaining Spine 4.1 world constraints whose
constrained bone has a singular parent. It reports the constraint, constrained bone,
parent, and determinant without mutating the document.

The remaining scale-control mismatch observed in Spine Editor is not hidden by the codec
and must be fixed in the target-aware rig policy later.

## Codec ownership

`Spine41JsonCodec` owns only verified schema representation:

- exact `skeleton.spine = 4.1.24`;
- `inherit` to legacy `transform` bone-field mapping;
- Spine 4.1 skin IK/transform membership arrays;
- removal of unsupported physics data;
- deterministic JSON encoding.

The codec deliberately preserves builder-authored constraint order. It never repairs
hierarchy, setup transforms, scale values, constraint targets, or dependency phases.

## Production registry

The production registry contains:

- `Spine41JsonCodec` for `SPINE_4_1`;
- `Spine42JsonCodec` for `SPINE_4_2`.

Registry readiness and descriptor readiness must remain exactly equal. Unsupported targets
3.8, 4.0, and 4.3 remain selectable for migration/testing but fail before geometry.

## Runtime oracle

`tools/spine41_runtime_oracle.mjs` remains the external acceptance gate. It imports the
vendored runtime as read-only input and creates atlas/texture objects only in memory.

The oracle checks:

1. exact 4.1 metadata;
2. globally unique contiguous constraint order;
3. parser and skeleton construction;
4. setup pose and update cache;
5. every runtime constraint scheduled exactly once;
6. finite world/applied bone transforms;
7. setup attachment agreement with JSON;
8. finite positive setup bounds.

The oracle does not prove UI control behavior. Spine Editor import and control testing are
still required for every new fixture.

## Known limitation

Spine 4.1 scale-constraint behavior is not fully equivalent to the Spine 4.2 result. The
current limited release allows export for continued integration testing, but the defect
must remain documented and must not be described as solved.

The follow-up must compare the generated per-object scale constraints against a native
Spine Editor 4.1.24 authoring fixture and verify control response, not only setup bounds.

## Next gates

Before expanding the Spine 4.1 capability matrix:

- run the complete pure-Python suite;
- run the complete real-bpy suite;
- export single and standalone multi-object JSON directly from Blender;
- rerun the exact 4.1 runtime oracle on those Blender-generated files;
- import those exact files into Spine Editor 4.1.24;
- fix and certify scale-control behavior;
- keep connected/mixed disabled until their own Editor fixtures and acceptance gates pass;
- keep 3-Axis disabled until its setup and compensator phases are proven;
- verify Spine 4.2 output remains byte-identical.

No CI/CD workflow is added or changed by this checkpoint.
