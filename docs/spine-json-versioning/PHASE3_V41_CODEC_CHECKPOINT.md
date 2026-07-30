# Phase 3 checkpoint: Spine 4.1 quarantined

## Status

The previous Spine 4.1 production checkpoint is invalidated.

`SPINE_4_1` remains a selectable target so requests, filenames, and UI migration stay
stable, but `serializer_ready` is `False` and no Spine 4.1 codec is registered in the
production facade. Analyze/export must fail before geometry preparation.

Spine 4.2 remains the only production-ready target and its serializer path is unchanged.

## Why the checkpoint was invalidated

The previous implementation serialized the canonical Spine 4.2-oriented document and
then rewrote a small set of JSON fields. That approach proved insufficient for the
connected `2-Axis Rotation + Scale` rig.

Repository evidence from `maximsokal/Spine2D_curve_optimization` shows that version
support is a complete pipeline concern:

1. select the exact runtime family;
2. select matching parser/reconstructor transformers;
3. reconstruct the document for the target family;
4. parse it again with the target family;
5. validate the resulting setup pose with that runtime.

Changing `skeleton.spine`, renaming `inherit`, or renumbering constraints after
serialization does not prove target-runtime compatibility.

## Removed production behavior

The following behavior is no longer part of the production contract:

- registration of `Spine41JsonCodec` in the production codec registry;
- `serializer_ready=True` for Spine 4.1;
- codec-owned constraint-order linearization;
- any zero-scale replacement or numerical stabilization inside the serializer;
- claims that the generated connected rig is native Spine 4.1 output.

The quarantined adapter retains only evidence-backed field transformations for focused
research. It deliberately preserves authored constraint order and cannot be reached
through `serialize_spine_document()`.

## Target-aware two-axis implementation

The builder now has a separate Spine 4.1 two-axis constraint policy. It does not modify
bone scales and does not use an epsilon.

The two changes are owned by the rig builder:

1. The uniform scale constraint remains relative but is evaluated in local applied space.
   This avoids a world-space decomposition of `<prefix>_rotate_X`, whose generated parent
   intentionally collapses one axis with `scaleX == 0`.
2. The depth correction constraint targets final layer bones rather than their
   `onlyTranslation` scale wrappers. The final layer bones have invertible parents, so
   Spine 4.1 can call `Bone.updateAppliedTransform()` safely.

Both per-object rigs and the connected global wrapper use this policy. Application is
idempotent so already-adapted objects cannot be adapted twice with different semantics.
Spine 4.2 continues to use the existing constraints unchanged.

A pure domain setup-matrix validator rejects any remaining Spine 4.1 world constraint
whose constrained bone has a singular parent matrix. This validator reports the exact
constraint, constrained bone, parent bone, and determinant; it never changes the rig.

## Constraint scheduling

Spine 4.1 connected two-axis output receives one globally unique order for every
constraint and the exact contiguous order range `0..N-1`. Order ownership is in the
connected builder, not the serializer.

Spine 4.2 retains its historical same-Z-layer order sharing for byte and behavior parity.
Spine 4.1 three-axis connected scheduling remains fail-closed because the standalone
scale compensator still needs a separately proven dependency phase.

## Architectural ownership

### Version codec

A version codec may own only target JSON schema representation:

- exact `skeleton.spine` metadata;
- verified field-name differences such as bone `inherit` versus `transform`;
- verified skin membership representation;
- removal of fields unsupported by the target schema;
- deterministic JSON encoding.

A codec must not repair hierarchy, setup transforms, or constraint scheduling.

### Connected rig builder

The connected builder owns:

- generated bone hierarchy;
- global and per-object control topology;
- constraint targets and affected bones;
- one global dependency-aware constraint schedule;
- setup-pose placement and correction.

### Runtime acceptance oracle

Production readiness requires validation with the vendored Spine 4.1 runtime, not the
4.2 runtime and not a pure JSON structural test.

The external local gate is `tools/spine41_runtime_oracle.mjs`; its usage and checked
invariants are documented in `SPINE41_RUNTIME_ORACLE.md`.

## Required runtime gate

For single, standalone multi-object, and connected multi-object output, and for both rig
profiles, the gate must:

1. load the exact Spine 4.1 runtime used by the project;
2. parse the generated JSON without compatibility shims;
3. instantiate the skeleton and apply setup pose;
4. execute the runtime update-cache/setup transform path;
5. assert finite `worldX`, `worldY`, `a`, `b`, `c`, and `d` for every bone;
6. assert every expected IK/transform/path constraint is scheduled exactly once;
7. compute bounds for all setup attachments;
8. compare setup-pose bounds and control anchors against approved target fixtures;
9. reject duplicate, missing, or gapped global constraint orders when the runtime would
   skip a constraint;
10. import the generated output into Spine Editor 4.1.24.

## Re-enable conditions

Spine 4.1 may return to the production registry only when all conditions below pass:

- complete pure-Python suite;
- complete real-bpy suite;
- target-aware connected-rig tests;
- exact-runtime 4.1 acceptance gate;
- single-object Editor import;
- standalone multi-object Editor import;
- connected multi-object Editor import;
- setup-pose comparison for `3-Axis Rotation`;
- setup-pose comparison for `2-Axis Rotation + Scale`;
- proof that Spine 4.2 output remains byte-identical.

No CI/CD workflow is added or changed by this quarantine checkpoint.
