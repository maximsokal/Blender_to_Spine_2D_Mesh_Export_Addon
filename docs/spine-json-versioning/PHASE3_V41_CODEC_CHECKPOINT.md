# Phase 3 checkpoint: Spine 4.1 native codec

## Scope

This checkpoint enables production serialization for `SPINE_4_1` while leaving the
existing Spine 4.2 serializer path unchanged. Spine 3.8, 4.0, and 4.3 remain
fail-closed before geometry preparation.

The target version written to JSON is `4.1.19`.

## Implemented transformations

The canonical Rewrite document remains version-independent. `Spine41JsonCodec`
serializes it through the existing strict serializer, parses the resulting JSON into a
fully detached mapping, and applies only the following target-boundary changes:

1. `skeleton.spine` is set to `4.1.19`.
2. Bone `inherit` is renamed to the Spine 4.1 `transform` field.
3. Generic skin `constraints` membership is split into Spine 4.1 `ik` and
   `transform` arrays by resolving names against the typed canonical constraints.
4. Top-level physics constraints and per-animation physics timelines are removed.
5. The 4.2-only `skeleton.referenceScale` field is removed.

## Preserved structures

Spine 4.1 already supports the structures below, so the codec preserves them without
conversion:

- IK and transform constraint setup data;
- mesh edge coordinate offsets;
- weighted vertices;
- array-shaped skins;
- linked meshes using `parent` and `timelines`;
- attachment `sequence` objects and sequence timelines;
- modern 4.x Bezier curve arrays;
- slot RGB/RGBA/alpha timeline families.

Preview animation remains disabled by the shared immutable Scene capture contract.
This does not disable real attachment sequences requested through frame baking.

## Failure policy

The codec rejects ambiguous or unknown skin constraint membership. It does not guess
whether an unknown constraint name represents IK, transform, path, or physics.

All conversion operates on a JSON mapping detached from `SpineDocument`; input bones,
skins, attachments, animations, and extras are not mutated.

## Regression gates

The focused tests cover:

- exact `4.1.19` metadata;
- `inherit` to `transform` conversion;
- skin constraint membership conversion;
- removal of 4.2-only physics and `referenceScale`;
- sequence preservation;
- linked-mesh `timelines` preservation;
- deterministic input immutability;
- validator forwarding;
- registry/readiness agreement;
- unchanged byte-identical Spine 4.2 output.

## Pending external validation

This checkpoint is not considered externally validated until all of the following pass:

1. the complete pure-Python suite;
2. the complete real-bpy suite;
3. import of generated single, standalone multi, and connected JSON into Spine 4.1;
4. setup-pose comparison of controls, weighted meshes, and connected ordering.

No CI/CD workflow is added or changed by this phase.
