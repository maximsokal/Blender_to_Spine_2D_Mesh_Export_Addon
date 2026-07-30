# Phase 3 checkpoint: Spine 4.1 native codec

## Scope

This checkpoint enables production serialization for `SPINE_4_1` while leaving the
existing Spine 4.2 serializer path unchanged. Spine 3.8, 4.0, and 4.3 remain
fail-closed before geometry preparation.

The target version written to JSON is `4.1.24`, the final Spine 4.1 Editor patch.
User-facing final JSON filenames include the exact target version, for example
`Cone_plus_2_objects_spine_4.1.24.json`.

## Implemented transformations

The canonical Rewrite document remains version-independent. `Spine41JsonCodec`
serializes it through the existing strict serializer, parses the resulting JSON into a
fully detached mapping, and applies only the following target-boundary changes:

1. `skeleton.spine` is set to `4.1.24`.
2. Bone `inherit` is renamed to the Spine 4.1 `transform` field.
3. Generic skin `constraints` membership is split into Spine 4.1 `ik` and
   `transform` arrays by resolving names against the typed canonical constraints.
4. Top-level physics constraints and per-animation physics timelines are removed.
5. The 4.2-only `skeleton.referenceScale` field is removed.
6. IK, transform, and path constraint orders are stably linearized to the exact
   contiguous range `0..N-1` after unsupported physics constraints are removed.
7. Zero scale components are stabilized only when they occur in the parent ancestry
   that Spine 4.1 world-space transform constraints must invert.

Constraint-order linearization is target-boundary behavior. The canonical connected
composition may retain same-Z-layer order ties for 4.2 parity, while Spine 4.1 receives
a unique deterministic ordinal for every constraint. Authored phase order is the
primary sort key and canonical encounter order is the stable tie-break.

The generated two-axis rig intentionally uses zero-scale axis-collapse bones. Spine
4.1 `Bone.updateAppliedTransform()` inverts a constrained bone's parent matrix without
the `onlyTranslation` special-case available in 4.2. A zero determinant can therefore
produce non-finite or extremely large applied transforms. The 4.1 codec replaces only
the unsafe zero components with `0.001`, preserves unrelated zero-scale bones, and
recomputes setup determinants before serialization. Any remaining singular parent
matrix fails closed instead of producing a visually corrupt skeleton.

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

## Output naming

The filename token is derived from the same canonical registry that writes
`skeleton.spine`. UI exports therefore produce names such as:

- `Hero_merged_spine_4.1.24.json`;
- `Cone_plus_2_objects_spine_4.1.24.json`;
- `Hero_merged_spine_4.2.43.json`.

Only the final JSON stem changes. Object texture stems, image directories, attachment
paths, and baked texture filenames remain unchanged.

## Failure policy

The codec rejects ambiguous or unknown skin constraint membership. It does not guess
whether an unknown constraint name represents IK, transform, path, or physics.
Malformed, negative, or non-integer constraint orders fail before JSON is committed.
Unsafe world-constraint ancestry must resolve to a finite parent determinant greater
than the codec stability threshold; otherwise serialization fails before atomic commit.

All conversion operates on a JSON mapping detached from `SpineDocument`; input bones,
skins, attachments, animations, and extras are not mutated.

## Regression gates

The focused tests cover:

- exact `4.1.24` metadata;
- `inherit` to `transform` conversion;
- skin constraint membership conversion;
- stable contiguous order normalization across IK, transform, and path constraints;
- targeted zero-scale stabilization for Spine 4.1 world constraints;
- preservation of unrelated and local-constraint zero scales;
- fail-closed handling of nested near-singular ancestry;
- exact-version tokens in single and multi JSON filenames;
- removal of 4.2-only physics and `referenceScale`;
- sequence preservation;
- linked-mesh `timelines` preservation;
- deterministic input immutability;
- validator forwarding;
- registry/readiness agreement;
- unchanged byte-identical Spine 4.2 serialization output.

## Pending external validation

This checkpoint is not considered externally validated until all of the following pass:

1. the complete pure-Python suite;
2. the complete real-bpy suite;
3. import of generated single, standalone multi, and connected JSON into Spine 4.1.24;
4. setup-pose comparison of controls, weighted meshes, and connected ordering.

No CI/CD workflow is added or changed by this phase.
