# Legacy A1 Rig Ownership Split

## Scope

The historical `legacy_rig_builder.py` mixed immutable contracts, scale policy,
Z-group layout, name resolution, bone generation, constraint generation, semantic
validation, and final assembly. It is now a compatibility facade over physical
owners.

This slice preserves the Spine 4.2.43 legacy rotatable-mesh payload while making
invalid numeric and namespace states fail before serialization.

## Physical ownership

```text
legacy_rig_error.py
    LegacyRigBuildError

legacy_rig_contracts.py
    UniformScaleMode
    LegacyZGroup
    LegacyRigBuildRequest
    LegacyZGroupBuildInfo
    LegacyRigInfo
    LegacyRigBuildResult

legacy_rig_scale.py
    calculate_uniform_scale
    resolve_main_position
    finite derived-value checks

legacy_rig_plan.py
    LegacyRigBuildPlan
    sorted Z metadata
    dense Z indices
    resolved names
    namespace collision detection

legacy_rig_bones.py
    core bones
    Z-group bones
    control bones
    IK-chain bones

legacy_rig_constraints.py
    exact one-IK/five-Transform legacy payload

legacy_rig_validation.py
    plan consistency
    exact result parity
    finite bone/constraint payload
    generic Spine cross-reference validation

legacy_rig_assembly.py
    build_legacy_rig orchestration

legacy_rig_builder.py
    compatibility re-exports only
```

## Build flow

```text
LegacyRigBuildRequest
-> resolve explicit/default LegacyRigProfile
-> build immutable LegacyRigBuildPlan
-> validate scale, main position, Z metadata, and names
-> build exact ordered bones
-> build exact legacy constraints
-> construct LegacyRigBuildResult
-> compare result with deterministic plan
-> validate finite numeric payload
-> validate Spine cross-references
```

## Preserved output contract

The generated bone order remains:

```text
root
<prefix>_main
<prefix>
<prefix>_scale_rotate_X
<prefix>_rotate_X
ordered Z scale/bone pairs
<prefix>_rotation_X
<prefix>_rotation_Y
<prefix>_rotation_Z
four IK-chain bones
```

Constraint names, orders, targets, extras, colors, icons, parent order, Z sorting,
main-position rounding, negative `height_real_pixels`, and all three scale modes are
unchanged.

## Hardening

### Bool is not numeric domain data

`bool` is rejected for:

- texture dimensions;
- Z values and height overrides;
- average/main positions;
- `bone_for_z` query and tolerance;
- profile Z/segment/vertex indices.

### Canonical request prefix

`LegacyRigBuildRequest.prefix` must not contain leading or trailing whitespace.
This prevents disagreement between `result.request.prefix`, `result.info.prefix`,
and names generated through `LegacyRigProfile`.

Internal spaces remain valid.

### Explicit profile semantics

Only `profile=None` selects the default profile. Explicit falsy values are rejected
instead of being silently replaced.

### Derived finite values

Scale, half-scale, doubled scale, main position, Z deltas, Z offsets, bone numeric
fields, and numeric constraint extras must remain finite.

AVERAGE scale first uses the historical `(width + height) / 2` arithmetic whenever
the sum is finite, preserving old rounding. The half-plus-half fallback is used only
when the legacy sum would overflow.

### Early namespace validation

All generated bone names are resolved before `Bone` construction and must be
unique. For example, prefix `root` with the default profile fails during planning
instead of after building the complete rig.

### Exact result validation

`LegacyRigBuildResult.validate()` no longer checks only generic Spine references.
It rebuilds the deterministic plan and verifies exact info, bones, IK constraints,
Transform constraints, numeric payload, and cross-references.

## Compatibility facade

The facade preserves:

```text
UniformScaleMode
LegacyZGroup
LegacyRigBuildRequest
LegacyZGroupBuildInfo
LegacyRigInfo
LegacyRigBuildResult
LegacyRigBuildError
calculate_uniform_scale
build_legacy_rig
```

Historical private entrypoints also remain:

```text
_main_position
_build_z_group_bones
_build_constraints
```

Their signatures and return shapes are unchanged.

## Production imports

The main A1 document preparation, single-object settings, Z-group assignment,
connected-group contracts/assembly, and package exports use physical owners rather
than the facade.

The facade remains available for external callers and compatibility imports.

## Validation performed for this slice

- local compileall of the focused mirror;
- 24 focused architecture and behavior tests;
- exact legacy bone/constraint order checks in the focused harness;
- acyclic physical top-level import graph.

The complete repository pytest suite and Blender matrices were not run for this
slice. GitHub Actions remain manual-only and were not triggered.
