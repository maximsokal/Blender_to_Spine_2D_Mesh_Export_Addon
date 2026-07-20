# Connected group composition split

## Scope

The former `domain/spine/legacy_connected_group.py` combined public contracts,
source validation, Z-layer layout, constraint scheduling, global rig construction,
typed document composition, placement rewriting and final validation in one
761-line module.

The public connected A1 output contract remains unchanged. The former module is
now a compatibility facade and production imports physical owners directly.

## Physical ownership

```text
connected_group_error.py
  -> ConnectedGroupBuildError

connected_group_contracts.py
  -> ConnectedObjectDocument
  -> ConnectedGroupSettings
  -> ConnectedZLayer
  -> ConnectedObjectPlacement
  -> ConnectedConstraintSchedule
  -> ConnectedGroupBuildResult

connected_group_validation.py
  -> source-document preflight
  -> exact legacy IK/Transform constraint schema
  -> global component and bone namespace preflight

connected_group_layout.py
  -> anchor resolution
  -> top-down Z clustering
  -> input-ordered placements
  -> layer-ordered component IDs

connected_group_schedule.py
  -> contiguous connected constraint schedule
  -> immutable object-constraint order replacement

connected_group_global_rig.py
  -> global all_objects bones-only document
  -> four global connected constraints

connected_group_assembly.py
  -> complete typed connected composition
  -> object-main placement rewrite
  -> final Spine validation

legacy_connected_group.py
  -> public and historical private compatibility re-exports only
```

## Production flow

```text
validate settings/profile and source documents
-> validate exact one-IK/five-Transform object constraint schema
-> resolve anchor, layers and placements
-> validate generated global component/bone namespace
-> build contiguous constraint schedule
-> calculate legacy uniform scale
-> build valid global bones-only component
-> reorder immutable object constraints
-> compose typed documents with requested animation namespace policy
-> reparent object main bones and apply XY offsets
-> append global constraints in scheduled order
-> validate final Spine document
-> ConnectedGroupBuildResult
```

## Fixed animation namespace contract

`A1MultiObjectExportSettings.namespace_animations` was previously honored by
STANDALONE composition but discarded by CONNECTED composition. The connected
builder always forced `namespace_animations=True`.

`ConnectedGroupSettings` now contains an additive keyword field:

```python
namespace_animations: bool = True
```

`a1_multi_object_composition.py` passes the actual application setting to the
physical connected builder.

```text
True
  -> <animation_namespace>/<animation_name>

False
  -> original animation name
  -> duplicate original names fail explicitly in typed composition
```

The new field follows the existing `animation_separator` field so the historical
seven-argument positional constructor keeps its exact meaning.

## Canonical connected namespaces

Identity values must be non-empty and cannot contain leading or trailing
whitespace:

```text
component_id
object prefix
group_prefix
anchor_component_id
animation_namespace
```

This closes the case where `"Hero"` and `" Hero "` passed raw-string uniqueness
checks but were both normalized by `LegacyRigProfile` into the same Spine bone
namespace.

Before composition, validation also rejects:

- an input component ID equal to the generated `__<group_prefix>_rig__` ID;
- generated global rig names colliding with source document bones;
- generated layer names colliding with source document bones;
- duplicate generated global names from a custom profile.

The shared root bone remains the only intentional cross-component bone name.

## Exact object constraint schema

Every connected source document must contain:

```text
IK
  -> exactly <prefix>_scale_constraint_IK

Transform
  -> <prefix>_rotation_X
  -> <prefix>_rotation_Y
  -> <prefix>_scale_constraint
  -> <prefix>_rotation_Z
  -> <prefix>_scale_compensator
```

Correct names in the wrong IK/Transform collection are rejected before schedule
construction. General `SpineValidator` cross-reference validation still runs
first and final document validation still runs after composition.

## Strict numeric and profile inputs

Connected contracts now reject Python booleans where integer/float domain values
are required, including dimensions, Z tolerance, world positions, layer indexes
and schedule orders.

An explicitly supplied falsy profile is no longer treated as absence:

```python
resolved_profile = LegacyRigProfile() if profile is None else profile
```

Therefore `profile=0` fails with `TypeError` instead of silently selecting the
default profile.

## Preserved golden behavior

The split does not change:

- default `namespace_animations=True`;
- root sharing;
- default `all_objects` prefix;
- first-object anchor default;
- explicit anchor behavior;
- top-down Z clustering and tolerance;
- input order inside one layer;
- zero-based connected layer indexes;
- global bone names and order;
- connected constraint phase order;
- object XY placement scaling and rounding;
- weighted attachment bone-index remapping;
- source document immutability;
- Spine version matching;
- B4 grouped rendering and output transactions.

## Compatibility facade

`legacy_connected_group.py` retains:

```text
ConnectedObjectDocument
ConnectedGroupSettings
ConnectedZLayer
ConnectedObjectPlacement
ConnectedConstraintSchedule
ConnectedGroupBuildResult
ConnectedGroupBuildError
build_connected_group_document
```

Historical private names remain aliases to physical functions:

```text
_validate_inputs
_anchor
_resolve_layers_and_placements
_ordered_component_ids
_build_constraint_schedule
_reorder_object_constraints
_build_global_bones_document
_build_global_constraints
_apply_object_placements
```

## Validation

Focused local split-mirror validation performed for this slice:

- all new/replaced connected-group modules compile;
- 12 architecture and behavior tests pass;
- the historical Z-layer layout is unchanged;
- the historical constraint schedule is unchanged;
- distinct animations remain un-namespaced when requested;
- duplicate un-namespaced animations fail explicitly;
- whitespace-equivalent identities fail before composition;
- global component and bone namespace collisions fail before composition;
- wrong IK/Transform constraint placement fails before schedule construction;
- bool numeric values fail contract validation;
- explicit falsy profiles fail type validation;
- public and historical private aliases resolve to physical owners.

A separate test protects historical positional `ConnectedGroupSettings`
construction while confirming `namespace_animations` is an additive keyword
contract.

The complete repository pytest suite and Blender 4.4 matrices have not been run
on this HEAD. GitHub Actions remain manual-only.
