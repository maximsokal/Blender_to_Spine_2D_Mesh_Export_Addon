"""Build and apply Legacy-compatible connected A1 constraint schedules."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from .connected_group_contracts import (
    ConnectedConstraintSchedule,
    ConnectedObjectDocument,
    ConnectedObjectPlacement,
)
from .legacy_profile import LegacyRigProfile
from .model import SpineDocument
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .two_axis_scale_profile import TwoAxisScaleRigProfile


# The old single-object builder always serialized the compensator at order 6. The
# historical connected merger did not renumber or remove it, so exact main-branch
# parity requires preserving this value for every connected three-axis object.
_LEGACY_SCALE_COMPENSATOR_ORDER = 6


def _layer_count(placements: Tuple[ConnectedObjectPlacement, ...]) -> int:
    if not isinstance(placements, tuple) or not placements:
        raise ValueError("placements must be a non-empty tuple")
    if not all(isinstance(item, ConnectedObjectPlacement) for item in placements):
        raise TypeError("placements must contain ConnectedObjectPlacement values")
    return max(item.layer_index for item in placements) + 1


def _assign_layer_phase(
    placements: Tuple[ConnectedObjectPlacement, ...],
    first_order: int,
) -> tuple[Tuple[Tuple[str, int], ...], int]:
    """Assign one order per Z layer while preserving source component order."""

    count = _layer_count(placements)
    assignments = tuple(
        (placement.component_id, first_order + placement.layer_index)
        for placement in placements
    )
    return assignments, first_order + count


def _build_legacy_schedule(
    placements: Tuple[ConnectedObjectPlacement, ...],
    profile: LegacyRigProfile,
) -> ConnectedConstraintSchedule:
    """Reproduce the connected order formula from historical ``main``."""

    next_order = 0
    global_rotation_x = next_order
    next_order += 1
    global_rotation_y = next_order
    next_order += 1
    global_rotation_z = next_order
    next_order += 1

    object_rotation_x, next_order = _assign_layer_phase(placements, next_order)
    object_rotation_y, next_order = _assign_layer_phase(placements, next_order)

    global_scale_ik = next_order
    next_order += 1
    object_scale_ik, next_order = _assign_layer_phase(placements, next_order)

    global_scale = next_order
    next_order += 1
    object_scale, next_order = _assign_layer_phase(placements, next_order)
    object_rotation_z, next_order = _assign_layer_phase(placements, next_order)

    return ConnectedConstraintSchedule(
        global_rotation_x=global_rotation_x,
        global_rotation_y=global_rotation_y,
        global_rotation_z=global_rotation_z,
        object_rotation_x=object_rotation_x,
        object_rotation_y=object_rotation_y,
        global_scale_ik=global_scale_ik,
        object_scale_ik=object_scale_ik,
        global_scale=global_scale,
        object_scale=object_scale,
        object_rotation_z=object_rotation_z,
        # Compensators keep their original standalone order and are intentionally not
        # part of the layer-phase model, exactly like main._renumber_object_constraints.
        object_scale_compensator=(),
        profile_id=profile.profile_id,
    )


def _build_two_axis_schedule(
    placements: Tuple[ConnectedObjectPlacement, ...],
    profile: TwoAxisScaleRigProfile,
) -> ConnectedConstraintSchedule:
    """Generalize Legacy layer phases to the five two-axis operations."""

    next_order = 0
    global_rotation_x = next_order
    next_order += 1
    object_rotation_x, next_order = _assign_layer_phase(placements, next_order)

    global_scale_ik = next_order
    next_order += 1
    object_scale_ik, next_order = _assign_layer_phase(placements, next_order)

    global_scale = next_order
    next_order += 1
    object_scale, next_order = _assign_layer_phase(placements, next_order)

    global_scale_depth = next_order
    next_order += 1
    object_scale_depth, next_order = _assign_layer_phase(placements, next_order)

    global_rotation_y = next_order
    next_order += 1
    object_rotation_y, next_order = _assign_layer_phase(placements, next_order)

    return ConnectedConstraintSchedule(
        global_rotation_x=global_rotation_x,
        global_rotation_y=global_rotation_y,
        global_rotation_z=None,
        object_rotation_x=object_rotation_x,
        object_rotation_y=object_rotation_y,
        global_scale_ik=global_scale_ik,
        object_scale_ik=object_scale_ik,
        global_scale=global_scale,
        object_scale=object_scale,
        object_rotation_z=(),
        object_scale_compensator=(),
        global_scale_depth=global_scale_depth,
        object_scale_depth=object_scale_depth,
        profile_id=profile.profile_id,
    )


def build_constraint_schedule(
    placements: Tuple[ConnectedObjectPlacement, ...],
    profile: LegacyRigProfile | None = None,
) -> ConnectedConstraintSchedule:
    """Assign the exact connected schedule for the selected rig profile."""

    resolved_profile = LegacyRigProfile() if profile is None else profile
    if not isinstance(resolved_profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    profile_id = resolve_a1_rig_profile(resolved_profile.profile_id)
    if profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        return _build_legacy_schedule(placements, resolved_profile)
    if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(resolved_profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE connected schedule requires "
                "TwoAxisScaleRigProfile"
            )
        return _build_two_axis_schedule(placements, resolved_profile)
    raise AssertionError(f"Unhandled connected rig profile: {profile_id}")


def _object_order_by_name(
    item: ConnectedObjectDocument,
    schedule: ConnectedConstraintSchedule,
    profile: LegacyRigProfile,
) -> dict[str, int]:
    profile_id = resolve_a1_rig_profile(profile.profile_id)
    if profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        return {
            profile.rotation_x_constraint(item.prefix): schedule.order_for(
                "object_rotation_x", item.component_id
            ),
            profile.rotation_y_constraint(item.prefix): schedule.order_for(
                "object_rotation_y", item.component_id
            ),
            profile.scale_ik_constraint(item.prefix): schedule.order_for(
                "object_scale_ik", item.component_id
            ),
            profile.scale_constraint(item.prefix): schedule.order_for(
                "object_scale", item.component_id
            ),
            profile.rotation_z_constraint(item.prefix): schedule.order_for(
                "object_rotation_z", item.component_id
            ),
            profile.scale_compensator_constraint(
                item.prefix
            ): _LEGACY_SCALE_COMPENSATOR_ORDER,
        }
    if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE constraint scheduling requires "
                "TwoAxisScaleRigProfile"
            )
        return {
            profile.rotation_x_constraint(item.prefix): schedule.order_for(
                "object_rotation_x", item.component_id
            ),
            profile.scale_ik_constraint(item.prefix): schedule.order_for(
                "object_scale_ik", item.component_id
            ),
            profile.scale_constraint(item.prefix): schedule.order_for(
                "object_scale", item.component_id
            ),
            profile.scale_depth_constraint(item.prefix): schedule.order_for(
                "object_scale_depth", item.component_id
            ),
            profile.rotation_y_constraint(item.prefix): schedule.order_for(
                "object_rotation_y", item.component_id
            ),
        }
    raise AssertionError(f"Unhandled connected rig profile: {profile_id}")


def reorder_object_constraints(
    item: ConnectedObjectDocument,
    schedule: ConnectedConstraintSchedule,
    profile: LegacyRigProfile,
) -> SpineDocument:
    """Return one object document with connected Legacy-compatible orders."""

    if not isinstance(item, ConnectedObjectDocument):
        raise TypeError("item must be ConnectedObjectDocument")
    if not isinstance(schedule, ConnectedConstraintSchedule):
        raise TypeError("schedule must be ConnectedConstraintSchedule")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    order_by_name = _object_order_by_name(item, schedule, profile)
    actual_names = {
        constraint.name for constraint in (*item.document.ik, *item.document.transform)
    }
    if actual_names != set(order_by_name):
        raise ValueError(
            f"Connected object '{item.component_id}' constraint names changed after "
            f"validation: expected={tuple(sorted(order_by_name))}, "
            f"actual={tuple(sorted(actual_names))}"
        )

    return replace(
        item.document,
        ik=tuple(
            replace(constraint, order=order_by_name[constraint.name])
            for constraint in item.document.ik
        ),
        transform=tuple(
            replace(constraint, order=order_by_name[constraint.name])
            for constraint in item.document.transform
        ),
    )


def apply_connected_constraint_schedule(
    document: SpineDocument,
    objects: Tuple[ConnectedObjectDocument, ...],
    schedule: ConnectedConstraintSchedule,
    profile: LegacyRigProfile,
    group_prefix: str,
) -> SpineDocument:
    """Apply final global and object orders after safe typed composition."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(objects, tuple) or not objects:
        raise ValueError("objects must be a non-empty tuple")
    if not isinstance(schedule, ConnectedConstraintSchedule):
        raise TypeError("schedule must be ConnectedConstraintSchedule")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")
    if not isinstance(group_prefix, str) or not group_prefix.strip():
        raise ValueError("group_prefix must be a non-empty string")

    profile_id = resolve_a1_rig_profile(profile.profile_id)
    order_by_name: dict[str, int] = {
        profile.rotation_x_constraint(group_prefix): schedule.global_rotation_x,
        profile.rotation_y_constraint(group_prefix): schedule.global_rotation_y,
        profile.scale_ik_constraint(group_prefix): schedule.global_scale_ik,
        profile.scale_constraint(group_prefix): schedule.global_scale,
    }
    if profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        if schedule.global_rotation_z is None:
            raise ValueError("legacy connected schedule has no global Rotation Z order")
        order_by_name[profile.rotation_z_constraint(group_prefix)] = (
            schedule.global_rotation_z
        )
    elif profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE final scheduling requires "
                "TwoAxisScaleRigProfile"
            )
        if schedule.global_scale_depth is None:
            raise ValueError("two-axis connected schedule has no global depth order")
        order_by_name[profile.scale_depth_constraint(group_prefix)] = (
            schedule.global_scale_depth
        )
    else:
        raise AssertionError(f"Unhandled connected rig profile: {profile_id}")

    for item in objects:
        object_orders = _object_order_by_name(item, schedule, profile)
        overlap = set(order_by_name).intersection(object_orders)
        if overlap:
            raise ValueError(
                "Connected schedule produced duplicate constraint names: "
                f"{tuple(sorted(overlap))}"
            )
        order_by_name.update(object_orders)

    actual_names = {
        constraint.name for constraint in (*document.ik, *document.transform)
    }
    if actual_names != set(order_by_name):
        raise ValueError(
            "Connected final constraint schema differs from the scheduled schema: "
            f"missing={tuple(sorted(set(order_by_name) - actual_names))}, "
            f"unexpected={tuple(sorted(actual_names - set(order_by_name)))}"
        )

    return replace(
        document,
        ik=tuple(
            replace(constraint, order=order_by_name[constraint.name])
            for constraint in document.ik
        ),
        transform=tuple(
            replace(constraint, order=order_by_name[constraint.name])
            for constraint in document.transform
        ),
    )


__all__ = [
    "apply_connected_constraint_schedule",
    "build_constraint_schedule",
    "reorder_object_constraints",
]
