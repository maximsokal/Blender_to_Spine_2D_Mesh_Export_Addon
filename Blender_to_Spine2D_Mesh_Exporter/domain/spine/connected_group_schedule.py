"""Build and apply collision-free connected A1 constraint schedules."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from .connected_group_contracts import (
    ConnectedConstraintSchedule,
    ConnectedObjectDocument,
    ConnectedObjectPlacement,
)
from .connected_group_layout import ordered_component_ids
from .legacy_profile import LegacyRigProfile
from .model import SpineDocument
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .two_axis_scale_profile import TwoAxisScaleRigProfile


def _assign_phase(
    component_ids: Tuple[str, ...],
    next_order: int,
) -> tuple[Tuple[Tuple[str, int], ...], int]:
    assignments = tuple(
        (component_id, next_order + offset)
        for offset, component_id in enumerate(component_ids)
    )
    return assignments, next_order + len(component_ids)


def _build_legacy_schedule(
    component_ids: Tuple[str, ...],
    profile: LegacyRigProfile,
) -> ConnectedConstraintSchedule:
    """Preserve the exact historical six-constraint connected phase order."""

    next_order = 0
    global_rotation_x = next_order
    next_order += 1
    global_rotation_y = next_order
    next_order += 1
    global_rotation_z = next_order
    next_order += 1

    object_rotation_x, next_order = _assign_phase(component_ids, next_order)
    object_rotation_y, next_order = _assign_phase(component_ids, next_order)
    global_scale_ik = next_order
    next_order += 1
    object_scale_ik, next_order = _assign_phase(component_ids, next_order)
    global_scale = next_order
    next_order += 1
    object_scale, next_order = _assign_phase(component_ids, next_order)
    object_rotation_z, next_order = _assign_phase(component_ids, next_order)
    object_scale_compensator, next_order = _assign_phase(
        component_ids,
        next_order,
    )

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
        object_scale_compensator=object_scale_compensator,
        profile_id=profile.profile_id,
    )


def _build_two_axis_schedule(
    component_ids: Tuple[str, ...],
    profile: TwoAxisScaleRigProfile,
) -> ConnectedConstraintSchedule:
    """Interleave global and object work in the real five-phase X/Y/Scale order."""

    next_order = 0
    global_rotation_x = next_order
    next_order += 1
    object_rotation_x, next_order = _assign_phase(component_ids, next_order)

    global_scale_ik = next_order
    next_order += 1
    object_scale_ik, next_order = _assign_phase(component_ids, next_order)

    global_scale = next_order
    next_order += 1
    object_scale, next_order = _assign_phase(component_ids, next_order)

    global_scale_depth = next_order
    next_order += 1
    object_scale_depth, next_order = _assign_phase(component_ids, next_order)

    global_rotation_y = next_order
    next_order += 1
    object_rotation_y, next_order = _assign_phase(component_ids, next_order)

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
    """Assign one exact schedule for the selected connected rig profile."""

    resolved_profile = LegacyRigProfile() if profile is None else profile
    if not isinstance(resolved_profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    component_ids = ordered_component_ids(placements)
    profile_id = resolve_a1_rig_profile(resolved_profile.profile_id)
    if profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        return _build_legacy_schedule(component_ids, resolved_profile)
    if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(resolved_profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE connected schedule requires "
                "TwoAxisScaleRigProfile"
            )
        return _build_two_axis_schedule(component_ids, resolved_profile)
    raise AssertionError(f"Unhandled connected rig profile: {profile_id}")


def reorder_object_constraints(
    item: ConnectedObjectDocument,
    schedule: ConnectedConstraintSchedule,
    profile: LegacyRigProfile,
) -> SpineDocument:
    """Rebuild one immutable object document with scheduled global orders."""

    if not isinstance(item, ConnectedObjectDocument):
        raise TypeError("item must be ConnectedObjectDocument")
    if not isinstance(schedule, ConnectedConstraintSchedule):
        raise TypeError("schedule must be ConnectedConstraintSchedule")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    profile_id = resolve_a1_rig_profile(profile.profile_id)
    if profile_id is A1RigProfile.THREE_AXIS_ROTATION:
        order_by_name = {
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
            profile.scale_compensator_constraint(item.prefix): schedule.order_for(
                "object_scale_compensator", item.component_id
            ),
        }
    elif profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        if not isinstance(profile, TwoAxisScaleRigProfile):
            raise TypeError(
                "TWO_AXIS_ROTATION_SCALE constraint reordering requires "
                "TwoAxisScaleRigProfile"
            )
        order_by_name = {
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
    else:
        raise AssertionError(f"Unhandled connected rig profile: {profile_id}")

    actual_names = {
        constraint.name for constraint in (*item.document.ik, *item.document.transform)
    }
    if actual_names != set(order_by_name):
        raise ValueError(
            f"Connected object '{item.component_id}' constraint names changed after "
            f"validation: expected={tuple(order_by_name)}, actual={tuple(sorted(actual_names))}"
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


__all__ = ["build_constraint_schedule", "reorder_object_constraints"]
