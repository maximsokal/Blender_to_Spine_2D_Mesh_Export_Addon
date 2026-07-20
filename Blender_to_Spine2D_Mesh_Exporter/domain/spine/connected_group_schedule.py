"""Build and apply the collision-free connected A1 constraint schedule."""

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


def build_constraint_schedule(
    placements: Tuple[ConnectedObjectPlacement, ...],
) -> ConnectedConstraintSchedule:
    """Assign the exact historical connected constraint phase order."""

    component_ids = ordered_component_ids(placements)
    next_order = 0

    global_rotation_x = next_order
    next_order += 1
    global_rotation_y = next_order
    next_order += 1
    global_rotation_z = next_order
    next_order += 1

    def assign_phase() -> Tuple[Tuple[str, int], ...]:
        nonlocal next_order
        assignments = tuple(
            (component_id, next_order + offset)
            for offset, component_id in enumerate(component_ids)
        )
        next_order += len(component_ids)
        return assignments

    object_rotation_x = assign_phase()
    object_rotation_y = assign_phase()
    global_scale_ik = next_order
    next_order += 1
    object_scale_ik = assign_phase()
    global_scale = next_order
    next_order += 1
    object_scale = assign_phase()
    object_rotation_z = assign_phase()
    object_scale_compensator = assign_phase()

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
    )


def reorder_object_constraints(
    item: ConnectedObjectDocument,
    schedule: ConnectedConstraintSchedule,
    profile: LegacyRigProfile,
) -> SpineDocument:
    """Rebuild one immutable object document with its scheduled global orders."""

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
