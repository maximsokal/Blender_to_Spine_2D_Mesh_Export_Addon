"""Resolve deterministic connected A1 Z-layers and object placements."""

from __future__ import annotations

from typing import Tuple

from .connected_group_contracts import (
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    ConnectedObjectPlacement,
    ConnectedZLayer,
)
from .legacy_profile import LegacyRigProfile


def resolve_anchor(
    objects: Tuple[ConnectedObjectDocument, ...],
    settings: ConnectedGroupSettings,
) -> ConnectedObjectDocument:
    """Return the explicit anchor or preserve the first-object legacy default."""

    if settings.anchor_component_id is None:
        return objects[0]
    return next(
        item for item in objects if item.component_id == settings.anchor_component_id
    )


def resolve_layers_and_placements(
    objects: Tuple[ConnectedObjectDocument, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
) -> tuple[Tuple[ConnectedZLayer, ...], Tuple[ConnectedObjectPlacement, ...]]:
    """Cluster relative Z top-down and keep placements in source input order."""

    anchor = resolve_anchor(objects, settings)
    anchor_x, anchor_y, anchor_z = map(float, anchor.world_position)
    input_order = {item.component_id: index for index, item in enumerate(objects)}
    offsets = tuple(
        (
            item,
            float(item.world_position[0]) - anchor_x,
            float(item.world_position[1]) - anchor_y,
            float(item.world_position[2]) - anchor_z,
        )
        for item in objects
    )

    sorted_by_z = sorted(
        offsets,
        key=lambda entry: (-entry[3], input_order[entry[0].component_id]),
    )
    clusters: list[dict[str, object]] = []
    for item, _, _, relative_z in sorted_by_z:
        if (
            not clusters
            or abs(relative_z - float(clusters[-1]["representative"]))
            > settings.z_tolerance
        ):
            clusters.append(
                {
                    "representative": relative_z,
                    "component_ids": [item.component_id],
                }
            )
        else:
            component_ids = clusters[-1]["component_ids"]
            if not isinstance(component_ids, list):
                raise TypeError("internal connected Z-cluster component list is invalid")
            component_ids.append(item.component_id)

    layers = tuple(
        ConnectedZLayer(
            layer_index=layer_index,
            representative_relative_z=float(cluster["representative"]),
            component_ids=tuple(
                sorted(
                    cluster["component_ids"],
                    key=lambda component_id: input_order[component_id],
                )
            ),
            scale_bone_name=f"{settings.group_prefix}_{layer_index}_scale",
            layer_bone_name=f"{settings.group_prefix}_layer_{layer_index}",
        )
        for layer_index, cluster in enumerate(clusters)
    )
    layer_by_component = {
        component_id: layer.layer_index
        for layer in layers
        for component_id in layer.component_ids
    }
    layer_name_by_index = {
        layer.layer_index: layer.layer_bone_name for layer in layers
    }
    offset_by_component = {
        item.component_id: (relative_x, relative_y, relative_z)
        for item, relative_x, relative_y, relative_z in offsets
    }
    placements = tuple(
        ConnectedObjectPlacement(
            component_id=item.component_id,
            prefix=item.prefix,
            relative_x=offset_by_component[item.component_id][0],
            relative_y=offset_by_component[item.component_id][1],
            relative_z=offset_by_component[item.component_id][2],
            layer_index=layer_by_component[item.component_id],
            main_bone_name=profile.main_bone(item.prefix),
            parent_layer_bone_name=layer_name_by_index[
                layer_by_component[item.component_id]
            ],
        )
        for item in objects
    )
    return layers, placements


def ordered_component_ids(
    placements: Tuple[ConnectedObjectPlacement, ...],
) -> Tuple[str, ...]:
    """Order components by layer and then original placement/input order."""

    input_order = {
        placement.component_id: index
        for index, placement in enumerate(placements)
    }
    return tuple(
        placement.component_id
        for placement in sorted(
            placements,
            key=lambda item: (item.layer_index, input_order[item.component_id]),
        )
    )


__all__ = [
    "ordered_component_ids",
    "resolve_anchor",
    "resolve_layers_and_placements",
]
