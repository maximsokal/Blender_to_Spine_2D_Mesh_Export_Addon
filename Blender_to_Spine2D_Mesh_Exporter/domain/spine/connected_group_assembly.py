"""Assemble connected A1 object documents under one global profile-aware rig."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from .composition import (
    ConstraintOrderPolicy,
    SpineCompositionSettings,
    SpineDocumentComponent,
    compose_spine_documents,
)
from .connected_group_contracts import (
    ConnectedGroupBuildResult,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    ConnectedObjectPlacement,
    ConnectedPlacementSpace,
)
from .connected_group_draw_order import apply_connected_setup_draw_order
from .connected_group_error import ConnectedGroupBuildError
from .connected_group_global_rig import (
    build_global_bones_document,
    build_global_constraints,
)
from .connected_group_layout import resolve_layers_and_placements
from .connected_group_schedule import (
    build_constraint_schedule,
    reorder_object_constraints,
)
from .connected_group_validation import (
    validate_connected_global_namespace,
    validate_connected_group_inputs,
)
from .legacy_profile import LegacyRigProfile
from .legacy_rig_scale import calculate_uniform_scale
from .model import Bone, SpineDocument
from .validator import SpineValidator


def apply_object_placements(
    document: SpineDocument,
    placements: Tuple[ConnectedObjectPlacement, ...],
    uniform_scale: float,
) -> SpineDocument:
    """Reparent object main bones in their declared connected coordinate space.

    Connected object-bake documents intentionally keep only document-local XY on
    ``<prefix>_main``. That local value compensates for centering attachment vertex bones
    around the geometry bounding-box midpoint. Connected composition adds the
    anchor-relative Object translation to this existing offset instead of replacing it.

    Camera-projection documents already encode screen-space XY in attachment vertices;
    their main-bone coordinates are preserved while only the Z-layer parent changes.
    """

    placement_by_main = {
        placement.main_bone_name: placement for placement in placements
    }
    found: set[str] = set()
    updated_bones: list[Bone] = []
    for bone in document.bones:
        placement = placement_by_main.get(bone.name)
        if placement is None:
            updated_bones.append(bone)
            continue
        found.add(bone.name)
        if placement.placement_space is ConnectedPlacementSpace.ANCHOR_RELATIVE_WORLD:
            updated_bones.append(
                replace(
                    bone,
                    parent=placement.parent_layer_bone_name,
                    x=round(
                        float(bone.x) + placement.relative_x * uniform_scale,
                        2,
                    ),
                    y=round(
                        float(bone.y) + placement.relative_y * uniform_scale,
                        2,
                    ),
                )
            )
            continue
        if placement.placement_space is ConnectedPlacementSpace.PRESERVE_DOCUMENT:
            updated_bones.append(
                replace(
                    bone,
                    parent=placement.parent_layer_bone_name,
                )
            )
            continue
        raise TypeError(
            f"Unsupported connected placement space: {placement.placement_space!r}"
        )

    missing = set(placement_by_main) - found
    if missing:
        raise ConnectedGroupBuildError(
            "Unable to apply placements; main bones missing: "
            f"{tuple(sorted(missing))}"
        )
    return replace(document, bones=tuple(updated_bones))


def build_connected_group_document(
    objects: Tuple[ConnectedObjectDocument, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile | None = None,
) -> ConnectedGroupBuildResult:
    """Compose A1 object documents under one collision-free global control rig."""

    if not isinstance(settings, ConnectedGroupSettings):
        raise TypeError("settings must be ConnectedGroupSettings")
    resolved_profile = LegacyRigProfile() if profile is None else profile
    if not isinstance(resolved_profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    validate_connected_group_inputs(objects, settings, resolved_profile)
    layers, placements = resolve_layers_and_placements(
        objects,
        settings,
        resolved_profile,
    )
    validate_connected_global_namespace(
        objects,
        settings,
        resolved_profile,
        layers,
    )
    schedule = build_constraint_schedule(placements, resolved_profile)
    uniform_scale = calculate_uniform_scale(
        settings.texture_width,
        settings.texture_height,
        settings.scale_mode,
    )
    global_document = build_global_bones_document(
        objects[0].document.skeleton,
        layers,
        settings,
        resolved_profile,
        uniform_scale,
    )

    object_components = tuple(
        SpineDocumentComponent(
            component_id=item.component_id,
            document=reorder_object_constraints(
                item,
                schedule,
                resolved_profile,
            ),
            animation_namespace=(item.animation_namespace or item.component_id),
        )
        for item in objects
    )
    composition = compose_spine_documents(
        (
            SpineDocumentComponent(
                component_id=f"__{settings.group_prefix}_rig__",
                document=global_document,
                animation_namespace=settings.group_prefix,
            ),
            *object_components,
        ),
        SpineCompositionSettings(
            shared_bone_names=(resolved_profile.root_bone(),),
            constraint_order_policy=ConstraintOrderPolicy.PRESERVE,
            namespace_animations=settings.namespace_animations,
            animation_separator=settings.animation_separator,
        ),
    )

    draw_order_document = apply_connected_setup_draw_order(
        composition.document,
        objects,
        placements,
    )
    placed_document = apply_object_placements(
        draw_order_document,
        placements,
        uniform_scale,
    )
    global_ik, global_transform = build_global_constraints(
        objects,
        layers,
        schedule,
        settings,
        resolved_profile,
        uniform_scale,
    )
    final_document = replace(
        placed_document,
        ik=tuple(
            sorted(
                (*placed_document.ik, *global_ik),
                key=lambda item: (item.order, item.name),
            )
        ),
        transform=tuple(
            sorted(
                (*placed_document.transform, *global_transform),
                key=lambda item: (item.order, item.name),
            )
        ),
    )
    try:
        SpineValidator().validate_or_raise(final_document)
    except Exception as exc:
        raise ConnectedGroupBuildError(
            f"Connected A1 group failed final validation: {exc}"
        ) from exc

    composition = replace(composition, document=final_document)
    return ConnectedGroupBuildResult(
        document=final_document,
        composition=composition,
        settings=settings,
        layers=layers,
        placements=placements,
        constraint_schedule=schedule,
        uniform_scale=uniform_scale,
    )


__all__ = ["apply_object_placements", "build_connected_group_document"]
