"""Assemble connected A1 object documents under one profile-aware global rig."""

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
from .connected_group_object_setup import normalize_connected_object_control_space
from .connected_group_schedule import (
    apply_connected_constraint_schedule,
    build_constraint_schedule,
)
from .connected_group_setup_correction import correct_connected_setup_pose
from .connected_group_validation import (
    validate_connected_global_namespace,
    validate_connected_group_inputs,
)
from .legacy_profile import LegacyRigProfile
from .legacy_rig_scale import calculate_uniform_scale
from .model import Bone, SpineDocument
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .validator import SpineValidator


def apply_object_placements(
    document: SpineDocument,
    placements: Tuple[ConnectedObjectPlacement, ...],
    uniform_scale: float,
) -> SpineDocument:
    """Reparent object mains and preserve the historical full XY offset.

    The Legacy connected wrapper stores no setup translation on its generated Z layers,
    so ``<prefix>_main`` receives the complete anchor-relative Blender X/Y translation,
    exactly as ``main._apply_offsets`` did. The two-axis wrapper still has profile-owned
    layer setup translation; that profile alone is compensated after global constraints
    are assembled.
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
                        float(bone.x or 0.0)
                        + placement.relative_x * float(uniform_scale),
                        2,
                    ),
                    y=round(
                        float(bone.y or 0.0)
                        + placement.relative_y * float(uniform_scale),
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


def _validate_connected_final(document: SpineDocument) -> None:
    """Validate everything except intentional Legacy same-layer order sharing."""

    issues = tuple(
        issue
        for issue in SpineValidator().validate(document)
        if issue.code != "DUPLICATE_CONSTRAINT_ORDER"
    )
    if issues:
        details = "\n".join(
            f"- [{issue.code}] {issue.path}: {issue.message}" for issue in issues
        )
        raise ConnectedGroupBuildError(
            "Connected A1 group failed final validation:\n" + details
        )


def build_connected_group_document(
    objects: Tuple[ConnectedObjectDocument, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile | None = None,
) -> ConnectedGroupBuildResult:
    """Compose A1 object documents under one Legacy-compatible connected wrapper."""

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

    normalized_objects = tuple(
        normalize_connected_object_control_space(item, resolved_profile)
        for item in objects
    )
    object_components = tuple(
        SpineDocumentComponent(
            component_id=item.component_id,
            document=item.document,
            animation_namespace=(item.animation_namespace or item.component_id),
        )
        for item in normalized_objects
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
            # Generic composition remains collision-strict. Final Legacy layer orders are
            # applied only after every weighted bone index has been safely remapped.
            constraint_order_policy=ConstraintOrderPolicy.REBASE_CONTIGUOUS,
            namespace_animations=settings.namespace_animations,
            animation_separator=settings.animation_separator,
        ),
    )

    draw_order_document = apply_connected_setup_draw_order(
        composition.document,
        normalized_objects,
        placements,
    )
    placed_document = apply_object_placements(
        draw_order_document,
        placements,
        uniform_scale,
    )
    global_ik, global_transform = build_global_constraints(
        normalized_objects,
        layers,
        schedule,
        settings,
        resolved_profile,
        uniform_scale,
    )
    # Keep the exact main-branch array ownership: object constraints already exist in
    # component order and global constraints are appended afterward. Only their ``order``
    # values are replaced; the arrays are deliberately not sorted.
    with_global_constraints = replace(
        placed_document,
        ik=(*placed_document.ik, *global_ik),
        transform=(*placed_document.transform, *global_transform),
    )

    profile_id = resolve_a1_rig_profile(resolved_profile.profile_id)
    if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        # Two-axis generated layers intentionally store depth in setup Y. Compensate that
        # profile only; the Legacy connected layers are neutral and need no correction.
        with_global_constraints = correct_connected_setup_pose(
            with_global_constraints,
            normalized_objects,
            layers,
            placements,
            resolved_profile,
            settings.group_prefix,
            uniform_scale,
        )

    final_document = apply_connected_constraint_schedule(
        with_global_constraints,
        normalized_objects,
        schedule,
        resolved_profile,
        settings.group_prefix,
    )
    _validate_connected_final(final_document)

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
