"""Assemble connected A1 object documents under one target-aware global rig."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

from .composition import (
    ConstraintOrderAssignment,
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
    validate_constraint_schedule_for_target,
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
from .spine41_setup_safety import (
    Spine41RigSafetyError,
    validate_spine41_setup_safety,
)
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .two_axis_scale_spine41 import (
    adapt_connected_two_axis_constraints_for_spine41,
    adapt_two_axis_document_for_spine41,
)
from .validator import SpineValidator
from .version_target import (
    DEFAULT_SPINE_JSON_TARGET,
    SpineJsonTarget,
    resolve_spine_json_target,
)


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


def _validate_connected_final(
    document: SpineDocument,
    spine_target: SpineJsonTarget,
) -> None:
    """Validate the final document using the selected runtime order contract."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(spine_target, SpineJsonTarget):
        raise TypeError("spine_target must be SpineJsonTarget")

    issues = SpineValidator().validate(document)
    if spine_target is SpineJsonTarget.SPINE_4_2:
        issues = tuple(
            issue for issue in issues if issue.code != "DUPLICATE_CONSTRAINT_ORDER"
        )

    if issues:
        details = "\n".join(
            f"- [{issue.code}] {issue.path}: {issue.message}" for issue in issues
        )
        raise ConnectedGroupBuildError(
            f"Connected A1 group failed {spine_target.exact_version} validation:\n"
            + details
        )


def _validate_target_runtime_safety(
    document: SpineDocument,
    spine_target: SpineJsonTarget,
) -> None:
    """Run target-specific setup checks owned by the target runtime boundary."""

    if spine_target is not SpineJsonTarget.SPINE_4_1:
        return
    try:
        validate_spine41_setup_safety(document)
    except Spine41RigSafetyError as exc:
        raise ConnectedGroupBuildError(
            "Connected A1 group is not safe for Spine 4.1 setup evaluation: "
            + str(exc)
        ) from exc


def _connected_constraint_assignments(
    document: SpineDocument,
    objects: Tuple[ConnectedObjectDocument, ...],
    group_component_id: str,
) -> Tuple[ConstraintOrderAssignment, ...]:
    """Describe the actual final connected orders instead of temporary rebase values."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(objects, tuple) or not objects:
        raise ValueError("objects must be a non-empty tuple")
    if not isinstance(group_component_id, str) or not group_component_id.strip():
        raise ValueError("group_component_id must be a non-empty string")

    owner_and_original: dict[str, tuple[str, str, int]] = {}
    for item in objects:
        for constraint_type, constraints in (
            ("ik", item.document.ik),
            ("transform", item.document.transform),
        ):
            for constraint in constraints:
                if constraint.name in owner_and_original:
                    raise ConnectedGroupBuildError(
                        f"Constraint metadata name is duplicated: {constraint.name}"
                    )
                owner_and_original[constraint.name] = (
                    item.component_id,
                    constraint_type,
                    constraint.order,
                )

    assignments: list[ConstraintOrderAssignment] = []
    found_names: set[str] = set()
    for constraint_type, constraints in (
        ("ik", document.ik),
        ("transform", document.transform),
    ):
        for constraint in constraints:
            found_names.add(constraint.name)
            owner = owner_and_original.get(constraint.name)
            if owner is None:
                component_id = group_component_id
                original_order = constraint.order
            else:
                component_id, expected_type, original_order = owner
                if expected_type != constraint_type:
                    raise ConnectedGroupBuildError(
                        f"Constraint '{constraint.name}' changed collection from "
                        f"{expected_type} to {constraint_type}"
                    )
            assignments.append(
                ConstraintOrderAssignment(
                    component_id=component_id,
                    constraint_type=constraint_type,
                    constraint_name=constraint.name,
                    original_order=original_order,
                    global_order=constraint.order,
                )
            )

    missing = set(owner_and_original) - found_names
    if missing:
        raise ConnectedGroupBuildError(
            "Connected composition lost constraint metadata for: "
            f"{tuple(sorted(missing))}"
        )
    return tuple(assignments)


def _target_normalized_objects(
    objects: Tuple[ConnectedObjectDocument, ...],
    profile: LegacyRigProfile,
    spine_target: SpineJsonTarget,
) -> Tuple[ConnectedObjectDocument, ...]:
    """Normalize control space and apply idempotent target rig semantics."""

    normalized = tuple(
        normalize_connected_object_control_space(item, profile) for item in objects
    )
    if spine_target is not SpineJsonTarget.SPINE_4_1:
        return normalized
    if not isinstance(profile, TwoAxisScaleRigProfile):
        raise ConnectedGroupBuildError(
            "Spine 4.1 connected output currently requires TWO_AXIS_ROTATION_SCALE"
        )
    return tuple(
        replace(
            item,
            document=adapt_two_axis_document_for_spine41(
                item.document,
                profile=profile,
                prefix=item.prefix,
            ),
        )
        for item in normalized
    )


def build_connected_group_document(
    objects: Tuple[ConnectedObjectDocument, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile | None = None,
    *,
    spine_target: object = DEFAULT_SPINE_JSON_TARGET,
) -> ConnectedGroupBuildResult:
    """Compose A1 documents under one target-aware connected wrapper."""

    if not isinstance(settings, ConnectedGroupSettings):
        raise TypeError("settings must be ConnectedGroupSettings")
    resolved_profile = LegacyRigProfile() if profile is None else profile
    if not isinstance(resolved_profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")
    resolved_target = resolve_spine_json_target(spine_target)
    if resolved_target not in {
        SpineJsonTarget.SPINE_4_1,
        SpineJsonTarget.SPINE_4_2,
    }:
        raise ValueError(
            f"Connected rig construction is not implemented for "
            f"{resolved_target.label} ({resolved_target.exact_version})"
        )

    profile_id = resolve_a1_rig_profile(resolved_profile.profile_id)
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
    schedule = build_constraint_schedule(
        placements,
        resolved_profile,
        spine_target=resolved_target,
    )
    validate_constraint_schedule_for_target(schedule, resolved_target)
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

    normalized_objects = _target_normalized_objects(
        objects,
        resolved_profile,
        resolved_target,
    )
    group_component_id = f"__{settings.group_prefix}_rig__"
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
                component_id=group_component_id,
                document=global_document,
                animation_namespace=settings.group_prefix,
            ),
            *object_components,
        ),
        SpineCompositionSettings(
            shared_bone_names=(resolved_profile.root_bone(),),
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
    if resolved_target is SpineJsonTarget.SPINE_4_1:
        if profile_id is not A1RigProfile.TWO_AXIS_ROTATION_SCALE or not isinstance(
            resolved_profile,
            TwoAxisScaleRigProfile,
        ):
            raise ConnectedGroupBuildError(
                "Spine 4.1 connected output currently requires "
                "TWO_AXIS_ROTATION_SCALE"
            )
        global_ik, global_transform = (
            adapt_connected_two_axis_constraints_for_spine41(
                global_ik,
                global_transform,
                profile=resolved_profile,
                group_prefix=settings.group_prefix,
                layers=layers,
            )
        )

    with_global_constraints = replace(
        placed_document,
        ik=(*placed_document.ik, *global_ik),
        transform=(*placed_document.transform, *global_transform),
    )

    if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:
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
    _validate_connected_final(final_document, resolved_target)
    _validate_target_runtime_safety(final_document, resolved_target)

    composition = replace(
        composition,
        document=final_document,
        constraint_orders=_connected_constraint_assignments(
            final_document,
            normalized_objects,
            group_component_id,
        ),
    )
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
