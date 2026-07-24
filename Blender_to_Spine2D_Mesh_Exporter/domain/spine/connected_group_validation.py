"""Fail-fast validation for connected A1 source documents and namespaces."""

from __future__ import annotations

from typing import Tuple

from .connected_group_contracts import (
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    ConnectedZLayer,
)
from .connected_group_error import ConnectedGroupBuildError
from .legacy_profile import LegacyRigProfile
from .validator import SpineValidator


def _duplicates(values: Tuple[str, ...]) -> Tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def validate_object_constraint_schema(
    item: ConnectedObjectDocument,
    profile: LegacyRigProfile,
) -> None:
    """Require the exact one-IK plus five-Transform legacy A1 schema."""

    expected_ik = (profile.scale_ik_constraint(item.prefix),)
    expected_transform = (
        profile.rotation_x_constraint(item.prefix),
        profile.rotation_y_constraint(item.prefix),
        profile.scale_constraint(item.prefix),
        profile.rotation_z_constraint(item.prefix),
        profile.scale_compensator_constraint(item.prefix),
    )
    actual_ik = tuple(constraint.name for constraint in item.document.ik)
    actual_transform = tuple(
        constraint.name for constraint in item.document.transform
    )

    ik_valid = len(actual_ik) == 1 and set(actual_ik) == set(expected_ik)
    transform_valid = (
        len(actual_transform) == 5
        and set(actual_transform) == set(expected_transform)
    )
    if ik_valid and transform_valid:
        return

    expected_all = set((*expected_ik, *expected_transform))
    actual_all = set((*actual_ik, *actual_transform))
    wrong_collection = tuple(
        sorted(
            name
            for name in actual_ik
            if name in set(expected_transform)
        )
    ) + tuple(
        sorted(
            name
            for name in actual_transform
            if name in set(expected_ik)
        )
    )
    raise ConnectedGroupBuildError(
        f"Object component '{item.component_id}' must contain exactly the six "
        "A1 constraints in their required IK/Transform collections; "
        f"missing={tuple(sorted(expected_all - actual_all))}, "
        f"unexpected={tuple(sorted(actual_all - expected_all))}, "
        f"wrong_collection={wrong_collection}, "
        f"actual_ik={actual_ik}, actual_transform={actual_transform}"
    )


def validate_connected_group_inputs(
    objects: Tuple[ConnectedObjectDocument, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
) -> None:
    """Validate source documents before layout or global-rig construction."""

    if not isinstance(objects, tuple) or len(objects) < 2:
        raise ValueError("objects must contain at least two connected documents")
    if not all(isinstance(item, ConnectedObjectDocument) for item in objects):
        raise TypeError("objects must contain ConnectedObjectDocument values")
    if not isinstance(settings, ConnectedGroupSettings):
        raise TypeError("settings must be ConnectedGroupSettings")
    if not isinstance(profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")

    component_ids = tuple(item.component_id for item in objects)
    prefixes = tuple(item.prefix for item in objects)
    duplicate_components = _duplicates(component_ids)
    if duplicate_components:
        raise ValueError(
            f"component_id values must be unique: {duplicate_components}"
        )
    duplicate_prefixes = _duplicates(prefixes)
    if duplicate_prefixes:
        raise ValueError(
            f"connected object prefixes must be unique: {duplicate_prefixes}"
        )
    if settings.group_prefix in set(prefixes):
        raise ValueError("group_prefix cannot equal an object prefix")
    if settings.anchor_component_id is not None and (
        settings.anchor_component_id not in set(component_ids)
    ):
        raise ValueError("anchor_component_id is not present in objects")

    placement_spaces = {item.placement_space for item in objects}
    if len(placement_spaces) != 1:
        details = tuple(
            (item.component_id, item.placement_space.value)
            for item in objects
        )
        raise ConnectedGroupBuildError(
            "Connected A1 composition cannot mix object-local attachments with "
            "camera screen-space attachments. Use one placement space for the whole "
            f"connected subgroup or use static grouped camera flattening; sources={details}"
        )

    internal_component_id = f"__{settings.group_prefix}_rig__"
    if internal_component_id in set(component_ids):
        raise ConnectedGroupBuildError(
            "Connected global rig component ID collides with an input component: "
            f"'{internal_component_id}'"
        )

    validator = SpineValidator()
    for item in objects:
        try:
            validator.validate_or_raise(item.document)
        except Exception as exc:
            raise ConnectedGroupBuildError(
                f"Object component '{item.component_id}' is invalid: {exc}"
            ) from exc

        required_bones = {
            profile.root_bone(),
            profile.main_bone(item.prefix),
            profile.base_bone(item.prefix),
        }
        actual_bones = {bone.name for bone in item.document.bones}
        missing_bones = required_bones - actual_bones
        if missing_bones:
            raise ConnectedGroupBuildError(
                f"Object component '{item.component_id}' is missing A1 bones: "
                f"{tuple(sorted(missing_bones))}"
            )
        validate_object_constraint_schema(item, profile)


def validate_connected_global_namespace(
    objects: Tuple[ConnectedObjectDocument, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    layers: Tuple[ConnectedZLayer, ...],
) -> None:
    """Reject generated global names that would collide during composition."""

    prefix = settings.group_prefix
    generated_names = (
        profile.main_bone(prefix),
        profile.base_bone(prefix),
        profile.scale_rotate_x_bone(prefix),
        profile.rotate_x_bone(prefix),
        *profile.control_bones(prefix),
        *profile.ik_chain_bones(prefix),
        *(layer.scale_bone_name for layer in layers),
        *(layer.layer_bone_name for layer in layers),
    )
    generated_duplicates = _duplicates(tuple(generated_names))
    if generated_duplicates:
        raise ConnectedGroupBuildError(
            "Connected global rig generated duplicate bone names: "
            f"{generated_duplicates}"
        )

    source_owner_by_bone: dict[str, str] = {}
    for item in objects:
        for bone in item.document.bones:
            if bone.name == profile.root_bone():
                continue
            source_owner_by_bone.setdefault(bone.name, item.component_id)

    collisions = tuple(
        sorted(
            (
                generated_name,
                source_owner_by_bone[generated_name],
            )
            for generated_name in generated_names
            if generated_name in source_owner_by_bone
        )
    )
    if collisions:
        raise ConnectedGroupBuildError(
            "Connected global rig bone namespace collides with source documents: "
            f"{collisions}"
        )


__all__ = [
    "validate_connected_global_namespace",
    "validate_connected_group_inputs",
    "validate_object_constraint_schema",
]
