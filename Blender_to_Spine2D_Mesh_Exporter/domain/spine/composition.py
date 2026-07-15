"""Compose several validated Spine documents before JSON serialization.

The legacy multi-object path serialized each object, reloaded JSON dictionaries, and
mutated compact weighted-vertex streams in place. This module performs the same
necessary index translation on typed immutable data. Unknown indices, duplicate
names, conflicting shared bones, and attachment-path collisions are hard errors;
there is no fallback to the root bone.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Tuple

from .model import (
    Bone,
    IKConstraint,
    MeshAttachment,
    Skin,
    Slot,
    SpineDocument,
    TransformConstraint,
)
from .validator import SpineValidator
from .weighted_vertices import (
    WeightedVertex,
    WeightedVertexInfluence,
    decode_weighted_vertices,
    encode_weighted_vertices,
)


class ConstraintOrderPolicy(str, Enum):
    """How constraint order values from independent documents are combined."""

    PRESERVE = "PRESERVE"
    REBASE_CONTIGUOUS = "REBASE_CONTIGUOUS"


@dataclass(frozen=True, slots=True)
class SpineDocumentComponent:
    component_id: str
    document: SpineDocument
    animation_namespace: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id.strip():
            raise ValueError("component_id must be a non-empty string")
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if self.animation_namespace is not None and (
            not isinstance(self.animation_namespace, str)
            or not self.animation_namespace.strip()
        ):
            raise ValueError(
                "animation_namespace must be a non-empty string or None"
            )


@dataclass(frozen=True, slots=True)
class SpineCompositionSettings:
    shared_bone_names: Tuple[str, ...] = ("root",)
    constraint_order_policy: ConstraintOrderPolicy = (
        ConstraintOrderPolicy.REBASE_CONTIGUOUS
    )
    namespace_animations: bool = True
    animation_separator: str = "/"
    require_matching_spine_version: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.shared_bone_names, tuple) or not all(
            isinstance(name, str) and name.strip() for name in self.shared_bone_names
        ):
            raise TypeError(
                "shared_bone_names must be a tuple of non-empty strings"
            )
        if len(self.shared_bone_names) != len(set(self.shared_bone_names)):
            raise ValueError("shared_bone_names cannot contain duplicates")
        if not isinstance(self.constraint_order_policy, ConstraintOrderPolicy):
            raise TypeError(
                "constraint_order_policy must be ConstraintOrderPolicy"
            )
        if not isinstance(self.namespace_animations, bool):
            raise TypeError("namespace_animations must be bool")
        if (
            not isinstance(self.animation_separator, str)
            or not self.animation_separator
        ):
            raise ValueError("animation_separator must be a non-empty string")
        if not isinstance(self.require_matching_spine_version, bool):
            raise TypeError("require_matching_spine_version must be bool")


@dataclass(frozen=True, slots=True)
class ComponentBoneIndexMap:
    component_id: str
    local_to_global: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id:
            raise ValueError("component_id must be a non-empty string")
        if not isinstance(self.local_to_global, tuple) or not all(
            isinstance(index, int) and index >= 0 for index in self.local_to_global
        ):
            raise TypeError(
                "local_to_global must be a tuple of non-negative integers"
            )

    def global_index_for(self, local_index: int) -> int:
        if not isinstance(local_index, int) or local_index < 0:
            raise ValueError("local_index must be a non-negative integer")
        try:
            return self.local_to_global[local_index]
        except IndexError as exc:
            raise KeyError(
                f"Component '{self.component_id}' has no local bone index "
                f"{local_index}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ConstraintOrderAssignment:
    component_id: str
    constraint_type: str
    constraint_name: str
    original_order: int
    global_order: int


@dataclass(frozen=True, slots=True)
class AnimationNameAssignment:
    component_id: str
    original_name: str
    global_name: str


@dataclass(frozen=True, slots=True)
class SpineDocumentCompositionResult:
    document: SpineDocument
    components: Tuple[SpineDocumentComponent, ...]
    bone_index_maps: Tuple[ComponentBoneIndexMap, ...]
    constraint_orders: Tuple[ConstraintOrderAssignment, ...]
    animation_names: Tuple[AnimationNameAssignment, ...]

    def bone_map_for(self, component_id: str) -> ComponentBoneIndexMap:
        matches = tuple(
            mapping
            for mapping in self.bone_index_maps
            if mapping.component_id == component_id
        )
        if len(matches) != 1:
            raise KeyError(
                f"Expected one bone index map for '{component_id}', found "
                f"{len(matches)}"
            )
        return matches[0]


class SpineCompositionError(ValueError):
    """Raised when independent documents cannot be combined safely."""


def _validate_components(
    components: Tuple[SpineDocumentComponent, ...],
) -> None:
    if not isinstance(components, tuple) or not components:
        raise ValueError("components must be a non-empty tuple")
    if not all(isinstance(item, SpineDocumentComponent) for item in components):
        raise TypeError("components must contain SpineDocumentComponent values")
    component_ids = tuple(item.component_id for item in components)
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("component_id values must be unique")
    validator = SpineValidator()
    for component in components:
        try:
            validator.validate_or_raise(component.document)
        except Exception as exc:
            raise SpineCompositionError(
                f"Component '{component.component_id}' is not a valid Spine "
                f"document: {exc}"
            ) from exc


def _spine_version(document: SpineDocument) -> str | None:
    value = document.skeleton.get("spine")
    return None if value is None else str(value)


def _validate_skeleton_versions(
    components: Tuple[SpineDocumentComponent, ...],
    settings: SpineCompositionSettings,
) -> None:
    if not settings.require_matching_spine_version:
        return
    expected = _spine_version(components[0].document)
    mismatches = tuple(
        (component.component_id, _spine_version(component.document))
        for component in components[1:]
        if _spine_version(component.document) != expected
    )
    if mismatches:
        raise SpineCompositionError(
            f"All components must use Spine version {expected!r}; mismatches="
            f"{mismatches}"
        )


def _compose_bones(
    components: Tuple[SpineDocumentComponent, ...],
    settings: SpineCompositionSettings,
) -> tuple[Tuple[Bone, ...], Tuple[ComponentBoneIndexMap, ...]]:
    shared = set(settings.shared_bone_names)
    global_bones: list[Bone] = []
    global_index_by_name: dict[str, int] = {}
    index_maps: list[ComponentBoneIndexMap] = []

    for component in components:
        local_to_global: list[int] = []
        for local_index, bone in enumerate(component.document.bones):
            existing_index = global_index_by_name.get(bone.name)
            if existing_index is not None:
                if bone.name not in shared:
                    raise SpineCompositionError(
                        f"Bone '{bone.name}' from component "
                        f"'{component.component_id}' collides with an existing "
                        "non-shared bone"
                    )
                existing_bone = global_bones[existing_index]
                if existing_bone != bone:
                    raise SpineCompositionError(
                        f"Shared bone '{bone.name}' differs in component "
                        f"'{component.component_id}': expected {existing_bone!r}, "
                        f"got {bone!r}"
                    )
                local_to_global.append(existing_index)
                continue

            if bone.parent is not None and bone.parent not in global_index_by_name:
                raise SpineCompositionError(
                    f"Bone '{bone.name}' in component '{component.component_id}' "
                    f"references parent '{bone.parent}' before that parent exists "
                    "in the global document"
                )
            global_index = len(global_bones)
            global_bones.append(bone)
            global_index_by_name[bone.name] = global_index
            local_to_global.append(global_index)

        if len(local_to_global) != len(component.document.bones):
            raise SpineCompositionError(
                f"Incomplete bone index map for component "
                f"'{component.component_id}'"
            )
        index_maps.append(
            ComponentBoneIndexMap(
                component_id=component.component_id,
                local_to_global=tuple(local_to_global),
            )
        )

    return tuple(global_bones), tuple(index_maps)


def _compose_slots(
    components: Tuple[SpineDocumentComponent, ...],
) -> Tuple[Slot, ...]:
    result: list[Slot] = []
    owner_by_name: dict[str, str] = {}
    for component in components:
        for slot in component.document.slots:
            previous_owner = owner_by_name.get(slot.name)
            if previous_owner is not None:
                raise SpineCompositionError(
                    f"Slot '{slot.name}' from component "
                    f"'{component.component_id}' collides with component "
                    f"'{previous_owner}'"
                )
            owner_by_name[slot.name] = component.component_id
            result.append(slot)
    return tuple(result)


def _constraint_records(component: SpineDocumentComponent):
    records = []
    for index, constraint in enumerate(component.document.ik):
        records.append((constraint.order, 0, index, "ik", constraint))
    for index, constraint in enumerate(component.document.transform):
        records.append((constraint.order, 1, index, "transform", constraint))
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(records)


def _compose_constraints(
    components: Tuple[SpineDocumentComponent, ...],
    settings: SpineCompositionSettings,
) -> tuple[
    Tuple[IKConstraint, ...],
    Tuple[TransformConstraint, ...],
    Tuple[ConstraintOrderAssignment, ...],
]:
    global_names: set[str] = set()
    used_orders: set[int] = set()
    ik_result: list[IKConstraint] = []
    transform_result: list[TransformConstraint] = []
    assignments: list[ConstraintOrderAssignment] = []
    next_order = 0

    for component in components:
        order_by_identity: dict[tuple[str, int], int] = {}
        for _, _, local_index, kind, constraint in _constraint_records(component):
            if constraint.name in global_names:
                raise SpineCompositionError(
                    f"Constraint '{constraint.name}' from component "
                    f"'{component.component_id}' is duplicated"
                )
            global_names.add(constraint.name)

            if (
                settings.constraint_order_policy
                is ConstraintOrderPolicy.REBASE_CONTIGUOUS
            ):
                global_order = next_order
                next_order += 1
            else:
                global_order = constraint.order
                if global_order in used_orders:
                    raise SpineCompositionError(
                        f"Constraint order {global_order} from component "
                        f"'{component.component_id}' collides with another "
                        "constraint under PRESERVE policy"
                    )
            used_orders.add(global_order)
            order_by_identity[(kind, local_index)] = global_order
            assignments.append(
                ConstraintOrderAssignment(
                    component_id=component.component_id,
                    constraint_type=kind,
                    constraint_name=constraint.name,
                    original_order=constraint.order,
                    global_order=global_order,
                )
            )

        ik_result.extend(
            replace(
                constraint,
                order=order_by_identity[("ik", local_index)],
            )
            for local_index, constraint in enumerate(component.document.ik)
        )
        transform_result.extend(
            replace(
                constraint,
                order=order_by_identity[("transform", local_index)],
            )
            for local_index, constraint in enumerate(component.document.transform)
        )

    return tuple(ik_result), tuple(transform_result), tuple(assignments)


def _remap_mesh_attachment(
    attachment: MeshAttachment,
    bone_map: ComponentBoneIndexMap,
) -> MeshAttachment:
    vertex_count = len(attachment.uvs) // 2
    try:
        decoded = decode_weighted_vertices(
            attachment.vertices,
            expected_vertex_count=vertex_count,
        )
    except Exception as exc:
        raise SpineCompositionError(
            f"Attachment '{attachment.name}' in component "
            f"'{bone_map.component_id}' has an invalid weighted vertex stream: "
            f"{exc}"
        ) from exc

    remapped_vertices: list[WeightedVertex] = []
    for vertex_index, vertex in enumerate(decoded):
        influences: list[WeightedVertexInfluence] = []
        for influence_index, influence in enumerate(vertex.influences):
            try:
                global_index = bone_map.global_index_for(influence.bone_index)
            except (KeyError, ValueError) as exc:
                raise SpineCompositionError(
                    f"Attachment '{attachment.name}' vertex {vertex_index}, "
                    f"influence {influence_index} references unknown local bone "
                    f"index {influence.bone_index} in component "
                    f"'{bone_map.component_id}'"
                ) from exc
            influences.append(
                WeightedVertexInfluence(
                    bone_index=global_index,
                    x=influence.x,
                    y=influence.y,
                    weight=influence.weight,
                )
            )
        remapped_vertices.append(WeightedVertex(tuple(influences)))

    return replace(
        attachment,
        vertices=encode_weighted_vertices(tuple(remapped_vertices)),
    )


def _copy_attachment(
    attachment: MeshAttachment | Mapping[str, Any],
    bone_map: ComponentBoneIndexMap,
):
    if isinstance(attachment, MeshAttachment):
        return _remap_mesh_attachment(attachment, bone_map)
    if not isinstance(attachment, Mapping):
        raise SpineCompositionError(
            f"Unsupported attachment value in component "
            f"'{bone_map.component_id}': {type(attachment).__name__}"
        )
    return deepcopy(dict(attachment))


def _append_unique_names(
    target: list[str],
    source: Tuple[str, ...],
) -> None:
    seen = set(target)
    for name in source:
        if name not in seen:
            target.append(name)
            seen.add(name)


def _compose_skins(
    components: Tuple[SpineDocumentComponent, ...],
    bone_maps: Tuple[ComponentBoneIndexMap, ...],
) -> Tuple[Skin, ...]:
    builders: dict[str, dict[str, Any]] = {}
    skin_order: list[str] = []

    for component, bone_map in zip(components, bone_maps):
        for skin in component.document.skins:
            builder = builders.get(skin.name)
            if builder is None:
                builder = {
                    "attachments": {},
                    "bones": [],
                    "constraints": [],
                    "extras": deepcopy(dict(skin.extras)),
                }
                builders[skin.name] = builder
                skin_order.append(skin.name)
            elif builder["extras"] != dict(skin.extras):
                raise SpineCompositionError(
                    f"Skin '{skin.name}' has conflicting extras in component "
                    f"'{component.component_id}'"
                )

            _append_unique_names(builder["bones"], skin.bones)
            _append_unique_names(builder["constraints"], skin.constraints)
            attachments = builder["attachments"]
            for slot_name, slot_attachments in skin.attachments.items():
                global_slot = attachments.setdefault(slot_name, {})
                for attachment_name, attachment in slot_attachments.items():
                    if attachment_name in global_slot:
                        raise SpineCompositionError(
                            f"Attachment path '{skin.name}/{slot_name}/"
                            f"{attachment_name}' is duplicated by component "
                            f"'{component.component_id}'"
                        )
                    global_slot[attachment_name] = _copy_attachment(
                        attachment,
                        bone_map,
                    )

    return tuple(
        Skin(
            name=skin_name,
            attachments=builders[skin_name]["attachments"],
            bones=tuple(builders[skin_name]["bones"]),
            constraints=tuple(builders[skin_name]["constraints"]),
            extras=builders[skin_name]["extras"],
        )
        for skin_name in skin_order
    )


def _compose_animations(
    components: Tuple[SpineDocumentComponent, ...],
    settings: SpineCompositionSettings,
) -> tuple[dict[str, Any], Tuple[AnimationNameAssignment, ...]]:
    animations: dict[str, Any] = {}
    assignments: list[AnimationNameAssignment] = []

    for component in components:
        namespace = component.animation_namespace or component.component_id
        for original_name, animation in component.document.animations.items():
            original = str(original_name)
            global_name = (
                f"{namespace}{settings.animation_separator}{original}"
                if settings.namespace_animations
                else original
            )
            if global_name in animations:
                raise SpineCompositionError(
                    f"Animation name '{global_name}' is duplicated"
                )
            animations[global_name] = deepcopy(animation)
            assignments.append(
                AnimationNameAssignment(
                    component_id=component.component_id,
                    original_name=original,
                    global_name=global_name,
                )
            )

    return animations, tuple(assignments)


def _merge_named_mappings(
    components: Tuple[SpineDocumentComponent, ...],
    attribute_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    owner_by_name: dict[str, str] = {}
    for component in components:
        mapping = getattr(component.document, attribute_name)
        for name, value in mapping.items():
            key = str(name)
            if key in result and result[key] != value:
                raise SpineCompositionError(
                    f"Top-level {attribute_name} entry '{key}' conflicts between "
                    f"components '{owner_by_name[key]}' and "
                    f"'{component.component_id}'"
                )
            if key not in result:
                result[key] = deepcopy(value)
                owner_by_name[key] = component.component_id
    return result


def compose_spine_documents(
    components: Tuple[SpineDocumentComponent, ...],
    settings: SpineCompositionSettings | None = None,
) -> SpineDocumentCompositionResult:
    """Compose validated documents and remap all mesh weights before serialization."""

    _validate_components(components)
    resolved_settings = settings or SpineCompositionSettings()
    if not isinstance(resolved_settings, SpineCompositionSettings):
        raise TypeError("settings must be SpineCompositionSettings")
    _validate_skeleton_versions(components, resolved_settings)

    bones, bone_maps = _compose_bones(components, resolved_settings)
    slots = _compose_slots(components)
    ik, transform, constraint_orders = _compose_constraints(
        components,
        resolved_settings,
    )
    skins = _compose_skins(components, bone_maps)
    animations, animation_names = _compose_animations(
        components,
        resolved_settings,
    )
    events = _merge_named_mappings(components, "events")
    extras = _merge_named_mappings(components, "extras")

    document = SpineDocument(
        skeleton=deepcopy(dict(components[0].document.skeleton)),
        bones=bones,
        slots=slots,
        skins=skins,
        ik=ik,
        transform=transform,
        animations=animations,
        events=events,
        extras=extras,
    )
    try:
        SpineValidator().validate_or_raise(document)
    except Exception as exc:
        raise SpineCompositionError(
            f"Composed Spine document failed validation: {exc}"
        ) from exc

    return SpineDocumentCompositionResult(
        document=document,
        components=components,
        bone_index_maps=bone_maps,
        constraint_orders=constraint_orders,
        animation_names=animation_names,
    )
