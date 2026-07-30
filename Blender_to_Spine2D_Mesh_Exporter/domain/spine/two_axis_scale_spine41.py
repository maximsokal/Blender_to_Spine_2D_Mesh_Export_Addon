"""Build Spine 4.1-safe two-axis scale documents without changing scale semantics.

The canonical Spine 4.2 hierarchy contains exact axis-collapse bones with ``scaleX == 0``.
Spine 4.1 world-space transform constraints call ``Bone.updateAppliedTransform`` and
invert the constrained bone's parent matrix, so two constrained parent relationships
must be made invertible without changing the visible control behavior:

- the uniform world-scale constraint is moved from ``<prefix>_rotate_X`` to its direct
  collapse parent ``<prefix>_scale_rotate_X``. Uniform world scale commutes through the
  child rotation, so the resulting descendant world matrix is unchanged;
- every ``onlyTranslation`` depth wrapper receives an internal ``onlyTranslation``
  bridge. The bridge carries the wrapper's original X/Y offset while the wrapper is
  reparented at local zero. This preserves the wrapper world pose and keeps the original
  depth constraint on the original ``*_scale`` bone, whose new parent is invertible.

Bridge insertion happens on the typed immutable ``SpineDocument`` before serialization.
Weighted mesh bone indices are remapped by bone name; no serialized JSON repair, epsilon,
or setup-scale replacement is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from .connected_group_contracts import ConnectedZLayer
from .legacy_rig_contracts import LegacyRigBuildResult
from .model import (
    Bone,
    IKConstraint,
    MeshAttachment,
    Skin,
    SpineDocument,
    TransformConstraint,
)
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile
from .spine41_setup_safety import validate_spine41_setup_safety
from .two_axis_scale_profile import TwoAxisScaleRigProfile
from .validator import SpineValidator
from .weighted_vertices import (
    WeightedVertex,
    WeightedVertexInfluence,
    decode_weighted_vertices,
    encode_weighted_vertices,
)


@dataclass(frozen=True, slots=True)
class Spine41TwoAxisDocumentAdaptation:
    """Complete immutable result of one target-specific document adaptation."""

    document: SpineDocument
    old_to_new_bone_indices: Mapping[int, int]
    bridge_bone_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if not isinstance(self.old_to_new_bone_indices, Mapping):
            raise TypeError("old_to_new_bone_indices must be a mapping")
        normalized: dict[int, int] = {}
        for old_index, new_index in self.old_to_new_bone_indices.items():
            if (
                isinstance(old_index, bool)
                or not isinstance(old_index, int)
                or old_index < 0
            ):
                raise ValueError("old bone indices must be non-negative integers")
            if (
                isinstance(new_index, bool)
                or not isinstance(new_index, int)
                or new_index < 0
            ):
                raise ValueError("new bone indices must be non-negative integers")
            normalized[old_index] = new_index
        if not isinstance(self.bridge_bone_names, tuple):
            raise TypeError("bridge_bone_names must be tuple")
        if not all(
            isinstance(name, str) and name.strip()
            for name in self.bridge_bone_names
        ):
            raise ValueError("bridge_bone_names must contain non-empty strings")
        object.__setattr__(
            self,
            "old_to_new_bone_indices",
            MappingProxyType(normalized),
        )


def _constraint_by_name(
    constraints: tuple[TransformConstraint, ...],
    name: str,
) -> TransformConstraint:
    matches = tuple(item for item in constraints if item.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one two-axis constraint named {name!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _spine41_depth_bridge_name(wrapper_name: str) -> str:
    if not isinstance(wrapper_name, str) or not wrapper_name.strip():
        raise ValueError("wrapper_name must be a non-empty string")
    return f"{wrapper_name}_spine41_bridge"


def _document_depth_mapping(
    document: SpineDocument,
    depth_constraint: TransformConstraint,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve canonical wrapper/layer pairs, including the superseded layer-target form."""

    bone_by_name = {bone.name: bone for bone in document.bones}
    children_by_parent: dict[str, list[Bone]] = {}
    for bone in document.bones:
        if bone.parent is not None:
            children_by_parent.setdefault(bone.parent, []).append(bone)

    wrappers: list[str] = []
    layers: list[str] = []
    for constrained_name in depth_constraint.bones:
        constrained = bone_by_name.get(constrained_name)
        if constrained is None:
            raise ValueError(
                f"Depth constraint {depth_constraint.name!r} references missing bone "
                f"{constrained_name!r}"
            )

        if constrained.extras.get("inherit") == "onlyTranslation":
            layer_children = tuple(children_by_parent.get(constrained.name, ()))
            if len(layer_children) != 1:
                raise ValueError(
                    f"Depth wrapper {constrained.name!r} must have exactly one layer "
                    f"child, found {len(layer_children)}"
                )
            wrappers.append(constrained.name)
            layers.append(layer_children[0].name)
            continue

        parent = bone_by_name.get(constrained.parent or "")
        if parent is None or parent.extras.get("inherit") != "onlyTranslation":
            raise ValueError(
                f"Depth constraint bone {constrained.name!r} is neither a generated "
                "onlyTranslation wrapper nor its direct layer child"
            )
        wrappers.append(parent.name)
        layers.append(constrained.name)

    return tuple(wrappers), tuple(layers)


def _adapt_uniform_scale_constraint(
    constraint: TransformConstraint,
    *,
    unsafe_rotation_bone: str,
    safe_collapse_bone: str,
) -> TransformConstraint:
    """Preserve relative-world scale while moving the singular-parent driver upward."""

    extras = dict(constraint.extras)
    if extras.get("relative") is not True:
        raise ValueError(
            f"Constraint {constraint.name!r} must be relative before Spine 4.1 adaptation"
        )
    for field_name in ("mixRotate", "mixX", "mixShearY"):
        if extras.get(field_name) != 0:
            raise ValueError(
                f"Constraint {constraint.name!r} requires {field_name}=0 before "
                "Spine 4.1 adaptation"
            )

    local_value = extras.pop("local", None)
    if local_value not in {None, False, True}:
        raise ValueError(
            f"Constraint {constraint.name!r} has a non-boolean local field"
        )

    unsafe_count = constraint.bones.count(unsafe_rotation_bone)
    safe_count = constraint.bones.count(safe_collapse_bone)
    if unsafe_count == 1 and safe_count == 0:
        bones = tuple(
            safe_collapse_bone if name == unsafe_rotation_bone else name
            for name in constraint.bones
        )
    elif unsafe_count == 0 and safe_count == 1:
        bones = constraint.bones
    else:
        raise ValueError(
            f"Constraint {constraint.name!r} must contain exactly one of "
            f"{unsafe_rotation_bone!r} or {safe_collapse_bone!r}; "
            f"actual={constraint.bones}"
        )

    return replace(constraint, bones=bones, extras=extras)


def _restore_depth_wrapper_targets(
    constraint: TransformConstraint,
    *,
    wrapper_bones: tuple[str, ...],
    layer_bones: tuple[str, ...],
) -> TransformConstraint:
    """Keep depth evaluation on the original wrapper bones, never on final layers."""

    if len(wrapper_bones) != len(layer_bones):
        raise ValueError("Spine 4.1 depth target mapping must be one-to-one")
    if constraint.bones == wrapper_bones:
        return constraint
    if constraint.bones == layer_bones:
        return replace(constraint, bones=wrapper_bones)
    raise ValueError(
        f"Constraint {constraint.name!r} bone schema changed: "
        f"expected wrappers={wrapper_bones} or layers={layer_bones}, "
        f"actual={constraint.bones}"
    )


def _insert_depth_bridges(
    bones: tuple[Bone, ...],
    *,
    wrapper_bones: tuple[str, ...],
    expected_parent_name: str,
) -> tuple[tuple[Bone, ...], tuple[str, ...], Mapping[int, int]]:
    """Insert invertible translation bridges before canonical depth wrappers."""

    bone_by_name = {bone.name: bone for bone in bones}
    if len(bone_by_name) != len(bones):
        raise ValueError("Spine document contains duplicate bone names")

    wrapper_set = set(wrapper_bones)
    if len(wrapper_set) != len(wrapper_bones):
        raise ValueError("Depth wrapper names must be unique")

    bridge_names = tuple(_spine41_depth_bridge_name(name) for name in wrapper_bones)
    bridge_by_wrapper = dict(zip(wrapper_bones, bridge_names, strict=True))
    adapted: list[Bone] = []

    for bone in bones:
        bridge_name = bridge_by_wrapper.get(bone.name)
        if bridge_name is None:
            adapted.append(bone)
            continue

        if bone.extras.get("inherit") != "onlyTranslation":
            raise ValueError(
                f"Depth wrapper {bone.name!r} must use inherit=onlyTranslation"
            )

        existing_bridge = bone_by_name.get(bridge_name)
        if bone.parent == bridge_name:
            if existing_bridge is None:
                raise ValueError(
                    f"Depth wrapper {bone.name!r} references missing bridge {bridge_name!r}"
                )
            if existing_bridge.parent != expected_parent_name:
                raise ValueError(
                    f"Bridge {bridge_name!r} parent changed: "
                    f"expected={expected_parent_name!r}, actual={existing_bridge.parent!r}"
                )
            if existing_bridge.extras.get("inherit") != "onlyTranslation":
                raise ValueError(
                    f"Bridge {bridge_name!r} must use inherit=onlyTranslation"
                )
            if float(bone.x or 0.0) != 0.0 or float(bone.y or 0.0) != 0.0:
                raise ValueError(
                    f"Adapted depth wrapper {bone.name!r} must be at local zero"
                )
            adapted.append(bone)
            continue

        if existing_bridge is not None:
            raise ValueError(f"Bridge bone name collision for {bridge_name!r}")
        if bone.parent != expected_parent_name:
            raise ValueError(
                f"Depth wrapper {bone.name!r} parent changed: "
                f"expected={expected_parent_name!r}, actual={bone.parent!r}"
            )

        bridge = Bone(
            name=bridge_name,
            parent=bone.parent,
            x=bone.x,
            y=bone.y,
            extras={"inherit": "onlyTranslation"},
        )
        adapted.append(bridge)
        adapted.append(
            replace(
                bone,
                parent=bridge_name,
                x=None if bone.x is None else 0.0,
                y=None if bone.y is None else 0.0,
            )
        )

    missing = wrapper_set - {bone.name for bone in bones}
    if missing:
        raise ValueError(f"Depth wrapper bones are missing: {tuple(sorted(missing))}")

    adapted_bones = tuple(adapted)
    new_index_by_name = {bone.name: index for index, bone in enumerate(adapted_bones)}
    old_to_new = {
        old_index: new_index_by_name[bone.name]
        for old_index, bone in enumerate(bones)
    }
    return adapted_bones, bridge_names, MappingProxyType(old_to_new)


def _remap_weighted_stream(
    vertices: Sequence[float | int],
    *,
    uvs: Sequence[float | int],
    old_to_new_bone_indices: Mapping[int, int],
    label: str,
) -> tuple[float | int, ...]:
    """Remap one weighted mesh stream; plain unweighted XY streams are unchanged."""

    if len(vertices) == len(uvs):
        return tuple(vertices)

    decoded = decode_weighted_vertices(
        vertices,
        expected_vertex_count=len(uvs) // 2,
    )
    remapped: list[WeightedVertex] = []
    for vertex_index, vertex in enumerate(decoded):
        influences: list[WeightedVertexInfluence] = []
        for influence_index, influence in enumerate(vertex.influences):
            new_index = old_to_new_bone_indices.get(influence.bone_index)
            if new_index is None:
                raise ValueError(
                    f"{label} vertex {vertex_index} influence {influence_index} "
                    f"references unknown old bone index {influence.bone_index}"
                )
            influences.append(replace(influence, bone_index=new_index))
        remapped.append(WeightedVertex(tuple(influences)))
    return encode_weighted_vertices(remapped)


def _remap_attachment_value(
    attachment: MeshAttachment | Mapping[str, object],
    *,
    old_to_new_bone_indices: Mapping[int, int],
    label: str,
) -> MeshAttachment | Mapping[str, object]:
    if isinstance(attachment, MeshAttachment):
        vertices = _remap_weighted_stream(
            attachment.vertices,
            uvs=attachment.uvs,
            old_to_new_bone_indices=old_to_new_bone_indices,
            label=label,
        )
        return (
            attachment
            if vertices == attachment.vertices
            else replace(attachment, vertices=vertices)
        )

    if not isinstance(attachment, Mapping):
        raise TypeError(f"{label} must be MeshAttachment or mapping")
    if attachment.get("type") != "mesh":
        return attachment

    raw_vertices = attachment.get("vertices")
    raw_uvs = attachment.get("uvs")
    if not isinstance(raw_vertices, (tuple, list)) or not isinstance(
        raw_uvs,
        (tuple, list),
    ):
        raise TypeError(f"{label} mesh vertices and uvs must be arrays")
    remapped = _remap_weighted_stream(
        raw_vertices,
        uvs=raw_uvs,
        old_to_new_bone_indices=old_to_new_bone_indices,
        label=label,
    )
    if tuple(raw_vertices) == remapped:
        return attachment
    copied = dict(attachment)
    copied["vertices"] = list(remapped) if isinstance(raw_vertices, list) else remapped
    return copied


def _remap_skins(
    skins: tuple[Skin, ...],
    *,
    old_to_new_bone_indices: Mapping[int, int],
) -> tuple[Skin, ...]:
    remapped_skins: list[Skin] = []
    for skin in skins:
        attachment_groups: dict[
            str,
            dict[str, MeshAttachment | Mapping[str, object]],
        ] = {}
        for slot_name, attachments in skin.attachments.items():
            attachment_groups[slot_name] = {
                attachment_name: _remap_attachment_value(
                    attachment,
                    old_to_new_bone_indices=old_to_new_bone_indices,
                    label=(
                        f"skin {skin.name!r} slot {slot_name!r} "
                        f"attachment {attachment_name!r}"
                    ),
                )
                for attachment_name, attachment in attachments.items()
            }
        remapped_skins.append(replace(skin, attachments=attachment_groups))
    return tuple(remapped_skins)


def adapt_two_axis_document_for_spine41_with_report(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
) -> Spine41TwoAxisDocumentAdaptation:
    """Return a target-safe document plus the exact bone-index remapping report."""

    if not isinstance(document, SpineDocument):
        raise TypeError("document must be SpineDocument")
    if not isinstance(profile, TwoAxisScaleRigProfile):
        raise TypeError("profile must be TwoAxisScaleRigProfile")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")

    depth_name = profile.scale_depth_constraint(prefix)
    scale_name = profile.scale_constraint(prefix)
    depth_constraint = _constraint_by_name(document.transform, depth_name)
    wrappers, layers = _document_depth_mapping(document, depth_constraint)

    adapted_bones, bridge_names, index_map = _insert_depth_bridges(
        document.bones,
        wrapper_bones=wrappers,
        expected_parent_name=profile.rotate_x_bone(prefix),
    )
    source_scale = _constraint_by_name(document.transform, scale_name)
    adapted_scale = _adapt_uniform_scale_constraint(
        source_scale,
        unsafe_rotation_bone=profile.rotate_x_bone(prefix),
        safe_collapse_bone=profile.scale_rotate_x_bone(prefix),
    )
    adapted_depth = _restore_depth_wrapper_targets(
        depth_constraint,
        wrapper_bones=wrappers,
        layer_bones=layers,
    )
    transformed_by_name = {item.name: item for item in document.transform}
    transformed_by_name[scale_name] = adapted_scale
    transformed_by_name[depth_name] = adapted_depth

    adapted = replace(
        document,
        bones=adapted_bones,
        skins=_remap_skins(
            document.skins,
            old_to_new_bone_indices=index_map,
        ),
        transform=tuple(
            transformed_by_name[item.name] for item in document.transform
        ),
    )
    SpineValidator().validate_or_raise(adapted)
    validate_spine41_setup_safety(adapted)
    return Spine41TwoAxisDocumentAdaptation(
        document=adapted,
        old_to_new_bone_indices=index_map,
        bridge_bone_names=bridge_names,
    )


def adapt_two_axis_document_for_spine41(
    document: SpineDocument,
    *,
    profile: TwoAxisScaleRigProfile,
    prefix: str,
) -> SpineDocument:
    """Compatibility wrapper returning only the adapted immutable document."""

    return adapt_two_axis_document_for_spine41_with_report(
        document,
        profile=profile,
        prefix=prefix,
    ).document


def adapt_two_axis_scale_rig_for_spine41(
    rig: LegacyRigBuildResult,
) -> LegacyRigBuildResult:
    """Return a detached research view of the document-level Spine 4.1 topology.

    Production attachment projection intentionally keeps the canonical rig and applies this
    topology only to the final typed document. This helper exists for pure domain inspection.
    """

    if not isinstance(rig, LegacyRigBuildResult):
        raise TypeError("rig must be LegacyRigBuildResult")
    profile_id = resolve_a1_rig_profile(rig.profile.profile_id)
    if profile_id is not A1RigProfile.TWO_AXIS_ROTATION_SCALE:
        raise TypeError("Spine 4.1 two-axis adaptation requires TWO_AXIS_ROTATION_SCALE")
    if not isinstance(rig.profile, TwoAxisScaleRigProfile):
        raise TypeError("rig.profile must be TwoAxisScaleRigProfile")

    document = SpineDocument(
        skeleton={"spine": "4.1.24"},
        bones=rig.bones,
        slots=(),
        skins=(),
        ik=rig.ik,
        transform=rig.transform,
    )
    adapted = adapt_two_axis_document_for_spine41(
        document,
        profile=rig.profile,
        prefix=rig.request.prefix,
    )
    return replace(
        rig,
        bones=adapted.bones,
        transform=adapted.transform,
    )


# Connected Spine 4.1 remains development-only. Keep its previously isolated global
# constraint adapter independent from the standalone bridge topology until connected
# authoring has its own runtime and Spine Editor acceptance evidence.
def _adapt_connected_uniform_scale_constraint(
    constraint: TransformConstraint,
) -> TransformConstraint:
    extras = dict(constraint.extras)
    if extras.get("relative") is not True:
        raise ValueError(
            f"Constraint {constraint.name!r} must be relative before Spine 4.1 adaptation"
        )
    for field_name in ("mixRotate", "mixX", "mixShearY"):
        if extras.get(field_name) != 0:
            raise ValueError(
                f"Constraint {constraint.name!r} requires {field_name}=0 before "
                "Spine 4.1 adaptation"
            )
    extras["local"] = True
    return replace(constraint, extras=extras)


def _adapt_connected_depth_constraint(
    constraint: TransformConstraint,
    *,
    source_wrapper_bones: tuple[str, ...],
    target_layer_bones: tuple[str, ...],
) -> TransformConstraint:
    if len(target_layer_bones) != len(source_wrapper_bones):
        raise ValueError("Spine 4.1 depth target mapping must be one-to-one")
    if constraint.bones == target_layer_bones:
        return constraint
    if constraint.bones != source_wrapper_bones:
        raise ValueError(
            f"Constraint {constraint.name!r} bone schema changed: "
            f"expected={source_wrapper_bones}, actual={constraint.bones}"
        )
    return replace(constraint, bones=target_layer_bones)


def adapt_connected_two_axis_constraints_for_spine41(
    ik: tuple[IKConstraint, ...],
    transform: tuple[TransformConstraint, ...],
    *,
    profile: TwoAxisScaleRigProfile,
    group_prefix: str,
    layers: tuple[ConnectedZLayer, ...],
) -> tuple[tuple[IKConstraint, ...], tuple[TransformConstraint, ...]]:
    """Return the quarantined target-safe global-wrapper constraint variant."""

    if not isinstance(ik, tuple) or not all(
        isinstance(item, IKConstraint) for item in ik
    ):
        raise TypeError("ik must contain IKConstraint values")
    if not isinstance(transform, tuple) or not all(
        isinstance(item, TransformConstraint) for item in transform
    ):
        raise TypeError("transform must contain TransformConstraint values")
    if not isinstance(profile, TwoAxisScaleRigProfile):
        raise TypeError("profile must be TwoAxisScaleRigProfile")
    if not isinstance(group_prefix, str) or not group_prefix.strip():
        raise ValueError("group_prefix must be a non-empty string")
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("layers must be a non-empty tuple")
    if not all(isinstance(layer, ConnectedZLayer) for layer in layers):
        raise TypeError("layers must contain ConnectedZLayer values")

    scale_name = profile.scale_constraint(group_prefix)
    depth_name = profile.scale_depth_constraint(group_prefix)
    adapted_by_name = {item.name: item for item in transform}
    adapted_by_name[scale_name] = _adapt_connected_uniform_scale_constraint(
        _constraint_by_name(transform, scale_name)
    )
    adapted_by_name[depth_name] = _adapt_connected_depth_constraint(
        _constraint_by_name(transform, depth_name),
        source_wrapper_bones=tuple(layer.scale_bone_name for layer in layers),
        target_layer_bones=tuple(layer.layer_bone_name for layer in layers),
    )
    return ik, tuple(adapted_by_name[item.name] for item in transform)


__all__ = [
    "Spine41TwoAxisDocumentAdaptation",
    "adapt_connected_two_axis_constraints_for_spine41",
    "adapt_two_axis_document_for_spine41",
    "adapt_two_axis_document_for_spine41_with_report",
    "adapt_two_axis_scale_rig_for_spine41",
]
