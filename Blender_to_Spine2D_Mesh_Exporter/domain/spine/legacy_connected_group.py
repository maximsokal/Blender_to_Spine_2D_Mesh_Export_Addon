"""Build the legacy-style ``all_objects`` rig on typed Spine documents.

Object documents remain independently valid until after composition. The builder
first creates a valid global bones-only component, gives every known A1 object
constraint a collision-free global order, composes all weighted attachments through
the typed document composer, then reparents each object main bone to its resolved
Z-layer and adds global constraints once all referenced bones exist.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from typing import Tuple

from .composition import (
    ConstraintOrderPolicy,
    SpineCompositionSettings,
    SpineDocumentComponent,
    SpineDocumentCompositionResult,
    compose_spine_documents,
)
from .legacy_profile import LegacyRigProfile
from .legacy_rig_builder import UniformScaleMode, calculate_uniform_scale
from .model import Bone, IKConstraint, SpineDocument, TransformConstraint
from .validator import SpineValidator


@dataclass(frozen=True, slots=True)
class ConnectedObjectDocument:
    component_id: str
    prefix: str
    document: SpineDocument
    world_position: Tuple[float, float, float]
    animation_namespace: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("component_id", "prefix"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if (
            not isinstance(self.world_position, tuple)
            or len(self.world_position) != 3
            or not all(
                isinstance(value, (int, float)) and isfinite(float(value))
                for value in self.world_position
            )
        ):
            raise ValueError("world_position must contain three finite values")
        if self.animation_namespace is not None and (
            not isinstance(self.animation_namespace, str)
            or not self.animation_namespace.strip()
        ):
            raise ValueError(
                "animation_namespace must be a non-empty string or None"
            )


@dataclass(frozen=True, slots=True)
class ConnectedGroupSettings:
    texture_width: int
    texture_height: int
    group_prefix: str = "all_objects"
    anchor_component_id: str | None = None
    z_tolerance: float = 1e-4
    scale_mode: UniformScaleMode = UniformScaleMode.AVERAGE
    animation_separator: str = "/"

    def __post_init__(self) -> None:
        for field_name in ("texture_width", "texture_height"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.group_prefix, str) or not self.group_prefix.strip():
            raise ValueError("group_prefix must be a non-empty string")
        if self.anchor_component_id is not None and (
            not isinstance(self.anchor_component_id, str)
            or not self.anchor_component_id.strip()
        ):
            raise ValueError(
                "anchor_component_id must be a non-empty string or None"
            )
        if not isinstance(self.z_tolerance, (int, float)) or not isfinite(
            float(self.z_tolerance)
        ):
            raise ValueError("z_tolerance must be finite")
        if self.z_tolerance < 0.0:
            raise ValueError("z_tolerance cannot be negative")
        if not isinstance(self.scale_mode, UniformScaleMode):
            raise TypeError("scale_mode must be UniformScaleMode")
        if not isinstance(self.animation_separator, str) or not self.animation_separator:
            raise ValueError("animation_separator must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ConnectedZLayer:
    layer_index: int
    representative_relative_z: float
    component_ids: Tuple[str, ...]
    scale_bone_name: str
    layer_bone_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer_index, int) or self.layer_index < 0:
            raise ValueError("layer_index must be a non-negative integer")
        if not self.component_ids:
            raise ValueError("component_ids cannot be empty")


@dataclass(frozen=True, slots=True)
class ConnectedObjectPlacement:
    component_id: str
    prefix: str
    relative_x: float
    relative_y: float
    relative_z: float
    layer_index: int
    main_bone_name: str
    parent_layer_bone_name: str


@dataclass(frozen=True, slots=True)
class ConnectedConstraintSchedule:
    global_rotation_x: int
    global_rotation_y: int
    global_rotation_z: int
    object_rotation_x: Tuple[Tuple[str, int], ...]
    object_rotation_y: Tuple[Tuple[str, int], ...]
    global_scale_ik: int
    object_scale_ik: Tuple[Tuple[str, int], ...]
    global_scale: int
    object_scale: Tuple[Tuple[str, int], ...]
    object_rotation_z: Tuple[Tuple[str, int], ...]
    object_scale_compensator: Tuple[Tuple[str, int], ...]

    @property
    def all_orders(self) -> Tuple[int, ...]:
        return (
            self.global_rotation_x,
            self.global_rotation_y,
            self.global_rotation_z,
            *(order for _, order in self.object_rotation_x),
            *(order for _, order in self.object_rotation_y),
            self.global_scale_ik,
            *(order for _, order in self.object_scale_ik),
            self.global_scale,
            *(order for _, order in self.object_scale),
            *(order for _, order in self.object_rotation_z),
            *(order for _, order in self.object_scale_compensator),
        )

    def order_for(self, phase: str, component_id: str) -> int:
        values = getattr(self, phase)
        mapping = dict(values)
        try:
            return mapping[component_id]
        except KeyError as exc:
            raise KeyError(
                f"No constraint order for phase '{phase}' and component "
                f"'{component_id}'"
            ) from exc


@dataclass(frozen=True, slots=True)
class ConnectedGroupBuildResult:
    document: SpineDocument
    composition: SpineDocumentCompositionResult
    settings: ConnectedGroupSettings
    layers: Tuple[ConnectedZLayer, ...]
    placements: Tuple[ConnectedObjectPlacement, ...]
    constraint_schedule: ConnectedConstraintSchedule
    uniform_scale: float


class ConnectedGroupBuildError(ValueError):
    """Raised when A1 object documents cannot form one connected global rig."""


def _validate_inputs(
    objects: Tuple[ConnectedObjectDocument, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
) -> None:
    if not isinstance(objects, tuple) or len(objects) < 2:
        raise ValueError("objects must contain at least two connected documents")
    if not all(isinstance(item, ConnectedObjectDocument) for item in objects):
        raise TypeError("objects must contain ConnectedObjectDocument values")
    component_ids = tuple(item.component_id for item in objects)
    prefixes = tuple(item.prefix for item in objects)
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("component_id values must be unique")
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("connected object prefixes must be unique")
    if settings.group_prefix in set(prefixes):
        raise ValueError("group_prefix cannot equal an object prefix")
    if settings.anchor_component_id is not None and (
        settings.anchor_component_id not in set(component_ids)
    ):
        raise ValueError("anchor_component_id is not present in objects")

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
        expected_constraints = set(profile.constraint_names(item.prefix))
        actual_constraints = {
            constraint.name
            for constraint in (*item.document.ik, *item.document.transform)
        }
        if actual_constraints != expected_constraints:
            raise ConnectedGroupBuildError(
                f"Object component '{item.component_id}' must contain exactly the "
                f"six A1 constraints; missing="
                f"{tuple(sorted(expected_constraints - actual_constraints))}, "
                f"unexpected="
                f"{tuple(sorted(actual_constraints - expected_constraints))}"
            )


def _anchor(
    objects: Tuple[ConnectedObjectDocument, ...],
    settings: ConnectedGroupSettings,
) -> ConnectedObjectDocument:
    if settings.anchor_component_id is None:
        return objects[0]
    return next(
        item for item in objects if item.component_id == settings.anchor_component_id
    )


def _resolve_layers_and_placements(
    objects: Tuple[ConnectedObjectDocument, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
) -> tuple[Tuple[ConnectedZLayer, ...], Tuple[ConnectedObjectPlacement, ...]]:
    anchor = _anchor(objects, settings)
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
            clusters[-1]["component_ids"].append(item.component_id)

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


def _ordered_component_ids(
    placements: Tuple[ConnectedObjectPlacement, ...],
) -> Tuple[str, ...]:
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


def _build_constraint_schedule(
    placements: Tuple[ConnectedObjectPlacement, ...],
) -> ConnectedConstraintSchedule:
    component_ids = _ordered_component_ids(placements)
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

    schedule = ConnectedConstraintSchedule(
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
    if schedule.all_orders != tuple(range(len(schedule.all_orders))):
        raise ConnectedGroupBuildError(
            "Connected constraint schedule is not contiguous and unique"
        )
    return schedule


def _reorder_object_constraints(
    item: ConnectedObjectDocument,
    schedule: ConnectedConstraintSchedule,
    profile: LegacyRigProfile,
) -> SpineDocument:
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


def _build_global_bones_document(
    source_skeleton,
    layers: Tuple[ConnectedZLayer, ...],
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> SpineDocument:
    half_scale = uniform_scale / 2.0
    prefix = settings.group_prefix
    root = profile.root_bone()
    main = profile.main_bone(prefix)
    base = profile.base_bone(prefix)
    scale = profile.scale_rotate_x_bone(prefix)
    rotate = profile.rotate_x_bone(prefix)
    control_x, control_y, control_z = profile.control_bones(prefix)
    constraint, constraint_scale, constraint_rotate, constraint_ik = (
        profile.ik_chain_bones(prefix)
    )

    bones: list[Bone] = [
        Bone(name=root),
        Bone(name=main, parent=root, length=half_scale),
        Bone(name=base, parent=main),
        Bone(name=scale, parent=base, length=half_scale),
        Bone(
            name=control_x,
            parent=root,
            length=uniform_scale,
            x=uniform_scale,
            y=half_scale,
            color="ff0000ff",
        ),
        Bone(
            name=control_y,
            parent=root,
            length=uniform_scale,
            x=uniform_scale,
            color="00ff18ff",
        ),
        Bone(
            name=control_z,
            parent=root,
            length=uniform_scale,
            x=uniform_scale,
            y=-half_scale,
            color="002cffff",
        ),
        Bone(name=rotate, parent=scale, length=half_scale * 0.1),
        Bone(
            name=constraint,
            parent=base,
            length=half_scale,
            rotation=-90.0,
            color="abe323ff",
        ),
        Bone(name=constraint_scale, parent=base, rotation=-90.0),
        Bone(name=constraint_rotate, parent=constraint_scale),
        Bone(
            name=constraint_ik,
            parent=constraint_rotate,
            rotation=90.0,
            color="ff3f00ff",
            icon="ik",
        ),
    ]
    for layer in layers:
        bones.extend(
            (
                Bone(
                    name=layer.scale_bone_name,
                    parent=rotate,
                    length=half_scale * 0.1,
                ),
                Bone(
                    name=layer.layer_bone_name,
                    parent=layer.scale_bone_name,
                    length=half_scale * 0.1,
                ),
            )
        )

    document = SpineDocument(
        skeleton=deepcopy(dict(source_skeleton)),
        bones=tuple(bones),
        slots=(),
        skins=(),
    )
    SpineValidator().validate_or_raise(document)
    return document


def _build_global_constraints(
    objects: Tuple[ConnectedObjectDocument, ...],
    layers: Tuple[ConnectedZLayer, ...],
    schedule: ConnectedConstraintSchedule,
    settings: ConnectedGroupSettings,
    profile: LegacyRigProfile,
    uniform_scale: float,
) -> tuple[Tuple[IKConstraint, ...], Tuple[TransformConstraint, ...]]:
    prefix = settings.group_prefix
    half_scale = uniform_scale / 2.0
    base = profile.base_bone(prefix)
    rotate = profile.rotate_x_bone(prefix)
    control_x, control_y, control_z = profile.control_bones(prefix)
    constraint, _, constraint_rotate, constraint_ik = profile.ik_chain_bones(prefix)
    scale_bones = tuple(layer.scale_bone_name for layer in layers)
    object_base_bones = tuple(profile.base_bone(item.prefix) for item in objects)

    ik = (
        IKConstraint(
            name=profile.scale_ik_constraint(prefix),
            order=schedule.global_scale_ik,
            bones=(constraint,),
            target=constraint_ik,
            extras={"compress": True, "stretch": True},
        ),
    )
    transform = (
        TransformConstraint(
            name=profile.rotation_x_constraint(prefix),
            order=schedule.global_rotation_x,
            bones=scale_bones + (base,),
            target=control_x,
            extras={
                "rotation": 90,
                "local": True,
                "relative": True,
                "x": -(uniform_scale * 2.0),
                "y": -half_scale,
                "scaleX": -1,
                "scaleY": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_y_constraint(prefix),
            order=schedule.global_rotation_y,
            bones=(rotate, constraint_rotate),
            target=control_y,
            extras={
                "local": True,
                "relative": True,
                "x": uniform_scale,
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.rotation_z_constraint(prefix),
            order=schedule.global_rotation_z,
            bones=object_base_bones,
            target=control_z,
            extras={
                "local": True,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
        TransformConstraint(
            name=profile.scale_constraint(prefix),
            order=schedule.global_scale,
            bones=scale_bones,
            target=constraint,
            extras={
                "scaleX": -1,
                "mixX": 0,
                "mixScaleX": 0,
                "mixShearY": 0,
            },
        ),
    )
    return ik, transform


def _apply_object_placements(
    document: SpineDocument,
    placements: Tuple[ConnectedObjectPlacement, ...],
    uniform_scale: float,
) -> SpineDocument:
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
        updated_bones.append(
            replace(
                bone,
                parent=placement.parent_layer_bone_name,
                x=round(placement.relative_x * uniform_scale, 2),
                y=round((bone.y or 0.0) + placement.relative_y * uniform_scale, 2),
            )
        )
    missing = set(placement_by_main) - found
    if missing:
        raise ConnectedGroupBuildError(
            f"Unable to apply placements; main bones missing: "
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
    resolved_profile = profile or LegacyRigProfile()
    if not isinstance(resolved_profile, LegacyRigProfile):
        raise TypeError("profile must be LegacyRigProfile")
    _validate_inputs(objects, settings, resolved_profile)

    layers, placements = _resolve_layers_and_placements(
        objects,
        settings,
        resolved_profile,
    )
    schedule = _build_constraint_schedule(placements)
    uniform_scale = calculate_uniform_scale(
        settings.texture_width,
        settings.texture_height,
        settings.scale_mode,
    )
    global_document = _build_global_bones_document(
        objects[0].document.skeleton,
        layers,
        settings,
        resolved_profile,
        uniform_scale,
    )

    object_components = tuple(
        SpineDocumentComponent(
            component_id=item.component_id,
            document=_reorder_object_constraints(
                item,
                schedule,
                resolved_profile,
            ),
            animation_namespace=(
                item.animation_namespace or item.component_id
            ),
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
            namespace_animations=True,
            animation_separator=settings.animation_separator,
        ),
    )
    placed_document = _apply_object_placements(
        composition.document,
        placements,
        uniform_scale,
    )
    global_ik, global_transform = _build_global_constraints(
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

    return ConnectedGroupBuildResult(
        document=final_document,
        composition=composition,
        settings=settings,
        layers=layers,
        placements=placements,
        constraint_schedule=schedule,
        uniform_scale=uniform_scale,
    )
