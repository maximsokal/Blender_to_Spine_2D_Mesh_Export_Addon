"""Immutable contracts for connected A1 document composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Tuple

from .composition import SpineDocumentCompositionResult
from .legacy_rig_contracts import UniformScaleMode
from .model import SpineDocument
from .rig_profiles import A1RigProfile, resolve_a1_rig_profile


def _require_canonical_string(value: object, field_name: str) -> str:
    """Require one non-empty identity string with no boundary whitespace."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain leading or trailing whitespace")
    return value


def _require_finite_number(value: object, field_name: str) -> float:
    """Return a finite numeric value while rejecting bool-as-int ambiguity."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be a finite number")
    return resolved


class ConnectedPlacementSpace(str, Enum):
    """Define where a component's visible XY placement is already encoded."""

    ANCHOR_RELATIVE_WORLD = "ANCHOR_RELATIVE_WORLD"
    PRESERVE_DOCUMENT = "PRESERVE_DOCUMENT"


@dataclass(frozen=True, slots=True)
class ConnectedObjectDocument:
    component_id: str
    prefix: str
    document: SpineDocument
    world_position: Tuple[float, float, float]
    animation_namespace: str | None = None
    placement_space: ConnectedPlacementSpace = (
        ConnectedPlacementSpace.ANCHOR_RELATIVE_WORLD
    )

    def __post_init__(self) -> None:
        _require_canonical_string(self.component_id, "component_id")
        _require_canonical_string(self.prefix, "prefix")
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if not isinstance(self.world_position, tuple) or len(self.world_position) != 3:
            raise ValueError("world_position must contain three finite values")
        for index, value in enumerate(self.world_position):
            _require_finite_number(value, f"world_position[{index}]")
        if self.animation_namespace is not None:
            _require_canonical_string(
                self.animation_namespace,
                "animation_namespace",
            )
        if not isinstance(self.placement_space, ConnectedPlacementSpace):
            raise TypeError("placement_space must be ConnectedPlacementSpace")


@dataclass(frozen=True, slots=True)
class ConnectedGroupSettings:
    texture_width: int
    texture_height: int
    group_prefix: str = "all_objects"
    anchor_component_id: str | None = None
    z_tolerance: float = 1e-4
    scale_mode: UniformScaleMode = UniformScaleMode.AVERAGE
    animation_separator: str = "/"
    namespace_animations: bool = True

    def __post_init__(self) -> None:
        for field_name in ("texture_width", "texture_height"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        _require_canonical_string(self.group_prefix, "group_prefix")
        if self.anchor_component_id is not None:
            _require_canonical_string(
                self.anchor_component_id,
                "anchor_component_id",
            )
        tolerance = _require_finite_number(self.z_tolerance, "z_tolerance")
        if tolerance < 0.0:
            raise ValueError("z_tolerance cannot be negative")
        if not isinstance(self.scale_mode, UniformScaleMode):
            raise TypeError("scale_mode must be UniformScaleMode")
        if not isinstance(self.namespace_animations, bool):
            raise TypeError("namespace_animations must be bool")
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
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer")
        _require_finite_number(
            self.representative_relative_z,
            "representative_relative_z",
        )
        if not isinstance(self.component_ids, tuple) or not self.component_ids:
            raise ValueError("component_ids must be a non-empty tuple")
        for index, component_id in enumerate(self.component_ids):
            _require_canonical_string(component_id, f"component_ids[{index}]")
        if len(self.component_ids) != len(set(self.component_ids)):
            raise ValueError("component_ids cannot contain duplicates")
        _require_canonical_string(self.scale_bone_name, "scale_bone_name")
        _require_canonical_string(self.layer_bone_name, "layer_bone_name")
        if self.scale_bone_name == self.layer_bone_name:
            raise ValueError("scale_bone_name and layer_bone_name must differ")


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
    placement_space: ConnectedPlacementSpace = (
        ConnectedPlacementSpace.ANCHOR_RELATIVE_WORLD
    )

    def __post_init__(self) -> None:
        _require_canonical_string(self.component_id, "component_id")
        _require_canonical_string(self.prefix, "prefix")
        for field_name in ("relative_x", "relative_y", "relative_z"):
            _require_finite_number(getattr(self, field_name), field_name)
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer")
        _require_canonical_string(self.main_bone_name, "main_bone_name")
        _require_canonical_string(
            self.parent_layer_bone_name,
            "parent_layer_bone_name",
        )
        if not isinstance(self.placement_space, ConnectedPlacementSpace):
            raise TypeError("placement_space must be ConnectedPlacementSpace")


@dataclass(frozen=True, slots=True)
class ConnectedConstraintSchedule:
    """Legacy-compatible global/object order grouped by connected Z layer.

    The historical ``main`` implementation assigned one order value to a whole Z layer.
    Multiple independent object constraints may therefore share an order when their
    objects belong to the same layer. The set of used order values must still be dense.
    """

    global_rotation_x: int
    global_rotation_y: int
    global_rotation_z: int | None
    object_rotation_x: Tuple[Tuple[str, int], ...]
    object_rotation_y: Tuple[Tuple[str, int], ...]
    global_scale_ik: int
    object_scale_ik: Tuple[Tuple[str, int], ...]
    global_scale: int
    object_scale: Tuple[Tuple[str, int], ...]
    object_rotation_z: Tuple[Tuple[str, int], ...]
    object_scale_compensator: Tuple[Tuple[str, int], ...]
    global_scale_depth: int | None = None
    object_scale_depth: Tuple[Tuple[str, int], ...] = ()
    profile_id: str = A1RigProfile.THREE_AXIS_ROTATION.value

    def __post_init__(self) -> None:
        profile = resolve_a1_rig_profile(self.profile_id)

        for field_name in (
            "global_rotation_x",
            "global_rotation_y",
            "global_scale_ik",
            "global_scale",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

        for field_name in ("global_rotation_z", "global_scale_depth"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be None or a non-negative integer")

        common_assignment_fields = (
            "object_rotation_x",
            "object_rotation_y",
            "object_scale_ik",
            "object_scale",
        )
        if profile is A1RigProfile.THREE_AXIS_ROTATION:
            required_assignment_fields = (
                *common_assignment_fields,
                "object_rotation_z",
            )
            forbidden_assignment_fields = (
                "object_scale_compensator",
                "object_scale_depth",
            )
            if self.global_rotation_z is None:
                raise ValueError("legacy connected schedule requires global_rotation_z")
            if self.global_scale_depth is not None:
                raise ValueError("legacy connected schedule cannot define global_scale_depth")
        else:
            required_assignment_fields = (
                *common_assignment_fields,
                "object_scale_depth",
            )
            forbidden_assignment_fields = (
                "object_rotation_z",
                "object_scale_compensator",
            )
            if self.global_rotation_z is not None:
                raise ValueError("two-axis connected schedule cannot define global_rotation_z")
            if self.global_scale_depth is None:
                raise ValueError("two-axis connected schedule requires global_scale_depth")

        for field_name in forbidden_assignment_fields:
            if getattr(self, field_name):
                raise ValueError(
                    f"{field_name} is not valid for connected profile {profile.value}"
                )

        expected_components: Tuple[str, ...] | None = None
        for field_name in required_assignment_fields:
            assignments = getattr(self, field_name)
            if not isinstance(assignments, tuple) or not assignments:
                raise ValueError(f"{field_name} must be a non-empty tuple")
            components: list[str] = []
            for index, assignment in enumerate(assignments):
                if not isinstance(assignment, tuple) or len(assignment) != 2:
                    raise TypeError(
                        f"{field_name}[{index}] must be a (component_id, order) tuple"
                    )
                component_id, order = assignment
                _require_canonical_string(
                    component_id,
                    f"{field_name}[{index}].component_id",
                )
                if isinstance(order, bool) or not isinstance(order, int) or order < 0:
                    raise ValueError(
                        f"{field_name}[{index}].order must be a non-negative integer"
                    )
                components.append(component_id)
            resolved_components = tuple(components)
            if len(resolved_components) != len(set(resolved_components)):
                raise ValueError(f"{field_name} cannot repeat component IDs")
            if expected_components is None:
                expected_components = resolved_components
            elif resolved_components != expected_components:
                raise ValueError(
                    "all object constraint phases must use the same component order"
                )

        if not self.unique_orders:
            raise ValueError("Connected constraint schedule cannot be empty")
        expected_dense = tuple(range(self.unique_orders[-1] + 1))
        if self.unique_orders != expected_dense:
            raise ValueError(
                "Connected constraint schedule order values must form a dense range"
            )

    @property
    def all_orders(self) -> Tuple[int, ...]:
        """Return every assigned order, including valid same-layer duplicates."""

        scalar_orders = [
            self.global_rotation_x,
            self.global_rotation_y,
            self.global_scale_ik,
            self.global_scale,
        ]
        if self.global_rotation_z is not None:
            scalar_orders.append(self.global_rotation_z)
        if self.global_scale_depth is not None:
            scalar_orders.append(self.global_scale_depth)

        assignment_fields = (
            self.object_rotation_x,
            self.object_rotation_y,
            self.object_scale_ik,
            self.object_scale,
            self.object_rotation_z,
            self.object_scale_compensator,
            self.object_scale_depth,
        )
        orders = scalar_orders + [
            order for assignments in assignment_fields for _, order in assignments
        ]
        return tuple(sorted(orders))

    @property
    def unique_orders(self) -> Tuple[int, ...]:
        """Return the dense evaluation phases used by the connected rig."""

        return tuple(sorted(set(self.all_orders)))

    def order_for(self, phase: str, component_id: str) -> int:
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be a non-empty string")
        _require_canonical_string(component_id, "component_id")
        if phase not in {
            "object_rotation_x",
            "object_rotation_y",
            "object_scale_ik",
            "object_scale",
            "object_rotation_z",
            "object_scale_compensator",
            "object_scale_depth",
        }:
            raise KeyError(f"Unknown connected constraint phase '{phase}'")
        mapping = dict(getattr(self, phase))
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

    def __post_init__(self) -> None:
        if not isinstance(self.document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if not isinstance(self.composition, SpineDocumentCompositionResult):
            raise TypeError("composition must be SpineDocumentCompositionResult")
        if not isinstance(self.settings, ConnectedGroupSettings):
            raise TypeError("settings must be ConnectedGroupSettings")
        if not isinstance(self.layers, tuple) or not self.layers:
            raise ValueError("layers must be a non-empty tuple")
        if not all(isinstance(item, ConnectedZLayer) for item in self.layers):
            raise TypeError("layers must contain ConnectedZLayer values")
        if not isinstance(self.placements, tuple) or not self.placements:
            raise ValueError("placements must be a non-empty tuple")
        if not all(
            isinstance(item, ConnectedObjectPlacement) for item in self.placements
        ):
            raise TypeError("placements must contain ConnectedObjectPlacement values")
        if not isinstance(self.constraint_schedule, ConnectedConstraintSchedule):
            raise TypeError(
                "constraint_schedule must be ConnectedConstraintSchedule"
            )
        resolved_scale = _require_finite_number(self.uniform_scale, "uniform_scale")
        if resolved_scale <= 0.0:
            raise ValueError("uniform_scale must be positive")


__all__ = [
    "ConnectedConstraintSchedule",
    "ConnectedGroupBuildResult",
    "ConnectedGroupSettings",
    "ConnectedObjectDocument",
    "ConnectedObjectPlacement",
    "ConnectedPlacementSpace",
    "ConnectedZLayer",
]
