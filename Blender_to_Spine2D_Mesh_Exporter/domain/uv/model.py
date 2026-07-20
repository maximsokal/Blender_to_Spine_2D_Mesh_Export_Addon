"""Typed settings and results for UV unwrap use cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..geometry import MeshSnapshot, MeshSnapshotValidator
from ..geometry.contracts import (
    require_exact_type,
    require_finite_number,
    require_integer,
    require_non_empty_string,
)
from .range import UvRangePolicy


class UvUnwrapMethod(str, Enum):
    SMART_PROJECT = "SMART_PROJECT"
    ANGLE_BASED = "ANGLE_BASED"
    CONFORMAL = "CONFORMAL"
    MINIMUM_STRETCH = "MINIMUM_STRETCH"


class UvMarginMethod(str, Enum):
    SCALED = "SCALED"
    ADD = "ADD"
    FRACTION = "FRACTION"


class UvSmartRotateMethod(str, Enum):
    AXIS_ALIGNED = "AXIS_ALIGNED"
    AXIS_ALIGNED_X = "AXIS_ALIGNED_X"
    AXIS_ALIGNED_Y = "AXIS_ALIGNED_Y"


class UvPackRotateMethod(str, Enum):
    ANY = "ANY"
    CARDINAL = "CARDINAL"
    AXIS_ALIGNED = "AXIS_ALIGNED"
    AXIS_ALIGNED_X = "AXIS_ALIGNED_X"
    AXIS_ALIGNED_Y = "AXIS_ALIGNED_Y"


class UvPackUdimSource(str, Enum):
    CLOSEST_UDIM = "CLOSEST_UDIM"
    ACTIVE_UDIM = "ACTIVE_UDIM"
    ORIGINAL_AABB = "ORIGINAL_AABB"


class UvPackPinMethod(str, Enum):
    SCALE = "SCALE"
    ROTATION = "ROTATION"
    ROTATION_SCALE = "ROTATION_SCALE"
    LOCKED = "LOCKED"


class UvPackShapeMethod(str, Enum):
    CONCAVE = "CONCAVE"
    CONVEX = "CONVEX"
    AABB = "AABB"


@dataclass(frozen=True, slots=True)
class UvUnwrapSettings:
    layer_name: str = "SpineBakeUV"
    method: UvUnwrapMethod = UvUnwrapMethod.SMART_PROJECT
    margin_method: UvMarginMethod = UvMarginMethod.SCALED
    smart_angle_limit_degrees: float = 66.0
    smart_rotate_method: UvSmartRotateMethod = UvSmartRotateMethod.AXIS_ALIGNED_Y
    island_margin: float = 0.001
    area_weight: float = 0.0
    correct_aspect: bool = True
    scale_to_bounds: bool = True
    fill_holes: bool = True
    use_subsurf_data: bool = False
    no_flip: bool = False
    iterations: int = 10
    use_weights: bool = False
    weight_group: str = "uv_importance"
    weight_factor: float = 1.0
    pack_islands: bool = True
    pack_udim_source: UvPackUdimSource = UvPackUdimSource.CLOSEST_UDIM
    pack_rotate: bool = True
    pack_rotate_method: UvPackRotateMethod = UvPackRotateMethod.ANY
    pack_scale: bool = True
    pack_merge_overlap: bool = False
    pack_margin: float = 0.001
    pack_pin: bool = False
    pack_pin_method: UvPackPinMethod = UvPackPinMethod.LOCKED
    pack_shape_method: UvPackShapeMethod = UvPackShapeMethod.CONCAVE
    # Appended to preserve the positional layout of the pre-hardening contract.
    range_policy: UvRangePolicy = UvRangePolicy.REQUIRE_UNIT_SQUARE
    range_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        require_non_empty_string(self.layer_name, "layer_name")
        enum_fields = {
            "method": UvUnwrapMethod,
            "margin_method": UvMarginMethod,
            "smart_rotate_method": UvSmartRotateMethod,
            "pack_udim_source": UvPackUdimSource,
            "pack_rotate_method": UvPackRotateMethod,
            "pack_pin_method": UvPackPinMethod,
            "pack_shape_method": UvPackShapeMethod,
            "range_policy": UvRangePolicy,
        }
        for field_name, enum_type in enum_fields.items():
            require_exact_type(getattr(self, field_name), enum_type, field_name)

        numeric_values = {
            field_name: require_finite_number(getattr(self, field_name), field_name)
            for field_name in (
                "smart_angle_limit_degrees",
                "island_margin",
                "area_weight",
                "pack_margin",
                "weight_factor",
                "range_epsilon",
            )
        }
        if not 0.0 < numeric_values["smart_angle_limit_degrees"] <= 90.0:
            raise ValueError("smart_angle_limit_degrees must be in (0, 90]")
        if not 0.0 <= numeric_values["island_margin"] <= 1.0:
            raise ValueError("island_margin must be in [0, 1]")
        if not 0.0 <= numeric_values["pack_margin"] <= 1.0:
            raise ValueError("pack_margin must be in [0, 1]")
        if not 0.0 <= numeric_values["area_weight"] <= 1.0:
            raise ValueError("area_weight must be in [0, 1]")
        if numeric_values["range_epsilon"] < 0.0:
            raise ValueError("range_epsilon cannot be negative")

        require_integer(self.iterations, "iterations", minimum=0, maximum=10000)
        require_non_empty_string(self.weight_group, "weight_group")

        for field_name in (
            "correct_aspect",
            "scale_to_bounds",
            "fill_holes",
            "use_subsurf_data",
            "no_flip",
            "use_weights",
            "pack_islands",
            "pack_rotate",
            "pack_scale",
            "pack_merge_overlap",
            "pack_pin",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")


@dataclass(frozen=True, slots=True)
class UvUnwrapStatistics:
    loop_count: int
    minimum_u: float
    maximum_u: float
    minimum_v: float
    maximum_v: float
    outside_unit_square_count: int

    def __post_init__(self) -> None:
        require_integer(self.loop_count, "loop_count", minimum=1)
        require_integer(
            self.outside_unit_square_count,
            "outside_unit_square_count",
            minimum=0,
            maximum=self.loop_count,
        )
        minimum_u = require_finite_number(self.minimum_u, "minimum_u")
        maximum_u = require_finite_number(self.maximum_u, "maximum_u")
        minimum_v = require_finite_number(self.minimum_v, "minimum_v")
        maximum_v = require_finite_number(self.maximum_v, "maximum_v")
        if minimum_u > maximum_u:
            raise ValueError("minimum_u cannot exceed maximum_u")
        if minimum_v > maximum_v:
            raise ValueError("minimum_v cannot exceed maximum_v")


@dataclass(frozen=True, slots=True)
class UvUnwrapResult:
    snapshot: MeshSnapshot
    settings: UvUnwrapSettings
    statistics: UvUnwrapStatistics

    def __post_init__(self) -> None:
        require_exact_type(self.snapshot, MeshSnapshot, "snapshot")
        require_exact_type(self.settings, UvUnwrapSettings, "settings")
        require_exact_type(self.statistics, UvUnwrapStatistics, "statistics")
        MeshSnapshotValidator().validate_or_raise(self.snapshot)
        if self.snapshot.active_uv_layer != self.settings.layer_name:
            raise ValueError(
                "snapshot.active_uv_layer must match settings.layer_name for an "
                "UvUnwrapResult"
            )
        expected_statistics = calculate_uv_statistics(
            self.snapshot,
            self.settings.layer_name,
        )
        if self.statistics != expected_statistics:
            raise ValueError(
                "statistics do not match the UV coordinates stored in snapshot; "
                f"expected={expected_statistics}, actual={self.statistics}"
            )


def calculate_uv_statistics(
    snapshot: MeshSnapshot,
    layer_name: str,
) -> UvUnwrapStatistics:
    require_exact_type(snapshot, MeshSnapshot, "snapshot")
    require_non_empty_string(layer_name, "layer_name")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    coordinates = tuple(
        coordinate
        for loop in snapshot.loops
        for coordinate in (loop.uv(layer_name),)
        if coordinate is not None
    )
    if len(coordinates) != len(snapshot.loops):
        raise ValueError(
            f"UV layer '{layer_name}' is not present on every loop in snapshot"
        )
    if not coordinates:
        raise ValueError("snapshot contains no loops")
    minimum_u = min(value[0] for value in coordinates)
    maximum_u = max(value[0] for value in coordinates)
    minimum_v = min(value[1] for value in coordinates)
    maximum_v = max(value[1] for value in coordinates)
    outside = sum(
        value[0] < 0.0 or value[0] > 1.0 or value[1] < 0.0 or value[1] > 1.0
        for value in coordinates
    )
    return UvUnwrapStatistics(
        loop_count=len(coordinates),
        minimum_u=minimum_u,
        maximum_u=maximum_u,
        minimum_v=minimum_v,
        maximum_v=maximum_v,
        outside_unit_square_count=outside,
    )
