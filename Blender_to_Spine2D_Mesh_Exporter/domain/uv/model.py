"""Typed settings and results for UV unwrap use cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from ..geometry import MeshSnapshot


class UvUnwrapMethod(str, Enum):
    SMART_PROJECT = "SMART_PROJECT"
    ANGLE_BASED = "ANGLE_BASED"
    CONFORMAL = "CONFORMAL"


@dataclass(frozen=True, slots=True)
class UvUnwrapSettings:
    layer_name: str = "SpineBakeUV"
    method: UvUnwrapMethod = UvUnwrapMethod.SMART_PROJECT
    smart_angle_limit_degrees: float = 66.0
    island_margin: float = 0.001
    area_weight: float = 0.0
    correct_aspect: bool = True
    scale_to_bounds: bool = True
    fill_holes: bool = True
    use_subsurf_data: bool = False
    pack_islands: bool = True
    pack_rotate: bool = True
    pack_scale: bool = True
    pack_margin: float = 0.001

    def __post_init__(self) -> None:
        if not isinstance(self.layer_name, str) or not self.layer_name.strip():
            raise ValueError("layer_name must be a non-empty string")
        if not isinstance(self.method, UvUnwrapMethod):
            raise TypeError("method must be UvUnwrapMethod")
        for field_name in (
            "smart_angle_limit_degrees",
            "island_margin",
            "area_weight",
            "pack_margin",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if not 0.0 < self.smart_angle_limit_degrees <= 180.0:
            raise ValueError("smart_angle_limit_degrees must be in (0, 180]")
        if self.island_margin < 0.0 or self.pack_margin < 0.0:
            raise ValueError("UV margins cannot be negative")
        if not 0.0 <= self.area_weight <= 1.0:
            raise ValueError("area_weight must be in [0, 1]")
        for field_name in (
            "correct_aspect",
            "scale_to_bounds",
            "fill_holes",
            "use_subsurf_data",
            "pack_islands",
            "pack_rotate",
            "pack_scale",
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


@dataclass(frozen=True, slots=True)
class UvUnwrapResult:
    snapshot: MeshSnapshot
    settings: UvUnwrapSettings
    statistics: UvUnwrapStatistics


def calculate_uv_statistics(
    snapshot: MeshSnapshot,
    layer_name: str,
) -> UvUnwrapStatistics:
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(layer_name, str) or not layer_name.strip():
        raise ValueError("layer_name must be a non-empty string")
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
