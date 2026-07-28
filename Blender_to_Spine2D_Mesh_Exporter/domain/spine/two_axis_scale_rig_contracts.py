"""Immutable contracts for the X/Y rotation plus uniform-scale rig."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class TwoAxisScaleRigLayout:
    """Finite model-independent dimensions for one generated control hierarchy."""

    control_length: float
    control_x: float
    control_y_spacing: float
    helper_span: float
    scale_control_x: float
    scale_control_y: float
    minimum_depth_y: float
    maximum_depth_y: float

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        for field_name in (
            "control_length",
            "control_x",
            "helper_span",
            "scale_control_x",
        ):
            if float(getattr(self, field_name)) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        if self.minimum_depth_y > self.maximum_depth_y:
            raise ValueError("minimum_depth_y cannot exceed maximum_depth_y")


__all__ = ["TwoAxisScaleRigLayout"]
