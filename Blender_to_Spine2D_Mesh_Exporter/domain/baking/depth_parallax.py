"""Immutable user-facing settings for Depth Camera Projection parallax reserve."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi


@dataclass(frozen=True, slots=True)
class DepthParallaxSettings:
    """Angular surface and texture reserve around the active-camera horizon.

    Blender RNA stores the value in radians and displays it as degrees. Zero preserves
    the 0.81.0 single-front-attachment behavior exactly. Positive values enable angular
    shared-edge expansion, eight fitted virtual texture views, and reserve attachments.
    """

    horizon_angle_radians: float = 0.0

    def __post_init__(self) -> None:
        value = self.horizon_angle_radians
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("horizon_angle_radians must be a finite number")
        resolved = float(value)
        if not isfinite(resolved):
            raise ValueError("horizon_angle_radians must be finite")
        if resolved < 0.0 or resolved >= pi / 2.0:
            raise ValueError(
                "horizon_angle_radians must be in [0, pi/2)"
            )
        object.__setattr__(self, "horizon_angle_radians", resolved)

    @property
    def enabled(self) -> bool:
        return self.horizon_angle_radians > 1.0e-12


__all__ = ["DepthParallaxSettings"]
