"""Scale and main-position policy for the legacy A1 rig."""

from __future__ import annotations

from math import isfinite

from .legacy_rig_contracts import LegacyRigBuildRequest, UniformScaleMode


def require_finite_derived(value: object, field_name: str) -> float:
    """Return one finite derived numeric value and reject bool ambiguity."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite")
    resolved = float(value)
    if not isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def _positive_dimension(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def calculate_uniform_scale(
    texture_width: int,
    texture_height: int,
    mode: UniformScaleMode = UniformScaleMode.AVERAGE,
) -> float:
    """Return the exact legacy texture-size scale with finite overflow checks."""

    width = _positive_dimension(texture_width, "texture_width")
    height = _positive_dimension(texture_height, "texture_height")
    if not isinstance(mode, UniformScaleMode):
        raise TypeError("mode must be UniformScaleMode")

    try:
        width_float = float(width)
        height_float = float(height)
    except (OverflowError, ValueError) as exc:
        raise ValueError("texture dimensions are too large for finite rig scale") from exc

    if mode is UniformScaleMode.AVERAGE:
        legacy_sum = width_float + height_float
        result = (
            legacy_sum / 2.0
            if isfinite(legacy_sum)
            else (width_float / 2.0) + (height_float / 2.0)
        )
    elif mode is UniformScaleMode.MAXIMUM:
        result = max(width_float, height_float)
    else:
        result = min(width_float, height_float)

    resolved = require_finite_derived(result, "uniform_scale")
    if resolved <= 0.0:
        raise ValueError("uniform_scale must be positive")
    return resolved


def resolve_main_position(
    request: LegacyRigBuildRequest,
) -> tuple[float, float]:
    """Resolve and round the historical main-bone pixel position."""

    if not isinstance(request, LegacyRigBuildRequest):
        raise TypeError("request must be LegacyRigBuildRequest")
    if request.main_position_pixels is not None:
        x_value, y_value = request.main_position_pixels
        resolved_x = require_finite_derived(float(x_value), "main_position_pixels[0]")
        resolved_y = require_finite_derived(float(y_value), "main_position_pixels[1]")
        rounded_x = require_finite_derived(round(resolved_x, 2), "main_position_x")
        rounded_y = require_finite_derived(round(resolved_y, 2), "main_position_y")
        return rounded_x, rounded_y

    rounded_y = require_finite_derived(
        round(float(request.average_y_pixels), 2),
        "main_position_y",
    )
    return 0.0, rounded_y


__all__ = [
    "calculate_uniform_scale",
    "require_finite_derived",
    "resolve_main_position",
]
