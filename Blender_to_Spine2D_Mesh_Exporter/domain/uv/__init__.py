"""Blender-independent UV unwrap contracts and immutable layouts."""

from .layout import (
    UvLayout,
    UvLayoutError,
    UvLoopCoordinate,
    apply_uv_layout,
    build_uv_layout,
)
from .model import (
    UvMarginMethod,
    UvPackPinMethod,
    UvPackRotateMethod,
    UvPackShapeMethod,
    UvPackUdimSource,
    UvSmartRotateMethod,
    UvUnwrapMethod,
    UvUnwrapResult,
    UvUnwrapSettings,
    UvUnwrapStatistics,
    calculate_uv_statistics,
)
from .range import (
    UvRangeError,
    UvRangePolicy,
    UvRangeReport,
    UvRangeViolation,
    enforce_uv_range,
    inspect_uv_range,
)

__all__ = [
    "UvLayout",
    "UvLayoutError",
    "UvLoopCoordinate",
    "UvMarginMethod",
    "UvPackPinMethod",
    "UvPackRotateMethod",
    "UvPackShapeMethod",
    "UvPackUdimSource",
    "UvRangeError",
    "UvRangePolicy",
    "UvRangeReport",
    "UvRangeViolation",
    "UvSmartRotateMethod",
    "UvUnwrapMethod",
    "UvUnwrapResult",
    "UvUnwrapSettings",
    "UvUnwrapStatistics",
    "apply_uv_layout",
    "build_uv_layout",
    "calculate_uv_statistics",
    "enforce_uv_range",
    "inspect_uv_range",
]
