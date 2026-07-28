"""Resolve model-independent layout values for the two-axis scale rig."""

from __future__ import annotations

from .legacy_rig_plan import LegacyRigBuildPlan
from .legacy_rig_scale import require_finite_derived
from .two_axis_scale_rig_contracts import TwoAxisScaleRigLayout


# Ratios are normalized from the 500x501 reference skeleton. They affect only
# editor-control placement and length. Deformation uses actual Z-group offsets.
_CONTROL_LENGTH_RATIO = 0.4
_CONTROL_X_RATIO = 1.635
# Keep every visible control at least one control length apart vertically.
_CONTROL_Y_SPACING_RATIO = 0.4
_HELPER_SPAN_RATIO = 1.2


def _rounded(value: float, field_name: str) -> float:
    return require_finite_derived(round(float(value), 2), field_name)


def build_two_axis_scale_layout(plan: LegacyRigBuildPlan) -> TwoAxisScaleRigLayout:
    """Resolve all finite layout values before any Spine model objects are built."""

    if not isinstance(plan, LegacyRigBuildPlan):
        raise TypeError("plan must be LegacyRigBuildPlan")
    if not plan.z_groups:
        raise ValueError("plan.z_groups cannot be empty")

    scale = require_finite_derived(plan.uniform_scale, "uniform_scale")
    depth_values = tuple(float(group.y_offset_pixels) for group in plan.z_groups)
    control_x = _rounded(scale * _CONTROL_X_RATIO, "control_x")
    control_spacing = _rounded(
        scale * _CONTROL_Y_SPACING_RATIO,
        "control_y_spacing",
    )
    return TwoAxisScaleRigLayout(
        control_length=_rounded(scale * _CONTROL_LENGTH_RATIO, "control_length"),
        control_x=control_x,
        control_y_spacing=control_spacing,
        helper_span=_rounded(scale * _HELPER_SPAN_RATIO, "helper_span"),
        # X, Y, and Scale controls form one editor column.
        scale_control_x=control_x,
        scale_control_y=_rounded(-control_spacing, "scale_control_y"),
        minimum_depth_y=_rounded(min(depth_values), "minimum_depth_y"),
        maximum_depth_y=_rounded(max(depth_values), "maximum_depth_y"),
    )


__all__ = ["build_two_axis_scale_layout"]
