"""Compatibility facade for the decomposed two-axis plus scale rig builder."""

from .two_axis_scale_rig_assembly import build_two_axis_scale_rig
from .two_axis_scale_rig_bones import build_two_axis_scale_bones
from .two_axis_scale_rig_constraints import build_two_axis_scale_constraints
from .two_axis_scale_rig_contracts import TwoAxisScaleRigLayout
from .two_axis_scale_rig_plan import build_two_axis_scale_layout
from .two_axis_scale_rig_validation import validate_two_axis_scale_rig_result


__all__ = [
    "TwoAxisScaleRigLayout",
    "build_two_axis_scale_bones",
    "build_two_axis_scale_constraints",
    "build_two_axis_scale_layout",
    "build_two_axis_scale_rig",
    "validate_two_axis_scale_rig_result",
]
