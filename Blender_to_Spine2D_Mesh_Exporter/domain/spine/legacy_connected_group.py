"""Compatibility facade for decomposed connected legacy A1 composition."""

from .connected_group_assembly import (
    apply_object_placements as _apply_object_placements,
    build_connected_group_document,
)
from .connected_group_contracts import (
    ConnectedConstraintSchedule,
    ConnectedGroupBuildResult,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    ConnectedObjectPlacement,
    ConnectedZLayer,
)
from .connected_group_error import ConnectedGroupBuildError
from .connected_group_global_rig import (
    build_global_bones_document as _build_global_bones_document,
    build_global_constraints as _build_global_constraints,
)
from .connected_group_layout import (
    ordered_component_ids as _ordered_component_ids,
    resolve_anchor as _anchor,
    resolve_layers_and_placements as _resolve_layers_and_placements,
)
from .connected_group_schedule import (
    build_constraint_schedule as _build_constraint_schedule,
    reorder_object_constraints as _reorder_object_constraints,
)
from .connected_group_validation import (
    validate_connected_group_inputs as _validate_inputs,
)


__all__ = [
    "ConnectedConstraintSchedule",
    "ConnectedGroupBuildError",
    "ConnectedGroupBuildResult",
    "ConnectedGroupSettings",
    "ConnectedObjectDocument",
    "ConnectedObjectPlacement",
    "ConnectedZLayer",
    "_anchor",
    "_apply_object_placements",
    "_build_constraint_schedule",
    "_build_global_bones_document",
    "_build_global_constraints",
    "_ordered_component_ids",
    "_reorder_object_constraints",
    "_resolve_layers_and_placements",
    "_validate_inputs",
    "build_connected_group_document",
]
