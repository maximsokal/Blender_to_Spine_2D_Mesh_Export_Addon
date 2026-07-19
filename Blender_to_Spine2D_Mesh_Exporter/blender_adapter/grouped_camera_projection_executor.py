"""Compatibility facade for the decomposed grouped B4 staging pipeline.

Physical ownership lives in grouped validation, visibility, reversible execution,
shared postprocessing, and caller-owned output staging modules.
"""

from .grouped_camera_projection_output import (
    GroupedCameraProjectionStageResult,
    _reserve_group_outputs,
    stage_grouped_camera_projection_outputs,
)
from .grouped_camera_projection_validation import (
    GroupedCameraProjectionRuntime,
    object_name as _object_name,
    rna_identity as _rna_identity,
    validate_grouped_projection_runtime as _validate_group_runtime,
)
from .grouped_camera_projection_visibility import (
    configure_group_camera_visibility as _configure_group_camera_visibility,
)


__all__ = [
    "GroupedCameraProjectionRuntime",
    "GroupedCameraProjectionStageResult",
    "_configure_group_camera_visibility",
    "_object_name",
    "_reserve_group_outputs",
    "_rna_identity",
    "_validate_group_runtime",
    "stage_grouped_camera_projection_outputs",
]
