"""Blender API adapters for the rewritten exporter."""

from .context_state import (
    BlenderContextError,
    BlenderContextState,
    activate_object_for_operator,
)
from .evaluated_mesh_reader import (
    EvaluatedMeshReadError,
    EvaluatedMeshSnapshotResult,
    LineageAttributeNames,
    read_evaluated_mesh_snapshot,
)
from .mesh_reader import MeshReadError, read_source_mesh_snapshot
from .mesh_writer import MeshWriteError, TemporaryMeshObject, temporary_mesh_object
from .uv_unwrap import (
    UvOperatorPlan,
    UvUnwrapError,
    build_uv_operator_plan,
    unwrap_snapshot_uv,
)

__all__ = [
    "BlenderContextError",
    "BlenderContextState",
    "EvaluatedMeshReadError",
    "EvaluatedMeshSnapshotResult",
    "LineageAttributeNames",
    "MeshReadError",
    "MeshWriteError",
    "TemporaryMeshObject",
    "UvOperatorPlan",
    "UvUnwrapError",
    "activate_object_for_operator",
    "build_uv_operator_plan",
    "read_evaluated_mesh_snapshot",
    "read_source_mesh_snapshot",
    "temporary_mesh_object",
    "unwrap_snapshot_uv",
]
