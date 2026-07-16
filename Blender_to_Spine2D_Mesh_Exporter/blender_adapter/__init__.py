"""Blender API adapters for the rewritten exporter."""

from .a1_mixed_object_export import (
    export_a1_mixed_object,
    prepare_a1_mixed_object,
)
from .a1_multi_object_export import (
    A1MultiObjectPreparationError,
    A1MultiObjectSource,
    PreparedA1MultiObject,
    export_a1_multi_object,
    prepare_a1_multi_object,
)
from .a1_object_preparation import (
    A1ObjectPreparationError,
    PreparedA1Object,
    StatisticsValue,
    prepare_a1_object,
)
from .a1_single_object_export import export_a1_single_object
from .bake_compositor import (
    BakeCompositeError,
    BakePixelBuffer,
    compose_bake_passes,
    read_bake_image_pixels,
    write_bake_image_pixels,
)
from .bake_executor import (
    BakeExecutionError,
    build_bake_execution_result,
    execute_bake_plan,
    stage_bake_plan_outputs,
)
from .bake_materials import (
    BakeMaterialError,
    PreparedBakeMaterials,
    temporary_bake_materials,
)
from .bake_scene_state import (
    BakeSceneState,
    BakeSceneStateError,
    configure_scene_for_bake,
    preserve_bake_scene_state,
)
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
from .material_analyzer import (
    MaterialAnalysisError,
    analyse_material_slot,
    analyse_object_materials,
)
from .mesh_reader import MeshReadError, read_source_mesh_snapshot
from .mesh_writer import MeshWriteError, TemporaryMeshObject, temporary_mesh_object
from .scene_bake_analyzer import (
    SceneBakeAnalysisError,
    analyse_bake_contexts,
    analyse_object_bake_context,
    analyse_scene_bake_context,
    validate_runtime_scene_context,
)
from .shader_graph_analyzer import (
    MaterialGraphAnalysisError,
    analyse_material_graph,
)
from .uv_unwrap import (
    UvOperatorPlan,
    UvUnwrapError,
    build_uv_operator_plan,
    unwrap_snapshot_uv,
)

__all__ = [
    "A1MultiObjectPreparationError",
    "A1MultiObjectSource",
    "A1ObjectPreparationError",
    "BakeCompositeError",
    "BakeExecutionError",
    "BakeMaterialError",
    "BakePixelBuffer",
    "BakeSceneState",
    "BakeSceneStateError",
    "BlenderContextError",
    "BlenderContextState",
    "EvaluatedMeshReadError",
    "EvaluatedMeshSnapshotResult",
    "LineageAttributeNames",
    "MaterialAnalysisError",
    "MaterialGraphAnalysisError",
    "MeshReadError",
    "MeshWriteError",
    "PreparedA1MultiObject",
    "PreparedA1Object",
    "PreparedBakeMaterials",
    "SceneBakeAnalysisError",
    "StatisticsValue",
    "TemporaryMeshObject",
    "UvOperatorPlan",
    "UvUnwrapError",
    "activate_object_for_operator",
    "analyse_bake_contexts",
    "analyse_material_graph",
    "analyse_material_slot",
    "analyse_object_bake_context",
    "analyse_object_materials",
    "analyse_scene_bake_context",
    "build_bake_execution_result",
    "build_uv_operator_plan",
    "compose_bake_passes",
    "configure_scene_for_bake",
    "execute_bake_plan",
    "export_a1_mixed_object",
    "export_a1_multi_object",
    "export_a1_single_object",
    "prepare_a1_mixed_object",
    "prepare_a1_multi_object",
    "prepare_a1_object",
    "preserve_bake_scene_state",
    "read_bake_image_pixels",
    "read_evaluated_mesh_snapshot",
    "read_source_mesh_snapshot",
    "stage_bake_plan_outputs",
    "temporary_bake_materials",
    "temporary_mesh_object",
    "unwrap_snapshot_uv",
    "validate_runtime_scene_context",
    "write_bake_image_pixels",
]
