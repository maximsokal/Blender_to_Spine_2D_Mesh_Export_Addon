"""Orchestrate the staged preparation of one Blender mesh for A1 export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from ..application import (
    A1DocumentAssemblyResult,
    A1GeometryPreparationResult,
    A1ResolvedOutputPaths,
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    A1TexturingTopology,
    A1UvPropagationResult,
    A1ZGroupAssignmentPlan,
    ExportIssue,
)
from ..domain.baking import BakePlan, ObjectMaterialAnalysis
from ..domain.geometry import MeshSnapshot
from ..domain.spine import LegacyRigBuildResult, SpineDocument
from ..domain.uv import UvUnwrapResult
from .a1_document_preparation import prepare_a1_document
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    StatisticsValue,
)
from .a1_source_geometry_preparation import prepare_a1_source_geometry
from .a1_texture_planning import prepare_a1_texture_plan
from .a1_uv_preparation import prepare_a1_uv


@dataclass(frozen=True, slots=True)
class PreparedA1Object:
    """Complete in-memory product of one A1 object preparation pipeline."""

    source_object: Any
    object_id: str
    prefix: str
    settings: A1SingleObjectExportSettings
    output_paths: A1ResolvedOutputPaths
    source_snapshot: MeshSnapshot
    z_groups: A1ZGroupAssignmentPlan
    geometry: A1GeometryPreparationResult
    texturing_topology: A1TexturingTopology
    unwrap_result: UvUnwrapResult
    uv_regions: A1UvPropagationResult
    material_analysis: ObjectMaterialAnalysis
    bake_plan: BakePlan
    rig: LegacyRigBuildResult
    document_assembly: A1DocumentAssemblyResult
    warnings: Tuple[ExportIssue, ...]
    statistics: Mapping[str, StatisticsValue]

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        for field_name in ("object_id", "prefix"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        expected_types = (
            ("settings", A1SingleObjectExportSettings),
            ("output_paths", A1ResolvedOutputPaths),
            ("source_snapshot", MeshSnapshot),
            ("z_groups", A1ZGroupAssignmentPlan),
            ("geometry", A1GeometryPreparationResult),
            ("texturing_topology", A1TexturingTopology),
            ("unwrap_result", UvUnwrapResult),
            ("uv_regions", A1UvPropagationResult),
            ("material_analysis", ObjectMaterialAnalysis),
            ("bake_plan", BakePlan),
            ("rig", LegacyRigBuildResult),
            ("document_assembly", A1DocumentAssemblyResult),
        )
        for field_name, expected_type in expected_types:
            if not isinstance(getattr(self, field_name), expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__}")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")
        if self.source_snapshot.source_object_id != self.object_id:
            raise ValueError("source_snapshot.source_object_id must match object_id")
        if self.bake_plan.source_object_id != self.object_id:
            raise ValueError("bake_plan.source_object_id must match object_id")
        if self.rig.request.prefix != self.prefix:
            raise ValueError("rig prefix must match prepared prefix")

    @property
    def document(self) -> SpineDocument:
        return self.document_assembly.document

    @property
    def bake_target_snapshot(self) -> MeshSnapshot:
        return self.unwrap_result.snapshot

    @property
    def world_position(self) -> Tuple[float, float, float]:
        matrix = self.source_snapshot.world_matrix
        if len(matrix) != 16:
            raise ValueError("source_snapshot.world_matrix must contain 16 values")
        return float(matrix[3]), float(matrix[7]), float(matrix[11])


def prepare_a1_object(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> PreparedA1Object:
    """Run the four typed A1 preparation stages without writing output files."""

    stage = A1SingleObjectStage.VALIDATE_REQUEST
    object_id: str | None = None
    statistics: Mapping[str, StatisticsValue] = {}
    warnings: Tuple[ExportIssue, ...] = ()
    try:
        source = prepare_a1_source_geometry(source_obj, settings, scene=scene)
        object_id = source.object_id
        statistics = source.statistics
        warnings = source.warnings

        stage = A1SingleObjectStage.BUILD_TEXTURING_TOPOLOGY
        uv = prepare_a1_uv(source, context=context, scene=scene)
        statistics = uv.statistics
        warnings = uv.warnings

        stage = A1SingleObjectStage.ANALYZE_MATERIALS
        texture = prepare_a1_texture_plan(uv, context=context, scene=scene)
        statistics = texture.statistics
        warnings = texture.warnings

        stage = A1SingleObjectStage.BUILD_RIG
        document = prepare_a1_document(texture)
        statistics = document.statistics
        warnings = document.warnings

        stage = A1SingleObjectStage.ASSEMBLE_DOCUMENT
        return PreparedA1Object(
            source_object=source.source_object,
            object_id=source.object_id,
            prefix=source.prefix,
            settings=source.settings,
            output_paths=source.output_paths,
            source_snapshot=source.source_snapshot,
            z_groups=source.z_groups,
            geometry=source.geometry,
            texturing_topology=uv.texturing_topology,
            unwrap_result=uv.unwrap_result,
            uv_regions=uv.uv_regions,
            material_analysis=texture.material_analysis,
            bake_plan=texture.bake_plan,
            rig=document.rig,
            document_assembly=document.document_assembly,
            warnings=warnings,
            statistics=statistics,
        )
    except A1ObjectPreparationError:
        raise
    except Exception as exc:
        raise A1ObjectPreparationError(
            stage=stage,
            object_id=object_id,
            cause=exc,
            statistics=statistics,
            warnings=warnings,
        ) from exc


__all__ = [
    "A1ObjectPreparationError",
    "PreparedA1Object",
    "StatisticsValue",
    "prepare_a1_object",
]
