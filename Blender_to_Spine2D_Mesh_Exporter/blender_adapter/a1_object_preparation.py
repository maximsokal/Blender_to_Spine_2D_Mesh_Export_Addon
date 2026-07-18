"""Orchestrate the staged preparation of one Blender mesh for A1 export."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..application import A1SingleObjectExportSettings, A1SingleObjectStage, ExportIssue
from .a1_document_preparation import prepare_a1_document
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    PreparedA1Object,
    StatisticsValue,
)
from .a1_source_geometry_preparation import prepare_a1_source_geometry
from .a1_texture_planning import prepare_a1_texture_plan
from .a1_uv_preparation import prepare_a1_uv


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
