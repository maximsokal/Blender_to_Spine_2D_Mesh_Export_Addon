"""Orchestrate the staged preparation of one Blender mesh for A1 export."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Iterator, Mapping, Tuple

from ..application import (
    A1ExportProgressCallback,
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    ExportIssue,
    emit_a1_export_progress,
)
from .a1_document_preparation import prepare_a1_document
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    PreparedA1Object,
    StatisticsValue,
)
from .a1_source_geometry_preparation import prepare_a1_source_geometry
from .a1_texture_planning import prepare_a1_texture_plan
from .a1_uv_preparation import prepare_a1_uv
from .source_uv_integrity import (
    capture_source_uv_fingerprint_if_mesh,
    require_object_mode,
    require_source_uv_unchanged_if_captured,
)


logger = logging.getLogger(__name__)


_PROGRESS_MESSAGES = {
    A1SingleObjectStage.VALIDATE_REQUEST: "Validating object export request",
    A1SingleObjectStage.READ_GEOMETRY: "Reading and preparing source geometry",
    A1SingleObjectStage.BUILD_TEXTURING_TOPOLOGY: "Building topology and UV",
    A1SingleObjectStage.ANALYZE_MATERIALS: "Planning materials and textures",
    A1SingleObjectStage.BUILD_RIG: "Building Spine rig and document",
    A1SingleObjectStage.ASSEMBLE_DOCUMENT: "Object preparation complete",
}


def _progress(
    callback: A1ExportProgressCallback | None,
    percent: int,
    stage: A1SingleObjectStage,
    object_id: str | None = None,
) -> None:
    emit_a1_export_progress(
        callback,
        percent=percent,
        stage=stage,
        message=_PROGRESS_MESSAGES[stage],
        object_id=object_id,
    )


def _source_object_name(source_obj: Any) -> str:
    return str(
        getattr(source_obj, "name_full", None)
        or getattr(source_obj, "name", None)
        or "<unknown>"
    )


@contextmanager
def _source_uv_integrity_guard(
    source_obj: Any,
    context: Any | None,
) -> Iterator[None]:
    """Protect real source Mesh UV state without pre-empting typed stage validation."""

    require_object_mode(context)
    before = capture_source_uv_fingerprint_if_mesh(source_obj)
    primary_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            require_source_uv_unchanged_if_captured(before, source_obj)
        except Exception as mutation_error:
            logger.exception("Rewrite source UV immutability contract failed")
            wrapped = A1ObjectPreparationError(
                stage=A1SingleObjectStage.READ_GEOMETRY,
                object_id=_source_object_name(source_obj),
                cause=mutation_error,
                statistics={},
                warnings=(),
            )
            if primary_error is not None:
                raise wrapped from primary_error
            raise wrapped from mutation_error


def prepare_a1_object(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    context: Any | None = None,
    scene: Any | None = None,
    progress_callback: A1ExportProgressCallback | None = None,
) -> PreparedA1Object:
    """Run the typed A1 stages while preserving the source Mesh UV state."""

    stage = A1SingleObjectStage.VALIDATE_REQUEST
    object_id: str | None = None
    statistics: Mapping[str, StatisticsValue] = {}
    warnings: Tuple[ExportIssue, ...] = ()
    try:
        with _source_uv_integrity_guard(source_obj, context):
            _progress(progress_callback, 0, stage)
            _progress(progress_callback, 10, A1SingleObjectStage.READ_GEOMETRY)
            source = prepare_a1_source_geometry(source_obj, settings, scene=scene)
            object_id, statistics, warnings = (
                source.object_id,
                source.statistics,
                source.warnings,
            )

            stage = A1SingleObjectStage.BUILD_TEXTURING_TOPOLOGY
            _progress(progress_callback, 45, stage, object_id)
            uv = prepare_a1_uv(source, context=context, scene=scene)
            statistics, warnings = uv.statistics, uv.warnings

            stage = A1SingleObjectStage.ANALYZE_MATERIALS
            _progress(progress_callback, 65, stage, object_id)
            texture = prepare_a1_texture_plan(uv, context=context, scene=scene)
            statistics, warnings = texture.statistics, texture.warnings

            stage = A1SingleObjectStage.BUILD_RIG
            _progress(progress_callback, 82, stage, object_id)
            document = prepare_a1_document(texture)
            statistics, warnings = document.statistics, document.warnings

            stage = A1SingleObjectStage.ASSEMBLE_DOCUMENT
            prepared = PreparedA1Object(
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
            _progress(progress_callback, 100, stage, object_id)
            return prepared
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
