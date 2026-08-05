"""Orchestrate the staged preparation of one Blender mesh for A1 export."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Any, Iterator, Mapping, Tuple

from ..application import (
    A1DocumentAssemblyResult,
    A1ExportProgressCallback,
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    ExportIssue,
    emit_a1_export_progress,
)
from ..domain.baking import (
    A1TextureExportMode,
    CameraProjectionPlan,
)
from ..domain.baking.camera_projection import (
    build_camera_projection_view_plan,
)
from ..domain.geometry import DepthParallaxGeometryPackage
from ..domain.spine.export_capabilities import (
    SpineJsonExportScope,
    require_spine_json_export_capability,
)
from .a1_depth_document_preparation import prepare_a1_depth_document
from .a1_depth_source_geometry_preparation import (
    prepare_a1_depth_source_geometry,
)
from .a1_document_preparation import prepare_a1_document
from .a1_preparation_contracts import (
    A1BlenderFinalizationContext,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedDepthA1Object(PreparedA1Object):
    """Prepared object with auxiliary parallax texture plans and topology."""

    depth_parallax_package: DepthParallaxGeometryPackage
    reserve_bake_plans: Tuple[CameraProjectionPlan, ...] = ()

    def __post_init__(self) -> None:
        PreparedA1Object.__post_init__(self)
        if not isinstance(
            self.depth_parallax_package,
            DepthParallaxGeometryPackage,
        ):
            raise TypeError(
                "depth_parallax_package must be DepthParallaxGeometryPackage"
            )
        if not isinstance(self.reserve_bake_plans, tuple) or not all(
            isinstance(plan, CameraProjectionPlan)
            for plan in self.reserve_bake_plans
        ):
            raise TypeError(
                "reserve_bake_plans must contain CameraProjectionPlan values"
            )
        expected = len(self.depth_parallax_package.reserve_surfaces)
        if len(self.reserve_bake_plans) != expected:
            raise ValueError(
                "reserve_bake_plans must match reserve surface count; "
                f"plans={len(self.reserve_bake_plans)}, surfaces={expected}"
            )
        if self.source_snapshot != self.depth_parallax_package.union_snapshot:
            raise ValueError(
                "Prepared Depth source_snapshot must equal parallax union_snapshot"
            )


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


def _build_prepared_object(
    source: Any,
    uv: Any,
    texture: Any,
    document: Any,
    reserve_bake_plans: Tuple[CameraProjectionPlan, ...],
    *,
    context: Any | None,
    scene: Any | None,
    warnings: Tuple[ExportIssue, ...],
    statistics: Mapping[str, StatisticsValue],
) -> PreparedA1Object:
    """Build the immutable stage product while preserving opaque unit-test doubles."""

    assembly = document.document_assembly
    prepared_rig = (
        assembly.rig
        if isinstance(assembly, A1DocumentAssemblyResult)
        else document.rig
    )
    common = dict(
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
        rig=prepared_rig,
        document_assembly=assembly,
        warnings=warnings,
        statistics=statistics,
        finalization_context=A1BlenderFinalizationContext(
            context=context,
            scene=scene,
        ),
    )
    package = getattr(source, "parallax_package", None)
    if isinstance(package, DepthParallaxGeometryPackage):
        return PreparedDepthA1Object(
            **common,
            depth_parallax_package=package,
            reserve_bake_plans=reserve_bake_plans,
        )
    if reserve_bake_plans:
        raise ValueError(
            "Non-Depth prepared object cannot carry reserve_bake_plans"
        )
    return PreparedA1Object(**common)


def _depth_mode(settings: A1SingleObjectExportSettings) -> bool:
    return (
        settings.bake_execution.texture_export_mode
        is A1TextureExportMode.DEPTH_CAMERA_PROJECTION
    )


def _prepare_source_geometry(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    scene: Any | None,
    progress_callback: A1ExportProgressCallback | None,
) -> Any:
    """Select the ordinary or depth-relief geometry source explicitly."""

    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if _depth_mode(settings):
        return prepare_a1_depth_source_geometry(
            source_obj,
            settings,
            scene=scene,
            progress_callback=progress_callback,
        )
    return prepare_a1_source_geometry(source_obj, settings, scene=scene)


def _prepare_texture(
    uv: Any,
    settings: A1SingleObjectExportSettings,
    *,
    context: Any | None,
    scene: Any | None,
) -> Any:
    """Plan texture output and validate the selected immutable export mode."""

    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    result = prepare_a1_texture_plan(uv, context=context, scene=scene)
    if _depth_mode(settings) and not isinstance(
        result.bake_plan,
        CameraProjectionPlan,
    ):
        raise ValueError(
            "Depth Camera Projection requires renderable source materials and a "
            "CameraProjectionPlan; generated-material object bake is not compatible"
        )
    return result


def _build_reserve_bake_plans(
    source: Any,
    texture: Any,
    settings: A1SingleObjectExportSettings,
) -> Tuple[CameraProjectionPlan, ...]:
    """Create one face-isolated camera plan for every retained reserve surface."""

    if not _depth_mode(settings):
        return ()
    package = getattr(source, "parallax_package", None)
    if not isinstance(package, DepthParallaxGeometryPackage):
        raise TypeError(
            "Depth Camera Projection source lost its parallax package"
        )
    if not isinstance(texture.bake_plan, CameraProjectionPlan):
        raise TypeError("Depth front texture plan must be CameraProjectionPlan")
    plans = tuple(
        build_camera_projection_view_plan(
            texture.bake_plan,
            view_id=surface.view.view_id.value,
            camera_world_matrix=surface.view.camera_world_matrix,
            lens_scale=surface.view.lens_scale,
            source_face_indices=surface.source_face_indices,
        )
        for surface in package.reserve_surfaces
    )
    if len(plans) != len(package.reserve_surfaces):
        raise ValueError("Reserve plan count differs from reserve surface count")
    return plans


def _prepare_document(
    texture: Any,
    settings: A1SingleObjectExportSettings,
    reserve_bake_plans: Tuple[CameraProjectionPlan, ...],
) -> Any:
    """Select flat camera or Normal-style depth attachment assembly."""

    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    if _depth_mode(settings):
        return prepare_a1_depth_document(texture, reserve_bake_plans)
    if reserve_bake_plans:
        raise ValueError("Only Depth mode may carry reserve_bake_plans")
    return prepare_a1_document(texture)


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
        if not isinstance(settings, A1SingleObjectExportSettings):
            raise TypeError("settings must be A1SingleObjectExportSettings")
        require_spine_json_export_capability(
            settings.export.spine_target,
            settings.export.rig_profile,
            SpineJsonExportScope.SINGLE_OBJECT,
        )

        with _source_uv_integrity_guard(source_obj, context):
            _progress(progress_callback, 0, stage)
            _progress(progress_callback, 10, A1SingleObjectStage.READ_GEOMETRY)
            source = _prepare_source_geometry(
                source_obj,
                settings,
                scene=scene,
                progress_callback=progress_callback,
            )
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
            texture = _prepare_texture(
                uv,
                settings,
                context=context,
                scene=scene,
            )
            statistics, warnings = texture.statistics, texture.warnings
            reserve_bake_plans = _build_reserve_bake_plans(
                source,
                texture,
                settings,
            )

            stage = A1SingleObjectStage.BUILD_RIG
            _progress(progress_callback, 82, stage, object_id)
            document = _prepare_document(
                texture,
                settings,
                reserve_bake_plans,
            )
            statistics, warnings = document.statistics, document.warnings

            stage = A1SingleObjectStage.ASSEMBLE_DOCUMENT
            prepared = _build_prepared_object(
                source,
                uv,
                texture,
                document,
                reserve_bake_plans,
                context=context,
                scene=scene,
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
    "PreparedDepthA1Object",
    "StatisticsValue",
    "prepare_a1_object",
]
