"""Read and prepare source geometry for one A1 object export."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Mapping, Tuple

from ..application import (
    A1GeometryPreparationResult,
    A1GeometryPreparationSettings,
    A1ResolvedOutputPaths,
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    A1SourceGeometryMode,
    A1SourceUvBoundaryMode,
    A1ZGroupAssignmentPlan,
    ExportIssue,
    build_a1_z_group_assignment,
    prepare_a1_geometry_regions,
    project_a1_prepared_geometry_camera,
    resolve_a1_names,
    resolve_a1_output_paths,
)
from ..domain.baking import A1TextureExportMode
from ..domain.camera_projection import A1CameraProjectionKind
from ..domain.geometry import (
    A1ProjectedSnapshotDepthRange,
    LineageSeverity,
    MeshSnapshot,
    MeshWorldTransformResult,
    calculate_a1_projected_snapshot_depth_range,
    normalize_mesh_snapshot_world_transform,
    project_a1_mesh_snapshot_axis,
    project_a1_mesh_snapshot_camera,
)
from ..domain.geometry.shared_pivot import rebase_a1_projected_snapshot_origin
from ..domain.projection import (
    A1ProjectedPoint,
    A1ProjectionDirection,
    resolve_a1_axis_projection_basis,
)
from ..domain.spine import calculate_uniform_scale
from .a1_preparation_contracts import (
    A1ObjectPreparationError,
    StatisticsValue,
    freeze_statistics,
    warning_issue,
)
from .active_camera_projection import resolve_a1_active_camera_projection_frame
from .evaluated_mesh_reader import read_evaluated_mesh_snapshot
from .mesh_reader import _matrix_tuple, read_source_mesh_snapshot
from .render_engine_contract import (
    RenderEngineContract,
    render_engine_contract_from_execution,
)
from .scene_context_contract import (
    BlenderSceneContextError,
    require_depsgraph_scene_consistency,
)
from .source_uv_integrity import (
    SourceUvIntegrityReport,
    resolve_readable_source_uv_layer_names,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A1SourceGeometryPreparationResult:
    """Validated source metadata and geometry products used by later stages.

    ``source_snapshot`` owns projected export geometry. ``material_bake_snapshot`` owns
    the unprojected Blender-local geometry used to evaluate source materials. Keeping
    those snapshots separate prevents projection direction from changing normals,
    positions, Fresnel, Generated coordinates, or scene-light response during Normal /
    UV Segments baking.
    """

    source_object: Any
    object_id: str
    prefix: str
    settings: A1SingleObjectExportSettings
    output_paths: A1ResolvedOutputPaths
    renderer: RenderEngineContract
    source_snapshot: MeshSnapshot
    z_groups: A1ZGroupAssignmentPlan
    geometry: A1GeometryPreparationResult
    warnings: Tuple[ExportIssue, ...]
    statistics: Mapping[str, StatisticsValue]
    # Appended to preserve positional compatibility with existing test doubles.
    camera_projection_kind: A1CameraProjectionKind | None = None
    # Appended so existing positional construction remains compatible.
    material_bake_snapshot: MeshSnapshot | None = None

    def __post_init__(self) -> None:
        if self.source_object is None:
            raise ValueError("source_object cannot be None")
        for field_name in ("object_id", "prefix"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        expected = (
            ("settings", A1SingleObjectExportSettings),
            ("output_paths", A1ResolvedOutputPaths),
            ("renderer", RenderEngineContract),
            ("source_snapshot", MeshSnapshot),
            ("z_groups", A1ZGroupAssignmentPlan),
            ("geometry", A1GeometryPreparationResult),
        )
        for field_name, expected_type in expected:
            if not isinstance(getattr(self, field_name), expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__}")
        if self.source_snapshot.source_object_id != self.object_id:
            raise ValueError("source_snapshot.source_object_id must match object_id")

        material_bake_snapshot = self.material_bake_snapshot
        if material_bake_snapshot is None:
            # Compatibility fallback for isolated test doubles. Production preparation
            # always supplies the unprojected source snapshot explicitly.
            material_bake_snapshot = self.source_snapshot
            object.__setattr__(
                self,
                "material_bake_snapshot",
                material_bake_snapshot,
            )
        if not isinstance(material_bake_snapshot, MeshSnapshot):
            raise TypeError("material_bake_snapshot must be MeshSnapshot or None")
        if material_bake_snapshot.source_object_id != self.object_id:
            raise ValueError(
                "material_bake_snapshot.source_object_id must match object_id"
            )

        if not isinstance(self.warnings, tuple) or not all(
            isinstance(issue, ExportIssue) for issue in self.warnings
        ):
            raise TypeError("warnings must be a tuple of ExportIssue values")
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")

        active_camera = (
            self.settings.projection_direction
            is A1ProjectionDirection.ACTIVE_CAMERA
        )
        if active_camera:
            if not isinstance(
                self.camera_projection_kind,
                A1CameraProjectionKind,
            ):
                raise TypeError(
                    "Active Camera source preparation requires "
                    "camera_projection_kind"
                )
        elif self.camera_projection_kind is not None:
            raise ValueError(
                "camera_projection_kind is valid only for Active Camera preparation"
            )


@dataclass(frozen=True, slots=True)
class _ResolvedSourceRequest:
    """Validated request metadata and optional evaluated Blender owners."""

    object_id: str
    prefix: str
    output_paths: A1ResolvedOutputPaths
    renderer: RenderEngineContract
    geometry_settings: A1GeometryPreparationSettings
    scene: Any | None
    depsgraph: Any | None
    statistics: Mapping[str, StatisticsValue]


@dataclass(frozen=True, slots=True)
class _NormalizedSourceGeometry:
    """World-normalized source plus diagnostics required by later stages."""

    snapshot: MeshSnapshot
    world_transform: MeshWorldTransformResult
    resolved_uv_boundary_layer: str | None
    warnings: Tuple[ExportIssue, ...]


@dataclass(frozen=True, slots=True)
class _ProjectionPreparation:
    """Projected source, optional prebuilt regions and projection diagnostics."""

    snapshot: MeshSnapshot
    geometry: A1GeometryPreparationResult | None
    projected_origin: A1ProjectedPoint
    depth_range: A1ProjectedSnapshotDepthRange
    camera_projection_kind: A1CameraProjectionKind | None
    statistics: Mapping[str, StatisticsValue]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MeshSnapshot):
            raise TypeError("snapshot must be MeshSnapshot")
        if self.geometry is not None and not isinstance(
            self.geometry,
            A1GeometryPreparationResult,
        ):
            raise TypeError(
                "geometry must be A1GeometryPreparationResult or None"
            )
        if not isinstance(self.projected_origin, A1ProjectedPoint):
            raise TypeError("projected_origin must be A1ProjectedPoint")
        if not isinstance(self.depth_range, A1ProjectedSnapshotDepthRange):
            raise TypeError("depth_range must be A1ProjectedSnapshotDepthRange")
        if self.camera_projection_kind is not None and not isinstance(
            self.camera_projection_kind,
            A1CameraProjectionKind,
        ):
            raise TypeError(
                "camera_projection_kind must be A1CameraProjectionKind or None"
            )
        if not isinstance(self.statistics, Mapping):
            raise TypeError("statistics must be a mapping")


def object_name(source_obj: Any) -> str:
    """Resolve and validate the stable Blender object name used by A1 contracts."""

    if source_obj is None or getattr(source_obj, "type", None) != "MESH":
        raise ValueError("source_obj must be a Blender MESH object")
    value = str(
        getattr(source_obj, "name_full", None)
        or getattr(source_obj, "name", None)
        or ""
    ).strip()
    if not value:
        raise ValueError("source_obj name is empty")
    if getattr(source_obj, "data", None) is None:
        raise ValueError("source_obj.data is missing")
    return value


def _resolved_evaluation_owners(
    scene: Any | None,
) -> tuple[Any, Any]:
    """Return one Blender Scene and its current evaluated dependency graph."""

    try:
        import bpy
    except Exception as exc:
        raise ValueError(
            "Blender bpy module is unavailable for evaluated geometry"
        ) from exc

    resolved_scene = scene or getattr(bpy.context, "scene", None)
    if resolved_scene is None:
        raise ValueError("A Blender Scene is required for evaluated geometry")
    try:
        resolved_depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception as exc:
        raise ValueError(
            "Unable to acquire Blender evaluated dependency graph"
        ) from exc
    if resolved_depsgraph is None:
        raise ValueError("Blender returned no evaluated dependency graph")
    try:
        require_depsgraph_scene_consistency(
            resolved_depsgraph,
            resolved_scene,
        )
    except BlenderSceneContextError as exc:
        raise ValueError(
            f"Evaluated geometry scene and dependency graph disagree: {exc}"
        ) from exc
    return resolved_scene, resolved_depsgraph


def _evaluated_source_world_matrix(
    source_obj: Any,
    depsgraph: Any,
) -> tuple[float, ...]:
    """Read matrix_world from the same depsgraph evaluation as the Mesh snapshot."""

    evaluated_get = getattr(source_obj, "evaluated_get", None)
    if not callable(evaluated_get):
        raise ValueError("source_obj.evaluated_get() is unavailable")
    try:
        evaluated_source = evaluated_get(depsgraph)
    except Exception as exc:
        raise ValueError("Unable to evaluate source object transform") from exc
    if evaluated_source is None:
        raise ValueError("source_obj.evaluated_get() returned None")
    matrix_world = getattr(evaluated_source, "matrix_world", None)
    if matrix_world is None:
        raise ValueError("Evaluated source object has no matrix_world")
    return _matrix_tuple(matrix_world)


def _ignored_uv_warnings(
    report: SourceUvIntegrityReport,
    *,
    object_id: str,
) -> Tuple[ExportIssue, ...]:
    return tuple(
        warning_issue(
            stage=A1SingleObjectStage.READ_GEOMETRY,
            code="IGNORED_MALFORMED_SOURCE_UV",
            message=(
                f"Ignoring malformed unused source UV layer '{layer.name}': "
                f"{layer.value_count} values for {layer.loop_count} mesh loops. "
                "Rewrite generated SpineBakeUV remains authoritative."
            ),
            object_id=object_id,
            context={
                "layer_name": layer.name,
                "value_count": layer.value_count,
                "loop_count": layer.loop_count,
            },
        )
        for layer in report.layers
        if layer.name in report.ignored_malformed_layer_names
    )


def _read_source_snapshot(
    source_obj: Any,
    object_id: str,
    settings: A1SingleObjectExportSettings,
    *,
    scene: Any | None,
    depsgraph: Any | None = None,
) -> tuple[
    MeshSnapshot,
    int,
    Tuple[ExportIssue, ...],
    SourceUvIntegrityReport,
]:
    stage = A1SingleObjectStage.READ_GEOMETRY
    uv_report = resolve_readable_source_uv_layer_names(source_obj, settings)
    readable_uv_layers = uv_report.readable_layer_names
    uv_warnings = _ignored_uv_warnings(uv_report, object_id=object_id)

    if settings.source_geometry_mode is A1SourceGeometryMode.EVALUATED:
        if depsgraph is None:
            resolved_scene, resolved_depsgraph = _resolved_evaluation_owners(scene)
        else:
            if scene is None:
                raise ValueError(
                    "Evaluated source geometry requires the Scene owning depsgraph"
                )
            resolved_scene = scene
            resolved_depsgraph = depsgraph
            try:
                require_depsgraph_scene_consistency(
                    resolved_depsgraph,
                    resolved_scene,
                )
            except BlenderSceneContextError as exc:
                raise ValueError(
                    f"Evaluated geometry scene and dependency graph disagree: {exc}"
                ) from exc

        evaluated = read_evaluated_mesh_snapshot(
            source_obj,
            scene=resolved_scene,
            depsgraph=resolved_depsgraph,
            source_object_id=object_id,
            snapshot_id=f"{object_id}:a1-evaluated",
            uv_layer_names=readable_uv_layers,
            lineage_policy=settings.modifier_lineage_policy,
        )
        evaluated_snapshot = replace(
            evaluated.snapshot,
            world_matrix=_evaluated_source_world_matrix(
                source_obj,
                resolved_depsgraph,
            ),
        )
        modifier_warnings = tuple(
            warning_issue(
                stage=stage,
                code=f"MODIFIER_{issue.code}",
                message=issue.message,
                object_id=object_id,
                context={"channel": issue.channel},
            )
            for issue in evaluated.lineage_report.issues
            if issue.severity is LineageSeverity.WARNING
        )
        return (
            evaluated_snapshot,
            len(evaluated.modifier_stack),
            uv_warnings + modifier_warnings,
            uv_report,
        )

    snapshot = read_source_mesh_snapshot(
        source_obj,
        source_object_id=object_id,
        snapshot_id=f"{object_id}:a1-source",
        uv_layer_names=readable_uv_layers,
    )
    return snapshot, 0, uv_warnings, uv_report


def _resolve_source_uv_boundary_layer(
    snapshot: MeshSnapshot,
    settings: A1SingleObjectExportSettings,
) -> str | None:
    """Resolve the exact pre-unwrap UV layer allowed to affect segmentation."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")

    mode = settings.source_uv_boundary_mode
    if mode is A1SourceUvBoundaryMode.DISABLED:
        return None
    if mode is A1SourceUvBoundaryMode.EXPLICIT_LAYER:
        layer_name = settings.source_uv_boundary_layer_name
        if layer_name is None:
            raise ValueError(
                "EXPLICIT_LAYER requires source_uv_boundary_layer_name"
            )
        if layer_name not in snapshot.uv_layer_names:
            raise ValueError(
                f"Source UV boundary layer '{layer_name}' is absent from "
                f"snapshot '{snapshot.snapshot_id}'"
            )
        return layer_name
    if mode is A1SourceUvBoundaryMode.ACTIVE_LAYER_LEGACY:
        layer_name = snapshot.active_uv_layer
        if layer_name is None:
            raise ValueError(
                "ACTIVE_LAYER_LEGACY requires the source mesh to have an active UV layer"
            )
        return layer_name
    raise TypeError(f"Unsupported source UV boundary mode: {mode!r}")


def _validate_projection_route(settings: A1SingleObjectExportSettings) -> None:
    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    if (
        settings.projection_direction is A1ProjectionDirection.ACTIVE_CAMERA
        and settings.bake_execution.texture_export_mode
        is not A1TextureExportMode.NORMAL_UV_SEGMENTS
    ):
        raise ValueError(
            "ACTIVE_CAMERA projection direction is available only for "
            "Normal / UV Segments. Existing Camera Projection mode owns its own "
            "rendered flattening pipeline."
        )


def _resolve_source_request(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    scene: Any | None,
) -> _ResolvedSourceRequest:
    """Validate request metadata and resolve Blender evaluation owners once."""

    if not isinstance(settings, A1SingleObjectExportSettings):
        raise TypeError("settings must be A1SingleObjectExportSettings")
    _validate_projection_route(settings)
    object_id = object_name(source_obj)
    prefix, _ = resolve_a1_names(object_id, settings)
    output_paths = resolve_a1_output_paths(object_id, settings)
    renderer = render_engine_contract_from_execution(settings.bake_execution)
    geometry_settings = settings.resolved_geometry_settings()

    resolved_scene = scene
    resolved_depsgraph = None
    if (
        settings.source_geometry_mode is A1SourceGeometryMode.EVALUATED
        or settings.projection_direction is A1ProjectionDirection.ACTIVE_CAMERA
    ):
        resolved_scene, resolved_depsgraph = _resolved_evaluation_owners(scene)

    statistics = freeze_statistics(
        {
            "source_object": object_id,
            "rig_prefix": prefix,
            "source_geometry_mode": settings.source_geometry_mode.value,
            "source_uv_boundary_mode": settings.source_uv_boundary_mode.value,
            "source_uv_boundary_configured_layer": (
                settings.source_uv_boundary_layer_name or ""
            ),
            "projection_direction": settings.projection_direction.value,
            "include_control_icons": int(settings.include_control_icons),
            "include_preview_animation": int(settings.include_preview_animation),
            "render_engine": renderer.blender_engine,
            "shader_render_target": renderer.shader_target,
        }
    )
    return _ResolvedSourceRequest(
        object_id=object_id,
        prefix=prefix,
        output_paths=output_paths,
        renderer=renderer,
        geometry_settings=geometry_settings,
        scene=resolved_scene,
        depsgraph=resolved_depsgraph,
        statistics=statistics,
    )


def _normalize_source_geometry(
    source_snapshot: MeshSnapshot,
    settings: A1SingleObjectExportSettings,
    warnings: Tuple[ExportIssue, ...],
    *,
    object_id: str,
) -> _NormalizedSourceGeometry:
    """Bake linear world transform and validate source-UV segmentation ownership."""

    world_transform = normalize_mesh_snapshot_world_transform(source_snapshot)
    normalized_snapshot = world_transform.snapshot
    resolved_uv_boundary_layer = _resolve_source_uv_boundary_layer(
        normalized_snapshot,
        settings,
    )
    resolved_warnings = warnings
    if world_transform.mirrored:
        resolved_warnings = warnings + (
            warning_issue(
                stage=A1SingleObjectStage.PREPARE_GEOMETRY,
                code="MIRRORED_OBJECT_TRANSFORM",
                message=(
                    "Object matrix_world has a negative determinant. Rewrite "
                    "preserved the mirrored geometry and oriented normals while "
                    "normalizing rotation/scale into the mesh snapshot."
                ),
                object_id=object_id,
                context={"determinant": world_transform.determinant},
            ),
        )
    return _NormalizedSourceGeometry(
        snapshot=normalized_snapshot,
        world_transform=world_transform,
        resolved_uv_boundary_layer=resolved_uv_boundary_layer,
        warnings=resolved_warnings,
    )


def _prepare_projection_route(
    source_snapshot: MeshSnapshot,
    settings: A1SingleObjectExportSettings,
    geometry_settings: A1GeometryPreparationSettings,
    *,
    scene: Any | None,
    depsgraph: Any | None,
) -> _ProjectionPreparation:
    """Execute the signed-axis or nonlinear Active Camera geometry route."""

    geometry: A1GeometryPreparationResult | None = None
    camera_projection_kind: A1CameraProjectionKind | None = None
    if settings.projection_direction.axis_aligned:
        axis_projection = project_a1_mesh_snapshot_axis(
            source_snapshot,
            settings.projection_direction,
        )
        projected_snapshot = axis_projection.snapshot
        projected_origin = axis_projection.projected_origin
        shared_pivot_applied = 0
        if settings.shared_pivot_world is not None:
            basis = resolve_a1_axis_projection_basis(settings.projection_direction)
            shared_projected_origin = basis.project_point(settings.shared_pivot_world)
            projected_snapshot = rebase_a1_projected_snapshot_origin(
                projected_snapshot,
                projected_origin,
                shared_projected_origin,
            )
            projected_origin = shared_projected_origin
            shared_pivot_applied = 1
        projection_statistics = {
            "projection_kind": "SIGNED_AXIS",
            "axis_projection_applied": int(axis_projection.changed),
            "active_camera_projection_applied": 0,
            "shared_pivot_applied": shared_pivot_applied,
            "active_camera_name": "",
            "active_camera_type": "",
            "active_camera_clip_start": 0.0,
            "active_camera_clip_end": 0.0,
            "active_camera_preprojection_triangulation": 0,
            "projection_canvas_width": settings.export.texture_width,
            "projection_canvas_height": settings.export.texture_height,
            "attachment_invert_y": 1,
        }
    else:
        if settings.projection_direction is not A1ProjectionDirection.ACTIVE_CAMERA:
            raise ValueError(
                f"Unsupported projection direction: {settings.projection_direction!r}"
            )
        if scene is None or depsgraph is None:
            raise ValueError(
                "ACTIVE_CAMERA projection lost its evaluated Scene context"
            )

        geometry = prepare_a1_geometry_regions(
            source_snapshot,
            geometry_settings,
        )
        frame = resolve_a1_active_camera_projection_frame(
            scene,
            texture_width=settings.export.texture_width,
            texture_height=settings.export.texture_height,
            depsgraph=depsgraph,
        )
        uniform_scale = calculate_uniform_scale(
            settings.export.texture_width,
            settings.export.texture_height,
            settings.rig_scale_mode,
        )
        camera_projection = project_a1_mesh_snapshot_camera(
            source_snapshot,
            frame,
            uniform_scale=uniform_scale,
        )
        projected_snapshot = camera_projection.snapshot
        geometry = project_a1_prepared_geometry_camera(
            geometry,
            frame,
            uniform_scale=uniform_scale,
        )
        projected_origin = camera_projection.projected_origin
        camera_projection_kind = frame.kind
        projection_statistics = {
            "projection_kind": "ACTIVE_CAMERA",
            "axis_projection_applied": 0,
            "active_camera_projection_applied": 1,
            "shared_pivot_applied": 0,
            "active_camera_name": frame.camera_id,
            "active_camera_type": frame.kind.value,
            "active_camera_clip_start": frame.clip_start,
            "active_camera_clip_end": frame.clip_end,
            "active_camera_preprojection_triangulation": 1,
            "projection_canvas_width": frame.texture_width,
            "projection_canvas_height": frame.texture_height,
            "attachment_invert_y": 0,
        }

    return _ProjectionPreparation(
        snapshot=projected_snapshot,
        geometry=geometry,
        projected_origin=projected_origin,
        depth_range=calculate_a1_projected_snapshot_depth_range(projected_snapshot),
        camera_projection_kind=camera_projection_kind,
        statistics=freeze_statistics(projection_statistics),
    )


def _complete_projected_geometry(
    projection: _ProjectionPreparation,
    geometry_settings: A1GeometryPreparationSettings,
) -> A1GeometryPreparationResult:
    """Build signed-axis regions after Z assignment or return camera regions."""

    if projection.geometry is not None:
        return projection.geometry
    return prepare_a1_geometry_regions(
        projection.snapshot,
        geometry_settings,
    )


def _build_prepared_statistics(
    base_statistics: Mapping[str, StatisticsValue],
    *,
    modifier_count: int,
    uv_report: SourceUvIntegrityReport,
    normalized: _NormalizedSourceGeometry,
    projection: _ProjectionPreparation,
    z_groups: A1ZGroupAssignmentPlan,
    geometry: A1GeometryPreparationResult,
) -> Mapping[str, StatisticsValue]:
    """Freeze complete source, projection, Z-group and region diagnostics."""

    snapshot = projection.snapshot
    world_transform = normalized.world_transform
    origin = projection.projected_origin
    depth_range = projection.depth_range
    return freeze_statistics(
        base_statistics,
        {
            "modifier_count": modifier_count,
            "source_vertices": len(snapshot.vertices),
            "source_edges": len(snapshot.edges),
            "source_faces": len(snapshot.faces),
            "source_uv_layer_count": len(uv_report.layers),
            "source_uv_readable_layer_count": len(uv_report.readable_layer_names),
            "source_uv_ignored_malformed_count": len(
                uv_report.ignored_malformed_layer_names
            ),
            "source_uv_required_layers": ",".join(uv_report.required_layer_names),
            "object_linear_transform_baked": int(world_transform.changed),
            "object_world_determinant": world_transform.determinant,
            "object_world_mirrored": int(world_transform.mirrored),
            "object_world_translation_x": world_transform.translation[0],
            "object_world_translation_y": world_transform.translation[1],
            "object_world_translation_z": world_transform.translation[2],
            "projected_origin_u": origin.u,
            "projected_origin_v": origin.v,
            "projected_origin_depth": origin.depth,
            "projected_nearest_vertex_index": depth_range.nearest_vertex_id.index,
            "projected_nearest_vertex_depth": depth_range.nearest_vertex_depth,
            "projected_farthest_vertex_index": depth_range.farthest_vertex_id.index,
            "projected_farthest_vertex_depth": depth_range.farthest_vertex_depth,
            "source_uv_boundary_resolved_layer": (
                normalized.resolved_uv_boundary_layer or ""
            ),
            "z_group_count": len(z_groups.groups),
            "segment_count": len(geometry.segmentation.segments),
            "region_count": len(geometry.regions),
            "decomposition_cut_count": len(geometry.decomposition.cuts),
        },
        projection.statistics,
    )


def _log_prepared_source(
    request: _ResolvedSourceRequest,
    settings: A1SingleObjectExportSettings,
    normalized: _NormalizedSourceGeometry,
    projection: _ProjectionPreparation,
    geometry: A1GeometryPreparationResult,
    uv_report: SourceUvIntegrityReport,
) -> None:
    """Emit one structured source-geometry diagnostic record."""

    logger.debug(
        "Prepared source geometry for %s: vertices=%d faces=%d regions=%d "
        "world_transform_baked=%s determinant=%s mirrored=%s "
        "projection_direction=%s projection_kind=%s shared_pivot=%s "
        "preprojection_triangulation=%s "
        "projected_origin=(%s, %s, %s) nearest=(%s, %s) "
        "source_uv_boundary_mode=%s source_uv_boundary_layer=%s "
        "ignored_malformed_uv=%s",
        request.object_id,
        len(projection.snapshot.vertices),
        len(projection.snapshot.faces),
        len(geometry.regions),
        normalized.world_transform.changed,
        normalized.world_transform.determinant,
        normalized.world_transform.mirrored,
        settings.projection_direction.value,
        projection.statistics["projection_kind"],
        bool(projection.statistics["shared_pivot_applied"]),
        projection.statistics["active_camera_preprojection_triangulation"],
        projection.projected_origin.u,
        projection.projected_origin.v,
        projection.projected_origin.depth,
        projection.depth_range.nearest_vertex_id.index,
        projection.depth_range.nearest_vertex_depth,
        settings.source_uv_boundary_mode.value,
        normalized.resolved_uv_boundary_layer,
        uv_report.ignored_malformed_layer_names,
    )


def prepare_a1_source_geometry(
    source_obj: Any,
    settings: A1SingleObjectExportSettings,
    *,
    scene: Any | None = None,
) -> A1SourceGeometryPreparationResult:
    """Validate one request and prepare immutable source geometry state."""

    stage = A1SingleObjectStage.VALIDATE_REQUEST
    object_id: str | None = None
    warnings: Tuple[ExportIssue, ...] = ()
    statistics: Mapping[str, StatisticsValue] = {}
    try:
        request = _resolve_source_request(source_obj, settings, scene)
        object_id = request.object_id
        statistics = request.statistics

        stage = A1SingleObjectStage.READ_GEOMETRY
        material_bake_snapshot, modifier_count, warnings, uv_report = (
            _read_source_snapshot(
                source_obj,
                request.object_id,
                settings,
                scene=request.scene,
                depsgraph=request.depsgraph,
            )
        )

        stage = A1SingleObjectStage.PREPARE_GEOMETRY
        normalized = _normalize_source_geometry(
            material_bake_snapshot,
            settings,
            warnings,
            object_id=request.object_id,
        )
        warnings = normalized.warnings
        projection = _prepare_projection_route(
            normalized.snapshot,
            settings,
            request.geometry_settings,
            scene=request.scene,
            depsgraph=request.depsgraph,
        )

        stage = A1SingleObjectStage.ASSIGN_Z_GROUPS
        z_groups = build_a1_z_group_assignment(projection.snapshot)

        stage = A1SingleObjectStage.PREPARE_GEOMETRY
        geometry = _complete_projected_geometry(
            projection,
            request.geometry_settings,
        )
        statistics = _build_prepared_statistics(
            statistics,
            modifier_count=modifier_count,
            uv_report=uv_report,
            normalized=normalized,
            projection=projection,
            z_groups=z_groups,
            geometry=geometry,
        )
        statistics = freeze_statistics(
            statistics,
            {
                "material_bake_vertices": len(material_bake_snapshot.vertices),
                "material_bake_edges": len(material_bake_snapshot.edges),
                "material_bake_faces": len(material_bake_snapshot.faces),
                "material_bake_projection_independent": 1,
            },
        )
        _log_prepared_source(
            request,
            settings,
            normalized,
            projection,
            geometry,
            uv_report,
        )
        return A1SourceGeometryPreparationResult(
            source_object=source_obj,
            object_id=request.object_id,
            prefix=request.prefix,
            settings=settings,
            output_paths=request.output_paths,
            renderer=request.renderer,
            source_snapshot=projection.snapshot,
            z_groups=z_groups,
            geometry=geometry,
            warnings=warnings,
            statistics=statistics,
            camera_projection_kind=projection.camera_projection_kind,
            material_bake_snapshot=material_bake_snapshot,
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
    "A1SourceGeometryPreparationResult",
    "object_name",
    "prepare_a1_source_geometry",
]
