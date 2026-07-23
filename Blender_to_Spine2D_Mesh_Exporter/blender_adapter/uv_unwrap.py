"""Transactional Blender 5.2 UV unwrap on an isolated temporary mesh.

All context-sensitive UV operators are confined to this module. The source
Object is never activated or modified. Operators are invoked once per stage,
never inside geometry loops, and Edit Mode is always left in ``finally``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import radians
from types import MappingProxyType
from typing import Any, Mapping

from ..domain.geometry import MeshSnapshot
from ..domain.uv import (
    UvLayout,
    UvLoopCoordinate,
    UvUnwrapMethod,
    UvUnwrapResult,
    UvUnwrapSettings,
    apply_uv_layout,
    calculate_uv_statistics,
)
from .context_state import BlenderContextError, activate_object_for_operator
from .mesh_uv_attributes import MeshUvAttributeError, read_uv_coordinate
from .mesh_writer import (
    MeshTopologyCorrespondence,
    MeshWriteError,
    build_mesh_topology_correspondence,
    temporary_mesh_object,
)


logger = logging.getLogger(__name__)


class UvUnwrapError(RuntimeError):
    """Raised when Blender cannot produce a complete validated UV layout."""


@dataclass(frozen=True, slots=True)
class UvOperatorPlan:
    unwrap_operator: str
    unwrap_arguments: Mapping[str, object]
    pack_arguments: Mapping[str, object] | None

    def __post_init__(self) -> None:
        if self.unwrap_operator not in {"smart_project", "unwrap"}:
            raise ValueError("unwrap_operator must be 'smart_project' or 'unwrap'")
        if not isinstance(self.unwrap_arguments, Mapping):
            raise TypeError("unwrap_arguments must be a mapping")
        if self.pack_arguments is not None and not isinstance(
            self.pack_arguments,
            Mapping,
        ):
            raise TypeError("pack_arguments must be a mapping or None")


def build_uv_operator_plan(settings: UvUnwrapSettings) -> UvOperatorPlan:
    """Translate typed settings to Blender 5.2 UV operator arguments."""

    if not isinstance(settings, UvUnwrapSettings):
        raise TypeError("settings must be UvUnwrapSettings")

    if settings.method is UvUnwrapMethod.SMART_PROJECT:
        unwrap_operator = "smart_project"
        unwrap_arguments = {
            "angle_limit": radians(settings.smart_angle_limit_degrees),
            "margin_method": settings.margin_method.value,
            "rotate_method": settings.smart_rotate_method.value,
            "island_margin": settings.island_margin,
            "area_weight": settings.area_weight,
            "correct_aspect": settings.correct_aspect,
            "scale_to_bounds": settings.scale_to_bounds,
        }
    else:
        unwrap_operator = "unwrap"
        unwrap_arguments = {
            "method": settings.method.value,
            "fill_holes": settings.fill_holes,
            "correct_aspect": settings.correct_aspect,
            "use_subsurf_data": settings.use_subsurf_data,
            "margin_method": settings.margin_method.value,
            "margin": settings.island_margin,
            "no_flip": settings.no_flip,
            "iterations": settings.iterations,
            "use_weights": settings.use_weights,
            "weight_group": settings.weight_group,
            "weight_factor": settings.weight_factor,
        }

    pack_arguments: dict[str, object] | None = None
    if settings.pack_islands:
        pack_arguments = {
            "udim_source": settings.pack_udim_source.value,
            "rotate": settings.pack_rotate,
            "rotate_method": settings.pack_rotate_method.value,
            "scale": settings.pack_scale,
            "merge_overlap": settings.pack_merge_overlap,
            "margin_method": settings.margin_method.value,
            "margin": settings.pack_margin,
            "pin": settings.pack_pin,
            "pin_method": settings.pack_pin_method.value,
            "shape_method": settings.pack_shape_method.value,
        }

    return UvOperatorPlan(
        unwrap_operator=unwrap_operator,
        unwrap_arguments=MappingProxyType(unwrap_arguments),
        pack_arguments=(
            None if pack_arguments is None else MappingProxyType(pack_arguments)
        ),
    )


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise UvUnwrapError("Blender bpy module is unavailable") from exc
    return bpy


def _require_finished(result: Any, operator_name: str) -> None:
    try:
        finished = "FINISHED" in result
    except Exception as exc:
        raise UvUnwrapError(
            f"Operator {operator_name} returned an invalid result: {result!r}"
        ) from exc
    if not finished:
        raise UvUnwrapError(f"Operator {operator_name} did not finish: {result!r}")


def _call_operator(
    operator: Any,
    operator_name: str,
    arguments: Mapping[str, object],
) -> None:
    poll = getattr(operator, "poll", None)
    if callable(poll) and not poll():
        raise UvUnwrapError(f"bpy.ops.uv.{operator_name}.poll() returned False")
    try:
        result = operator(**dict(arguments))
    except Exception as exc:
        raise UvUnwrapError(
            f"bpy.ops.uv.{operator_name} failed with arguments {dict(arguments)!r}"
        ) from exc
    _require_finished(result, f"bpy.ops.uv.{operator_name}")


def _set_mode(bpy_module: Any, mode: str) -> None:
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("mode must be a non-empty string")
    resolved_mode = mode.strip().upper()
    operator = bpy_module.ops.object.mode_set
    poll = getattr(operator, "poll", None)
    if callable(poll) and not poll():
        raise UvUnwrapError(
            f"bpy.ops.object.mode_set cannot enter {resolved_mode}"
        )
    try:
        result = operator(mode=resolved_mode)
    except Exception as exc:
        raise UvUnwrapError(
            f"Unable to switch temporary object to {resolved_mode}"
        ) from exc
    _require_finished(
        result,
        f"bpy.ops.object.mode_set(mode={resolved_mode!r})",
    )


def _select_all_mesh_and_uv(bpy_module: Any) -> None:
    mesh_operator = bpy_module.ops.mesh.select_all
    mesh_poll = getattr(mesh_operator, "poll", None)
    if callable(mesh_poll) and not mesh_poll():
        raise UvUnwrapError("bpy.ops.mesh.select_all.poll() returned False")
    try:
        mesh_result = mesh_operator(action="SELECT")
    except Exception as exc:
        raise UvUnwrapError("bpy.ops.mesh.select_all failed") from exc
    _require_finished(mesh_result, "bpy.ops.mesh.select_all")

    uv_operator = bpy_module.ops.uv.select_all
    uv_poll = getattr(uv_operator, "poll", None)
    if callable(uv_poll) and not uv_poll():
        raise UvUnwrapError("bpy.ops.uv.select_all.poll() returned False")
    try:
        uv_result = uv_operator(action="SELECT")
    except Exception as exc:
        raise UvUnwrapError("bpy.ops.uv.select_all failed") from exc
    _require_finished(uv_result, "bpy.ops.uv.select_all")


def _activate_target_uv_layer(mesh: Any, layer_name: str) -> Any:
    if not isinstance(layer_name, str) or not layer_name.strip():
        raise ValueError("layer_name must be a non-empty string")
    resolved_name = layer_name.strip()
    layers = getattr(mesh, "uv_layers", None)
    if layers is None:
        raise UvUnwrapError("Temporary mesh has no UV layer collection")
    try:
        layer = layers.get(resolved_name)
    except Exception as exc:
        raise UvUnwrapError(
            f"Unable to find target UV layer '{resolved_name}'"
        ) from exc
    if layer is None:
        try:
            layer = layers.new(name=resolved_name)
        except Exception as exc:
            raise UvUnwrapError(
                f"Unable to create target UV layer '{resolved_name}'"
            ) from exc
    try:
        layers.active = layer
        layer.active_render = True
    except Exception as exc:
        raise UvUnwrapError(
            f"Unable to activate Blender 5.2 UV layer '{resolved_name}'"
        ) from exc
    return layer


def _capture_uv_layout(
    snapshot: MeshSnapshot,
    mesh: Any,
    layer_name: str,
    correspondence: MeshTopologyCorrespondence | None = None,
) -> UvLayout:
    resolved = correspondence or build_mesh_topology_correspondence(
        snapshot,
        mesh,
        stage="UV-layout-capture",
    )
    if not isinstance(resolved, MeshTopologyCorrespondence):
        raise TypeError("correspondence must be MeshTopologyCorrespondence or None")
    if resolved.snapshot_id != snapshot.snapshot_id:
        raise UvUnwrapError(
            "Topology correspondence belongs to a different snapshot"
        )

    try:
        layer = mesh.uv_layers.get(layer_name)
    except Exception as exc:
        raise UvUnwrapError(
            f"Unable to find result UV layer '{layer_name}'"
        ) from exc
    if layer is None:
        raise UvUnwrapError(f"Result UV layer '{layer_name}' is missing")

    mesh_loop_count = len(mesh.loops)
    coordinates: list[UvLoopCoordinate] = []
    loop_map = snapshot.loop_by_id()
    for loop_id, mesh_loop_index in resolved.loop_to_mesh_index:
        try:
            coordinate = read_uv_coordinate(
                layer,
                mesh_loop_index,
                expected_length=mesh_loop_count,
            )
        except MeshUvAttributeError as exc:
            raise UvUnwrapError(
                f"Unable to read UV for loop {loop_id.index} from Blender "
                f"loop {mesh_loop_index}: {exc}"
            ) from exc
        coordinates.append(
            UvLoopCoordinate(
                loop_id=loop_id,
                source_loop_id=loop_map[loop_id].source_id,
                coordinate=coordinate,
            )
        )

    coordinates.sort(key=lambda entry: entry.loop_id.index)
    if len(coordinates) != len(snapshot.loops):
        raise UvUnwrapError(
            f"UV operation returned {len(coordinates)} loop coordinates for "
            f"{len(snapshot.loops)} snapshot loops"
        )
    return UvLayout(
        snapshot_id=snapshot.snapshot_id,
        layer_name=layer_name,
        coordinates=tuple(coordinates),
    )


def _execute_uv_operator_plan(
    bpy_module: Any,
    plan: UvOperatorPlan,
) -> None:
    """Run Blender UV operators and always leave the temporary Object in Object Mode."""

    _set_mode(bpy_module, "EDIT")
    primary_error: BaseException | None = None
    try:
        _select_all_mesh_and_uv(bpy_module)
        uv_operators = bpy_module.ops.uv
        unwrap_operator = getattr(uv_operators, plan.unwrap_operator, None)
        if unwrap_operator is None:
            raise UvUnwrapError(
                f"bpy.ops.uv.{plan.unwrap_operator} is unavailable"
            )
        _call_operator(
            unwrap_operator,
            plan.unwrap_operator,
            plan.unwrap_arguments,
        )
        if plan.pack_arguments is not None:
            pack_operator = getattr(uv_operators, "pack_islands", None)
            if pack_operator is None:
                raise UvUnwrapError("bpy.ops.uv.pack_islands is unavailable")
            _call_operator(
                pack_operator,
                "pack_islands",
                plan.pack_arguments,
            )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _set_mode(bpy_module, "OBJECT")
        except Exception:
            if primary_error is None:
                raise
            logger.exception(
                "Unable to leave Edit Mode while handling a UV operator failure"
            )


def unwrap_snapshot_uv(
    snapshot: MeshSnapshot,
    settings: UvUnwrapSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> UvUnwrapResult:
    """Unwrap one immutable snapshot without changing its source Blender Object."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    resolved_settings = settings or UvUnwrapSettings()
    if not isinstance(resolved_settings, UvUnwrapSettings):
        raise TypeError("settings must be UvUnwrapSettings")

    bpy_module = _load_bpy()
    resolved_context = context or bpy_module.context
    resolved_scene = scene or getattr(resolved_context, "scene", None)
    if resolved_scene is None:
        raise UvUnwrapError("A Blender Scene is required for UV unwrap")
    plan = build_uv_operator_plan(resolved_settings)

    try:
        with temporary_mesh_object(
            snapshot,
            scene=resolved_scene,
            name_prefix="__Spine2D_Unwrap",
        ) as temporary:
            _activate_target_uv_layer(temporary.mesh, resolved_settings.layer_name)
            with activate_object_for_operator(
                temporary.object,
                context=resolved_context,
            ):
                _execute_uv_operator_plan(bpy_module, plan)

            try:
                temporary.mesh.update()
            except Exception as exc:
                raise UvUnwrapError(
                    "Unable to update temporary Mesh after UV operators"
                ) from exc
            try:
                correspondence = build_mesh_topology_correspondence(
                    snapshot,
                    temporary.mesh,
                    stage="post-UV-operators",
                )
            except MeshWriteError as exc:
                raise UvUnwrapError(
                    "Temporary mesh topology no longer corresponds to the "
                    f"snapshot after UV operations: {exc}"
                ) from exc
            layout = _capture_uv_layout(
                snapshot,
                temporary.mesh,
                resolved_settings.layer_name,
                correspondence,
            )
            updated_snapshot = apply_uv_layout(snapshot, layout)
            statistics = calculate_uv_statistics(
                updated_snapshot,
                resolved_settings.layer_name,
            )
            logger.info(
                "Unwrapped Blender 5.2 snapshot '%s' using %s: %d loops, "
                "bounds U[%s, %s] V[%s, %s]",
                snapshot.snapshot_id,
                resolved_settings.method.value,
                statistics.loop_count,
                statistics.minimum_u,
                statistics.maximum_u,
                statistics.minimum_v,
                statistics.maximum_v,
            )
            return UvUnwrapResult(
                snapshot=updated_snapshot,
                settings=resolved_settings,
                statistics=statistics,
            )
    except UvUnwrapError:
        raise
    except (MeshWriteError, BlenderContextError) as exc:
        raise UvUnwrapError(
            f"Unable to prepare transactional UV unwrap for '{snapshot.snapshot_id}'"
        ) from exc
    except Exception as exc:
        logger.exception("UV unwrap failed for snapshot '%s'", snapshot.snapshot_id)
        raise UvUnwrapError(
            f"UV unwrap failed for snapshot '{snapshot.snapshot_id}': {exc}"
        ) from exc


__all__ = [
    "UvOperatorPlan",
    "UvUnwrapError",
    "build_uv_operator_plan",
    "unwrap_snapshot_uv",
]
