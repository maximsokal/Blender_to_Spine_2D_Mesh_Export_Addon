"""Reversible Blender render state and operator-boundary helpers for B4."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Tuple

from ..domain.baking import BakeExecutionSettings, CameraProjectionPlan, TextureFormat
from ..infrastructure import AtomicOutputReservation
from .render_engine_contract import (
    render_engine_contract,
    render_engine_contract_from_execution,
)
from .scene_bake_analyzer import validate_runtime_scene_context
from .view_layer_contract import validate_source_view_layer_for_camera_projection

logger = logging.getLogger(__name__)


class CameraProjectionExecutionError(RuntimeError):
    """Raised when an active-camera projection cannot be staged safely."""


@dataclass(frozen=True, slots=True)
class _SceneValue:
    path: str
    value: Any


@dataclass(frozen=True, slots=True)
class _ObjectVisibility:
    obj: Any
    hide_render: bool
    visible_camera: bool | None


@dataclass(frozen=True, slots=True)
class ProjectionRuntimeState:
    scene_values: Tuple[_SceneValue, ...]
    frame_current: int
    visibility: Tuple[_ObjectVisibility, ...]

    @classmethod
    def capture(cls, scene: Any) -> "ProjectionRuntimeState":
        if scene is None:
            raise CameraProjectionExecutionError("scene cannot be None")
        values: list[_SceneValue] = []
        for path in _SCENE_PATHS:
            try:
                values.append(_SceneValue(path, _get_path(scene, path)))
            except Exception:
                logger.debug("Optional render property '%s' is unavailable", path)
        try:
            objects = tuple(scene.objects)
        except Exception as exc:
            raise CameraProjectionExecutionError("Unable to inspect scene objects") from exc
        visibility = []
        for obj in objects:
            try:
                hide_render = bool(getattr(obj, "hide_render", False))
            except Exception:
                hide_render = False
            try:
                visible_camera = (
                    bool(obj.visible_camera) if hasattr(obj, "visible_camera") else None
                )
            except Exception:
                visible_camera = None
            visibility.append(_ObjectVisibility(obj, hide_render, visible_camera))
        return cls(
            tuple(values),
            int(getattr(scene, "frame_current", 0) or 0),
            tuple(visibility),
        )

    def restore(self, scene: Any) -> None:
        failures: list[str] = []
        for entry in reversed(self.scene_values):
            try:
                _set_path(scene, entry.path, entry.value)
            except Exception as exc:
                failures.append(f"{entry.path}: {exc}")
        for entry in self.visibility:
            name = str(getattr(entry.obj, "name", "Object"))
            try:
                entry.obj.hide_render = entry.hide_render
            except Exception as exc:
                failures.append(f"{name}.hide_render: {exc}")
            if entry.visible_camera is not None:
                try:
                    entry.obj.visible_camera = entry.visible_camera
                except Exception as exc:
                    failures.append(f"{name}.visible_camera: {exc}")
        try:
            setter = getattr(scene, "frame_set", None)
            if callable(setter):
                setter(self.frame_current)
            else:
                scene.frame_current = self.frame_current
        except Exception as exc:
            failures.append(f"frame_current: {exc}")
        if failures:
            raise CameraProjectionExecutionError(
                "Unable to restore camera projection state: " + "; ".join(failures)
            )


_SCENE_PATHS = (
    "render.engine",
    "render.resolution_x",
    "render.resolution_y",
    "render.resolution_percentage",
    "render.filepath",
    "render.film_transparent",
    "render.use_file_extension",
    "render.use_compositing",
    "render.use_sequencer",
    "render.image_settings.file_format",
    "render.image_settings.color_mode",
    "render.image_settings.color_depth",
    "cycles.samples",
    "cycles.film_transparent_glass",
)
_RENDERABLE_TYPES = frozenset({"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"})


def _get_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _set_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    current = root
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def _set_if_available(root: Any, path: str, value: Any) -> None:
    try:
        _set_path(root, path, value)
    except Exception:
        logger.debug("Optional render property '%s' is not writable", path, exc_info=True)


def load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise CameraProjectionExecutionError("Blender bpy module is unavailable") from exc
    return bpy


@contextmanager
def preserve_camera_projection_state(scene: Any) -> Iterator[ProjectionRuntimeState]:
    state = ProjectionRuntimeState.capture(scene)
    primary_error: BaseException | None = None
    try:
        yield state
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            state.restore(scene)
        except Exception:
            if primary_error is None:
                raise
            logger.exception(
                "Failed to restore camera projection state while handling another error"
            )


def configure_camera_visibility(source_obj: Any, scene: Any, *, isolate: bool) -> None:
    try:
        objects = tuple(scene.objects)
    except Exception as exc:
        raise CameraProjectionExecutionError("Unable to iterate scene objects") from exc
    if source_obj not in objects:
        raise CameraProjectionExecutionError("source object is not linked to the render scene")
    for obj in objects:
        if obj is source_obj:
            try:
                obj.hide_render = False
                if hasattr(obj, "visible_camera"):
                    obj.visible_camera = True
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    "Unable to make source object camera-visible"
                ) from exc
            continue
        if (
            isolate
            and str(getattr(obj, "type", "") or "") in _RENDERABLE_TYPES
            and hasattr(obj, "visible_camera")
        ):
            try:
                obj.visible_camera = False
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    f"Unable to isolate camera visibility for '{getattr(obj, 'name', obj)}'"
                ) from exc


def configure_scene_for_camera_projection(
    scene: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings,
    staged_path: Path,
) -> None:
    if scene is None:
        raise CameraProjectionExecutionError("scene cannot be None")
    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if not isinstance(execution_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings")
    if not isinstance(staged_path, Path):
        raise TypeError("staged_path must be pathlib.Path")

    renderer = render_engine_contract_from_execution(execution_settings)
    if plan.scene_context is None:
        raise CameraProjectionExecutionError(
            "CameraProjectionPlan is missing its renderer-specific SceneBakeContext"
        )
    planned_renderer = render_engine_contract(plan.scene_context.render_engine)
    if renderer != planned_renderer:
        raise CameraProjectionExecutionError(
            "camera projection execution engine differs from the analyzed renderer; "
            f"planned={planned_renderer.blender_engine}, "
            f"execution={renderer.blender_engine}"
        )

    scene.render.engine = renderer.blender_engine
    scene.render.resolution_x = plan.settings.width
    scene.render.resolution_y = plan.settings.height
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(staged_path)
    scene.render.film_transparent = plan.transparent_background
    scene.render.use_file_extension = True
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = plan.settings.texture_format.value
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = (
        "32" if plan.settings.texture_format is TextureFormat.OPEN_EXR else "8"
    )
    _set_if_available(scene, "cycles.samples", execution_settings.samples)
    _set_if_available(scene, "cycles.film_transparent_glass", False)


def set_timeline_frame(scene: Any, context: Any, frame: int | None) -> None:
    if frame is None:
        return
    try:
        scene.frame_set(frame)
        update = getattr(getattr(context, "view_layer", None), "update", None)
        if callable(update):
            update()
    except Exception as exc:
        raise CameraProjectionExecutionError(
            f"Unable to evaluate camera projection frame {frame}"
        ) from exc


def call_public_render_operator(bpy_module: Any) -> None:
    from . import bake_executor as public_executor

    public_executor._call_render_operator(bpy_module)


def require_reservations(
    plan: CameraProjectionPlan,
    reservations: Iterable[AtomicOutputReservation],
) -> Tuple[AtomicOutputReservation, ...]:
    resolved = tuple(reservations)
    if len(resolved) != len(plan.frame_tasks):
        raise CameraProjectionExecutionError(
            f"Expected {len(plan.frame_tasks)} projection reservations, got {len(resolved)}"
        )
    for task, reservation in zip(plan.frame_tasks, resolved):
        if not isinstance(reservation, AtomicOutputReservation):
            raise TypeError("reservations must contain AtomicOutputReservation")
        expected = task.output_path.expanduser().resolve(strict=False)
        if reservation.final_path != expected:
            raise CameraProjectionExecutionError(
                f"Projection task {task.task_index} expected '{expected}', got "
                f"'{reservation.final_path}'"
            )
    return resolved


def validate_projection_runtime(
    source_obj: Any,
    plan: CameraProjectionPlan,
    *,
    context: Any | None,
    scene: Any | None,
) -> tuple[Any, Any, Any]:
    if source_obj is None or getattr(source_obj, "type", None) != "MESH":
        raise CameraProjectionExecutionError("source_obj must be a Blender MESH object")
    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if str(getattr(source_obj, "name", "")) != plan.source_object_id:
        raise CameraProjectionExecutionError(
            "source object identity does not match CameraProjectionPlan"
        )
    bpy_module = load_bpy()
    resolved_context = context or bpy_module.context
    resolved_scene = scene or getattr(resolved_context, "scene", None)
    if resolved_scene is None:
        raise CameraProjectionExecutionError("A Blender Scene is required")
    if getattr(resolved_scene, "camera", None) is None:
        raise CameraProjectionExecutionError("Scene has no active camera")
    validate_source_view_layer_for_camera_projection(
        source_obj,
        getattr(resolved_context, "view_layer", None),
    )
    validate_runtime_scene_context(
        source_obj,
        plan.object_context,
        plan.scene_context,
        scene=resolved_scene,
        context=resolved_context,
    )
    return bpy_module, resolved_context, resolved_scene
