"""Execute B4 active-camera renders into caller-owned atomic reservations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Tuple

from ..domain.baking import (
    BakeArtifact,
    BakeExecutionResult,
    BakeExecutionSettings,
    CameraProjectionPlan,
    TextureFormat,
)
from ..infrastructure import (
    AtomicFileTransaction,
    AtomicOutputReservation,
    atomic_file_transaction,
)
from .scene_bake_analyzer import validate_runtime_scene_context

logger = logging.getLogger(__name__)


class CameraProjectionExecutionError(RuntimeError):
    """Raised when an active-camera projection render cannot be staged safely."""


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
class _ProjectionRuntimeState:
    scene_values: Tuple[_SceneValue, ...]
    frame_current: int
    visibility: Tuple[_ObjectVisibility, ...]

    @classmethod
    def capture(cls, scene: Any) -> "_ProjectionRuntimeState":
        if scene is None:
            raise CameraProjectionExecutionError("scene cannot be None")
        scene_values = []
        for path in _SCENE_PATHS:
            try:
                scene_values.append(_SceneValue(path=path, value=_get_path(scene, path)))
            except Exception:
                logger.debug("Optional render property '%s' is unavailable", path)
        visibility = []
        try:
            objects = tuple(scene.objects)
        except Exception as exc:
            raise CameraProjectionExecutionError("Unable to inspect scene objects") from exc
        for obj in objects:
            try:
                hide_render = bool(getattr(obj, "hide_render", False))
            except Exception:
                hide_render = False
            try:
                visible_camera = (
                    bool(getattr(obj, "visible_camera"))
                    if hasattr(obj, "visible_camera")
                    else None
                )
            except Exception:
                visible_camera = None
            visibility.append(
                _ObjectVisibility(
                    obj=obj,
                    hide_render=hide_render,
                    visible_camera=visible_camera,
                )
            )
        return cls(
            scene_values=tuple(scene_values),
            frame_current=int(getattr(scene, "frame_current", 0) or 0),
            visibility=tuple(visibility),
        )

    def restore(self, scene: Any) -> None:
        failures: list[str] = []
        for entry in reversed(self.scene_values):
            try:
                _set_path(scene, entry.path, entry.value)
            except Exception as exc:
                failures.append(f"{entry.path}: {exc}")
        for entry in self.visibility:
            try:
                entry.obj.hide_render = entry.hide_render
            except Exception as exc:
                failures.append(
                    f"{getattr(entry.obj, 'name', 'Object')}.hide_render: {exc}"
                )
            if entry.visible_camera is not None:
                try:
                    entry.obj.visible_camera = entry.visible_camera
                except Exception as exc:
                    failures.append(
                        f"{getattr(entry.obj, 'name', 'Object')}.visible_camera: {exc}"
                    )
        try:
            frame_set = getattr(scene, "frame_set", None)
            if callable(frame_set):
                frame_set(self.frame_current)
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


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise CameraProjectionExecutionError("Blender bpy module is unavailable") from exc
    return bpy


def _set_if_available(root: Any, path: str, value: Any) -> None:
    try:
        _set_path(root, path, value)
    except Exception:
        logger.debug("Optional render property '%s' is not writable", path, exc_info=True)


@contextmanager
def preserve_camera_projection_state(scene: Any) -> Iterator[_ProjectionRuntimeState]:
    state = _ProjectionRuntimeState.capture(scene)
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


def _configure_camera_visibility(source_obj: Any, scene: Any) -> None:
    try:
        objects = tuple(scene.objects)
    except Exception as exc:
        raise CameraProjectionExecutionError("Unable to iterate scene objects") from exc
    if source_obj not in objects:
        raise CameraProjectionExecutionError("source object is not linked to the render scene")

    for obj in objects:
        object_type = str(getattr(obj, "type", "") or "")
        if obj is source_obj:
            try:
                obj.hide_render = False
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    "Unable to make source object render-visible"
                ) from exc
            if hasattr(obj, "visible_camera"):
                try:
                    obj.visible_camera = True
                except Exception as exc:
                    raise CameraProjectionExecutionError(
                        "Unable to enable source camera-ray visibility"
                    ) from exc
            continue

        # Keep every dependency in the render graph.  Only direct camera rays are disabled;
        # glossy, transmission, diffuse and shadow visibility remain untouched.
        if object_type in _RENDERABLE_TYPES and hasattr(obj, "visible_camera"):
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

    scene.render.engine = execution_settings.render_engine
    scene.render.resolution_x = plan.settings.width
    scene.render.resolution_y = plan.settings.height
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(staged_path)
    scene.render.film_transparent = plan.transparent_background
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = plan.settings.texture_format.value
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = (
        "32" if plan.settings.texture_format is TextureFormat.OPEN_EXR else "8"
    )
    scene.cycles.samples = execution_settings.samples
    _set_if_available(scene, "cycles.film_transparent_glass", False)


def _set_timeline_frame(scene: Any, context: Any, frame: int | None) -> None:
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


def _call_public_render_operator(bpy_module: Any) -> None:
    from . import bake_executor as public_executor

    public_executor._call_render_operator(bpy_module)


def _require_reservations(
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


def _render_to_reservations(
    source_obj: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings,
    reservations: Tuple[AtomicOutputReservation, ...],
    *,
    context: Any | None,
    scene: Any | None,
) -> None:
    if source_obj is None or getattr(source_obj, "type", None) != "MESH":
        raise CameraProjectionExecutionError("source_obj must be a Blender MESH object")
    if not isinstance(plan, CameraProjectionPlan):
        raise TypeError("plan must be CameraProjectionPlan")
    if str(getattr(source_obj, "name", "")) != plan.source_object_id:
        raise CameraProjectionExecutionError(
            "source object identity does not match CameraProjectionPlan"
        )

    bpy_module = _load_bpy()
    resolved_context = context or bpy_module.context
    resolved_scene = scene or getattr(resolved_context, "scene", None)
    if resolved_scene is None:
        raise CameraProjectionExecutionError("A Blender Scene is required")
    if getattr(resolved_scene, "camera", None) is None:
        raise CameraProjectionExecutionError("Scene has no active camera")

    validate_runtime_scene_context(
        source_obj,
        plan.object_context,
        plan.scene_context,
        scene=resolved_scene,
        context=resolved_context,
    )
    resolved_reservations = _require_reservations(plan, reservations)

    with preserve_camera_projection_state(resolved_scene):
        _configure_camera_visibility(source_obj, resolved_scene)
        for task, reservation in zip(plan.frame_tasks, resolved_reservations):
            _set_timeline_frame(resolved_scene, resolved_context, task.timeline_frame)
            configure_scene_for_camera_projection(
                resolved_scene,
                plan,
                execution_settings,
                reservation.staged_path,
            )
            logger.info(
                "Rendering B4 camera projection '%s' frame %d/%d camera='%s'",
                plan.source_object_id,
                task.task_index + 1,
                len(plan.frame_tasks),
                plan.camera_object_id,
            )
            _call_public_render_operator(bpy_module)
            if not reservation.staged_path.is_file():
                raise CameraProjectionExecutionError(
                    "Blender reported a finished render but the staged file is missing: "
                    f"{reservation.staged_path}"
                )
            if reservation.staged_path.stat().st_size <= 0:
                raise CameraProjectionExecutionError(
                    f"Camera projection output is empty: {reservation.staged_path}"
                )


def stage_camera_projection_outputs(
    source_obj: Any,
    plan: CameraProjectionPlan,
    output_transaction: AtomicFileTransaction,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> Tuple[AtomicOutputReservation, ...]:
    if not isinstance(output_transaction, AtomicFileTransaction):
        raise TypeError("output_transaction must be AtomicFileTransaction")
    resolved_settings = execution_settings or BakeExecutionSettings()
    if not isinstance(resolved_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings or None")
    try:
        reservations = tuple(
            output_transaction.reserve(task.output_path) for task in plan.frame_tasks
        )
        _render_to_reservations(
            source_obj,
            plan,
            resolved_settings,
            reservations,
            context=context,
            scene=scene,
        )
        return reservations
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        logger.exception("Unexpected B4 projection failure for '%s'", plan.source_object_id)
        raise CameraProjectionExecutionError(
            f"Camera projection failed for '{plan.source_object_id}': {exc}"
        ) from exc


def _build_execution_result(
    plan: CameraProjectionPlan,
    committed_paths: Iterable[Path],
) -> BakeExecutionResult:
    resolved = tuple(Path(path).expanduser().resolve(strict=False) for path in committed_paths)
    expected = tuple(task.output_path.expanduser().resolve(strict=False) for task in plan.frame_tasks)
    if resolved != expected:
        raise CameraProjectionExecutionError(
            f"Committed projection paths do not match plan; expected={expected}, got={resolved}"
        )
    artifacts = tuple(
        BakeArtifact(
            task_index=task.task_index,
            timeline_frame=task.timeline_frame,
            image_name=task.image_name,
            output_path=path,
            width=plan.settings.width,
            height=plan.settings.height,
        )
        for task, path in zip(plan.frame_tasks, resolved)
    )
    return BakeExecutionResult(plan=plan, artifacts=artifacts)


def execute_camera_projection_plan(
    source_obj: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings | None = None,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> BakeExecutionResult:
    with atomic_file_transaction() as transaction:
        reservations = stage_camera_projection_outputs(
            source_obj,
            plan,
            transaction,
            execution_settings,
            context=context,
            scene=scene,
        )
        committed = transaction.commit()
    return _build_execution_result(
        plan,
        tuple(
            path
            for reservation, path in zip(reservations, committed)
            if path == reservation.final_path
        ),
    )


__all__ = [
    "CameraProjectionExecutionError",
    "configure_scene_for_camera_projection",
    "execute_camera_projection_plan",
    "preserve_camera_projection_state",
    "stage_camera_projection_outputs",
]
