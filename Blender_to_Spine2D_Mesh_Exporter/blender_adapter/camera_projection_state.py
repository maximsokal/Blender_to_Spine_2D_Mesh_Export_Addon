"""Reversible Blender 5.2 camera-projection render state."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterator, Tuple

from ..domain.baking import (
    BakeExecutionSettings,
    CameraProjectionPlan,
    resolve_projection_output_policy,
)
from .camera_projection_error import CameraProjectionExecutionError
from .render_engine_contract import (
    render_engine_contract,
    render_engine_contract_from_execution,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SceneValue:
    path: str
    value: Any


@dataclass(frozen=True, slots=True)
class _ObjectVisibility:
    obj: Any
    hide_render: bool
    visible_camera: bool


@dataclass(frozen=True, slots=True)
class ProjectionRuntimeState:
    """Scene, frame, and visibility values restored after one render scope."""

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
            raise CameraProjectionExecutionError(
                "Unable to inspect scene objects"
            ) from exc

        visibility: list[_ObjectVisibility] = []
        for obj in objects:
            object_name = str(getattr(obj, "name", "Object") or "Object")
            try:
                hide_render = bool(obj.hide_render)
                visible_camera = bool(obj.visible_camera)
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    f"Unable to capture Blender 5.2 visibility for '{object_name}'"
                ) from exc
            visibility.append(
                _ObjectVisibility(
                    obj=obj,
                    hide_render=hide_render,
                    visible_camera=visible_camera,
                )
            )

        try:
            frame_current = int(scene.frame_current)
        except Exception as exc:
            raise CameraProjectionExecutionError(
                "Unable to capture Scene frame_current"
            ) from exc
        return cls(
            scene_values=tuple(values),
            frame_current=frame_current,
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
            object_name = str(getattr(entry.obj, "name", "Object") or "Object")
            try:
                entry.obj.hide_render = entry.hide_render
            except Exception as exc:
                failures.append(f"{object_name}.hide_render: {exc}")
            try:
                entry.obj.visible_camera = entry.visible_camera
            except Exception as exc:
                failures.append(f"{object_name}.visible_camera: {exc}")

        try:
            scene.frame_set(self.frame_current)
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
_RENDERABLE_TYPES = frozenset(
    {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME"}
)


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
        logger.debug(
            "Optional render property '%s' is not writable",
            path,
            exc_info=True,
        )


@contextmanager
def preserve_camera_projection_state(
    scene: Any,
) -> Iterator[ProjectionRuntimeState]:
    """Restore all captured Blender 5.2 state even when rendering fails."""

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


def configure_camera_visibility(
    source_obj: Any,
    scene: Any,
    *,
    isolate: bool,
) -> None:
    """Expose the source to the camera while retaining dependency-ray participation."""

    if not isinstance(isolate, bool):
        raise TypeError("isolate must be bool")

    try:
        objects = tuple(scene.objects)
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to iterate scene objects"
        ) from exc

    if source_obj not in objects:
        raise CameraProjectionExecutionError(
            "source object is not linked to the render scene"
        )

    for obj in objects:
        object_name = str(getattr(obj, "name", "Object") or "Object")
        if obj is source_obj:
            try:
                obj.hide_render = False
                obj.visible_camera = True
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    "Unable to make source object camera-visible"
                ) from exc
            continue

        if isolate and str(getattr(obj, "type", "") or "") in _RENDERABLE_TYPES:
            try:
                obj.visible_camera = False
            except Exception as exc:
                raise CameraProjectionExecutionError(
                    f"Unable to isolate camera visibility for '{object_name}'"
                ) from exc


def configure_scene_for_camera_projection(
    scene: Any,
    plan: CameraProjectionPlan,
    execution_settings: BakeExecutionSettings,
    staged_path: Path,
) -> None:
    """Apply one frame's validated Blender 5.2 render settings."""

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

    output_policy = resolve_projection_output_policy(
        execution_settings.projection_output_policy,
        plan.settings.texture_format,
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
    scene.render.image_settings.color_depth = output_policy.color_depth
    _set_if_available(scene, "cycles.samples", execution_settings.samples)
    _set_if_available(scene, "cycles.film_transparent_glass", False)


def set_timeline_frame(
    scene: Any,
    context: Any,
    frame: int | None,
) -> None:
    """Evaluate one planned frame and update the active View Layer."""

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


__all__ = [
    "CameraProjectionExecutionError",
    "ProjectionRuntimeState",
    "configure_camera_visibility",
    "configure_scene_for_camera_projection",
    "preserve_camera_projection_state",
    "set_timeline_frame",
]
