"""Capture, configure, and restore Blender scene state used by texture baking."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Any, Iterator, Tuple

from ..domain.baking import BakeExecutionSettings, BakeMode, BakePlan

logger = logging.getLogger(__name__)


class BakeSceneStateError(RuntimeError):
    """Raised when bake-related scene state cannot be captured or restored."""


@dataclass(frozen=True, slots=True)
class ScenePropertyValue:
    path: str
    value: Any


@dataclass(frozen=True, slots=True)
class BakeSceneState:
    properties: Tuple[ScenePropertyValue, ...]
    frame_current: int

    @classmethod
    def capture(cls, scene: Any) -> "BakeSceneState":
        if scene is None:
            raise BakeSceneStateError("scene cannot be None")
        values = []
        for path in _CAPTURE_PATHS:
            try:
                values.append(ScenePropertyValue(path, _get_path(scene, path)))
            except (AttributeError, TypeError):
                # Blender builds may omit optional properties. Only captured
                # properties are restored; required ones are checked at configure.
                continue
        try:
            frame_current = int(scene.frame_current)
        except Exception as exc:
            raise BakeSceneStateError("Unable to capture scene.frame_current") from exc
        return cls(properties=tuple(values), frame_current=frame_current)

    def restore(self, scene: Any) -> None:
        failures = []
        for entry in reversed(self.properties):
            try:
                _set_path(scene, entry.path, entry.value)
            except Exception as exc:
                failures.append(f"{entry.path}: {exc}")
        try:
            frame_set = getattr(scene, "frame_set", None)
            if callable(frame_set):
                frame_set(self.frame_current)
            else:
                scene.frame_current = self.frame_current
        except Exception as exc:
            failures.append(f"frame_current: {exc}")
        if failures:
            raise BakeSceneStateError(
                "Unable to restore Blender bake scene state: " + "; ".join(failures)
            )


_CAPTURE_PATHS = (
    "render.engine",
    "render.image_settings.file_format",
    "render.image_settings.color_mode",
    "render.bake.margin",
    "render.bake.use_clear",
    "render.bake.use_selected_to_active",
    "render.bake.use_cage",
    "render.bake.cage_extrusion",
    "render.bake.use_pass_direct",
    "render.bake.use_pass_indirect",
    "render.bake.use_pass_color",
    "cycles.bake_type",
    "cycles.samples",
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


def _require_path(scene: Any, path: str) -> None:
    try:
        _get_path(scene, path)
    except Exception as exc:
        raise BakeSceneStateError(
            f"Required Blender bake setting '{path}' is unavailable"
        ) from exc


def configure_scene_for_bake(
    scene: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
) -> None:
    """Apply all render settings required by one immutable BakePlan."""

    if scene is None:
        raise BakeSceneStateError("scene cannot be None")
    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")
    if not isinstance(execution_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings")

    required = (
        "render.engine",
        "render.image_settings.file_format",
        "render.image_settings.color_mode",
        "render.bake.margin",
        "render.bake.use_clear",
        "render.bake.use_selected_to_active",
        "render.bake.use_cage",
        "render.bake.cage_extrusion",
        "render.bake.use_pass_direct",
        "render.bake.use_pass_indirect",
        "render.bake.use_pass_color",
        "cycles.bake_type",
        "cycles.samples",
    )
    for path in required:
        _require_path(scene, path)

    scene.render.engine = execution_settings.render_engine
    scene.render.image_settings.file_format = plan.settings.texture_format.value
    scene.render.image_settings.color_mode = execution_settings.color_mode
    scene.render.bake.margin = plan.settings.margin_pixels
    scene.render.bake.use_clear = execution_settings.use_clear
    scene.render.bake.use_selected_to_active = plan.settings.selected_to_active
    scene.render.bake.use_cage = plan.settings.selected_to_active
    scene.render.bake.cage_extrusion = plan.settings.cage_extrusion
    flat_color_pass = plan.bake_mode in {BakeMode.DIFFUSE, BakeMode.EMIT}
    scene.render.bake.use_pass_direct = not flat_color_pass
    scene.render.bake.use_pass_indirect = not flat_color_pass
    scene.render.bake.use_pass_color = True
    scene.cycles.bake_type = plan.bake_mode.value
    scene.cycles.samples = execution_settings.samples


@contextmanager
def preserve_bake_scene_state(scene: Any) -> Iterator[BakeSceneState]:
    """Restore scene settings on success and failure without hiding primary errors."""

    state = BakeSceneState.capture(scene)
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
                "Failed to restore bake scene state while handling another exception"
            )
