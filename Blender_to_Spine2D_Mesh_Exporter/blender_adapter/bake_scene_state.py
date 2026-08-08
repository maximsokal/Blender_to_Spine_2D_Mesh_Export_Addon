"""Capture, configure, and restore Blender 5.2 Scene state used by baking."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Any, Iterator, Tuple

from ..domain.baking import (
    BakeEvaluationScope,
    BakeExecutionSettings,
    BakeMode,
    BakePlan,
)
from ..domain.baking.texture_format_policy import resolve_texture_color_mode


logger = logging.getLogger(__name__)


class BakeSceneStateError(RuntimeError):
    """Raised when bake-related Scene state cannot be captured or restored."""


@dataclass(frozen=True, slots=True)
class ScenePropertyValue:
    path: str
    value: Any

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("path must be a non-empty string")


@dataclass(frozen=True, slots=True)
class BakeSceneState:
    properties: Tuple[ScenePropertyValue, ...]
    frame_current: int

    def __post_init__(self) -> None:
        if not isinstance(self.properties, tuple) or not all(
            isinstance(value, ScenePropertyValue) for value in self.properties
        ):
            raise TypeError("properties must contain ScenePropertyValue values")
        if not isinstance(self.frame_current, int) or isinstance(
            self.frame_current,
            bool,
        ):
            raise TypeError("frame_current must be int")

    @classmethod
    def capture(cls, scene: Any) -> "BakeSceneState":
        if scene is None:
            raise BakeSceneStateError("scene cannot be None")
        values: list[ScenePropertyValue] = []
        for path in _CAPTURE_PATHS:
            try:
                values.append(ScenePropertyValue(path, _get_path(scene, path)))
            except Exception as exc:
                raise BakeSceneStateError(
                    f"Required Blender 5.2 bake setting '{path}' is unavailable"
                ) from exc
        try:
            frame_current = int(scene.frame_current)
        except Exception as exc:
            raise BakeSceneStateError("Unable to capture scene.frame_current") from exc
        return cls(properties=tuple(values), frame_current=frame_current)

    def restore(self, scene: Any) -> None:
        if scene is None:
            raise BakeSceneStateError("scene cannot be None")
        failures: list[str] = []
        for entry in reversed(self.properties):
            try:
                _set_path(scene, entry.path, entry.value)
            except Exception as exc:
                failures.append(f"{entry.path}: {exc}")
        try:
            scene.frame_set(self.frame_current)
        except Exception as exc:
            failures.append(f"frame_current: {exc}")
        if failures:
            raise BakeSceneStateError(
                "Unable to restore Blender 5.2 bake Scene state: "
                + "; ".join(failures)
            )


# Blender 5.2 BakeSettings exposes only these Combined contribution toggles:
# EMIT, DIRECT, INDIRECT, COLOR, DIFFUSE, GLOSSY and TRANSMISSION. Ambient
# Occlusion and Subsurface are not BakeSettings pass-filter properties in 5.2.
_CAPTURE_PATHS = (
    "render.engine",
    # Capture color mode first so reverse-order restoration puts the file format
    # back before assigning a mode whose enum depends on that format (JPEG has no RGBA).
    "render.image_settings.color_mode",
    "render.image_settings.file_format",
    "render.bake.margin",
    "render.bake.use_clear",
    "render.bake.use_selected_to_active",
    "render.bake.use_cage",
    "render.bake.cage_extrusion",
    "render.bake.view_from",
    "render.bake.use_pass_direct",
    "render.bake.use_pass_indirect",
    "render.bake.use_pass_color",
    "render.bake.use_pass_diffuse",
    "render.bake.use_pass_glossy",
    "render.bake.use_pass_transmission",
    "render.bake.use_pass_emit",
    "cycles.bake_type",
    "cycles.samples",
)


def _get_path(root: Any, path: str) -> Any:
    if root is None:
        raise BakeSceneStateError("root cannot be None")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    current = root
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _set_path(root: Any, path: str, value: Any) -> None:
    if root is None:
        raise BakeSceneStateError("root cannot be None")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    parts = path.split(".")
    current = root
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def configure_scene_for_bake(
    scene: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    *,
    bake_mode: BakeMode,
    evaluation_scope: BakeEvaluationScope,
) -> None:
    """Apply explicit Blender 5.2 settings for one immutable bake pass.

    Camera-scoped semantic passes use Blender's ``ACTIVE_CAMERA`` bake view so
    reflection and transmission rays are evaluated from the same camera context used by
    the material plan. Every Blender-5.2 Combined contribution is set explicitly;
    results therefore cannot depend on whichever Render > Bake toggles the user last
    changed. All settings are owned by :func:`preserve_bake_scene_state` and restored
    after the bake.
    """

    if scene is None:
        raise BakeSceneStateError("scene cannot be None")
    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")
    if not isinstance(execution_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings")
    if not isinstance(bake_mode, BakeMode):
        raise TypeError("bake_mode must be BakeMode")
    if not isinstance(evaluation_scope, BakeEvaluationScope):
        raise TypeError("evaluation_scope must be BakeEvaluationScope")

    for path in _CAPTURE_PATHS:
        try:
            _get_path(scene, path)
        except Exception as exc:
            raise BakeSceneStateError(
                f"Required Blender 5.2 bake setting '{path}' is unavailable"
            ) from exc

    if evaluation_scope is BakeEvaluationScope.CAMERA and getattr(
        scene,
        "camera",
        None,
    ) is None:
        raise BakeSceneStateError(
            "Camera-scoped semantic bake requires scene.camera"
        )

    scene.render.engine = execution_settings.render_engine
    scene.render.image_settings.file_format = plan.settings.texture_format.value
    scene.render.image_settings.color_mode = (
        resolve_texture_color_mode(
            plan.settings.texture_format,
            execution_settings.color_mode,
        )
    )
    scene.render.bake.margin = plan.settings.margin_pixels
    scene.render.bake.use_clear = execution_settings.use_clear
    scene.render.bake.use_selected_to_active = plan.settings.selected_to_active
    scene.render.bake.use_cage = plan.settings.selected_to_active
    scene.render.bake.cage_extrusion = plan.settings.cage_extrusion
    scene.render.bake.view_from = (
        "ACTIVE_CAMERA"
        if evaluation_scope is BakeEvaluationScope.CAMERA
        else "ABOVE_SURFACE"
    )

    flat_color_pass = bake_mode in {BakeMode.DIFFUSE, BakeMode.EMIT}
    scene.render.bake.use_pass_direct = not flat_color_pass
    scene.render.bake.use_pass_indirect = not flat_color_pass
    scene.render.bake.use_pass_color = True

    # Blender 5.2's Combined pass-filter exposes exactly these surface contribution
    # toggles. Set all of them explicitly so Metallic/Coat/Transmission appearance is
    # deterministic and independent of the user's previous Render > Bake settings.
    scene.render.bake.use_pass_diffuse = True
    scene.render.bake.use_pass_glossy = True
    scene.render.bake.use_pass_transmission = True
    scene.render.bake.use_pass_emit = True

    scene.cycles.bake_type = bake_mode.value
    scene.cycles.samples = execution_settings.samples


@contextmanager
def preserve_bake_scene_state(scene: Any) -> Iterator[BakeSceneState]:
    """Restore all Blender 5.2 bake settings without hiding a primary error."""

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
                "Failed to restore bake Scene state while handling another exception"
            )


__all__ = [
    "BakeSceneState",
    "BakeSceneStateError",
    "ScenePropertyValue",
    "configure_scene_for_bake",
    "preserve_bake_scene_state",
]
