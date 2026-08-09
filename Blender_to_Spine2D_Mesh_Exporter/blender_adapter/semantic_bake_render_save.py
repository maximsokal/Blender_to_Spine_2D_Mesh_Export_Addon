"""Save semantic bake images according to texture-data vs render-appearance semantics.

Blender bake buffers live in scene-linear space. Local surface/emission passes and
camera-context surface-color EMIT passes are texture data and intentionally retain the
historical ``Image.save()`` path. Real Scene/Camera COMBINED passes are render appearance:
display formats such as PNG must be written with Blender's render color management so
the Scene view transform, exposure, gamma, and display transform are applied instead of
clipping HDR scene-linear values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..domain.baking import (
    BakeEvaluationScope,
    BakeMode,
    BakePlan,
    TextureFormat,
)
from ..infrastructure import AtomicOutputReservation
from .bake_execution_error import BakeExecutionError
from .semantic_bake_image_io import (
    _flip_image_rows_for_spine,
    _save_bake_image as _save_texture_data_image,
)


logger = logging.getLogger(__name__)


def _scene_output_file_format(scene: Any) -> str:
    """Return the active Blender Scene output file format or fail explicitly."""

    if scene is None:
        raise BakeExecutionError("render-managed semantic bake requires a Scene")
    try:
        value = str(scene.render.image_settings.file_format or "").strip().upper()
    except Exception as exc:
        raise BakeExecutionError(
            "Unable to read Scene render image file format for render-managed bake"
        ) from exc
    if not value:
        raise BakeExecutionError("Scene render image file format is empty")
    return value


def _render_color_management_summary(scene: Any) -> tuple[str, str, float, float]:
    """Return diagnostic Scene view settings without mutating color management."""

    if scene is None:
        raise BakeExecutionError("scene cannot be None")
    view_settings = getattr(scene, "view_settings", None)
    if view_settings is None:
        raise BakeExecutionError("Scene has no view_settings for render color management")
    try:
        return (
            str(getattr(view_settings, "view_transform", "") or ""),
            str(getattr(view_settings, "look", "") or ""),
            float(getattr(view_settings, "exposure", 0.0)),
            float(getattr(view_settings, "gamma", 1.0)),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise BakeExecutionError("Scene exposes invalid render color-management values") from exc


def _save_render_managed_image(
    image: Any,
    staged_path: Path,
    *,
    texture_format: TextureFormat,
    scene: Any,
) -> None:
    """Write one scene-linear render-appearance buffer through Blender Save as Render."""

    if image is None:
        raise BakeExecutionError("image cannot be None")
    if not isinstance(staged_path, Path):
        raise TypeError("staged_path must be pathlib.Path")
    if not isinstance(texture_format, TextureFormat):
        raise TypeError("texture_format must be TextureFormat")

    expected_format = texture_format.value.strip().upper()
    actual_format = _scene_output_file_format(scene)
    if actual_format != expected_format:
        raise BakeExecutionError(
            "Scene render output format differs from the semantic bake plan before "
            f"render-managed save: expected={expected_format!r}, actual={actual_format!r}"
        )

    save_render = getattr(image, "save_render", None)
    if not callable(save_render):
        raise BakeExecutionError("Blender Image.save_render() is unavailable")

    try:
        image.file_format = expected_format
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to configure render-managed bake image format {expected_format!r}"
        ) from exc

    view_transform, look, exposure, gamma = _render_color_management_summary(scene)
    logger.info(
        "Saving render-appearance semantic bake with render color management: "
        "format=%s view_transform=%s look=%s exposure=%s gamma=%s path=%s",
        expected_format,
        view_transform,
        look,
        exposure,
        gamma,
        staged_path,
    )
    try:
        save_render(str(staged_path), scene=scene)
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to save render-managed semantic bake image '{staged_path}'"
        ) from exc


def _require_written_file(staged_path: Path) -> None:
    """Require one non-empty staged file after Blender reports a successful save."""

    if not isinstance(staged_path, Path):
        raise TypeError("staged_path must be pathlib.Path")
    try:
        exists = staged_path.is_file()
        size = staged_path.stat().st_size if exists else 0
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to inspect staged bake image '{staged_path}'"
        ) from exc
    if not exists or size <= 0:
        raise BakeExecutionError(
            "Blender reported a successful save but staged file is missing or empty: "
            f"{staged_path}"
        )


def _requires_render_color_management(plan: BakePlan) -> bool:
    """Return whether the final buffer represents lit Scene/Camera render appearance.

    Evaluation scope alone is insufficient: Normal/UV may evaluate a Base Color graph in
    CAMERA context while still producing straight texture data through EMIT. Render color
    management is required only for a true COMBINED pass whose Scene/Camera contributions
    are intentionally part of the exported appearance.
    """

    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")
    return any(
        pass_plan.bake_mode is BakeMode.COMBINED
        and pass_plan.evaluation_scope
        in {BakeEvaluationScope.SCENE, BakeEvaluationScope.CAMERA}
        for pass_plan in plan.passes
    )


def save_semantic_bake_image(
    image: Any,
    reservation: AtomicOutputReservation,
    plan: BakePlan,
    *,
    scene: Any,
) -> None:
    """Save one staged bake without conflating texture data and rendered appearance.

    Contextual evaluation and rendered appearance are deliberately separate. Local passes
    and CAMERA-scoped EMIT surface-color passes use the texture-data save path. Only a
    real Scene/Camera COMBINED pass uses Blender ``Image.save_render``.
    """

    if image is None:
        raise BakeExecutionError("image cannot be None")
    if not isinstance(reservation, AtomicOutputReservation):
        raise TypeError("reservation must be AtomicOutputReservation")
    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")

    if not _requires_render_color_management(plan):
        _save_texture_data_image(image, reservation, plan)
        return

    staged_path = Path(reservation.staged_path)
    try:
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        _flip_image_rows_for_spine(image)
        _save_render_managed_image(
            image,
            staged_path,
            texture_format=plan.settings.texture_format,
            scene=scene,
        )
    except BakeExecutionError:
        raise
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to save render-appearance staged bake image '{staged_path}'"
        ) from exc

    _require_written_file(staged_path)


__all__ = [
    "_requires_render_color_management",
    "_save_render_managed_image",
    "save_semantic_bake_image",
]
