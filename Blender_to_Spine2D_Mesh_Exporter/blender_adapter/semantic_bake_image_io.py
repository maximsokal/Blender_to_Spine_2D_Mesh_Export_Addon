"""Blender image, UV, and timeline primitives used by semantic object baking."""

from __future__ import annotations

import logging
from typing import Any

from ..domain.baking import BakeExecutionSettings, BakePlan, TextureFormat
from ..infrastructure import AtomicOutputReservation
from .bake_execution_error import BakeExecutionError


logger = logging.getLogger(__name__)


def _activate_uv_layer(mesh: Any, layer_name: str) -> None:
    """Activate the requested UV map on a temporary bake target mesh."""

    layers = getattr(mesh, "uv_layers", None)
    layer = layers.get(layer_name) if layers is not None else None
    if layer is None:
        raise BakeExecutionError(
            f"Temporary target mesh is missing UV layer '{layer_name}'"
        )
    try:
        layers.active = layer
    except Exception:
        try:
            layers.active_index = next(
                index for index, candidate in enumerate(layers) if candidate == layer
            )
        except Exception as exc:
            raise BakeExecutionError(
                f"Unable to activate bake UV layer '{layer_name}'"
            ) from exc
    try:
        layer.active_render = True
    except Exception:
        logger.debug("UV active_render flag is not writable", exc_info=True)


def _create_bake_image(
    bpy_module: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    image_name: str,
    *,
    force_float_buffer: bool = False,
) -> Any:
    """Create one temporary Blender image matching the planned output contract."""

    float_buffer = force_float_buffer or (
        plan.settings.texture_format is TextureFormat.OPEN_EXR
    )
    try:
        image = bpy_module.data.images.new(
            name=f"__Spine2D_{image_name}",
            width=plan.settings.width,
            height=plan.settings.height,
            alpha=execution_settings.color_mode == "RGBA",
            float_buffer=float_buffer,
        )
        image.generated_color = execution_settings.generated_color
        image.file_format = plan.settings.texture_format.value
        return image
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to create bake image '{image_name}'"
        ) from exc


def _remove_image(bpy_module: Any, image: Any | None) -> None:
    """Best-effort removal for one temporary Blender image datablock."""

    if image is None:
        return
    try:
        bpy_module.data.images.remove(image)
    except Exception:
        logger.exception("Failed to remove temporary bake image")


def _set_timeline_frame(scene: Any, context: Any, frame: int | None) -> None:
    """Set one planned timeline frame and update the active view layer."""

    if frame is None:
        return
    try:
        scene.frame_set(frame)
        update = getattr(context.view_layer, "update", None)
        if callable(update):
            update()
    except Exception as exc:
        raise BakeExecutionError(f"Unable to set timeline frame {frame}") from exc


def _save_bake_image(
    image: Any,
    reservation: AtomicOutputReservation,
    plan: BakePlan,
) -> None:
    """Save one Blender image only to its already reserved staged path."""

    if not isinstance(reservation, AtomicOutputReservation):
        raise TypeError("reservation must be AtomicOutputReservation")
    staged_path = reservation.staged_path
    try:
        image.filepath_raw = str(staged_path)
        image.file_format = plan.settings.texture_format.value
        image.save()
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to save staged bake image '{staged_path}'"
        ) from exc
    if not staged_path.is_file():
        raise BakeExecutionError(
            "Blender reported a successful save but staged file is missing: "
            f"{staged_path}"
        )


__all__ = [
    "_activate_uv_layer",
    "_create_bake_image",
    "_remove_image",
    "_save_bake_image",
    "_set_timeline_frame",
]
