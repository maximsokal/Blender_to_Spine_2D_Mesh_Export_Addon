"""Blender 5.2 Image, UV, and timeline primitives for semantic baking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..domain.baking import BakeExecutionSettings, BakePlan, TextureFormat
from ..infrastructure import AtomicOutputReservation
from .bake_execution_error import BakeExecutionError


logger = logging.getLogger(__name__)


def _activate_uv_layer(mesh: Any, layer_name: str) -> None:
    """Activate one exact Blender 5.2 UV layer for editing and rendering."""

    if mesh is None:
        raise BakeExecutionError("mesh cannot be None")
    if not isinstance(layer_name, str) or not layer_name.strip():
        raise ValueError("layer_name must be a non-empty string")
    resolved_name = layer_name.strip()
    layers = getattr(mesh, "uv_layers", None)
    if layers is None:
        raise BakeExecutionError("Temporary target mesh has no UV layer collection")
    getter = getattr(layers, "get", None)
    if not callable(getter):
        raise BakeExecutionError("Blender 5.2 UVLoopLayers.get() is unavailable")
    try:
        layer = getter(resolved_name)
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to resolve bake UV layer '{resolved_name}'"
        ) from exc
    if layer is None:
        raise BakeExecutionError(
            f"Temporary target mesh is missing UV layer '{resolved_name}'"
        )

    try:
        layers.active = layer
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to activate bake UV layer '{resolved_name}'"
        ) from exc
    try:
        for candidate in layers:
            candidate.active_render = candidate is layer or candidate.name == resolved_name
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to assign render UV layer '{resolved_name}'"
        ) from exc

    active = getattr(layers, "active", None)
    if active is not layer and getattr(active, "name", None) != resolved_name:
        raise BakeExecutionError(
            f"Blender did not keep UV layer '{resolved_name}' active"
        )
    if not bool(getattr(layer, "active_render", False)):
        raise BakeExecutionError(
            f"Blender did not keep UV layer '{resolved_name}' active for rendering"
        )


def _create_bake_image(
    bpy_module: Any,
    plan: BakePlan,
    execution_settings: BakeExecutionSettings,
    image_name: str,
    *,
    force_float_buffer: bool = False,
) -> Any:
    """Create one temporary Blender 5.2 Image matching the output contract."""

    if bpy_module is None:
        raise BakeExecutionError("bpy_module cannot be None")
    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")
    if not isinstance(execution_settings, BakeExecutionSettings):
        raise TypeError("execution_settings must be BakeExecutionSettings")
    if not isinstance(image_name, str) or not image_name.strip():
        raise ValueError("image_name must be a non-empty string")
    if not isinstance(force_float_buffer, bool):
        raise TypeError("force_float_buffer must be bool")

    float_buffer = force_float_buffer or (
        plan.settings.texture_format is TextureFormat.OPEN_EXR
    )
    image = None
    try:
        image = bpy_module.data.images.new(
            name=f"__Spine2D_{image_name.strip()}",
            width=plan.settings.width,
            height=plan.settings.height,
            alpha=execution_settings.color_mode == "RGBA",
            float_buffer=float_buffer,
        )
        image.generated_color = execution_settings.generated_color
        image.file_format = plan.settings.texture_format.value
        return image
    except Exception as exc:
        if image is not None:
            try:
                bpy_module.data.images.remove(image, do_unlink=True)
            except Exception:
                logger.exception("Failed to remove partially configured bake image")
        raise BakeExecutionError(
            f"Unable to create bake image '{image_name}'"
        ) from exc


def _remove_image(bpy_module: Any, image: Any | None) -> None:
    """Best-effort unlink and removal for one temporary Blender Image."""

    if image is None:
        return
    try:
        bpy_module.data.images.remove(image, do_unlink=True)
    except Exception:
        logger.exception("Failed to remove temporary bake image")


def _set_timeline_frame(scene: Any, context: Any, frame: int | None) -> None:
    """Set one planned timeline frame and update the active View Layer."""

    if frame is None:
        return
    if not isinstance(frame, int) or isinstance(frame, bool):
        raise TypeError("frame must be int or None")
    try:
        scene.frame_set(frame)
        context.view_layer.update()
    except Exception as exc:
        raise BakeExecutionError(f"Unable to set timeline frame {frame}") from exc


def _save_bake_image(
    image: Any,
    reservation: AtomicOutputReservation,
    plan: BakePlan,
) -> None:
    """Save one Blender Image only to its reserved staged path."""

    if image is None:
        raise BakeExecutionError("image cannot be None")
    if not isinstance(reservation, AtomicOutputReservation):
        raise TypeError("reservation must be AtomicOutputReservation")
    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")

    staged_path = Path(reservation.staged_path)
    try:
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        image.filepath_raw = str(staged_path)
        image.file_format = plan.settings.texture_format.value
        image.save()
    except Exception as exc:
        raise BakeExecutionError(
            f"Unable to save staged bake image '{staged_path}'"
        ) from exc
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


__all__ = [
    "_activate_uv_layer",
    "_create_bake_image",
    "_remove_image",
    "_save_bake_image",
    "_set_timeline_frame",
]
