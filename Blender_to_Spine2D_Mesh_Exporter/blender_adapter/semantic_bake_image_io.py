"""Blender 5.2 Image, UV, and timeline primitives for semantic baking."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any

from ..domain.baking import BakeExecutionSettings, BakePlan, TextureFormat
from ..domain.baking.texture_format_policy import resolve_texture_color_mode
from ..infrastructure import AtomicOutputReservation
from .bake_execution_error import BakeExecutionError


logger = logging.getLogger(__name__)
_PIXEL_CHANNEL_COUNT = 4
_SPINE_FILE_SPACE_FLIP_MARKER = "spine2d_spine_file_space_rows_flipped_v1"


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


def _configure_image_alpha_mode(image: Any, *, color_mode: str) -> None:
    """Require straight alpha for every RGBA semantic-bake image."""

    if image is None:
        raise BakeExecutionError("image cannot be None")
    if not isinstance(color_mode, str) or not color_mode.strip():
        raise ValueError("color_mode must be a non-empty string")
    resolved_mode = color_mode.strip().upper()
    if resolved_mode not in {"RGB", "RGBA", "BW"}:
        raise BakeExecutionError(f"Unsupported Blender image color mode: {color_mode!r}")
    if resolved_mode != "RGBA":
        return
    try:
        image.alpha_mode = "STRAIGHT"
    except Exception as exc:
        raise BakeExecutionError(
            "Unable to configure Blender 5.2 bake image alpha_mode='STRAIGHT'"
        ) from exc
    if str(getattr(image, "alpha_mode", "") or "").upper() != "STRAIGHT":
        raise BakeExecutionError(
            "Blender did not keep bake image alpha_mode='STRAIGHT'"
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
    color_mode = resolve_texture_color_mode(
        plan.settings.texture_format,
        execution_settings.color_mode,
    )
    image = None
    try:
        image = bpy_module.data.images.new(
            name=f"__Spine2D_{image_name.strip()}",
            width=plan.settings.width,
            height=plan.settings.height,
            alpha=color_mode == "RGBA",
            float_buffer=float_buffer,
        )
        image.generated_color = execution_settings.generated_color
        image.file_format = plan.settings.texture_format.value
        _configure_image_alpha_mode(
            image,
            color_mode=color_mode,
        )
        return image
    except Exception as exc:
        if image is not None:
            try:
                bpy_module.data.images.remove(image, do_unlink=True)
            except Exception:
                logger.exception("Failed to remove partially configured bake image")
        if isinstance(exc, BakeExecutionError):
            raise
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


def _image_dimensions(image: Any) -> tuple[int, int]:
    """Read and validate the immutable dimensions of one Blender image."""

    if image is None:
        raise BakeExecutionError("image cannot be None")
    size = getattr(image, "size", None)
    try:
        width = int(size[0])
        height = int(size[1])
    except (TypeError, ValueError, OverflowError, IndexError) as exc:
        raise BakeExecutionError("Bake image exposes an invalid size") from exc
    if width <= 0 or height <= 0:
        raise BakeExecutionError(
            f"Bake image dimensions must be positive, got {(width, height)}"
        )
    return width, height


def _read_image_rgba_pixels(image: Any, width: int, height: int) -> array:
    """Read one complete bottom-up Blender RGBA buffer into compact float storage."""

    pixels = getattr(image, "pixels", None)
    if pixels is None:
        raise BakeExecutionError("Bake image has no pixel collection")
    expected = width * height * _PIXEL_CHANNEL_COUNT
    values = array("f", [0.0]) * expected
    foreach_get = getattr(pixels, "foreach_get", None)
    try:
        if callable(foreach_get):
            foreach_get(values)
        else:
            values = array("f", (float(value) for value in pixels))
            if len(values) != expected:
                raise BakeExecutionError(
                    f"Bake image contains {len(values)} values; expected {expected}"
                )
    except BakeExecutionError:
        raise
    except Exception as exc:
        raise BakeExecutionError("Unable to read bake image pixels") from exc
    return values


def _write_image_rgba_pixels(image: Any, values: Sequence[float]) -> None:
    """Replace one complete Blender RGBA buffer and notify the image datablock."""

    pixels = getattr(image, "pixels", None)
    if pixels is None:
        raise BakeExecutionError("Bake image has no pixel collection")
    foreach_set = getattr(pixels, "foreach_set", None)
    try:
        if callable(foreach_set):
            foreach_set(values)
        else:
            pixels[:] = values
    except Exception as exc:
        raise BakeExecutionError("Unable to write flipped bake image pixels") from exc

    update = getattr(image, "update", None)
    if not callable(update):
        raise BakeExecutionError("Bake image update() is unavailable")
    try:
        update()
    except Exception as exc:
        raise BakeExecutionError("Unable to update flipped bake image") from exc


def _spine_file_space_flip_applied(image: Any) -> bool:
    """Return whether this temporary image was already converted for Spine file-space."""

    getter = getattr(image, "get", None)
    if callable(getter):
        try:
            return bool(getter(_SPINE_FILE_SPACE_FLIP_MARKER, False))
        except Exception as exc:
            raise BakeExecutionError(
                "Unable to read the Spine file-space image marker"
            ) from exc
    return bool(getattr(image, _SPINE_FILE_SPACE_FLIP_MARKER, False))


def _mark_spine_file_space_flip_applied(image: Any) -> None:
    """Persist an idempotence marker on one temporary image datablock."""

    try:
        image[_SPINE_FILE_SPACE_FLIP_MARKER] = True
        return
    except Exception:
        logger.debug(
            "Image ID properties are unavailable for the Spine file-space marker",
            exc_info=True,
        )
    try:
        setattr(image, _SPINE_FILE_SPACE_FLIP_MARKER, True)
    except Exception as exc:
        raise BakeExecutionError(
            "Unable to mark the Spine file-space image conversion"
        ) from exc


def _flip_image_rows_for_spine(image: Any) -> bool:
    """Convert Blender's bottom-up bake buffer to Spine PNG top-down file-space.

    Blender UV coordinates use ``v=0`` at the bottom. Spine mesh UV values are retained
    numerically, while the exported image is consumed in top-down file-space. Reversing
    the rows once restores the legacy exporter contract without mutating source UVs.
    The temporary Image marker makes retries idempotent and prevents a double flip.
    """

    if _spine_file_space_flip_applied(image):
        return False

    width, height = _image_dimensions(image)
    values = _read_image_rgba_pixels(image, width, height)
    row_stride = width * _PIXEL_CHANNEL_COUNT
    for first_row in range(height // 2):
        second_row = height - 1 - first_row
        first_start = first_row * row_stride
        second_start = second_row * row_stride
        first_values = values[first_start : first_start + row_stride]
        values[first_start : first_start + row_stride] = values[
            second_start : second_start + row_stride
        ]
        values[second_start : second_start + row_stride] = first_values

    _write_image_rgba_pixels(image, values)
    _mark_spine_file_space_flip_applied(image)
    logger.debug(
        "Converted semantic bake image '%s' to Spine file-space (%dx%d)",
        str(getattr(image, "name", "<unnamed>")),
        width,
        height,
    )
    return True


def _save_bake_image(
    image: Any,
    reservation: AtomicOutputReservation,
    plan: BakePlan,
) -> None:
    """Save one Spine-oriented Blender Image only to its reserved staged path."""

    if image is None:
        raise BakeExecutionError("image cannot be None")
    if not isinstance(reservation, AtomicOutputReservation):
        raise TypeError("reservation must be AtomicOutputReservation")
    if not isinstance(plan, BakePlan):
        raise TypeError("plan must be BakePlan")

    staged_path = Path(reservation.staged_path)
    try:
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        _flip_image_rows_for_spine(image)
        image.filepath_raw = str(staged_path)
        image.file_format = plan.settings.texture_format.value
        image.save()
    except BakeExecutionError:
        raise
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
    "_configure_image_alpha_mode",
    "_create_bake_image",
    "_flip_image_rows_for_spine",
    "_remove_image",
    "_save_bake_image",
    "_set_timeline_frame",
]
