"""Decode staged B4 images, extract alpha masks, and rewrite stable crops."""

from __future__ import annotations

from array import array
import logging
from typing import Any

from ..domain.baking import CameraProjectionPlan, TextureFormat
from ..domain.baking.projection_layout import CameraProjectionLayout
from ..infrastructure import AtomicOutputReservation
from .camera_projection_state import CameraProjectionExecutionError

logger = logging.getLogger(__name__)


def remove_image(bpy_module: Any, image: Any | None) -> None:
    if image is None:
        return
    try:
        bpy_module.data.images.remove(image)
    except Exception:
        logger.exception("Failed to remove temporary projection image")


def read_image_pixels(image: Any, width: int, height: int) -> array:
    actual = tuple(int(value) for value in image.size[:2])
    if actual != (width, height):
        raise CameraProjectionExecutionError(
            f"Rendered image size {actual} does not match planned {(width, height)}"
        )
    pixels = array("f", [0.0]) * (width * height * 4)
    try:
        image.pixels.foreach_get(pixels)
    except Exception as exc:
        raise CameraProjectionExecutionError("Unable to read rendered image pixels") from exc
    return pixels


def read_staged_alpha_mask(
    bpy_module: Any,
    staged_path,
    *,
    width: int,
    height: int,
    threshold: float,
) -> bytes:
    image = None
    try:
        image = bpy_module.data.images.load(str(staged_path), check_existing=False)
        pixels = read_image_pixels(image, width, height)
        mask = bytearray(width * height)
        for pixel_index in range(width * height):
            if float(pixels[pixel_index * 4 + 3]) >= threshold:
                mask[pixel_index] = 1
        return bytes(mask)
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        raise CameraProjectionExecutionError(
            f"Unable to decode staged projection alpha '{staged_path}': {exc}"
        ) from exc
    finally:
        remove_image(bpy_module, image)


def crop_pixel_buffer(
    pixels: array,
    *,
    full_width: int,
    full_height: int,
    layout: CameraProjectionLayout,
) -> array:
    if len(pixels) != full_width * full_height * 4:
        raise CameraProjectionExecutionError("rendered pixel buffer has invalid length")
    crop = layout.crop
    result = array("f", [0.0]) * (layout.cropped_width * layout.cropped_height * 4)
    row_components = layout.cropped_width * 4
    for target_y, source_y in enumerate(range(crop.minimum_y, crop.maximum_y)):
        source_start = (source_y * full_width + crop.minimum_x) * 4
        target_start = target_y * row_components
        result[target_start : target_start + row_components] = pixels[
            source_start : source_start + row_components
        ]
    return result


def rewrite_staged_image_with_crop(
    bpy_module: Any,
    plan: CameraProjectionPlan,
    reservation: AtomicOutputReservation,
    layout: CameraProjectionLayout,
) -> None:
    loaded = None
    cropped = None
    try:
        loaded = bpy_module.data.images.load(
            str(reservation.staged_path),
            check_existing=False,
        )
        pixels = read_image_pixels(loaded, plan.settings.width, plan.settings.height)
        cropped_pixels = crop_pixel_buffer(
            pixels,
            full_width=plan.settings.width,
            full_height=plan.settings.height,
            layout=layout,
        )
        color_space = None
        try:
            color_space = str(loaded.colorspace_settings.name)
        except Exception:
            logger.debug("Unable to read source image color space", exc_info=True)
        remove_image(bpy_module, loaded)
        loaded = None

        cropped = bpy_module.data.images.new(
            name=f"__Spine2D_ProjectionCrop_{reservation.final_path.stem}",
            width=layout.cropped_width,
            height=layout.cropped_height,
            alpha=True,
            float_buffer=plan.settings.texture_format is TextureFormat.OPEN_EXR,
        )
        if color_space:
            try:
                cropped.colorspace_settings.name = color_space
            except Exception:
                logger.debug("Unable to restore cropped image color space", exc_info=True)
        cropped.pixels.foreach_set(cropped_pixels)
        cropped.update()
        cropped.file_format = plan.settings.texture_format.value
        cropped.filepath_raw = str(reservation.staged_path)
        cropped.save()
        if not reservation.staged_path.is_file() or reservation.staged_path.stat().st_size <= 0:
            raise CameraProjectionExecutionError(
                f"Cropped projection output is missing or empty: {reservation.staged_path}"
            )
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        raise CameraProjectionExecutionError(
            f"Unable to crop staged projection image '{reservation.staged_path}': {exc}"
        ) from exc
    finally:
        remove_image(bpy_module, loaded)
        remove_image(bpy_module, cropped)
