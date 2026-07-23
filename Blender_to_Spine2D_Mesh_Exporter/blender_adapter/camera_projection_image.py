"""Decode, crop, and rewrite Blender 5.2 projection images."""

from __future__ import annotations

from array import array
import logging
from math import isfinite
from typing import Any

from ..domain.baking import (
    CameraProjectionPlan,
    GroupedCameraProjectionPlan,
    ResolvedProjectionOutputPolicy,
    convert_rgba_alpha_representation,
)
from ..domain.baking.projection_layout import CameraProjectionLayout
from ..infrastructure import AtomicOutputReservation
from .camera_projection_error import CameraProjectionExecutionError


logger = logging.getLogger(__name__)


def remove_image(bpy_module: Any, image: Any | None) -> None:
    """Best-effort unlink and remove one temporary Blender Image."""

    if image is None:
        return
    try:
        bpy_module.data.images.remove(image, do_unlink=True)
    except Exception:
        logger.exception("Failed to remove temporary projection image")


def read_image_pixels(image: Any, width: int, height: int) -> array:
    if image is None:
        raise CameraProjectionExecutionError("image cannot be None")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive integer")

    try:
        actual = tuple(int(value) for value in image.size[:2])
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to inspect rendered image dimensions"
        ) from exc
    if actual != (width, height):
        raise CameraProjectionExecutionError(
            f"Rendered image size {actual} does not match planned {(width, height)}"
        )
    pixels = array("f", [0.0]) * (width * height * 4)
    try:
        image.pixels.foreach_get(pixels)
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to read rendered image pixels"
        ) from exc
    return pixels


def _alpha_coverage_byte(value: float, *, pixel_index: int) -> int:
    resolved = float(value)
    if not isfinite(resolved):
        raise CameraProjectionExecutionError(
            f"Rendered alpha at pixel {pixel_index} is not finite: {resolved!r}"
        )
    return int(round(max(0.0, min(1.0, resolved)) * 255.0))


def read_staged_alpha_coverage(
    bpy_module: Any,
    staged_path,
    *,
    width: int,
    height: int,
) -> bytes:
    """Decode one staged render into deterministic 8-bit alpha coverage."""

    image = None
    try:
        image = bpy_module.data.images.load(
            str(staged_path),
            check_existing=False,
        )
        pixels = read_image_pixels(image, width, height)
        coverage = bytearray(width * height)
        for pixel_index in range(width * height):
            coverage[pixel_index] = _alpha_coverage_byte(
                pixels[pixel_index * 4 + 3],
                pixel_index=pixel_index,
            )
        return bytes(coverage)
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
    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("layout must be CameraProjectionLayout")
    if len(pixels) != full_width * full_height * 4:
        raise CameraProjectionExecutionError(
            "rendered pixel buffer has invalid length"
        )

    crop = layout.crop
    result = array("f", [0.0]) * (
        layout.cropped_width * layout.cropped_height * 4
    )
    row_components = layout.cropped_width * 4
    for target_y, source_y in enumerate(
        range(crop.minimum_y, crop.maximum_y)
    ):
        source_start = (
            source_y * full_width + crop.minimum_x
        ) * 4
        target_start = target_y * row_components
        result[target_start : target_start + row_components] = pixels[
            source_start : source_start + row_components
        ]
    return result


def _image_color_space_name(image: Any) -> str:
    try:
        value = str(image.colorspace_settings.name or "").strip()
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to read staged image color space"
        ) from exc
    if not value:
        raise CameraProjectionExecutionError(
            "Staged image color space name is empty"
        )
    return value


def _image_alpha_mode(image: Any) -> str:
    try:
        value = str(image.alpha_mode or "").strip().upper()
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to read staged image alpha mode"
        ) from exc
    if not value:
        raise CameraProjectionExecutionError(
            "Staged image alpha mode is empty"
        )
    return value


def rewrite_staged_image_with_crop(
    bpy_module: Any,
    plan: CameraProjectionPlan | GroupedCameraProjectionPlan,
    reservation: AtomicOutputReservation,
    layout: CameraProjectionLayout,
    output_policy: ResolvedProjectionOutputPolicy,
) -> None:
    """Crop one staged frame with explicit dynamic-range and alpha semantics."""

    if not isinstance(
        plan,
        (CameraProjectionPlan, GroupedCameraProjectionPlan),
    ):
        raise TypeError(
            "plan must be CameraProjectionPlan or GroupedCameraProjectionPlan"
        )
    if not isinstance(reservation, AtomicOutputReservation):
        raise TypeError("reservation must be AtomicOutputReservation")
    if not isinstance(layout, CameraProjectionLayout):
        raise TypeError("layout must be CameraProjectionLayout")
    if not isinstance(
        output_policy,
        ResolvedProjectionOutputPolicy,
    ):
        raise TypeError(
            "output_policy must be ResolvedProjectionOutputPolicy"
        )
    if output_policy.texture_format is not plan.settings.texture_format:
        raise CameraProjectionExecutionError(
            "resolved output policy texture format does not match camera plan"
        )

    loaded = None
    cropped = None
    try:
        loaded = bpy_module.data.images.load(
            str(reservation.staged_path),
            check_existing=False,
        )
        pixels = read_image_pixels(
            loaded,
            plan.settings.width,
            plan.settings.height,
        )
        source_alpha_mode = _image_alpha_mode(loaded)
        source_color_space = _image_color_space_name(loaded)
        cropped_pixels = crop_pixel_buffer(
            pixels,
            full_width=plan.settings.width,
            full_height=plan.settings.height,
            layout=layout,
        )
        cropped_pixels = convert_rgba_alpha_representation(
            cropped_pixels,
            source_alpha_mode=source_alpha_mode,
            target=output_policy.alpha_representation,
        )

        remove_image(bpy_module, loaded)
        loaded = None

        cropped = bpy_module.data.images.new(
            name=(
                "__Spine2D_ProjectionCrop_"
                f"{reservation.final_path.stem}"
            ),
            width=layout.cropped_width,
            height=layout.cropped_height,
            alpha=True,
            float_buffer=output_policy.float_buffer,
        )
        try:
            cropped.colorspace_settings.name = source_color_space
        except Exception as exc:
            raise CameraProjectionExecutionError(
                f"Unable to restore cropped image color space '{source_color_space}'"
            ) from exc
        try:
            cropped.alpha_mode = output_policy.blender_alpha_mode
        except Exception as exc:
            raise CameraProjectionExecutionError(
                "Unable to apply resolved Blender alpha mode "
                f"'{output_policy.blender_alpha_mode}'"
            ) from exc

        try:
            cropped.pixels.foreach_set(cropped_pixels)
            cropped.update()
            cropped.file_format = plan.settings.texture_format.value
            cropped.filepath_raw = str(reservation.staged_path)
            cropped.save()
        except Exception as exc:
            raise CameraProjectionExecutionError(
                "Unable to write cropped projection pixels or save the staged image"
            ) from exc

        try:
            exists = reservation.staged_path.is_file()
            size = reservation.staged_path.stat().st_size if exists else 0
        except Exception as exc:
            raise CameraProjectionExecutionError(
                "Unable to inspect cropped staged projection output"
            ) from exc
        if not exists or size <= 0:
            raise CameraProjectionExecutionError(
                "Cropped projection output is missing or empty: "
                f"{reservation.staged_path}"
            )
    except CameraProjectionExecutionError:
        raise
    except Exception as exc:
        raise CameraProjectionExecutionError(
            "Unable to crop staged projection image "
            f"'{reservation.staged_path}': {exc}"
        ) from exc
    finally:
        remove_image(bpy_module, loaded)
        remove_image(bpy_module, cropped)


__all__ = [
    "crop_pixel_buffer",
    "read_image_pixels",
    "read_staged_alpha_coverage",
    "remove_image",
    "rewrite_staged_image_with_crop",
]
