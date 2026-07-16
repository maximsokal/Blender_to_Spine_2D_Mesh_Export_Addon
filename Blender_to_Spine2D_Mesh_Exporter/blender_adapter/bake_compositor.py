"""Decode, combine, and write Blender bake image pixels without source mutation."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import logging
from typing import Any, Tuple

from ..domain.baking import BakeCompositeMode, BakeCompositePlan

logger = logging.getLogger(__name__)


class BakeCompositeError(RuntimeError):
    """Raised when pass images cannot be combined into one deterministic output."""


@dataclass(frozen=True, slots=True)
class BakePixelBuffer:
    width: int
    height: int
    channels: int
    pixels: Any

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer")
        if self.channels != 4:
            raise ValueError("BakePixelBuffer currently requires four RGBA channels")
        expected = self.width * self.height * self.channels
        try:
            actual = len(self.pixels)
        except Exception as exc:
            raise TypeError("pixels must be a sized numeric buffer") from exc
        if actual != expected:
            raise ValueError(
                f"pixel buffer contains {actual} values; expected {expected}"
            )


def _image_dimensions(image: Any) -> tuple[int, int]:
    size = getattr(image, "size", None)
    try:
        width = int(size[0])
        height = int(size[1])
    except Exception as exc:
        raise BakeCompositeError("Unable to read Blender image dimensions") from exc
    if width <= 0 or height <= 0:
        raise BakeCompositeError(
            f"Blender image has invalid dimensions {width}x{height}"
        )
    return width, height


def read_bake_image_pixels(image: Any) -> BakePixelBuffer:
    """Copy one Blender image into a contiguous float32-compatible buffer."""

    if image is None:
        raise BakeCompositeError("image cannot be None")
    width, height = _image_dimensions(image)
    value_count = width * height * 4
    pixels = getattr(image, "pixels", None)
    if pixels is None:
        raise BakeCompositeError("Blender image has no pixel collection")

    try:
        import numpy as np

        data = np.empty(value_count, dtype=np.float32)
        foreach_get = getattr(pixels, "foreach_get", None)
        if callable(foreach_get):
            foreach_get(data)
        else:
            data[:] = pixels[:]
        return BakePixelBuffer(width=width, height=height, channels=4, pixels=data)
    except ImportError:
        logger.debug("NumPy is unavailable; using array('f') pixel fallback")
    except Exception as exc:
        raise BakeCompositeError("Unable to read Blender image pixels") from exc

    data = array("f", [0.0]) * value_count
    try:
        foreach_get = getattr(pixels, "foreach_get", None)
        if callable(foreach_get):
            foreach_get(data)
        else:
            data[:] = array("f", pixels[:])
    except Exception as exc:
        raise BakeCompositeError("Unable to read Blender image pixels") from exc
    return BakePixelBuffer(width=width, height=height, channels=4, pixels=data)


def _validate_compatible_buffers(
    buffers: Tuple[BakePixelBuffer, ...],
) -> tuple[int, int]:
    if not isinstance(buffers, tuple) or not buffers:
        raise BakeCompositeError("buffers must be a non-empty tuple")
    if not all(isinstance(item, BakePixelBuffer) for item in buffers):
        raise TypeError("buffers must contain BakePixelBuffer values")
    first = buffers[0]
    for index, item in enumerate(buffers[1:], start=1):
        if (
            item.width != first.width
            or item.height != first.height
            or item.channels != first.channels
        ):
            raise BakeCompositeError(
                f"Pass buffer {index} dimensions/channels do not match pass 0"
            )
    return first.width, first.height


def _resolved_color_indices(
    buffers: Tuple[BakePixelBuffer, ...],
    plan: BakeCompositePlan,
) -> Tuple[int, ...]:
    if plan.color_pass_indices:
        indices = plan.color_pass_indices
    elif plan.mode is BakeCompositeMode.ADD_RGB_MAX_ALPHA:
        indices = tuple(range(len(buffers)))
    else:
        indices = ()
    if indices and max(indices) >= len(buffers):
        raise BakeCompositeError("Composite color pass index is outside buffers")
    return indices


def _compose_with_numpy(
    buffers: Tuple[BakePixelBuffer, ...],
    plan: BakeCompositePlan,
) -> BakePixelBuffer | None:
    try:
        import numpy as np
    except ImportError:
        return None

    try:
        width, height = _validate_compatible_buffers(buffers)
        result = np.zeros(width * height * 4, dtype=np.float32)
        color_indices = _resolved_color_indices(buffers, plan)
        for index in color_indices:
            values = np.asarray(buffers[index].pixels, dtype=np.float32)
            result[0::4] += values[0::4]
            result[1::4] += values[1::4]
            result[2::4] += values[2::4]

        if plan.mode is BakeCompositeMode.ADD_RGB_MAX_ALPHA:
            for item in buffers:
                values = np.asarray(item.pixels, dtype=np.float32)
                np.maximum(result[3::4], values[3::4], out=result[3::4])
        elif plan.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA:
            assert plan.alpha_pass_index is not None
            if plan.alpha_pass_index >= len(buffers):
                raise BakeCompositeError("Composite alpha pass index is outside buffers")
            alpha_values = np.asarray(
                buffers[plan.alpha_pass_index].pixels,
                dtype=np.float32,
            )
            result[3::4] = alpha_values[0::4]
        else:
            raise BakeCompositeError(f"Unsupported composite mode: {plan.mode.value}")

        if plan.clamp_rgb:
            np.clip(result[0::4], 0.0, 1.0, out=result[0::4])
            np.clip(result[1::4], 0.0, 1.0, out=result[1::4])
            np.clip(result[2::4], 0.0, 1.0, out=result[2::4])
        np.clip(result[3::4], 0.0, 1.0, out=result[3::4])
        return BakePixelBuffer(width=width, height=height, channels=4, pixels=result)
    except BakeCompositeError:
        raise
    except Exception as exc:
        raise BakeCompositeError("NumPy bake-pass composition failed") from exc


def _compose_with_array(
    buffers: Tuple[BakePixelBuffer, ...],
    plan: BakeCompositePlan,
) -> BakePixelBuffer:
    width, height = _validate_compatible_buffers(buffers)
    value_count = width * height * 4
    result = array("f", [0.0]) * value_count
    color_indices = _resolved_color_indices(buffers, plan)

    try:
        for index in color_indices:
            values = buffers[index].pixels
            for offset in range(0, value_count, 4):
                result[offset] += float(values[offset])
                result[offset + 1] += float(values[offset + 1])
                result[offset + 2] += float(values[offset + 2])

        if plan.mode is BakeCompositeMode.ADD_RGB_MAX_ALPHA:
            for item in buffers:
                values = item.pixels
                for offset in range(0, value_count, 4):
                    result[offset + 3] = max(
                        result[offset + 3],
                        float(values[offset + 3]),
                    )
        elif plan.mode is BakeCompositeMode.ADD_RGB_REPLACE_ALPHA:
            assert plan.alpha_pass_index is not None
            if plan.alpha_pass_index >= len(buffers):
                raise BakeCompositeError("Composite alpha pass index is outside buffers")
            values = buffers[plan.alpha_pass_index].pixels
            for offset in range(0, value_count, 4):
                result[offset + 3] = float(values[offset])
        else:
            raise BakeCompositeError(f"Unsupported composite mode: {plan.mode.value}")

        for offset in range(0, value_count, 4):
            if plan.clamp_rgb:
                result[offset] = min(1.0, max(0.0, result[offset]))
                result[offset + 1] = min(1.0, max(0.0, result[offset + 1]))
                result[offset + 2] = min(1.0, max(0.0, result[offset + 2]))
            result[offset + 3] = min(1.0, max(0.0, result[offset + 3]))
    except BakeCompositeError:
        raise
    except Exception as exc:
        raise BakeCompositeError("Fallback bake-pass composition failed") from exc
    return BakePixelBuffer(width=width, height=height, channels=4, pixels=result)


def compose_bake_passes(
    buffers: Tuple[BakePixelBuffer, ...],
    plan: BakeCompositePlan,
) -> BakePixelBuffer:
    if not isinstance(plan, BakeCompositePlan):
        raise TypeError("plan must be BakeCompositePlan")
    _validate_compatible_buffers(buffers)
    if plan.mode is BakeCompositeMode.SINGLE:
        if len(buffers) != 1:
            raise BakeCompositeError("SINGLE composition requires exactly one buffer")
        return buffers[0]

    numpy_result = _compose_with_numpy(buffers, plan)
    return numpy_result if numpy_result is not None else _compose_with_array(buffers, plan)


def write_bake_image_pixels(image: Any, buffer: BakePixelBuffer) -> None:
    if image is None:
        raise BakeCompositeError("image cannot be None")
    if not isinstance(buffer, BakePixelBuffer):
        raise TypeError("buffer must be BakePixelBuffer")
    width, height = _image_dimensions(image)
    if (width, height) != (buffer.width, buffer.height):
        raise BakeCompositeError(
            f"Target image is {width}x{height}, buffer is {buffer.width}x{buffer.height}"
        )
    pixels = getattr(image, "pixels", None)
    if pixels is None:
        raise BakeCompositeError("Blender image has no pixel collection")
    try:
        foreach_set = getattr(pixels, "foreach_set", None)
        if callable(foreach_set):
            foreach_set(buffer.pixels)
        else:
            pixels[:] = buffer.pixels
        update = getattr(image, "update", None)
        if callable(update):
            update()
    except Exception as exc:
        raise BakeCompositeError("Unable to write composed pixels to Blender image") from exc
