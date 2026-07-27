from __future__ import annotations

import inspect

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_image_io import (
    _flip_image_rows_for_spine,
    _save_bake_image,
)


_FLOAT32_ABSOLUTE_TOLERANCE = 1.0e-7


class _Pixels(list):
    def foreach_get(self, target) -> None:
        for index, value in enumerate(self):
            target[index] = value

    def foreach_set(self, values) -> None:
        self[:] = list(values)


class _Image(dict):
    def __init__(self, width: int, height: int, values) -> None:
        super().__init__()
        self.name = "DirectionalBake"
        self.size = (width, height)
        self.pixels = _Pixels(values)
        self.update_count = 0

    def update(self) -> None:
        self.update_count += 1


def _row(value: float, width: int = 2) -> list[float]:
    return [component for _ in range(width) for component in (value, 0.0, 0.0, 1.0)]


def _assert_float32_pixels_equal(actual, expected) -> None:
    """Compare Blender-style float32 pixel buffers without hiding row-order errors."""

    assert len(actual) == len(expected)
    assert list(actual) == pytest.approx(
        list(expected),
        rel=0.0,
        abs=_FLOAT32_ABSOLUTE_TOLERANCE,
    )


def test_spine_file_space_flip_reverses_complete_rgba_rows_exactly_once():
    image = _Image(
        2,
        3,
        _row(0.1) + _row(0.5) + _row(0.9),
    )
    expected = _row(0.9) + _row(0.5) + _row(0.1)

    assert _flip_image_rows_for_spine(image)
    _assert_float32_pixels_equal(image.pixels, expected)
    assert image.update_count == 1

    # A save retry on the same temporary Blender Image must not restore the old
    # orientation by flipping the rows a second time.
    assert not _flip_image_rows_for_spine(image)
    _assert_float32_pixels_equal(image.pixels, expected)
    assert image.update_count == 1


def test_save_owner_converts_rows_before_writing_the_staged_file():
    source = inspect.getsource(_save_bake_image)

    assert source.index("_flip_image_rows_for_spine(image)") < source.index(
        "image.save()"
    )
