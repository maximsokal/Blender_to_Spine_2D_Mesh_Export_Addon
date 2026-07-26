from __future__ import annotations

import inspect

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_image_io import (
    _flip_image_rows_for_spine,
    _save_bake_image,
)


class _Pixels(list):
    def foreach_get(self, target) -> None:
        target[:] = self

    def foreach_set(self, values) -> None:
        self[:] = values


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


def test_spine_file_space_flip_reverses_complete_rgba_rows_exactly_once():
    image = _Image(
        2,
        3,
        _row(0.1) + _row(0.5) + _row(0.9),
    )

    assert _flip_image_rows_for_spine(image)
    assert list(image.pixels) == _row(0.9) + _row(0.5) + _row(0.1)
    assert image.update_count == 1

    # A save retry on the same temporary Blender Image must not restore the old
    # orientation by flipping the rows a second time.
    assert not _flip_image_rows_for_spine(image)
    assert list(image.pixels) == _row(0.9) + _row(0.5) + _row(0.1)
    assert image.update_count == 1


def test_save_owner_converts_rows_before_writing_the_staged_file():
    source = inspect.getsource(_save_bake_image)

    assert source.index("_flip_image_rows_for_spine(image)") < source.index(
        "image.save()"
    )
