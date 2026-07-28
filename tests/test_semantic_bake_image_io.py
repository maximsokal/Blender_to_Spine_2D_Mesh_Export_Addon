from __future__ import annotations

import inspect

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_execution_error import (
    BakeExecutionError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_image_io import (
    _activate_uv_layer,
    _flip_image_rows_for_spine,
    _save_bake_image,
)


_FLOAT32_ABSOLUTE_TOLERANCE = 1.0e-7


class _UvLayer:
    def __init__(self, name: str, *, active_render: bool = False) -> None:
        self.name = name
        self.active_render = active_render


class _UvLayers:
    def __init__(self, *layers: _UvLayer) -> None:
        self._layers = list(layers)
        self.active = layers[0] if layers else None

    def __iter__(self):
        return iter(self._layers)

    def get(self, name: str):
        return next((layer for layer in self._layers if layer.name == name), None)


class _Mesh:
    def __init__(self, *layers: _UvLayer) -> None:
        self.uv_layers = _UvLayers(*layers)


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


def test_bake_destination_and_source_render_uv_roles_remain_independent():
    source = _UvLayer("UVMap", active_render=True)
    destination = _UvLayer("SpineBakeUV", active_render=False)
    mesh = _Mesh(source, destination)

    _activate_uv_layer(
        mesh,
        "SpineBakeUV",
        render_layer_name="UVMap",
    )

    assert mesh.uv_layers.active is destination
    assert source.active_render is True
    assert destination.active_render is False


def test_uv_role_activation_can_reuse_one_layer_when_source_and_destination_match():
    shared = _UvLayer("UVMap", active_render=True)
    mesh = _Mesh(shared)

    _activate_uv_layer(
        mesh,
        "UVMap",
        render_layer_name="UVMap",
    )

    assert mesh.uv_layers.active is shared
    assert shared.active_render is True


def test_uv_role_activation_rejects_missing_source_render_layer():
    mesh = _Mesh(_UvLayer("SpineBakeUV"))

    with pytest.raises(BakeExecutionError, match="missing source render UV layer"):
        _activate_uv_layer(
            mesh,
            "SpineBakeUV",
            render_layer_name="UVMap",
        )


def test_uv_role_activation_uses_existing_unique_render_role_when_not_explicit():
    source = _UvLayer("UVMap", active_render=True)
    destination = _UvLayer("SpineBakeUV", active_render=False)
    mesh = _Mesh(source, destination)

    _activate_uv_layer(mesh, "SpineBakeUV")

    assert mesh.uv_layers.active is destination
    assert source.active_render is True
    assert destination.active_render is False


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
