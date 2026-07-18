from math import nan
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_image import (
    read_staged_alpha_coverage,
    read_staged_alpha_mask,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.camera_projection_state import (
    CameraProjectionExecutionError,
)


class FakePixels:
    def __init__(self, values):
        self._values = tuple(values)

    def foreach_get(self, target):
        target[:] = self._values


class FakeImage:
    def __init__(self, width, height, alphas):
        self.size = (width, height)
        values = []
        for alpha in alphas:
            values.extend((0.2, 0.4, 0.6, alpha))
        self.pixels = FakePixels(values)


class FakeImages:
    def __init__(self, image):
        self.image = image
        self.loaded_paths = []
        self.removed = []

    def load(self, path, check_existing=False):
        self.loaded_paths.append((path, check_existing))
        return self.image

    def remove(self, image):
        self.removed.append(image)


def _bpy(image):
    images = FakeImages(image)
    return SimpleNamespace(data=SimpleNamespace(images=images)), images


def test_staged_alpha_decode_quantizes_and_clamps_to_coverage_bytes(tmp_path):
    bpy_module, images = _bpy(FakeImage(4, 1, (-0.2, 1.0 / 255.0, 0.5, 1.2)))

    coverage = read_staged_alpha_coverage(
        bpy_module,
        tmp_path / "frame.png",
        width=4,
        height=1,
    )

    assert coverage == bytes((0, 1, 128, 255))
    assert len(images.loaded_paths) == 1
    assert images.removed == [images.image]


def test_staged_alpha_decode_rejects_non_finite_values_and_still_cleans_image(tmp_path):
    bpy_module, images = _bpy(FakeImage(1, 1, (nan,)))

    with pytest.raises(CameraProjectionExecutionError, match="not finite"):
        read_staged_alpha_coverage(
            bpy_module,
            tmp_path / "frame.png",
            width=1,
            height=1,
        )

    assert images.removed == [images.image]


def test_binary_mask_wrapper_applies_normalized_threshold(tmp_path):
    bpy_module, _ = _bpy(FakeImage(4, 1, (0.0, 0.25, 0.5, 1.0)))

    mask = read_staged_alpha_mask(
        bpy_module,
        tmp_path / "frame.png",
        width=4,
        height=1,
        threshold=0.5,
    )

    assert mask == bytes((0, 0, 1, 1))


def test_zero_threshold_binary_wrapper_includes_zero_alpha(tmp_path):
    bpy_module, _ = _bpy(FakeImage(2, 1, (0.0, 0.0)))

    mask = read_staged_alpha_mask(
        bpy_module,
        tmp_path / "frame.png",
        width=2,
        height=1,
        threshold=0.0,
    )

    assert mask == bytes((1, 1))
