"""Regressions for strict Blender 5.2 semantic-bake Image ownership."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_execution_error import (
    BakeExecutionError,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.semantic_bake_image_io import (
    _configure_image_alpha_mode,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_adapter"


class _RejectingAlphaImage:
    @property
    def alpha_mode(self):
        return "PREMUL"

    @alpha_mode.setter
    def alpha_mode(self, _value):
        raise RuntimeError("read only")


def test_rgba_image_requires_straight_alpha_and_verifies_readback():
    image = SimpleNamespace(alpha_mode="PREMUL")

    _configure_image_alpha_mode(image, color_mode="RGBA")

    assert image.alpha_mode == "STRAIGHT"


def test_rgb_and_bw_images_do_not_require_alpha_mode():
    rgb_image = SimpleNamespace()
    bw_image = SimpleNamespace()

    _configure_image_alpha_mode(rgb_image, color_mode="RGB")
    _configure_image_alpha_mode(bw_image, color_mode="BW")

    assert not hasattr(rgb_image, "alpha_mode")
    assert not hasattr(bw_image, "alpha_mode")


def test_rgba_image_rejects_unwritable_alpha_mode():
    with pytest.raises(BakeExecutionError, match="alpha_mode='STRAIGHT'"):
        _configure_image_alpha_mode(
            _RejectingAlphaImage(),
            color_mode="RGBA",
        )


def test_image_contract_rejects_unknown_color_mode():
    with pytest.raises(BakeExecutionError, match="Unsupported Blender image color mode"):
        _configure_image_alpha_mode(
            SimpleNamespace(alpha_mode="STRAIGHT"),
            color_mode="CMYK",
        )


def test_semantic_execution_does_not_reconfigure_alpha_outside_image_owner():
    execution = (ADAPTER / "semantic_bake_execution.py").read_text(encoding="utf-8")
    image_io = (ADAPTER / "semantic_bake_image_io.py").read_text(encoding="utf-8")

    assert 'image.alpha_mode = "STRAIGHT"' not in execution
    assert 'final_image.alpha_mode = "STRAIGHT"' not in execution
    assert "_configure_image_alpha_mode(" in image_io
    assert 'image.alpha_mode = "STRAIGHT"' in image_io
    assert 'do_unlink=True' in image_io
