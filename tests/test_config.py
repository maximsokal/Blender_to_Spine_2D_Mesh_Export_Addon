"""Current logging/config behavior plus Blender 5.2 Scene RNA ownership."""

from __future__ import annotations

import logging
from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter import config


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def test_calc_uniform_scale_modes_and_invalid_input():
    assert config.calc_uniform_scale(1024, 512) == 768.0
    assert config.calc_uniform_scale(1024, 512, mode="max") == 1024.0
    assert config.calc_uniform_scale(1024, 512, mode="min") == 512.0
    assert config.calc_uniform_scale("invalid", None) == 1.0


def test_short_name_formatter_preserves_logger_identity():
    formatter = config.ShortNameFormatter("%(short_name)s|%(message)s")
    record = logging.LogRecord(
        name=f"{config.PACKAGE_LOGGER_ROOT}.blender_adapter.mesh_reader",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="message",
        args=(),
        exc_info=None,
    )

    output = formatter.format(record)

    assert output == "blender_adapter.mesh_reader|message"
    assert record.name == f"{config.PACKAGE_LOGGER_ROOT}.blender_adapter.mesh_reader"


def test_scene_rna_uses_standard_blender_52_storage():
    source = (PACKAGE / "blender_adapter" / "scene_properties.py").read_text(
        encoding="utf-8"
    )

    assert '"spine2d_frames_for_render"' in source
    assert '"spine2d_texture_size"' in source
    assert "get=get_frames_for_render" not in source
    assert "set=set_frames_for_render" not in source
    assert "get=get_texture_size" not in source
    assert "set=set_texture_size" not in source
    assert 'self["spine2d_' not in source
    assert ".get(\"spine2d_" not in source


def test_scene_texture_size_property_has_blender_side_bounds():
    source = (PACKAGE / "blender_adapter" / "scene_properties.py").read_text(
        encoding="utf-8"
    )

    texture_section = source.split('"spine2d_texture_size"', 1)[1]
    assert "default=1024" in texture_section
    assert "min=64" in texture_section
    assert "max=4096" in texture_section
    assert "update=_update_texture_size" in texture_section
