"""Logging/config behavior plus Blender 5.2 Scene RNA ownership."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter import config


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


class _ModuleCollection(list):
    def clear(self) -> None:
        super().clear()

    def add(self):
        item = SimpleNamespace(name="", level="ERROR")
        self.append(item)
        return item


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


def test_get_default_output_dir_prefers_saved_blend_directory(monkeypatch):
    monkeypatch.setattr(config.bpy.data, "filepath", "/projects/hero/hero.blend")

    assert config.get_default_output_dir() == "/projects/hero"


def test_get_default_output_dir_falls_back_to_user_home(monkeypatch):
    monkeypatch.setattr(config.bpy.data, "filepath", "")
    monkeypatch.setattr(config.os.path, "expanduser", lambda value: "/home/test" if value == "~" else value)

    assert config.get_default_output_dir() == "/home/test"


def test_logging_update_callback_applies_current_configuration(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(config, "_LOGGING_PREFERENCES_SYNCING", False)
    monkeypatch.setattr(config, "setup_logging", lambda: calls.append("setup"))

    config._update_logging_config(None, SimpleNamespace())

    assert calls == ["setup"]


def test_logging_update_callback_is_suppressed_during_collection_sync(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(config, "_LOGGING_PREFERENCES_SYNCING", True)
    monkeypatch.setattr(config, "setup_logging", lambda: calls.append("setup"))

    config._update_logging_config(None, SimpleNamespace())

    assert calls == []


def test_synchronize_logging_preferences_rebuilds_discovered_modules(monkeypatch):
    modules = _ModuleCollection(
        [SimpleNamespace(name="old.module", level="DEBUG")]
    )
    settings = SimpleNamespace(log_file_path="", modules=modules)
    preferences = SimpleNamespace(logging_settings=settings)
    discovered = ("config", "blender_adapter.mesh_reader")

    monkeypatch.setattr(config, "discover_logging_modules", lambda: discovered)
    monkeypatch.setattr(
        config,
        "merge_module_levels",
        lambda *_args, **_kwargs: (
            SimpleNamespace(module_name="config", level="INFO"),
            SimpleNamespace(module_name="blender_adapter.mesh_reader", level="ERROR"),
        ),
    )
    monkeypatch.setattr(config.os.path, "expanduser", lambda value: "/home/test" if value == "~" else value)

    result = config.synchronize_logging_preferences(preferences)

    assert result == discovered
    assert settings.log_file_path.endswith("Blender_to_Spine2D_Mesh_Exporter.log")
    assert tuple((item.name, item.level) for item in modules) == (
        ("config", "INFO"),
        ("blender_adapter.mesh_reader", "ERROR"),
    )
    assert config._LOGGING_PREFERENCES_SYNCING is False


def test_setup_logging_without_preferences_uses_safe_defaults(monkeypatch):
    diagnostics: list[tuple[bool, bool]] = []
    logging_calls: list[str] = []

    monkeypatch.setattr(config, "_addon_preferences", lambda: None)
    monkeypatch.setattr(
        config,
        "configure_export_diagnostics",
        lambda *, preserve_failed_work_files, recover_stale_work_files: diagnostics.append(
            (preserve_failed_work_files, recover_stale_work_files)
        ),
    )
    monkeypatch.setattr(
        config,
        "_setup_default_logging",
        lambda: logging_calls.append("default"),
    )

    config.setup_logging()

    assert diagnostics == [(False, True)]
    assert logging_calls == ["default"]


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
    assert '.get("spine2d_' not in source


def test_scene_texture_size_property_has_blender_side_bounds():
    source = (PACKAGE / "blender_adapter" / "scene_properties.py").read_text(
        encoding="utf-8"
    )

    texture_section = source.split('"spine2d_texture_size"', 1)[1]
    assert "default=1024" in texture_section
    assert "min=64" in texture_section
    assert "max=4096" in texture_section
    assert "update=_update_texture_size" in texture_section
