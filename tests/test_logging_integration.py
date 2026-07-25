"""Integration coverage for the current strict per-module logging pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import Blender_to_Spine2D_Mesh_Exporter as addon
from Blender_to_Spine2D_Mesh_Exporter import config


class ModuleCollection(list):
    """Minimal Blender CollectionProperty-compatible test container."""

    def clear(self) -> None:
        super().clear()

    def add(self):
        item = SimpleNamespace(name="", level="ERROR")
        self.append(item)
        return item


def module(name: str, level: str):
    return SimpleNamespace(name=name, level=level)


def preferences(
    tmp_path: Path,
    *,
    modules=(),
    enable_file_logging: bool = False,
):
    return SimpleNamespace(
        logging_settings=SimpleNamespace(
            enable_file_logging=enable_file_logging,
            log_file_path=str(tmp_path / "addon.log"),
            module_filter="",
            preserve_failed_work_files=False,
            recover_stale_work_files=True,
            modules=ModuleCollection(modules),
        )
    )


def close_package_handlers() -> None:
    prefix = config.PACKAGE_LOGGER_ROOT
    for name, candidate in tuple(logging.root.manager.loggerDict.items()):
        if name != prefix and not name.startswith(prefix + "."):
            continue
        if not isinstance(candidate, logging.Logger):
            continue
        for handler in tuple(candidate.handlers):
            handler.flush()
            handler.close()
            candidate.removeHandler(handler)


@pytest.fixture(autouse=True)
def cleanup_logging_state():
    yield
    close_package_handlers()


def test_initialize_preferences_discovers_modules_and_preserves_existing_levels(
    tmp_path,
    monkeypatch,
):
    prefs = preferences(
        tmp_path,
        modules=(module("config", "INFO"), module("removed.module", "DEBUG")),
    )
    prefs.logging_settings.log_file_path = ""
    monkeypatch.setattr(
        config,
        "discover_logging_modules",
        lambda: (
            config.ADDON_DISPLAY_NAME,
            "blender_adapter.mesh_reader",
            "config",
        ),
    )
    monkeypatch.setattr(config.os.path, "expanduser", lambda value: str(tmp_path))

    discovered = addon.initialize_logging_preferences(prefs)

    assert discovered == (
        config.ADDON_DISPLAY_NAME,
        "blender_adapter.mesh_reader",
        "config",
    )
    assert prefs.logging_settings.log_file_path == str(
        tmp_path / f"{config.ADDON_DISPLAY_NAME}.log"
    )
    assert tuple((item.name, item.level) for item in prefs.logging_settings.modules) == (
        (config.ADDON_DISPLAY_NAME, "ERROR"),
        ("blender_adapter.mesh_reader", "ERROR"),
        ("config", "INFO"),
    )


def test_setup_logging_builds_exact_runtime_names_and_file_handler(
    tmp_path,
    monkeypatch,
):
    prefs = preferences(
        tmp_path,
        enable_file_logging=True,
        modules=(
            module(config.ADDON_DISPLAY_NAME, "DEBUG"),
            module("config", "INFO"),
            module("blender_adapter.mesh_reader", "WARNING"),
        ),
    )
    captured = []
    diagnostics = []
    monkeypatch.setattr(config, "_addon_preferences", lambda: prefs)
    monkeypatch.setattr(config.bpy.path, "abspath", lambda value: value)
    monkeypatch.setattr(config.logging.config, "dictConfig", captured.append)
    monkeypatch.setattr(
        config,
        "configure_export_diagnostics",
        lambda **values: diagnostics.append(values),
    )

    config.setup_logging()

    assert diagnostics == [
        {
            "preserve_failed_work_files": False,
            "recover_stale_work_files": True,
        }
    ]
    logging_config = captured[0]
    assert logging_config["handlers"]["file"]["filename"] == str(
        tmp_path / "addon.log"
    )
    assert logging_config["loggers"][config.PACKAGE_LOGGER_ROOT]["level"] == "DEBUG"
    assert (
        logging_config["loggers"][f"{config.PACKAGE_LOGGER_ROOT}.config"]["level"]
        == "INFO"
    )
    assert (
        logging_config["loggers"][
            f"{config.PACKAGE_LOGGER_ROOT}.blender_adapter.mesh_reader"
        ]["level"]
        == "WARNING"
    )


def test_real_file_logging_writes_package_messages(tmp_path, monkeypatch):
    prefs = preferences(
        tmp_path,
        enable_file_logging=True,
        modules=(module(config.ADDON_DISPLAY_NAME, "DEBUG"),),
    )
    monkeypatch.setattr(config, "_addon_preferences", lambda: prefs)
    monkeypatch.setattr(config.bpy.path, "abspath", lambda value: value)

    config.setup_logging()
    logger = logging.getLogger(config.PACKAGE_LOGGER_ROOT)
    logger.debug("debug integration message")
    logger.error("error integration message")
    for handler in tuple(logger.handlers):
        handler.flush()

    content = (tmp_path / "addon.log").read_text(encoding="utf-8")
    assert "debug integration message" in content
    assert "error integration message" in content
    assert config.ADDON_DISPLAY_NAME in content
    assert "DEBUG" in content


def test_logging_update_callback_applies_current_configuration(monkeypatch):
    setup = MagicMock()
    monkeypatch.setattr(config, "_LOGGING_PREFERENCES_SYNCING", False)
    monkeypatch.setattr(config, "setup_logging", setup)

    config._update_logging_config(None, SimpleNamespace())

    setup.assert_called_once_with()


def test_dictconfig_failure_falls_back_without_masking_export(monkeypatch, tmp_path):
    prefs = preferences(
        tmp_path,
        modules=(module(config.ADDON_DISPLAY_NAME, "INFO"),),
    )
    fallback = MagicMock()
    monkeypatch.setattr(config, "_addon_preferences", lambda: prefs)
    monkeypatch.setattr(
        config.logging.config,
        "dictConfig",
        MagicMock(side_effect=RuntimeError("invalid logging config")),
    )
    monkeypatch.setattr(config, "_setup_default_logging", fallback)

    config.setup_logging()

    fallback.assert_called_once_with()


def test_empty_module_collection_is_synchronized_before_configuration(
    monkeypatch,
    tmp_path,
):
    prefs = preferences(tmp_path, modules=())
    synchronize = MagicMock(
        side_effect=lambda current: current.logging_settings.modules.append(
            module(config.ADDON_DISPLAY_NAME, "ERROR")
        )
    )
    captured = []
    monkeypatch.setattr(config, "_addon_preferences", lambda: prefs)
    monkeypatch.setattr(config, "synchronize_logging_preferences", synchronize)
    monkeypatch.setattr(config.logging.config, "dictConfig", captured.append)

    config.setup_logging()

    synchronize.assert_called_once_with(prefs)
    assert config.PACKAGE_LOGGER_ROOT in captured[0]["loggers"]


def test_malformed_module_setting_fails_closed(monkeypatch, tmp_path):
    prefs = preferences(tmp_path, modules=(module("", "DEBUG"),))
    monkeypatch.setattr(config, "_addon_preferences", lambda: prefs)

    with pytest.raises(ValueError, match="module_name must be a non-empty string"):
        config.setup_logging()


def test_rapid_preference_callbacks_apply_each_user_change(monkeypatch):
    setup = MagicMock()
    monkeypatch.setattr(config, "_LOGGING_PREFERENCES_SYNCING", False)
    monkeypatch.setattr(config, "setup_logging", setup)

    for _ in range(50):
        config._update_logging_config(None, SimpleNamespace())

    assert setup.call_count == 50
