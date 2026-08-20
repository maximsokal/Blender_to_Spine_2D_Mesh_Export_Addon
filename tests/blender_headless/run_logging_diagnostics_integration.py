"""Real Blender 4.4 checks for per-file logging and failed-work diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace
from unittest import mock

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Blender_to_Spine2D_Mesh_Exporter as addon  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter import config  # noqa: E402
from Blender_to_Spine2D_Mesh_Exporter.infrastructure import (  # noqa: E402
    atomic_file_transaction,
    get_export_diagnostics_policy,
)


_TEST_LOGGING_SETTINGS_PROPERTY = "_spine2d_test_logging_settings"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _module_by_name(settings, name: str):
    for module in settings.modules:
        if module.name == name:
            return module
    raise AssertionError(f"logging module is missing: {name}")


def _create_test_preferences():
    if hasattr(bpy.types.Scene, _TEST_LOGGING_SETTINGS_PROPERTY):
        raise RuntimeError("temporary logging settings property already exists")
    setattr(
        bpy.types.Scene,
        _TEST_LOGGING_SETTINGS_PROPERTY,
        bpy.props.PointerProperty(type=config.AddonLoggingSettings),
    )
    settings = getattr(bpy.context.scene, _TEST_LOGGING_SETTINGS_PROPERTY)
    return SimpleNamespace(logging_settings=settings)


def _remove_test_preferences() -> None:
    if hasattr(bpy.types.Scene, _TEST_LOGGING_SETTINGS_PROPERTY):
        delattr(bpy.types.Scene, _TEST_LOGGING_SETTINGS_PROPERTY)


def test_per_file_logging_and_diagnostics_preferences() -> None:
    preferences = _create_test_preferences()
    try:
        with mock.patch.object(
            config,
            "_addon_preferences",
            return_value=preferences,
        ):
            discovered = addon.initialize_logging_preferences(preferences)
            settings = preferences.logging_settings

            required = {
                "blender_adapter.a1_ui_bridge",
                "blender_adapter.a1_multi_object_output",
                "domain.baking.projection_layout",
                "infrastructure.atomic_files",
                "infrastructure.export_events",
            }
            _assert(
                required.issubset(set(discovered)),
                f"missing modules: {required - set(discovered)}",
            )

            selected = _module_by_name(settings, "infrastructure.atomic_files")
            neighbour = _module_by_name(settings, "infrastructure.export_events")
            selected.level = "DEBUG"
            neighbour.level = "ERROR"
            config.setup_logging()

            selected_logger = logging.getLogger(
                f"{config.PACKAGE_LOGGER_ROOT}.infrastructure.atomic_files"
            )
            neighbour_logger = logging.getLogger(
                f"{config.PACKAGE_LOGGER_ROOT}.infrastructure.export_events"
            )
            _assert(
                selected_logger.level == logging.DEBUG,
                "selected module did not become DEBUG",
            )
            _assert(
                neighbour_logger.level == logging.ERROR,
                "neighbour module level was overwritten",
            )

            settings.preserve_failed_work_files = True
            settings.recover_stale_work_files = True
            config.setup_logging()
            policy = get_export_diagnostics_policy()
            _assert(
                policy.preserve_failed_work_files,
                "preservation preference was not applied",
            )
            _assert(
                policy.recover_stale_work_files,
                "recovery preference was not applied",
            )

            settings.preserve_failed_work_files = False
            config.setup_logging()
            with tempfile.TemporaryDirectory(prefix="spine2d-diagnostics-") as directory:
                final_path = Path(directory) / "result.json"
                staged_path = None
                try:
                    with atomic_file_transaction(
                        recover_stale_work_files=False
                    ) as transaction:
                        reservation = transaction.reserve(final_path)
                        staged_path = reservation.staged_path
                        staged_path.write_text("partial", encoding="utf-8")
                        raise RuntimeError("forced diagnostics rollback")
                except RuntimeError:
                    pass
                _assert(
                    staged_path is not None,
                    "transaction did not reserve a staged path",
                )
                _assert(
                    not staged_path.exists(),
                    "failed stage file was not removed",
                )
                _assert(
                    not final_path.exists(),
                    "failed transaction exposed a final output",
                )
    finally:
        _remove_test_preferences()


def main() -> None:
    print(f"Blender version: {bpy.app.version_string}")
    addon.register()
    try:
        test_per_file_logging_and_diagnostics_preferences()
        print("[DIAGNOSTICS] PASS per-file logging and failed-work cleanup")
    finally:
        _remove_test_preferences()
        addon.unregister()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
