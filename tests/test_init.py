"""Blender 5.2+ extension entry-point and registration source contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _load_version_module():
    path = PACKAGE / "infrastructure" / "blender_version.py"
    spec = importlib.util.spec_from_file_location("spine2d_blender_version_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extension_manifest_targets_blender_52_and_declares_files_permission():
    manifest_path = PACKAGE / "blender_manifest.toml"
    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)

    assert manifest["blender_version_min"] == "5.2.0"
    assert manifest["type"] == "add-on"
    assert str(manifest["permissions"]["files"]).strip()
    assert not (ROOT / "blender_manifest.toml").exists()


def test_entry_point_uses_manifest_metadata_and_runtime_gate():
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")

    assert "bl_info" not in source
    assert "require_supported_blender_runtime(bpy)" in source
    assert source.index("require_supported_blender_runtime(bpy)") < source.index(
        "config._setup_default_logging()"
    )
    assert "scene_properties.PROPERTIES" in source
    assert "config.PROPERTIES" not in source


def test_addon_preferences_do_not_self_disable_or_remove_extension():
    source = (PACKAGE / "addon_preferences.py").read_text(encoding="utf-8")

    assert "WM_OT_UninstallAddon" not in source
    assert "addon_disable" not in source
    assert "addon_remove" not in source
    assert "Preferences > Extensions" in source


def test_blender_version_gate_accepts_52_and_newer():
    module = _load_version_module()

    class App:
        version = (5, 2, 0)

    class Bpy:
        app = App()

    assert module.require_supported_blender_runtime(Bpy()) == (5, 2, 0)
    App.version = (6, 0, 1)
    assert module.require_supported_blender_runtime(Bpy()) == (6, 0, 1)


def test_blender_version_gate_rejects_older_runtime_before_registration():
    module = _load_version_module()

    class App:
        version = (5, 1, 9)

    class Bpy:
        app = App()

    with pytest.raises(module.UnsupportedBlenderVersionError, match="requires Blender 5.2.0"):
        module.require_supported_blender_runtime(Bpy())


def test_blender_version_gate_rejects_missing_or_malformed_version():
    module = _load_version_module()

    class MissingBpy:
        app = object()

    with pytest.raises(module.UnsupportedBlenderVersionError, match="unavailable"):
        module.require_supported_blender_runtime(MissingBpy())

    with pytest.raises(module.UnsupportedBlenderVersionError, match="at least three"):
        module.normalize_blender_version((5, 2))
