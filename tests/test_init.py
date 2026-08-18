"""Blender 5.2+ extension metadata and standard lifecycle regressions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib
from types import SimpleNamespace

import pytest
import Blender_to_Spine2D_Mesh_Exporter as extension


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _load_version_module():
    path = PACKAGE / "infrastructure" / "blender_version.py"
    spec = importlib.util.spec_from_file_location("spine2d_blender_version_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_preferences() -> object:
    preferences = object()
    extension.bpy.context.preferences = SimpleNamespace(
        addons={extension.__name__: SimpleNamespace(preferences=preferences)}
    )
    return preferences


def _runtime_owners() -> tuple[tuple[str, object], ...]:
    return (
        ("addon_preferences", extension.addon_preferences),
        ("scene_settings_migration", extension.scene_settings_migration),
        ("ui", extension.ui),
        ("rig_ui", extension.rig_ui),
        ("a1_readiness_invalidation", extension.a1_readiness_invalidation),
        ("auto_readiness", extension.auto_readiness),
        ("generated_material_ui", extension.generated_material_ui),
        ("ui_layout", extension.ui_layout),
        ("single_object_operator", extension.single_object_operator),
    )


def test_extension_manifest_targets_blender_52_and_declares_files_permission():
    manifest_path = PACKAGE / "blender_manifest.toml"
    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)

    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
    assert manifest["version"] == "0.155.0"
    assert manifest["name"] == "Spine2D Mesh Exporter"
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
    assert "REGISTRATION_STEPS" not in source
    assert "ExtensionRegistrationState" not in source
    assert "repolish_ui" not in source


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


def test_register_runs_runtime_gate_then_explicit_owners_then_logging(monkeypatch):
    events: list[str] = []
    preferences = _install_preferences()

    monkeypatch.setattr(
        extension,
        "require_supported_blender_runtime",
        lambda _bpy: events.append("runtime-gate"),
    )
    monkeypatch.setattr(
        extension.config,
        "_setup_default_logging",
        lambda: events.append("default-logging"),
    )
    monkeypatch.setattr(
        extension,
        "_register_config_rna",
        lambda: events.append("register:config-rna"),
    )
    for label, owner in _runtime_owners():
        monkeypatch.setattr(
            owner,
            "register",
            lambda label=label: events.append(f"register:{label}"),
        )

    monkeypatch.setattr(extension.config, "_addon_preferences", lambda: preferences)
    monkeypatch.setattr(
        extension,
        "initialize_logging_preferences",
        lambda value: events.append("preferences") if value is preferences else None,
    )
    monkeypatch.setattr(
        extension.config,
        "setup_logging",
        lambda: events.append("configured-logging"),
    )

    extension.register()

    assert events == [
        "runtime-gate",
        "default-logging",
        "register:addon_preferences",
        "register:config-rna",
        "register:scene_settings_migration",
        "register:ui",
        "register:rig_ui",
        "register:a1_readiness_invalidation",
        "register:auto_readiness",
        "register:generated_material_ui",
        "register:ui_layout",
        "register:single_object_operator",
        "preferences",
        "configured-logging",
    ]


def test_runtime_gate_failure_prevents_any_registration_mutation(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        extension,
        "require_supported_blender_runtime",
        lambda _bpy: (_ for _ in ()).throw(RuntimeError("unsupported runtime")),
    )
    monkeypatch.setattr(
        extension.config,
        "_setup_default_logging",
        lambda: events.append("default-logging"),
    )
    monkeypatch.setattr(
        extension,
        "_register_config_rna",
        lambda: events.append("register:config-rna"),
    )
    for label, owner in _runtime_owners():
        monkeypatch.setattr(
            owner,
            "register",
            lambda label=label: events.append(f"register:{label}"),
        )

    with pytest.raises(RuntimeError, match="unsupported runtime"):
        extension.register()

    assert events == []


def test_owner_registration_failure_is_not_hidden_by_generic_root_recovery(monkeypatch):
    events: list[str] = []
    failure = RuntimeError("ui registration failed")

    monkeypatch.setattr(extension, "require_supported_blender_runtime", lambda _bpy: None)
    monkeypatch.setattr(extension.config, "_setup_default_logging", lambda: None)
    monkeypatch.setattr(
        extension,
        "_register_config_rna",
        lambda: events.append("register:config-rna"),
    )

    for label, owner in _runtime_owners():
        if label == "ui":
            monkeypatch.setattr(
                owner,
                "register",
                lambda: (_ for _ in ()).throw(failure),
            )
        else:
            monkeypatch.setattr(
                owner,
                "register",
                lambda label=label: events.append(f"register:{label}"),
            )

    with pytest.raises(RuntimeError, match="ui registration failed") as raised:
        extension.register()

    assert raised.value is failure
    assert events == [
        "register:addon_preferences",
        "register:config-rna",
        "register:scene_settings_migration",
    ]


def test_unregister_runs_explicit_owners_in_reverse_order(monkeypatch):
    events: list[str] = []

    monkeypatch.setattr(
        extension,
        "_unregister_config_rna",
        lambda: events.append("unregister:config-rna"),
    )
    for label, owner in _runtime_owners():
        monkeypatch.setattr(
            owner,
            "unregister",
            lambda label=label: events.append(f"unregister:{label}"),
        )

    extension.unregister()

    assert events == [
        "unregister:single_object_operator",
        "unregister:ui_layout",
        "unregister:generated_material_ui",
        "unregister:auto_readiness",
        "unregister:a1_readiness_invalidation",
        "unregister:rig_ui",
        "unregister:ui",
        "unregister:scene_settings_migration",
        "unregister:config-rna",
        "unregister:addon_preferences",
    ]


def test_registration_structure_has_current_runtime_owners_only():
    module_names = tuple(module.__name__.rsplit(".", 1)[-1] for module in extension.MODULES)
    class_names = tuple(cls.__name__ for cls in extension.CLASSES_TO_REGISTER)

    assert module_names == (
        "addon_preferences",
        "scene_settings_migration",
        "ui",
        "rig_ui",
        "a1_readiness_invalidation",
        "auto_readiness",
        "generated_material_ui",
        "ui_layout",
        "single_object_operator",
    )
    assert "repolish_ui" not in module_names
    assert "WM_OT_UninstallAddon" not in class_names
    assert "SPINE2D_OT_RefreshLoggingModules" in class_names


def test_config_rna_owner_uses_direct_scene_properties():
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")

    assert "for name, value in CONFIG_RNA_PROPERTIES" in source
    assert "setattr(bpy.types.Scene, name, value)" in source
    assert "for name, _value in reversed(CONFIG_RNA_PROPERTIES)" in source
    assert "delattr(bpy.types.Scene, name)" in source
