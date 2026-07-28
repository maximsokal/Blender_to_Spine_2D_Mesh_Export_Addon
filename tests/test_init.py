"""Blender 5.2+ extension metadata and transactional lifecycle regressions."""

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


def _run_cleanup_actions(actions, **_kwargs) -> None:
    for action in actions:
        action.callback()


def _set_root_registration_state(state_name: str) -> None:
    state_type = getattr(extension, "ExtensionRegistrationState", None)
    if state_type is not None:
        extension._REGISTRATION_STATE = getattr(state_type, state_name)


@pytest.fixture(autouse=True)
def _isolate_root_registration_state():
    """Keep lifecycle tests independent from idempotent registration state."""

    _set_root_registration_state("UNREGISTERED")
    try:
        yield
    finally:
        _set_root_registration_state("UNREGISTERED")


def test_extension_manifest_targets_blender_52_and_declares_files_permission():
    manifest_path = PACKAGE / "blender_manifest.toml"
    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)

    assert manifest["id"] == "blender_to_spine2d_mesh_exporter"
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


def test_register_runs_runtime_gate_then_every_stage_then_preferences(monkeypatch):
    events: list[str] = []
    preferences = _install_preferences()

    def stage(label: str):
        return (
            label,
            lambda: events.append(f"register:{label}"),
            lambda: events.append(f"unregister:{label}"),
        )

    monkeypatch.setattr(extension, "REGISTRATION_STEPS", (stage("one"), stage("two")))
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
        "register:one",
        "register:two",
        "preferences",
        "configured-logging",
    ]


def test_register_rolls_back_only_completed_stages_in_reverse_order(monkeypatch):
    events: list[str] = []
    _install_preferences()
    primary_errors: list[BaseException | None] = []

    failure = RuntimeError("second stage failed")

    steps = (
        (
            "one",
            lambda: events.append("register:one"),
            lambda: events.append("unregister:one"),
        ),
        (
            "two",
            lambda: (_ for _ in ()).throw(failure),
            lambda: events.append("unregister:two"),
        ),
        (
            "three",
            lambda: events.append("register:three"),
            lambda: events.append("unregister:three"),
        ),
    )

    def cleanup(actions, *, primary_error=None, **_kwargs):
        primary_errors.append(primary_error)
        _run_cleanup_actions(actions)

    monkeypatch.setattr(extension, "REGISTRATION_STEPS", steps)
    monkeypatch.setattr(extension, "require_supported_blender_runtime", lambda _bpy: None)
    monkeypatch.setattr(extension.config, "_setup_default_logging", lambda: None)
    monkeypatch.setattr(extension, "unregister_all_best_effort", cleanup)

    with pytest.raises(RuntimeError, match="second stage failed"):
        extension.register()

    assert events == ["register:one", "unregister:one"]
    assert primary_errors == [failure]


def test_register_rolls_back_all_stages_when_preferences_fail(monkeypatch):
    events: list[str] = []
    _install_preferences()

    def stage(label: str):
        return (
            label,
            lambda: events.append(f"register:{label}"),
            lambda: events.append(f"unregister:{label}"),
        )

    monkeypatch.setattr(extension, "REGISTRATION_STEPS", (stage("one"), stage("two")))
    monkeypatch.setattr(extension, "require_supported_blender_runtime", lambda _bpy: None)
    monkeypatch.setattr(extension.config, "_setup_default_logging", lambda: None)
    monkeypatch.setattr(
        extension,
        "initialize_logging_preferences",
        lambda _prefs: (_ for _ in ()).throw(RuntimeError("preferences failed")),
    )
    monkeypatch.setattr(extension, "unregister_all_best_effort", _run_cleanup_actions)

    with pytest.raises(RuntimeError, match="preferences failed"):
        extension.register()

    assert events == [
        "register:one",
        "register:two",
        "unregister:two",
        "unregister:one",
    ]


def test_runtime_gate_failure_prevents_any_registration_mutation(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        extension,
        "REGISTRATION_STEPS",
        (("one", lambda: events.append("register:one"), lambda: None),),
    )
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

    with pytest.raises(RuntimeError, match="unsupported runtime"):
        extension.register()

    assert events == []


def test_unregister_runs_every_stage_in_reverse_order(monkeypatch):
    events: list[str] = []

    def stage(label: str):
        return (
            label,
            lambda: events.append(f"register:{label}"),
            lambda: events.append(f"unregister:{label}"),
        )

    monkeypatch.setattr(
        extension,
        "REGISTRATION_STEPS",
        (stage("one"), stage("two"), stage("three")),
    )
    monkeypatch.setattr(extension, "unregister_all_best_effort", _run_cleanup_actions)
    _set_root_registration_state("REGISTERED")

    extension.unregister()

    assert events == ["unregister:three", "unregister:two", "unregister:one"]


def test_registration_structure_has_current_runtime_owners_only():
    module_names = tuple(module.__name__.rsplit(".", 1)[-1] for module in extension.MODULES)
    class_names = tuple(cls.__name__ for cls in extension.CLASSES_TO_REGISTER)
    step_labels = tuple(step[0] for step in extension.REGISTRATION_STEPS)

    assert module_names == (
        "addon_preferences",
        "scene_settings_migration",
        "ui",
        "rig_ui",
        "a1_readiness_invalidation",
        "auto_readiness",
        "generated_material_ui",
        "ui_layout",
        "repolish_ui",
        "single_object_operator",
    )
    assert "WM_OT_UninstallAddon" not in class_names
    assert "SPINE2D_OT_RefreshLoggingModules" in class_names
    assert step_labels == (
        "addon preferences",
        "Scene RNA properties",
        "Scene settings migration",
        "UI",
        "Rig UI",
        "readiness invalidation",
        "automatic readiness",
        "generated material UI",
        "ordered UI layout",
        "Re-Polish UI",
        "single-object operator",
    )
