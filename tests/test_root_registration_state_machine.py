"""Regression contracts proving the rejected root registration state machine stays gone."""

from __future__ import annotations

from pathlib import Path

import pytest

import Blender_to_Spine2D_Mesh_Exporter as extension


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _owners() -> tuple[tuple[str, object], ...]:
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


def _prepare_register(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr(extension, "require_supported_blender_runtime", lambda _bpy: None)
    monkeypatch.setattr(extension.config, "_setup_default_logging", lambda: None)
    monkeypatch.setattr(extension.config, "_addon_preferences", lambda: None)
    monkeypatch.setattr(extension.config, "setup_logging", lambda: None)
    monkeypatch.setattr(
        extension,
        "_register_config_rna",
        lambda: calls.append("register:config-rna"),
    )


def test_root_registration_state_machine_symbols_are_absent() -> None:
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")

    for forbidden in (
        "ExtensionRegistrationState",
        "_REGISTRATION_STATE",
        "get_registration_state",
        "REGISTRATION_STEPS",
        "RegistrationCleanupAction",
        "unregister_all_best_effort",
    ):
        assert forbidden not in source


def test_root_registration_stops_at_first_owner_failure_without_hidden_recovery(
    monkeypatch,
) -> None:
    calls: list[str] = []
    _prepare_register(monkeypatch, calls)

    failure = RuntimeError("forced UI owner failure")
    for label, owner in _owners():
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
                lambda label=label: calls.append(f"register:{label}"),
            )

    with pytest.raises(RuntimeError, match="forced UI owner failure") as raised:
        extension.register()

    assert raised.value is failure
    assert calls == [
        "register:addon_preferences",
        "register:config-rna",
        "register:scene_settings_migration",
    ]


def test_preflight_failure_occurs_before_logging_or_owner_mutation(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        extension,
        "require_supported_blender_runtime",
        lambda _bpy: (_ for _ in ()).throw(RuntimeError("unsupported runtime")),
    )
    monkeypatch.setattr(
        extension.config,
        "_setup_default_logging",
        lambda: calls.append("default-logging"),
    )
    monkeypatch.setattr(
        extension,
        "_register_config_rna",
        lambda: calls.append("register:config-rna"),
    )
    for label, owner in _owners():
        monkeypatch.setattr(
            owner,
            "register",
            lambda label=label: calls.append(f"register:{label}"),
        )

    with pytest.raises(RuntimeError, match="unsupported runtime"):
        extension.register()

    assert calls == []


def test_root_unregister_is_plain_reverse_owner_sequence(monkeypatch) -> None:
    calls: list[str] = []
    for label, owner in _owners():
        monkeypatch.setattr(
            owner,
            "unregister",
            lambda label=label: calls.append(f"unregister:{label}"),
        )
    monkeypatch.setattr(
        extension,
        "_unregister_config_rna",
        lambda: calls.append("unregister:config-rna"),
    )

    extension.unregister()

    assert calls == [
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
