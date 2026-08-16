"""Regression coverage for Blender Extension AddonPreferences namespace lookup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    spine_version_preferences as preferences_module,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _context_with_addons(addons: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(preferences=SimpleNamespace(addons=addons))


def _spine_preferences(**overrides: str) -> SimpleNamespace:
    values = {
        "spine2d_exact_version_3_8": "3.8.99",
        "spine2d_exact_version_4_0": "4.0.64",
        "spine2d_exact_version_4_1": "4.1.24",
        "spine2d_exact_version_4_2": "4.2.30",
        "spine2d_exact_version_4_3": "4.3.23",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_installed_extension_uses_authoritative_root_package_for_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_package = "bl_ext.user_default.blender_to_spine2d_mesh_exporter"
    addon_preferences = _spine_preferences()
    context = _context_with_addons(
        {
            runtime_package: SimpleNamespace(
                module=runtime_package,
                preferences=addon_preferences,
            ),
        }
    )

    monkeypatch.setattr(preferences_module, "_ADDON_BASE_PACKAGE", runtime_package)

    assert preferences_module.addon_root_package_name() == runtime_package
    assert (
        preferences_module.get_spine_addon_preferences(context, required=True)
        is addon_preferences
    )
    assert (
        preferences_module.resolve_spine_project_exact_version(
            SpineJsonTarget.SPINE_4_2,
            context=context,
        )
        == "4.2.30"
    )


def test_ui_context_miss_falls_back_to_global_production_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_package = "bl_ext.user_default.blender_to_spine2d_mesh_exporter"
    addon_preferences = _spine_preferences(spine2d_exact_version_4_2="4.2.32")
    panel_context = _context_with_addons({})
    global_context = _context_with_addons(
        {
            runtime_package: SimpleNamespace(
                module=runtime_package,
                preferences=addon_preferences,
            ),
        }
    )

    monkeypatch.setattr(preferences_module, "_ADDON_BASE_PACKAGE", runtime_package)
    monkeypatch.setattr(preferences_module.bpy, "context", global_context)

    assert (
        preferences_module.resolve_spine_project_exact_version(
            SpineJsonTarget.SPINE_4_2,
            context=panel_context,
        )
        == "4.2.32"
    )


def test_installed_extension_finds_semantic_preferences_after_namespace_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_package = "bl_ext.user_default.blender_to_spine2d_mesh_exporter"
    exposed_key = "bl_ext.local.blender_to_spine2d_mesh_exporter"
    addon_preferences = _spine_preferences(spine2d_exact_version_4_2="4.2.31")
    context = _context_with_addons(
        {
            exposed_key: SimpleNamespace(
                module=exposed_key,
                preferences=addon_preferences,
            ),
        }
    )

    monkeypatch.setattr(preferences_module, "_ADDON_BASE_PACKAGE", runtime_package)

    assert (
        preferences_module.get_spine_addon_preferences(context, required=True)
        is addon_preferences
    )
    assert (
        preferences_module.resolve_spine_project_exact_version(
            SpineJsonTarget.SPINE_4_2,
            context=context,
        )
        == "4.2.31"
    )


def test_installed_extension_rejects_ambiguous_semantic_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_package = "bl_ext.user_default.blender_to_spine2d_mesh_exporter"
    context = _context_with_addons(
        {
            "bl_ext.repo_a.blender_to_spine2d_mesh_exporter": SimpleNamespace(
                module="bl_ext.repo_a.blender_to_spine2d_mesh_exporter",
                preferences=_spine_preferences(),
            ),
            "bl_ext.repo_b.blender_to_spine2d_mesh_exporter": SimpleNamespace(
                module="bl_ext.repo_b.blender_to_spine2d_mesh_exporter",
                preferences=_spine_preferences(),
            ),
        }
    )

    monkeypatch.setattr(preferences_module, "_ADDON_BASE_PACKAGE", runtime_package)

    with pytest.raises(RuntimeError, match="Multiple Spine2D AddonPreferences"):
        preferences_module.get_spine_addon_preferences(context, required=True)


def test_installed_extension_never_masks_missing_preferences_with_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_package = "bl_ext.user_default.blender_to_spine2d_mesh_exporter"
    context = _context_with_addons({})

    monkeypatch.setattr(preferences_module, "_ADDON_BASE_PACKAGE", runtime_package)
    monkeypatch.setattr(preferences_module.bpy, "context", context)

    with pytest.raises(RuntimeError, match="cannot resolve its AddonPreferences"):
        preferences_module.resolve_spine_project_exact_version(
            SpineJsonTarget.SPINE_4_2,
            context=context,
        )


def test_source_registered_development_context_keeps_descriptor_default_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context_with_addons({})

    monkeypatch.setattr(
        preferences_module,
        "_ADDON_BASE_PACKAGE",
        "Blender_to_Spine2D_Mesh_Exporter",
    )
    monkeypatch.setattr(preferences_module.bpy, "context", context)

    assert (
        preferences_module.resolve_spine_project_exact_version(
            SpineJsonTarget.SPINE_4_2,
            context=context,
        )
        == SpineJsonTarget.SPINE_4_2.exact_version
    )


def test_source_registered_incomplete_shared_preferences_mock_keeps_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_package = "Blender_to_Spine2D_Mesh_Exporter"
    incomplete_preferences = SimpleNamespace(logging_settings=SimpleNamespace())
    context = _context_with_addons(
        {
            runtime_package: SimpleNamespace(
                module=runtime_package,
                preferences=incomplete_preferences,
            ),
        }
    )

    monkeypatch.setattr(preferences_module, "_ADDON_BASE_PACKAGE", runtime_package)
    monkeypatch.setattr(preferences_module.bpy, "context", context)

    assert (
        preferences_module.get_spine_addon_preferences(context, required=True)
        is incomplete_preferences
    )
    assert (
        preferences_module.resolve_spine_project_exact_version(
            SpineJsonTarget.SPINE_4_2,
            context=context,
        )
        == SpineJsonTarget.SPINE_4_2.exact_version
    )