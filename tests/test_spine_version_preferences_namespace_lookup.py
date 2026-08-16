"""Regression coverage for Blender Extension AddonPreferences namespace lookup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    spine_version_preferences as preferences_module,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _context_with_addons(addons: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        preferences=SimpleNamespace(addons=addons),
    )


def test_installed_extension_uses_authoritative_root_package_for_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_package = "bl_ext.user_default.blender_to_spine2d_mesh_exporter"
    addon_preferences = SimpleNamespace(
        spine2d_exact_version_4_2="4.2.30",
    )
    context = _context_with_addons(
        {
            runtime_package: SimpleNamespace(preferences=addon_preferences),
        }
    )

    monkeypatch.setattr(
        preferences_module,
        "_ADDON_BASE_PACKAGE",
        runtime_package,
    )

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


def test_installed_extension_never_masks_missing_preferences_with_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_package = "bl_ext.user_default.blender_to_spine2d_mesh_exporter"
    context = _context_with_addons({})

    monkeypatch.setattr(
        preferences_module,
        "_ADDON_BASE_PACKAGE",
        runtime_package,
    )

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

    assert (
        preferences_module.resolve_spine_project_exact_version(
            SpineJsonTarget.SPINE_4_2,
            context=context,
        )
        == SpineJsonTarget.SPINE_4_2.exact_version
    )
