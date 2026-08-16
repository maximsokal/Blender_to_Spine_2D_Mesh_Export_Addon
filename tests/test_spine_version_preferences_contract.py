"""Static contracts for the persistent exact Spine project-version wiring."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _source(relative_path: str) -> str:
    return (PACKAGE / relative_path).read_text(encoding="utf-8")


def _assigned_string_property_names(source: str) -> set[str]:
    tree = ast.parse(source)
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        function = value.func
        if isinstance(function, ast.Attribute) and function.attr == "StringProperty":
            result.add(node.target.id)
    return result


def test_addon_preferences_owns_one_exact_version_field_per_supported_family() -> None:
    source = _source("addon_preferences.py")
    actual = _assigned_string_property_names(source)
    expected = {
        "spine2d_exact_version_3_8",
        "spine2d_exact_version_4_0",
        "spine2d_exact_version_4_1",
        "spine2d_exact_version_4_2",
        "spine2d_exact_version_4_3",
    }
    assert expected <= actual


def test_production_preferences_never_force_save_all_blender_preferences() -> None:
    source = _source("addon_preferences.py")
    assert "save_userpref" not in source
    assert "clear_a1_export_readiness" in source
    assert "SPINE_EXACT_VERSION_PREFERENCE_SPECS" in source


def test_export_settings_resolve_effective_exact_version_from_preferences() -> None:
    source = _source("blender_adapter/a1_ui_settings.py")
    assert "resolve_spine_project_exact_version(scene.spine_target)" in source
    assert "spine_version=resolved_spine_version" in source
    assert "spine_json_version_filename_token(resolved_version)" in source


def test_skeleton_metadata_uses_effective_exact_export_version() -> None:
    source = _source("blender_adapter/a1_preparation_contracts.py")
    assert '"spine": settings.export.spine_version' in source
    assert '"spine": settings.export.spine_target.exact_version' not in source


def test_multi_object_sources_share_one_exact_version_preference_snapshot() -> None:
    source = _source("blender_adapter/a1_ui_settings.py")
    assert "spine_version = resolve_spine_project_exact_version(scene.spine_target)" in source
    assert "spine_version=spine_version" in source
    assert "Build all sources from one exact-version preference snapshot" in source


def test_multi_object_filename_uses_resolved_export_exact_version() -> None:
    source = _source("blender_adapter/a1_ui_export_plan.py")
    assert "sources[0].settings.export.spine_version" in source
    assert "sources[0].settings.export.spine_target" not in source


def test_viewport_exact_version_label_uses_same_preference_resolver() -> None:
    source = _source("ui.py")
    assert "resolve_spine_project_exact_version(" in source
    assert 'text=f"Exact JSON version: {exact_version}"' in source
    assert 'text=f"Exact JSON version: {target.exact_version}"' not in source
    assert "Invalid Spine version settings" in source


def test_preference_resolver_covers_every_current_family_without_fallback_aliases() -> None:
    source = _source("blender_adapter/spine_version_preferences.py")
    tree = ast.parse(source)
    literal_names = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("spine2d_exact_version_")
    }
    assert literal_names == {
        "spine2d_exact_version_3_8",
        "spine2d_exact_version_4_0",
        "spine2d_exact_version_4_1",
        "spine2d_exact_version_4_2",
        "spine2d_exact_version_4_3",
    }


def test_readiness_signature_tracks_raw_exact_spine_version() -> None:
    source = _source("blender_adapter/a1_export_readiness.py")

    assert "read_spine_project_exact_version_raw" in source
    assert "spine_target = _resolve_spine_target(scene)" in source
    assert "context=context" in source
    assert '"spine_exact_version_raw": spine_exact_version_raw' in source
