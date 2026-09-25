"""Static contracts for the persistent exact Spine project-version wiring."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _source(relative_path: str) -> str:
    return (PACKAGE / relative_path).read_text(encoding="utf-8")


def _is_string_property_call(node: ast.AST | None) -> bool:
    """Return whether ``node`` is a Blender ``StringProperty(...)`` call.

    Blender RNA properties are normally declared with annotation syntax, for example
    ``value: bpy.props.StringProperty(...)``. Supporting both annotation and assigned
    call forms keeps this static contract focused on the property declaration itself
    instead of one Python spelling of it.
    """

    if not isinstance(node, ast.Call):
        return False
    function = node.func
    if isinstance(function, ast.Attribute):
        return function.attr == "StringProperty"
    return isinstance(function, ast.Name) and function.id == "StringProperty"


def _declared_string_property_names(source: str) -> set[str]:
    """Collect class-level names declared as Blender StringProperty RNA fields."""

    tree = ast.parse(source)
    result: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if _is_string_property_call(node.annotation) or _is_string_property_call(node.value):
            result.add(node.target.id)
    return result


def _exact_version_preference_spec_property_names(source: str) -> set[str]:
    """Read property names from the canonical exact-version preference registry."""

    tree = ast.parse(source)
    for node in ast.walk(tree):
        value: ast.AST | None = None
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "SPINE_EXACT_VERSION_PREFERENCE_SPECS"
        ):
            value = node.value
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "SPINE_EXACT_VERSION_PREFERENCE_SPECS"
            for target in node.targets
        ):
            value = node.value

        if value is None:
            continue
        if not isinstance(value, (ast.Tuple, ast.List)):
            raise AssertionError(
                "SPINE_EXACT_VERSION_PREFERENCE_SPECS must be a tuple/list literal"
            )

        result: set[str] = set()
        for entry in value.elts:
            if not isinstance(entry, ast.Call):
                raise AssertionError(
                    "Every exact-version preference spec must be a constructor call"
                )
            function = entry.func
            function_name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            if function_name != "SpineExactVersionPreferenceSpec":
                raise AssertionError(
                    "Unexpected exact-version preference registry entry"
                )
            if len(entry.args) < 2:
                raise AssertionError(
                    "SpineExactVersionPreferenceSpec must receive a property name"
                )
            property_name = entry.args[1]
            if not (
                isinstance(property_name, ast.Constant)
                and isinstance(property_name.value, str)
            ):
                raise AssertionError(
                    "Exact-version preference property name must be a string literal"
                )
            result.add(property_name.value)
        return result

    raise AssertionError("Missing SPINE_EXACT_VERSION_PREFERENCE_SPECS registry")


def test_addon_preferences_owns_one_exact_version_field_per_supported_family() -> None:
    source = _source("addon_preferences.py")
    actual = _declared_string_property_names(source)
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


def test_canonical_main_panel_is_the_single_exact_version_display_owner() -> None:
    source = _source("ui.py")
    child_panels = _source("ui_layout.py")

    assert "resolve_spine_project_exact_version(" in source
    assert 'text=f"Exact JSON version: {exact_version}"' in source
    assert 'text=f"Exact JSON version: {target.exact_version}"' not in source
    assert "Invalid Spine version settings" in source

    # Standard child panels must not duplicate the Paths / Spine version controls.
    assert "resolve_spine_project_exact_version(" not in child_panels
    assert "Exact JSON version:" not in child_panels
    assert "spine2d_target_spine_version" not in child_panels


def test_preference_resolver_covers_every_current_family_without_fallback_aliases() -> None:
    source = _source("blender_adapter/spine_version_preferences.py")
    property_names = _exact_version_preference_spec_property_names(source)

    assert property_names == {
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
