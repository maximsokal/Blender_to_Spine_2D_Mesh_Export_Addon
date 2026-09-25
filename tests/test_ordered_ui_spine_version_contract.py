"""Static contracts for Spine target and Bake-owned Texture size in the main panel."""

from __future__ import annotations

import ast
from pathlib import Path


UI_PATH = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "ui.py"
)


def _main_panel_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(
        UI_PATH.read_text(encoding="utf-8"),
        filename="ui.py",
    )
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "OBJECT_PT_Spine2DMeshPanel":
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == name:
                return member
    raise AssertionError(f"Main panel method {name!r} is missing")


def _property_call_line(method: ast.FunctionDef, property_name: str) -> int | None:
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(
            node.func,
            ast.Attribute,
        ):
            continue
        if node.func.attr != "prop" or len(node.args) < 2:
            continue
        property_argument = node.args[1]
        if (
            isinstance(property_argument, ast.Constant)
            and property_argument.value == property_name
        ):
            return node.lineno
    return None


def test_spine_target_stays_in_paths_and_texture_size_moves_to_bake() -> None:
    export_method = _main_panel_method("_draw_export_settings")
    bake_method = _main_panel_method("_draw_bake_settings")

    assert _property_call_line(export_method, "spine2d_target_spine_version") is not None
    assert _property_call_line(export_method, "spine2d_texture_size") is None
    assert _property_call_line(bake_method, "spine2d_texture_size") is not None


def test_main_export_panel_resolves_and_reports_target_capabilities() -> None:
    method = _main_panel_method("_draw_export_settings")
    called_names = {
        node.func.id
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attributes = {
        node.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute)
    }
    strings = {
        node.value
        for node in ast.walk(method)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "resolve_spine_json_target" in called_names
    assert "resolve_spine_project_exact_version" in called_names
    assert "exact_version" not in attributes
    assert "descriptor" in attributes
    assert "serializer_ready" in attributes
    assert "Spine version" in strings
    assert "Codec implementation unavailable for this target" in strings
    assert "Analyze blocks export" not in strings
