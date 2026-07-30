"""Static contract for the final ordered Blender panel's Spine target selector."""

from __future__ import annotations

import ast
from pathlib import Path


UI_LAYOUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "ui_layout.py"
)


def _ordered_export_method() -> ast.FunctionDef:
    tree = ast.parse(
        UI_LAYOUT_PATH.read_text(encoding="utf-8"),
        filename="ui_layout.py",
    )
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "OBJECT_PT_Spine2DOrderedMeshPanel":
            continue
        for member in node.body:
            if (
                isinstance(member, ast.FunctionDef)
                and member.name == "_draw_export_settings"
            ):
                return member
    raise AssertionError("Ordered panel _draw_export_settings method is missing")


def _property_call_line(method: ast.FunctionDef, property_name: str) -> int:
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
    raise AssertionError(f"Ordered export UI does not draw {property_name!r}")


def test_final_ordered_export_panel_draws_spine_target_before_texture_size() -> None:
    method = _ordered_export_method()

    spine_target_line = _property_call_line(
        method,
        "spine2d_target_spine_version",
    )
    texture_size_line = _property_call_line(method, "spine2d_texture_size")

    assert spine_target_line < texture_size_line


def test_final_ordered_export_panel_resolves_and_reports_target_capabilities() -> None:
    method = _ordered_export_method()
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
    assert "exact_version" in attributes
    assert "descriptor" in attributes
    assert "serializer_ready" in attributes
    assert "Spine version" in strings
    assert "Codec implementation in progress; Analyze blocks export" in strings
