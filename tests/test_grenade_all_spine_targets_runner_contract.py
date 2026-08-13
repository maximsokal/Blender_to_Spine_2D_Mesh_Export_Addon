"""Static contracts for the real grenade all-Spine-target Blender runner."""

from __future__ import annotations

import ast
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RUNNER = (
    _REPOSITORY_ROOT
    / "tests"
    / "blender_headless"
    / "run_grenade_all_spine_targets_real_export.py"
)
_SCENE_CAPTURE = (
    _REPOSITORY_ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_ui_scene_capture.py"
)
_SCENE_PROPERTIES = (
    _REPOSITORY_ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "scene_properties.py"
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assigned_string(module: ast.Module, name: str) -> str:
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            raise AssertionError(f"{name} must be assigned one string literal")
        return node.value.value
    raise AssertionError(f"Missing assignment for {name}")


def _function_property_strings(module: ast.Module, function_name: str) -> set[str]:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and child.value.startswith("spine2d_")
            }
    raise AssertionError(f"Missing function {function_name}")


def _registered_scene_property_names(module: ast.Module) -> set[str]:
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "PROPERTIES" for target in node.targets):
            continue
        if not isinstance(node.value, (ast.Tuple, ast.List)):
            raise AssertionError("scene_properties.PROPERTIES must be a tuple/list literal")

        names: set[str] = set()
        for entry in node.value.elts:
            if not isinstance(entry, ast.Tuple) or not entry.elts:
                continue
            first = entry.elts[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names.add(first.value)
        return names
    raise AssertionError("Missing scene_properties.PROPERTIES assignment")


def test_grenade_matrix_uses_the_same_target_rna_as_public_scene_capture() -> None:
    runner_module = _parse(_RUNNER)
    capture_module = _parse(_SCENE_CAPTURE)
    properties_module = _parse(_SCENE_PROPERTIES)

    runner_property = _assigned_string(runner_module, "_SCENE_TARGET_PROPERTY")
    capture_properties = _function_property_strings(
        capture_module,
        "_resolve_spine_target",
    )
    registered_properties = _registered_scene_property_names(properties_module)

    assert capture_properties == {runner_property}, (
        "grenade matrix must drive the exact Scene RNA property consumed by "
        f"_resolve_spine_target; runner={runner_property!r}, "
        f"capture={sorted(capture_properties)!r}"
    )
    assert runner_property in registered_properties, (
        "grenade matrix target property must be registered in Scene PROPERTIES; "
        f"runner={runner_property!r}"
    )
