import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADLESS_ROOT = ROOT / "tests" / "blender_headless"

ROLLBACK_CASES = (
    (
        "run_bake_integration.py",
        "test_complete_a1_service_rolls_back_png_and_json_together",
    ),
    (
        "run_multi_object_export_integration.py",
        "test_second_bake_failure_rolls_back_json_and_both_textures",
    ),
)


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _is_relative_as_posix_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "as_posix":
        return False

    relative_call = node.func.value
    if not isinstance(relative_call, ast.Call):
        return False
    if (
        not isinstance(relative_call.func, ast.Attribute)
        or relative_call.func.attr != "relative_to"
    ):
        return False
    if not isinstance(relative_call.func.value, ast.Name):
        return False
    if relative_call.func.value.id != "path":
        return False

    return (
        len(relative_call.args) == 1
        and isinstance(relative_call.args[0], ast.Name)
        and relative_call.args[0].id == "output_directory"
    )


def test_headless_rollback_file_lists_are_platform_independent():
    for filename, function_name in ROLLBACK_CASES:
        function = _function(HEADLESS_ROOT / filename, function_name)
        calls = tuple(node for node in ast.walk(function) if isinstance(node, ast.Call))

        assert any(_is_relative_as_posix_call(call) for call in calls), filename
        assert not any(
            isinstance(call.func, ast.Name)
            and call.func.id == "str"
            and any(
                isinstance(nested, ast.Attribute)
                and nested.attr == "relative_to"
                for nested in ast.walk(call)
            )
            for call in calls
        ), filename
