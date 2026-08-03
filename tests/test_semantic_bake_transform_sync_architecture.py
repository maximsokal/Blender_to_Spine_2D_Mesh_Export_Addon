"""Architecture guard for frame transform synchronization in semantic baking."""

from __future__ import annotations

import ast
from pathlib import Path


MODULE = (
    Path(__file__).resolve().parents[1]
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "semantic_bake_execution.py"
)


def _source() -> str:
    return MODULE.read_text(encoding="utf-8")


def _function(name: str) -> tuple[str, ast.FunctionDef]:
    source = _source()
    tree = ast.parse(source, filename=str(MODULE))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    text = "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
    return text, node


def _call_name(call: ast.Call) -> str:
    current: ast.AST = call.func
    parts: list[str] = []
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def test_frame_transform_is_synchronized_before_any_bake_path() -> None:
    source, _ = _function("_bake_frame_task")

    frame_set = source.index("_set_timeline_frame(")
    synchronize = source.index("synchronize_runtime_object_transform(")
    composed = source.index("_bake_composed_frame(")
    single = source.index("_bake_single_frame(")

    assert frame_set < synchronize
    assert synchronize < composed
    assert synchronize < single
    assert "validate_runtime_object_transform(" not in source


def test_run_semantic_bake_passes_temporary_object_to_frame_task() -> None:
    _, node = _function("run_semantic_bake")
    calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and _call_name(call) == "_bake_frame_task"
    ]

    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    target = keywords["bake_target_object"]
    assert isinstance(target, ast.Attribute)
    assert target.attr == "object"
    assert isinstance(target.value, ast.Name)
    assert target.value.id == "temporary"
