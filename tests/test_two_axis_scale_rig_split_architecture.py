"""Keep the selectable two-axis rig decomposed into explicit domain owners."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "domain" / "spine"


def _tree(name: str) -> ast.Module:
    path = SPINE / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _definitions(name: str) -> set[str]:
    return {
        node.name
        for node in _tree(name).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }


def test_two_axis_rig_uses_split_plan_bone_constraint_validation_owners():
    expected = {
        "two_axis_scale_rig_contracts.py": {"TwoAxisScaleRigLayout"},
        "two_axis_scale_rig_plan.py": {"build_two_axis_scale_layout"},
        "two_axis_scale_rig_bones.py": {"build_two_axis_scale_bones"},
        "two_axis_scale_rig_constraints.py": {
            "build_two_axis_scale_constraints"
        },
        "two_axis_scale_rig_validation.py": {
            "validate_two_axis_scale_rig_result"
        },
        "two_axis_scale_rig_assembly.py": {"build_two_axis_scale_rig"},
    }
    for filename, definitions in expected.items():
        assert definitions.issubset(_definitions(filename)), filename


def test_two_axis_rig_facade_contains_no_implementation_functions():
    path = SPINE / "two_axis_scale_rig.py"
    tree = _tree(path.name)

    assert len(path.read_text(encoding="utf-8").splitlines()) < 40
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in tree.body
    )


def test_two_axis_runtime_modules_remain_blender_independent():
    findings = []
    for path in SPINE.glob("two_axis_scale_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
                if "bpy" in imported or "bmesh" in imported:
                    findings.append(path.name)
            elif isinstance(node, ast.ImportFrom) and node.module in {"bpy", "bmesh"}:
                findings.append(path.name)
    assert findings == []
