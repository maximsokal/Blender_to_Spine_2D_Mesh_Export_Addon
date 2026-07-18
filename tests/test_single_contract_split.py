import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_adapter"


def _tree(name: str) -> ast.Module:
    path = ADAPTER / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(name: str) -> set[str]:
    return {
        node.module
        for node in ast.walk(_tree(name))
        if isinstance(node, ast.ImportFrom) and node.module
    }


def test_object_orchestrator_no_longer_owns_result_contract():
    tree = _tree("a1_object_preparation.py")
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "PreparedA1Object" not in classes
    assert any(value.endswith("a1_preparation_contracts") for value in _imports("a1_object_preparation.py"))


def test_multi_contracts_depend_on_contracts_not_orchestrator():
    imports = _imports("a1_multi_object_contracts.py")
    assert any(value.endswith("a1_preparation_contracts") for value in imports)
    assert not any(value.endswith("a1_object_preparation") for value in imports)


def test_skeleton_metadata_has_one_owner():
    contracts = _tree("a1_preparation_contracts.py")
    definitions = {
        node.name for node in contracts.body if isinstance(node, ast.FunctionDef)
    }
    assert "build_skeleton_metadata" in definitions
    for name in ("a1_document_preparation.py", "a1_projection_finalization.py"):
        tree = _tree(name)
        local_definitions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert "_skeleton_metadata" not in local_definitions
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "build_skeleton_metadata" in calls


def test_projection_statistics_use_shared_freezer_and_no_raw_mapping_proxy():
    tree = _tree("a1_projection_finalization.py")
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert "freeze_statistics" in names
    assert "MappingProxyType" not in names
    source = (ADAPTER / "a1_projection_finalization.py").read_text(encoding="utf-8")
    assert '"projection_contour_concave": int(layout.concave)' in source
    assert '"projection_output_float_buffer": int(output_policy.float_buffer)' in source
