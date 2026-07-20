import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "domain" / "geometry"


def _source(name: str) -> str:
    return (GEOMETRY / name).read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    path = GEOMETRY / name
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_geometry_contract_helpers_remain_blender_independent_and_reject_bool():
    source = _source("contracts.py")
    assert "import bpy" not in source
    assert "import bmesh" not in source
    assert "isinstance(value, bool)" in source
    assert "require_finite_vector" in source
    assert "require_exact_type" in source
    assert "require_tuple_items" in source


def test_identifier_and_model_boundaries_share_geometry_contract_owner():
    for name in ("ids.py", "model.py"):
        imports = {
            node.module
            for node in ast.walk(_tree(name))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "contracts" in imports, name

    model_source = _source("model.py")
    assert "require_exact_type(self.id, VertexId" in model_source
    assert "require_tuple_items(self.uvs, LoopUV" in model_source
    assert 'require_integer(self.material_index, "material_index", minimum=0)' in model_source
    assert 'require_identity(self.snapshot_id, "snapshot_id")' in model_source
    assert 'require_identity(self.source_object_id, "source_object_id")' in model_source


def test_cross_reference_validator_owns_zero_normal_diagnostics_not_scalar_checks():
    source = _source("validator.py")
    assert "isfinite" not in source
    assert '"ZERO_VERTEX_NORMAL"' in source
    assert '"ZERO_FACE_NORMAL"' in source
    assert "MeshValidationSeverity.WARNING" in source
