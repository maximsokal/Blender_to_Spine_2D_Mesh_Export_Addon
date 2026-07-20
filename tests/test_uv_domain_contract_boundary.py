from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UV_LAYOUT = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "domain" / "uv" / "layout.py"
UV_MODEL = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "domain" / "uv" / "model.py"
CORRESPONDENCE = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "correspondence.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_uv_layout_owns_exact_loop_and_vector_contracts():
    source = read(UV_LAYOUT)

    assert "require_exact_type(self.loop_id, LoopId" in source
    assert "require_exact_type(self.source_loop_id, SourceLoopId" in source
    assert 'require_finite_vector(self.coordinate, 2, "coordinate")' in source
    assert "a partial layout cannot introduce a new UV layer" in source
    assert "require_complete must be bool" in source


def test_uv_unwrap_model_reuses_geometry_numeric_contracts():
    source = read(UV_MODEL)

    assert "from math import isfinite" not in source
    assert "require_finite_number" in source
    assert "require_integer(self.iterations" in source
    assert "statistics do not match the UV coordinates stored in snapshot" in source
    assert "snapshot.active_uv_layer must match settings.layer_name" in source


def test_source_loop_correspondence_rejects_permissive_python_scalar_rules():
    source = read(CORRESPONDENCE)

    assert "require_finite_number(\n        duplicate_tolerance" in source
    assert "require_complete must be bool" in source
    assert "entries contain duplicate SourceLoopId values" in source
    assert "missing_source_loop_ids and unused_source_loop_ids cannot overlap" in source
    assert "isinstance(value, (int, float))" not in source
