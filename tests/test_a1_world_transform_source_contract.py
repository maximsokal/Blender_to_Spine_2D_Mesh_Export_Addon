from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_source_geometry_preparation.py"
)


def _source_text():
    return SOURCE.read_text(encoding="utf-8")


def test_a1_source_geometry_normalizes_world_transform_before_uv_z_and_segmentation():
    source = _source_text()

    normalize_index = source.index("normalize_mesh_snapshot_world_transform(")
    uv_boundary_index = source.index("_resolve_source_uv_boundary_layer(", normalize_index)
    z_group_index = source.index("build_a1_z_group_assignment(", normalize_index)
    geometry_index = source.index("prepare_a1_geometry_regions(", normalize_index)

    assert normalize_index < uv_boundary_index < z_group_index < geometry_index
    assert "source_snapshot = world_transform.snapshot" in source


def test_a1_source_geometry_records_transform_diagnostics():
    source = _source_text()

    assert '"object_linear_transform_baked"' in source
    assert '"object_world_determinant"' in source
    assert '"object_world_mirrored"' in source
    assert 'code="MIRRORED_OBJECT_TRANSFORM"' in source


def test_evaluated_geometry_binds_scene_and_dependency_graph_before_reading():
    source = _source_text()

    assert "_resolved_evaluation_owners(scene)" in source
    assert "require_depsgraph_scene_consistency" in source
    assert "depsgraph=resolved_depsgraph" in source
