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


def test_evaluated_geometry_binds_scene_dependency_graph_mesh_and_matrix():
    source = _source_text()

    owners_index = source.index("_resolved_evaluation_owners(scene)")
    read_index = source.index("read_evaluated_mesh_snapshot(", owners_index)
    matrix_index = source.index("_evaluated_source_world_matrix(", read_index)
    normalize_index = source.index("normalize_mesh_snapshot_world_transform(", matrix_index)

    assert owners_index < read_index < matrix_index < normalize_index
    assert "require_depsgraph_scene_consistency" in source
    assert "depsgraph=resolved_depsgraph" in source
    assert "evaluated_get(depsgraph)" in source
    assert "evaluated_snapshot = replace(" in source
    assert "world_matrix=_evaluated_source_world_matrix(" in source
