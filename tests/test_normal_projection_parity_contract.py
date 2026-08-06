from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "Blender_to_Spine2D_Mesh_Exporter"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_normal_document_assembly_retains_edge_on_regions():
    source = _read(
        "Blender_to_Spine2D_Mesh_Exporter/application/"
        "a1_projected_region_filter.py"
    )

    assert "return (snapshot,)" in source
    assert "Removing edge-on faces" not in source
    assert "split_xy_visible_region_snapshots" in source
    assert "later X/Y rotation" in source


def test_normal_attachment_projection_retains_setup_degenerate_meshes():
    source = _read(
        "Blender_to_Spine2D_Mesh_Exporter/application/"
        "a1_attachment_projection_service.py"
    )

    for setup_mode in (
        "A1RigSetupPoseMode.NORMALIZED_SINGLE",
        "A1RigSetupPoseMode.PRESERVE_COMPOSITION",
        "A1RigSetupPoseMode.CAMERA_VIEW_NORMAL",
        "A1RigSetupPoseMode.CAMERA_DEPTH_SURFACE",
    ):
        assert setup_mode in source

    assert "allow_setup_degenerate: bool = False" in source
    assert "if collapsed_triangles and not allow_setup_degenerate" in source
    assert "if allow_setup_degenerate and len(collapsed_triangles) == triangle_count" in source
    assert "return projection" in source
    assert "rig.request.setup_pose_mode in _DEFORMABLE_SETUP_MODES" in source
    assert "A1RigSetupPoseMode.PREPROJECTED_SCREEN" not in source.split(
        "_DEFORMABLE_SETUP_MODES = frozenset(", 1
    )[1].split(")\n", 1)[0]


def test_active_camera_normal_keeps_object_pivot_and_vertex_depth():
    projection = _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/geometry/"
        "camera_projection.py"
    )
    document = _read(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "a1_document_preparation.py"
    )
    profiles = _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/spine/rig_profiles.py"
    )
    two_axis = _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/spine/"
        "two_axis_scale_rig_constraints.py"
    )
    legacy = _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/spine/"
        "legacy_rig_constraints.py"
    )

    assert "projected_world.depth - projected_origin.depth" in projection
    assert "origin.u / resolved_scale" in projection
    assert "origin.v / resolved_scale" in projection
    assert "origin.depth," in projection
    assert 'CAMERA_VIEW_NORMAL = "CAMERA_VIEW_NORMAL"' in profiles
    assert "A1RigSetupPoseMode.CAMERA_VIEW_NORMAL" in document
    assert "if active_camera_normal" in document
    assert "camera_layer_projection_kind=None" in document
    assert "compensate_depth_setup_y=False" in document
    assert '"normal_active_camera_setup_neutral": int(active_camera_normal)' in document
    assert '"camera_relative_depth_group_count": 0' in document
    assert '"depth_setup_y_compensated": 0' in document
    assert "A1RigSetupPoseMode.CAMERA_VIEW_NORMAL" in two_axis
    assert "neutral_model_space_camera_setup" in two_axis
    assert "A1RigSetupPoseMode.CAMERA_VIEW_NORMAL" in legacy


def test_normal_material_bake_uses_unprojected_source_geometry():
    source_geometry = _read(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "a1_source_geometry_preparation.py"
    )
    uv = _read(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "a1_uv_preparation.py"
    )
    contracts = _read(
        "Blender_to_Spine2D_Mesh_Exporter/blender_adapter/"
        "a1_preparation_contracts.py"
    )

    assert "material_bake_snapshot=material_bake_snapshot" in source_geometry
    assert "transfer_normal_uv_to_material_bake_snapshot" in uv
    assert "transfer_uv_by_source_loop" in uv
    assert "updated.vertices != material_snapshot.vertices" in uv
    assert "updated.world_matrix != material_snapshot.world_matrix" in uv
    assert "return material_bake_snapshot" in contracts


def test_real_coin_projection_parity_gate_covers_all_three_regressions():
    source = _read(
        "tests/blender_headless/"
        "run_coin_star_normal_projection_parity_integration.py"
    )

    for marker in (
        "[COIN-NORMAL-PROJECTION-PARITY] PASS",
        "segments=",
        "neutral_constraints=",
        "setup=CAMERA_VIEW_NORMAL",
        "pivot=OBJECT_ORIGIN",
        "material_geometry=projection-independent",
        "Normal projection direction changed source-material bake geometry",
        "real coin parity gate did not retain expected side regions",
        "Active Camera Normal did not use neutral object-pivot setup",
        "Active Camera Normal retained setup rotation",
        "Normal projection changed material brightness beyond tolerance",
    ):
        assert marker in source
