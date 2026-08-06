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

    assert "split_xy_visible_region_snapshots" in source
    assert "_validate_triangulated_region(snapshot)" in source
    assert "return (snapshot,)" in source
    assert "Removing edge-on faces" not in source


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


def test_active_camera_normal_uses_full_rank_depth_with_inverse_setup_parents():
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
    naming = _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/spine/legacy_profile.py"
    )
    two_axis_bones = _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/spine/"
        "two_axis_scale_rig_bones.py"
    )
    legacy_bones = _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/spine/legacy_rig_bones.py"
    )
    attachment = _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/spine/"
        "legacy_attachment_builder.py"
    )
    two_axis_constraints = _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/spine/"
        "two_axis_scale_rig_constraints.py"
    )
    legacy_constraints = _read(
        "Blender_to_Spine2D_Mesh_Exporter/domain/spine/"
        "legacy_rig_constraints.py"
    )

    assert "projected_world.depth - projected_origin.depth" in projection
    assert "origin.u / resolved_scale" in projection
    assert "origin.v / resolved_scale" in projection
    assert "origin.depth," in projection
    assert 'CAMERA_VIEW_NORMAL = "CAMERA_VIEW_NORMAL"' in profiles

    assert "object_root_normal = active_camera_normal and not camera_root_normal" in document
    assert "A1RigSetupPoseMode.CAMERA_VIEW_NORMAL" in document
    assert "A1RigSetupPoseMode.PREPROJECTED_SCREEN" in document
    assert "camera_layer_projection_kind=camera_layer_kind" in document
    assert "compensate_depth_setup_y=camera_root_normal" in document

    assert "def z_camera_setup_bone(" in naming
    assert 'f"{self._require_prefix(prefix)}_{resolved_index}_camera_setup"' in naming

    assert "camera_view_normal = (" in two_axis_bones
    assert "plan.profile.z_camera_setup_bone(" in two_axis_bones
    assert "-float(group.y_offset_pixels)" in two_axis_bones
    assert "build_camera_view_setup_compensation_bones(" in legacy_bones
    assert "*build_camera_view_setup_compensation_bones(plan)" in legacy_bones

    assert "if rig.request.setup_pose_mode is A1RigSetupPoseMode.CAMERA_VIEW_NORMAL" in attachment
    assert "rig.profile.z_camera_setup_bone(" in attachment
    assert "CAMERA_VIEW_NORMAL rig is missing inverse setup parent" in attachment

    assert "neutral_depth_setup = (" in two_axis_constraints
    assert "preprojected_screen or neutral_model_space_camera_setup" in two_axis_constraints
    assert "neutral_depth_scale_setup = neutral_camera_setup" in legacy_constraints


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


def test_real_coin_projection_parity_gate_validates_inverse_setup():
    source = _read(
        "tests/blender_headless/"
        "run_coin_star_normal_projection_parity_integration.py"
    )

    for marker in (
        "[COIN-NORMAL-PROJECTION-PARITY] PASS",
        "segments=",
        "axis_depth_groups=",
        "camera_depth_groups=",
        "neutral_constraints=",
        "inverse_setup_bones=",
        "setup=CAMERA_VIEW_NORMAL",
        "pivot=OBJECT_ORIGIN",
        "depth_setup=neutral+inverse",
        "material_geometry=projection-independent",
        "Normal projection direction changed source-material bake geometry",
        "real coin parity gate did not retain expected side regions",
        "Active Camera Normal did not use neutral object-pivot setup",
        "Normal projection changed material brightness beyond tolerance",
    ):
        assert marker in source

    assert "def _assert_serialized_active_camera_normal_setup(" in source
    assert "abs(rotation) <= _NEUTRAL_TOLERANCE" in source
    assert "abs(depth_x) <= _NEUTRAL_TOLERANCE" in source
    assert "abs(depth_scale_x) <= _NEUTRAL_TOLERANCE" in source
    assert "profile.z_camera_setup_bone(prefix, group.index)" in source
    assert "inverse setup does not cancel depth translation" in source

    assert "def _assert_prepared_depth_groups(" in source
    assert "rig_group_count == plan_group_count" in source
    assert "tuple(rig.request.z_groups) == tuple(plan.groups)" in source
    assert "binding_count == vertex_count" in source
    assert "bound_group_indices == expected_group_indices" in source


def test_real_coin_object_root_gate_checks_every_inverse_chain_and_vertex_parent():
    source = _read(
        "tests/blender_headless/"
        "run_coin_star_normal_object_root_setup_compensation_integration.py"
    )

    for marker in (
        "[COIN-NORMAL-OBJECT-ROOT-SETUP] PASS",
        "inverse_setup_bones=",
        "weighted_vertices=",
        "depth_x=0",
        "depth_scale_x=0",
        "setup_chain=depth+inverse+projected_xy",
        "def _assert_typed_inverse_setup(",
        "def _assert_serialized_inverse_setup(",
        "typed vertex bypasses its inverse setup bone",
        "serialized vertex bypasses inverse setup bone",
        "translations do not sum to zero",
    ):
        assert marker in source
