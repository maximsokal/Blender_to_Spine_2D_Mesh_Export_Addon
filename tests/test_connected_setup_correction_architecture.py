import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
SPINE = PACKAGE / "domain" / "spine"
CORRECTION = SPINE / "connected_group_setup_correction.py"
ASSEMBLY = SPINE / "connected_group_assembly.py"
GLOBAL_RIG = SPINE / "connected_group_global_rig.py"
SCHEDULE = SPINE / "connected_group_schedule.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_connected_setup_correction_is_blender_independent_and_data_only():
    source = _source(CORRECTION)
    tree = ast.parse(source, filename=str(CORRECTION))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "bpy" not in imported_modules
    assert "bmesh" not in imported_modules
    assert "compose_spine_documents" not in source
    assert "SpineSerializer" not in source
    assert "MeshAttachment" not in source
    assert "decode_weighted_vertices" not in source
    assert "encode_weighted_vertices" not in source
    assert "uvs" not in source
    assert "triangles" not in source


def test_connected_assembly_skips_setup_correction_for_legacy_wrapper():
    source = _source(ASSEMBLY)

    profile_branch = source.index(
        "if profile_id is A1RigProfile.TWO_AXIS_ROTATION_SCALE:"
    )
    correction_call = source.index("correct_connected_setup_pose(", profile_branch)
    scheduling_call = source.index("apply_connected_constraint_schedule(")
    validation_call = source.index(
        "_validate_connected_final(final_document, resolved_target)"
    )
    runtime_safety_call = source.index(
        "_validate_target_runtime_safety(final_document, resolved_target)"
    )

    assert (
        profile_branch
        < correction_call
        < scheduling_call
        < validation_call
        < runtime_safety_call
    )
    assert "if profile_id is A1RigProfile.THREE_AXIS_ROTATION" not in source[
        profile_branch:correction_call
    ]
    assert "from .connected_group_setup_correction import" in source


def test_legacy_global_wrapper_is_owned_separately_from_object_rig_builder():
    global_source = _source(GLOBAL_RIG)
    schedule_source = _source(SCHEDULE)

    assert "def _legacy_connected_bones(" in global_source
    assert "def _legacy_global_constraints(" in global_source
    assert "build_legacy_rig" not in global_source
    assert "def _assign_layer_phase(" in schedule_source
    assert "placement.layer_index" in schedule_source
    assert "_LEGACY_SCALE_COMPENSATOR_ORDER = 6" in schedule_source


def test_correction_rebuilds_immutable_documents_without_mutating_inputs():
    source = _source(CORRECTION)

    assert "from dataclasses import replace" in source
    assert "return replace(document, bones=tuple(bones))" in source
    assert "return replace(" in source
    assert ".append(" in source
    assert "bpy.ops" not in source
