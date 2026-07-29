from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyRigProfile,
    build_connected_group_document,
)

from test_connected_group_document import connected_objects, settings


ROOT = Path(__file__).resolve().parents[1]
GLOBAL_RIG_SOURCE = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "connected_group_global_rig.py"
)


def _bone(document, name):
    return next(item for item in document.bones if item.name == name)


def test_connected_three_axis_global_wrapper_uses_main_branch_helper_geometry():
    profile = LegacyRigProfile()
    result = build_connected_group_document(
        connected_objects(),
        settings(),
        profile=profile,
    )
    prefix = "all_objects"
    scale = _bone(result.document, profile.scale_rotate_x_bone(prefix))
    constraint, constraint_scale, constraint_rotate, constraint_target = (
        profile.ik_chain_bones(prefix)
    )

    assert (scale.parent, scale.length, scale.y, scale.scale_x) == (
        profile.base_bone(prefix),
        50.0,
        0.0,
        None,
    )
    assert (
        _bone(result.document, constraint).parent,
        _bone(result.document, constraint).length,
        _bone(result.document, constraint).y,
        _bone(result.document, constraint).rotation,
    ) == (profile.base_bone(prefix), 50.0, 0.0, -90.0)
    assert (
        _bone(result.document, constraint_scale).parent,
        _bone(result.document, constraint_scale).y,
        _bone(result.document, constraint_scale).rotation,
        _bone(result.document, constraint_scale).scale_x,
    ) == (profile.base_bone(prefix), 0.0, -90.0, None)
    assert (
        _bone(result.document, constraint_rotate).parent,
        _bone(result.document, constraint_rotate).x,
    ) == (constraint_scale, 0.0)
    assert (
        _bone(result.document, constraint_target).parent,
        _bone(result.document, constraint_target).x,
        _bone(result.document, constraint_target).rotation,
    ) == (constraint_rotate, 0.0, 90.0)

    for control_name in profile.control_bones(prefix):
        assert _bone(result.document, control_name).parent == profile.root_bone()


def test_connected_global_owner_keeps_dedicated_legacy_wrapper_and_profiled_two_axis():
    source = GLOBAL_RIG_SOURCE.read_text(encoding="utf-8")

    assert "def _legacy_connected_bones(" in source
    assert "def _legacy_global_constraints(" in source
    assert "build_legacy_rig" not in source
    assert "from .two_axis_scale_rig_assembly import build_two_axis_scale_rig" in source
    assert "def _two_axis_global_constraints(" in source
