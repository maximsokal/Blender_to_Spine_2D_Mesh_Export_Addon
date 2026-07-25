from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from Blender_to_Spine2D_Mesh_Exporter import ui


def _scene(**overrides):
    values = {
        "spine2d_seam_maker_mode": "AUTO",
        "spine2d_angle_limit": 30,
        "spine2d_angular_mode": "SEED_CONE",
        "spine2d_local_angle_limit": 30.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _prop_names(column):
    return tuple(call.args[1] for call in column.prop.call_args_list)


def test_angular_scene_properties_are_registered_symmetrically():
    names = tuple(name for name, _ in ui.SCENE_PROPERTIES)

    assert names.count("spine2d_angular_mode") == 1
    assert names.count("spine2d_local_angle_limit") == 1
    assert len(names) == len(set(names))


def test_reset_restores_seed_cone_contract():
    operator = SimpleNamespace(report=MagicMock())
    context = SimpleNamespace(scene=SimpleNamespace())

    with patch(
        "Blender_to_Spine2D_Mesh_Exporter.ui.get_default_output_dir",
        return_value="/tmp/spine",
    ):
        result = ui.SPINE2D_OT_ResetSettings.execute(operator, context)

    assert result == {"FINISHED"}
    assert context.scene.spine2d_angular_mode == "SEED_CONE"
    assert context.scene.spine2d_local_angle_limit == 30.0


def test_seed_cone_cut_ui_does_not_draw_local_limit():
    column = MagicMock()

    ui.OBJECT_PT_Spine2DMeshPanel._draw_cut_settings(
        column,
        _scene(),
    )

    assert _prop_names(column) == (
        "spine2d_seam_maker_mode",
        "spine2d_angle_limit",
        "spine2d_angular_mode",
    )


def test_hybrid_cut_ui_draws_independent_local_limit():
    column = MagicMock()

    ui.OBJECT_PT_Spine2DMeshPanel._draw_cut_settings(
        column,
        _scene(spine2d_angular_mode="SEED_CONE_AND_LOCAL_DIHEDRAL"),
    )

    assert _prop_names(column) == (
        "spine2d_seam_maker_mode",
        "spine2d_angle_limit",
        "spine2d_angular_mode",
        "spine2d_local_angle_limit",
    )


def test_custom_seam_mode_hides_unused_angular_controls():
    column = MagicMock()

    ui.OBJECT_PT_Spine2DMeshPanel._draw_cut_settings(
        column,
        _scene(spine2d_seam_maker_mode="CUSTOM"),
    )

    assert _prop_names(column) == ("spine2d_seam_maker_mode",)
    column.label.assert_called_once_with(
        text="Angular splitting is disabled in Custom seam mode",
        icon="INFO",
    )
