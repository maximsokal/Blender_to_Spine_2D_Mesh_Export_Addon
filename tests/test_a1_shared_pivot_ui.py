from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from Blender_to_Spine2D_Mesh_Exporter import rig_ui
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import A1TextureExportMode
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionDirection


def _mesh(name: str):
    return SimpleNamespace(type="MESH", name=name, data=object())


def _context(
    *,
    direction: A1ProjectionDirection = A1ProjectionDirection.POSITIVE_Z,
    selected_count: int = 2,
    enabled: bool = True,
):
    scene = SimpleNamespace(
        spine2d_projection_direction=direction.value,
        spine2d_shared_selection_pivot=enabled,
    )
    return SimpleNamespace(
        scene=scene,
        selected_objects=tuple(_mesh(f"Mesh_{index}") for index in range(selected_count)),
    )


def _prop_names(layout: MagicMock) -> tuple[str, ...]:
    return tuple(call.args[1] for call in layout.prop.call_args_list)


def test_shared_pivot_control_is_visible_for_multiple_signed_axis_meshes() -> None:
    layout = MagicMock()
    context = _context(selected_count=3)

    rig_ui._draw_shared_selection_pivot(layout, context)

    assert _prop_names(layout) == ("spine2d_shared_selection_pivot",)
    layout.label.assert_called_once_with(
        text="Pivot: center of all selected exported Mesh geometry",
        icon="CON_PIVOT",
    )


def test_shared_pivot_control_is_hidden_for_one_mesh() -> None:
    layout = MagicMock()
    context = _context(selected_count=1)

    rig_ui._draw_shared_selection_pivot(layout, context)

    layout.prop.assert_not_called()
    layout.label.assert_not_called()


def test_shared_pivot_control_is_hidden_for_active_camera() -> None:
    layout = MagicMock()
    context = _context(direction=A1ProjectionDirection.ACTIVE_CAMERA, selected_count=4)

    rig_ui._draw_shared_selection_pivot(layout, context)

    layout.prop.assert_not_called()
    layout.label.assert_not_called()


def test_shared_pivot_available_is_false_for_camera_export_modes() -> None:
    context = _context(selected_count=4)

    assert not rig_ui._shared_pivot_available(
        context,
        A1TextureExportMode.CAMERA_PROJECTION,
    )
    assert not rig_ui._shared_pivot_available(
        context,
        A1TextureExportMode.DEPTH_CAMERA_PROJECTION,
    )


def test_shared_pivot_control_stays_visible_when_user_disables_it() -> None:
    layout = MagicMock()
    context = _context(selected_count=2, enabled=False)

    rig_ui._draw_shared_selection_pivot(layout, context)

    assert _prop_names(layout) == ("spine2d_shared_selection_pivot",)
    layout.label.assert_not_called()
