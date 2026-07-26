"""Focused tests for the related re-polish project link."""

from __future__ import annotations

from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter import repolish_ui


class _RecordingLayout:
    def __init__(self) -> None:
        self.labels: list[dict[str, str]] = []
        self.operators: list[tuple[str, dict[str, str], object]] = []

    def label(self, **kwargs) -> None:
        self.labels.append(dict(kwargs))

    def operator(self, operator_id: str, **kwargs):
        operator = SimpleNamespace(url=None)
        self.operators.append((operator_id, dict(kwargs), operator))
        return operator


def test_repolish_button_is_headerless_last_child_of_main_exporter_panel():
    panel = repolish_ui.OBJECT_PT_Spine2DRePolishPanel

    assert panel.bl_label == "re-polish"
    assert panel.bl_parent_id == "OBJECT_PT_spine2d_mesh"
    assert panel.bl_space_type == "VIEW_3D"
    assert panel.bl_region_type == "UI"
    assert panel.bl_category == "Blender to Spine2D Mesh Exporter"
    assert panel.bl_options == {"HIDE_HEADER"}
    assert panel.bl_order == 1000


def test_repolish_panel_draws_caption_and_lowercase_button_with_exact_url():
    layout = _RecordingLayout()
    panel_instance = SimpleNamespace(layout=layout)

    repolish_ui.OBJECT_PT_Spine2DRePolishPanel.draw(panel_instance, None)

    assert layout.labels == [{"text": "Try animation optimization"}]
    assert len(layout.operators) == 1
    operator_id, properties, operator = layout.operators[0]
    assert operator_id == "wm.url_open"
    assert properties == {
        "text": "re-polish",
        "icon": "URL",
    }
    assert operator.url == "https://www.re-polish.com/"
    assert operator.url == repolish_ui.REPOLISH_URL


def test_repolish_module_owns_only_its_panel_registration():
    assert repolish_ui.CLASSES == (
        repolish_ui.OBJECT_PT_Spine2DRePolishPanel,
    )
