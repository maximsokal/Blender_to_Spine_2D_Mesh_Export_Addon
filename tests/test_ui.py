"""Focused tests for the Blender 5.2+ Rewrite UI boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from mathutils import Vector

from Blender_to_Spine2D_Mesh_Exporter import ui


class _OperatorResult:
    def __init__(self, *, success: bool, issues=(), output_files=()):
        self.success = success
        self.issues = tuple(issues)
        self.output_files = tuple(output_files)


def test_scale_applied_uses_tolerance():
    assert ui.OBJECT_PT_Spine2DMeshPanel._scale_applied(
        SimpleNamespace(scale=(1.0, 1.0, 1.0))
    )
    assert ui.OBJECT_PT_Spine2DMeshPanel._scale_applied(
        SimpleNamespace(scale=(1.00001, 0.99999, 1.0))
    )
    assert not ui.OBJECT_PT_Spine2DMeshPanel._scale_applied(
        SimpleNamespace(scale=(1.1, 1.0, 1.0))
    )


def test_face_orientation_uses_inverse_transpose_normal_matrix():
    matrix_world = MagicMock()
    matrix_world.translation = Vector((0.0, 0.0, 0.0))
    matrix_world.__matmul__ = lambda _self, value: value

    matrix_3x3 = MagicMock()
    inverse = MagicMock()
    normal_matrix = MagicMock()
    matrix_world.to_3x3.return_value = matrix_3x3
    matrix_3x3.inverted_safe.return_value = inverse
    inverse.transposed.return_value = normal_matrix
    normal_matrix.__matmul__ = lambda _self, value: value

    outward = SimpleNamespace(
        center=Vector((1.0, 0.0, 0.0)),
        normal=Vector((1.0, 0.0, 0.0)),
    )
    inward = SimpleNamespace(
        center=Vector((-1.0, 0.0, 0.0)),
        normal=Vector((1.0, 0.0, 0.0)),
    )
    obj = SimpleNamespace(
        data=SimpleNamespace(polygons=(outward, inward)),
        matrix_world=matrix_world,
    )

    inverted, correct = ui.OBJECT_PT_Spine2DMeshPanel._face_orientation_stats(obj)

    assert (inverted, correct) == (1, 1)
    matrix_3x3.inverted_safe.assert_called_once_with()
    inverse.transposed.assert_called_once_with()


def test_reset_settings_resets_only_rewrite_properties(monkeypatch):
    scene = SimpleNamespace(
        spine2d_texture_size=256,
        spine2d_json_path="old",
        spine2d_images_path="old-images",
        spine2d_control_icons=False,
        spine2d_export_preview_animation=False,
        spine2d_angle_limit=5,
        spine2d_angular_mode="SEED_CONE_AND_LOCAL_DIHEDRAL",
        spine2d_local_angle_limit=90.0,
        spine2d_seam_maker_mode="CUSTOM",
        spine2d_frames_for_render=10,
        spine2d_bake_frame_start=8,
    )
    context = SimpleNamespace(scene=scene)
    operator = ui.SPINE2D_OT_ResetSettings()
    operator.report = MagicMock()
    monkeypatch.setattr(ui, "get_default_output_dir", lambda: "/exports")

    result = operator.execute(context)

    assert result == {"FINISHED"}
    assert scene.spine2d_texture_size == 1024
    assert scene.spine2d_json_path == "/exports"
    assert scene.spine2d_images_path == "images/"
    assert scene.spine2d_angle_limit == 30
    assert scene.spine2d_angular_mode == "LEGACY_SEED_CONE"
    assert scene.spine2d_local_angle_limit == 30.0
    assert scene.spine2d_seam_maker_mode == "AUTO"
    assert scene.spine2d_frames_for_render == 0
    assert scene.spine2d_bake_frame_start == 0
    operator.report.assert_called_once_with(
        {"INFO"},
        "Spine2D Rewrite settings have been reset.",
    )


def test_refresh_info_rejects_missing_active_mesh():
    operator = ui.OBJECT_OT_Spine2DRefreshInfo()
    operator.report = MagicMock()

    result = operator.execute(SimpleNamespace(active_object=None))

    assert result == {"CANCELLED"}
    operator.report.assert_called_once_with(
        {"ERROR"},
        "A valid active Mesh object is required",
    )


def test_multi_export_always_calls_rewrite(monkeypatch):
    expected = _OperatorResult(
        success=True,
        output_files=("/exports/result.json",),
    )
    calls: list[object] = []
    context = object()
    operator = ui.OBJECT_OT_Spine2DMultiExport()
    operator.report = MagicMock()

    monkeypatch.setattr(
        ui,
        "export_selected_objects_a1",
        lambda value: calls.append(value) or expected,
    )

    result = operator.execute(context)

    assert result == {"FINISHED"}
    assert calls == [context]
    operator.report.assert_called_once_with(
        {"INFO"},
        "Export finished → /exports/result.json",
    )


def test_ui_runtime_has_no_legacy_backend_property():
    property_names = tuple(registration.name for registration in ui.RNA_PROPERTIES)
    class_names = tuple(value.__name__ for value in ui.CLASSES)

    assert "spine2d_multi_export_backend" not in property_names
    assert "OBJECT_OT_Spine2DMultiExport" in class_names
    assert not hasattr(ui, "resolve_multi_backend")
    assert not hasattr(ui, "MULTI_BACKEND_PROPERTY")
