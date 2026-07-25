from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.single_object_operator import (
    OBJECT_OT_SaveUVAsJSON,
)
from Blender_to_Spine2D_Mesh_Exporter.ui import (
    OBJECT_OT_Spine2DMultiExport,
    OBJECT_OT_Spine2DSingleExport,
)


def _obj(object_type, *, data=object()):
    return SimpleNamespace(type=object_type, data=data)


def test_single_operator_poll_requires_active_mesh_with_data():
    valid = SimpleNamespace(active_object=_obj("MESH"))
    no_active = SimpleNamespace(active_object=None)
    camera = SimpleNamespace(active_object=_obj("CAMERA"))
    missing_data = SimpleNamespace(active_object=_obj("MESH", data=None))

    for operator in (OBJECT_OT_SaveUVAsJSON, OBJECT_OT_Spine2DSingleExport):
        assert operator.poll(valid)
        assert not operator.poll(no_active)
        assert not operator.poll(camera)
        assert not operator.poll(missing_data)


def test_multi_operator_poll_accepts_any_selected_mesh_with_data():
    assert OBJECT_OT_Spine2DMultiExport.poll(
        SimpleNamespace(selected_objects=(_obj("CAMERA"), _obj("MESH")))
    )
    assert not OBJECT_OT_Spine2DMultiExport.poll(
        SimpleNamespace(selected_objects=(_obj("CAMERA"), _obj("LIGHT")))
    )
    assert not OBJECT_OT_Spine2DMultiExport.poll(
        SimpleNamespace(selected_objects=(_obj("MESH", data=None),))
    )
    assert not OBJECT_OT_Spine2DMultiExport.poll(SimpleNamespace(selected_objects=()))
    assert not OBJECT_OT_Spine2DMultiExport.poll(SimpleNamespace())
