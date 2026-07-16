from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_materials import (
    BakeMaterialError,
    _apply_face_material_indices,
)


class FakePolygon:
    def __init__(self, material_index=0):
        self.material_index = material_index


class FakeMesh:
    def __init__(self, count):
        self.polygons = tuple(FakePolygon() for _ in range(count))


def test_face_material_indices_are_restored_after_slots_exist():
    mesh = FakeMesh(3)

    _apply_face_material_indices(
        mesh,
        (0, 2, 1),
        material_slot_count=3,
    )

    assert tuple(polygon.material_index for polygon in mesh.polygons) == (0, 2, 1)


def test_face_material_binding_rejects_polygon_count_mismatch():
    with pytest.raises(BakeMaterialError, match="face material indices"):
        _apply_face_material_indices(
            FakeMesh(2),
            (0,),
            material_slot_count=1,
        )


@pytest.mark.parametrize("indices, slots", (((0, 2), 2), ((-1, 0), 2)))
def test_face_material_binding_rejects_out_of_range_indices(indices, slots):
    with pytest.raises(BakeMaterialError, match="references material slot"):
        _apply_face_material_indices(
            FakeMesh(len(indices)),
            indices,
            material_slot_count=slots,
        )
