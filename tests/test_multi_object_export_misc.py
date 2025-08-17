# tests/test_multi_object_export_misc.py
from Blender_to_Spine2D_Mesh_Exporter import multi_object_export as moe
from mathutils import Vector  # Import the "fake" Vector from conftest

# --- Helper classes for tests ---
# These classes are needed because they are specific to these tests


class MockMatrix:
    """Mock for matrix_world"""

    def __init__(self, t):
        self.translation = t


class MockObject:
    """Mock for a Blender object"""

    def __init__(self, name, x, y, z):
        self.name = name
        self.matrix_world = MockMatrix(Vector((x, y, z)))


# --- Tests ---


def test_calc_offsets():
    """Tests the calculation of offsets between objects."""
    objs = [MockObject("a", 0, 0, 0), MockObject("b", 1, 2, 3)]
    res = moe._calc_offsets(objs)
    assert res["b"]["dx"] == 1
    assert res["b"]["dy"] == 2
    assert res["b"]["dz"] == 3
    assert "a" in res
    assert res["a"]["dx"] == 0


def test_group_by_z():
    """Tests grouping objects by Z-coordinate."""
    offsets = {"a": {"dz": 0.0}, "b": {"dz": 0.1}, "c": {"dz": 0.00001}}
    z_map, obj_map = moe._group_by_z(offsets, eps=1e-4)
    assert obj_map["b"] != obj_map["a"]
    assert obj_map["a"] == obj_map["c"]


def test_mk_bone():
    """Tests bone creation."""
    bone = moe._mk_bone("my_bone", "root", length=10.5, x=5.2, custom_prop="value")
    assert bone["name"] == "my_bone"
    assert bone["parent"] == "root"
    assert bone["length"] == 10.5
    assert bone["x"] == 5.2
    assert bone["custom_prop"] == "value"


def test_add_unique():
    """Tests adding unique elements to a list of dictionaries."""
    # Test by 'name' key
    dst = [{"name": "a", "val": 1}]
    moe._add_unique(dst, {"name": "b", "val": 2})
    assert len(dst) == 2
    moe._add_unique(dst, {"name": "a", "val": 3})
    assert len(dst) == 2
    assert dst[0]["val"] == 1

    # Test by another key 'id'
    dst_keyed = [{"id": 1, "val": 10}]
    moe._add_unique(dst_keyed, {"id": 2, "val": 20}, key="id")
    assert len(dst_keyed) == 2
    moe._add_unique(dst_keyed, {"id": 1, "val": 30}, key="id")
    assert len(dst_keyed) == 2
