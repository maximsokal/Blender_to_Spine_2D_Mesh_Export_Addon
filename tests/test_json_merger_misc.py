from Blender_to_Spine2D_Mesh_Exporter import json_merger


def test_ensure_key_adds():
    d = {}
    val = json_merger.ensure_key(d, "key", 1)
    assert d["key"] == 1 and val == 1


def test_build_global_bone_index():
    bones = [{"name": "root"}, {"name": "child"}]
    idx = json_merger.build_global_bone_index(bones)
    assert idx == {"root": 0, "child": 1}


def test_merge_bones_and_recalc_indices():
    global_bones = [{"name": "root"}]
    name_map = {"root": 0}
    seg_bones = [{"name": "child"}, {"name": "root"}]
    mapping = json_merger.merge_bones_and_recalc_indices(
        global_bones, name_map, seg_bones
    )
    assert mapping == {0: 1, 1: 0}
    assert global_bones[-1]["name"] == "child"


def test_fix_attachment_bone_indices():
    att = {"type": "mesh", "vertices": [1, 2, 0.1, 0.2, 0.3]}
    json_merger.fix_attachment_bone_indices(att, {2: 5})
    assert att["vertices"] == [1, 5, 0.1, 0.2, 0.3]
