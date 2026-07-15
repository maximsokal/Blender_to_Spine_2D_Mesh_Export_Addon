from unittest.mock import MagicMock, patch
import os

from Blender_to_Spine2D_Mesh_Exporter import main, pipeline_v2


class Poly:
    def __init__(self, idx):
        self.index = idx


class Mesh:
    def __init__(self, count):
        self.polygons = [Poly(index) for index in range(count)]
        self.storage = {}

    def __setitem__(self, key, value):
        self.storage[key] = value

    def __getitem__(self, key):
        return self.storage[key]


class Obj:
    def __init__(self, name, count):
        self.name = name
        self.data = Mesh(count)


class Image:
    def __init__(self, width, height):
        self.size = (width, height)


class Node:
    def __init__(self, image=None):
        self.type = "TEX_IMAGE"
        self.image = image


class NodeTree:
    def __init__(self, nodes):
        self.nodes = nodes


class Material:
    def __init__(self, nodes):
        self.node_tree = NodeTree(nodes)


class ObjMat(Obj):
    def __init__(self, name, count, material):
        super().__init__(name, count)
        self.active_material = material


def test_assign_face_ids():
    obj = Obj("o", 3)
    main.assign_face_ids(obj)
    assert obj.data["face_id_map"] == {"0": 0, "1": 1, "2": 2}


def test_get_texture_dimensions():
    image = Image(128, 256)
    material = Material([Node(image)])
    obj = ObjMat("o", 1, material)
    assert main.get_texture_dimensions(obj, 64, 64) == (128, 256)


def test_save_uv_as_json_uses_custom_output_dir(tmp_path):
    from contextlib import ExitStack

    custom_dir = tmp_path / "custom_json_output"
    custom_dir.mkdir()

    source = MagicMock(type="MESH")
    source.name = "TestCube"
    source.matrix_world.translation.copy.return_value = (0, 0, 0)
    source.data.__contains__.return_value = False

    working_copy = MagicMock(type="MESH")
    working_copy.name = "TestCube_copy_for_uv"
    source.copy.return_value = working_copy
    source.data.copy.return_value = working_copy.data

    textured = MagicMock(type="MESH")
    textured.name = "TestCube_texturing_copy"

    export_result = {
        "bones": [{"name": "root"}],
        "_uv3d_pairs": [[0, [0.1, 0.1], [0, 0, 0]]],
        "textured_uv3d_pairs": [[0, [0.1, 0.1], [0, 0, 0]]],
    }

    patches = [
        patch.object(pipeline_v2, "_mesh_object", return_value=source),
        patch.object(pipeline_v2, "_resolve_output_directory", return_value=str(custom_dir)),
        patch.object(pipeline_v2, "_ensure_source_uv", return_value="UVMap"),
        patch.object(pipeline_v2, "_copy_seams"),
        patch.object(pipeline_v2, "activate_object"),
        patch.object(pipeline_v2, "_object_ids", return_value=set()),
        patch.object(pipeline_v2, "_new_segment_objects", return_value=[]),
        patch.object(pipeline_v2, "apply_segmentation_seams"),
        patch.object(pipeline_v2, "copy_orig_face_id_layer", return_value=True),
        patch.object(pipeline_v2, "_prepare_textured_uv", return_value="UVMap_for_texturing"),
        patch.object(pipeline_v2, "_export_segments", return_value=[export_result]),
        patch.object(pipeline_v2, "scene_bool", return_value=False),
        patch.object(pipeline_v2.ExportSession, "cleanup"),
        patch.object(pipeline_v2.ExportSession, "restore_source_metadata"),
        patch.object(main, "assign_face_ids"),
        patch.object(main, "main_preprocessing", return_value={"z_groups_info": {0.0: {}}}),
        patch.object(main.plane_cut, "execute_smart_cut", return_value=[]),
        patch.object(main, "mark_seams_on_copy", return_value=(textured, [])),
        patch.object(main, "bake_textures_for_object", return_value=True),
        patch.object(main, "transfer_baked_uvs_to_segments"),
        patch.object(main, "merge_spine_json_dicts", return_value={"skeleton": {}}),
        patch.object(main, "write_json"),
    ]

    with ExitStack() as stack:
        entered = [stack.enter_context(item) for item in patches]
        write_json = entered[-1]
        result = main.save_uv_as_json(source, 512, 512, output_dir=str(custom_dir))

    expected_path = os.path.join(str(custom_dir), "TestCube_merged.json")
    assert result == expected_path
    write_json.assert_called_once_with({"skeleton": {}}, expected_path)
