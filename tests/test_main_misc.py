from Blender_to_Spine2D_Mesh_Exporter import main
from unittest.mock import patch, MagicMock
import os


class Poly:
    def __init__(self, idx):
        self.index = idx


class Mesh:
    def __init__(self, n):
        self.polygons = [Poly(i) for i in range(n)]
        self.storage = {}

    def __setitem__(self, key, val):
        self.storage[key] = val

    def __getitem__(self, key):
        return self.storage[key]


class Obj:
    def __init__(self, name, n):
        self.name = name
        self.data = Mesh(n)


class Image:
    def __init__(self, w, h):
        self.size = (w, h)


class Node:
    def __init__(self, img=None):
        self.type = "TEX_IMAGE"
        self.image = img


class NodeTree:
    def __init__(self, nodes):
        self.nodes = nodes


class Material:
    def __init__(self, nodes):
        self.node_tree = NodeTree(nodes)


class ObjMat(Obj):
    def __init__(self, name, n, mat):
        super().__init__(name, n)
        self.active_material = mat


def test_assign_face_ids():
    obj = Obj("o", 3)
    main.assign_face_ids(obj)
    assert obj.data["face_id_map"] == {"0": 0, "1": 1, "2": 2}


# Duplicate function definition removed
def test_get_texture_dimensions():
    img = Image(128, 256)
    mat = Material([Node(img)])
    obj = ObjMat("o", 1, mat)
    w, h = main.get_texture_dimensions(obj, 64, 64)
    assert (w, h) == (128, 256)


def test_save_uv_as_json_uses_custom_output_dir(tmp_path):
    """
    This test checks that save_uv_as_json uses the specified
    'output_dir' directory to save the final JSON,
    and not the default directory.
    """
    # 1. Setup
    custom_dir = tmp_path / "custom_json_output"
    custom_dir.mkdir()

    mock_uv_layers = MagicMock()
    mock_uv_layers.active = MagicMock()

    mock_obj = MagicMock()
    mock_obj.name = "TestCube"
    mock_obj.type = "MESH"
    mock_obj.matrix_world.translation.copy.return_value = (0, 0, 0)
    mock_obj.data.uv_layers = mock_uv_layers

    mock_textured_obj = MagicMock()
    mock_textured_obj.name = "textured_mock"
    mock_textured_obj.type = "MESH"
    mock_textured_obj.data.uv_layers = mock_uv_layers

    mock_export_return = {
        "bones": [{"name": "root"}],
        "_uv3d_pairs": [[0, [0.1, 0.1], [0, 0, 0]]],  # Add dummy data
        "textured_uv3d_pairs": [[0, [0.1, 0.1], [0, 0, 0]]],
    }

    with patch(
        "Blender_to_Spine2D_Mesh_Exporter.main._export_segment",
        return_value=mock_export_return,
    ) as mock_export, patch(
        "Blender_to_Spine2D_Mesh_Exporter.main.merge_spine_json_dicts",
        return_value={"skeleton": {}},
    ) as mock_merge, patch(
        "Blender_to_Spine2D_Mesh_Exporter.main.write_json"
    ) as mock_write_json, patch(
        "Blender_to_Spine2D_Mesh_Exporter.main.plane_cut.execute_smart_cut",
        return_value=[],
    ), patch(
        "Blender_to_Spine2D_Mesh_Exporter.main.main_preprocessing",
        return_value={"z_groups_info": {0.0: {}}},
    ), patch(
        "Blender_to_Spine2D_Mesh_Exporter.main.bake_textures_for_object",
        return_value=True,
    ), patch(
        "Blender_to_Spine2D_Mesh_Exporter.main.transfer_baked_uvs_to_segments"
    ), patch(
        "Blender_to_Spine2D_Mesh_Exporter.main.mark_seams_on_copy",
        return_value=(mock_textured_obj, []),
    ), patch("bpy.context.scene.objects", return_value=[]):
        # 2. Execution
        main.save_uv_as_json(mock_obj, 512, 512, output_dir=str(custom_dir))

        # 3. Verification
        mock_write_json.assert_called_once()

        call_args = mock_write_json.call_args
        final_path = call_args[0][1]

        expected_path = os.path.join(str(custom_dir), f"{mock_obj.name}_merged.json")

        assert final_path == expected_path
