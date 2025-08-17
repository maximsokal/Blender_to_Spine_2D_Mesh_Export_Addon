from Blender_to_Spine2D_Mesh_Exporter import json_export
import pytest
import bpy
from unittest.mock import patch
import sys


class P:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def test_generate_hash_length():
    h = json_export.generate_hash("some/file.blend")
    assert isinstance(h, str) and len(h) == 10


def test_check_euler_characteristic_triangle():
    vertices = [0.0] * 6  # 3 vertices
    edge_map = {(0, 1): 1, (1, 2): 1, (2, 0): 1}
    triangles = [0, 1, 2]
    assert json_export.check_euler_characteristic(vertices, edge_map, triangles) == 1


def test_match_segment_vertex_to_original_found():
    seg = P(1.0, 2.0, 3.0)
    orig = [[0, [1.0, 2.0, 3.0], []]]
    assert json_export.match_segment_vertex_to_original(seg, orig) == 0


def test_match_uv_from_original():
    seg_coord = (1.0, 1.0, 1.0)
    data = [
        [0, [0.1, 0.2], [1.0, 1.0, 1.0]],
        [1, [0.2, 0.3], [2.0, 2.0, 2.0]],
    ]
    used = set()
    uv = json_export.match_uv_from_original(seg_coord, data, used, 2)
    assert uv == [0.1, 0.2] and 0 in used


def test_unique_uv3d_pairs():
    pairs = [
        [0, [0.1234, 0.5678], [0, 0, 0]],
        [1, [0.12341, 0.56781], [0, 0, 0]],
        [2, [0.5, 0.5], [0, 0, 0]],
    ]
    unique = json_export.unique_uv3d_pairs(pairs)
    assert len(unique) == 2


def test_verify_function_replacement_true():
    def dummy():
        uniform_scale = 1.0
        z_displacement = 0.5
        calculation_method = "direct_3d_scaling"
        return json_export.verify_function_replacement()

    assert dummy() is True


def test_verify_function_replacement_false():
    def dummy():
        return json_export.verify_function_replacement()

    assert dummy() is False


@pytest.mark.parametrize(
    "json_path_in, images_path_in, expected_prefix",
    [
        ("E:/projects/my_game/", "./images/", "images/"),
        ("/var/tmp/output/json_files", "textures/", "textures/"),
        # Windows
        pytest.param(
            "E:/projects/my_game/json/",
            "E:/projects/my_game/assets/images/",
            "../assets/images/",
            marks=pytest.mark.skipif(
                sys.platform != "win32", reason="Windows-specific path test"
            ),
        ),
        pytest.param(
            "C:/json_folder/",
            "D:/image_folder/",
            "D:/image_folder/",
            marks=pytest.mark.skipif(
                sys.platform != "win32", reason="Windows-specific path test"
            ),
        ),
    ],
)
def test_construct_spine_json_uses_images_path_prefix(
    json_path_in, images_path_in, expected_prefix
):
    """
    Checks that construct_spine_json correctly forms the path to the images:
    - Creates a relative path if the folders are on the same drive.
    - Leaves the path absolute if they are on different drives (for Windows).
    """

    # Arrange
    def mock_scene_get(key, default=None):
        if key == "spine2d_images_path":
            return images_path_in
        if key == "spine2d_json_path":
            return json_path_in
        return default

    bpy.context.scene.get.side_effect = mock_scene_get
    bpy.context.scene.spine2d_frames_for_render = 0

    # Mock get_default_output_dir to return our test path
    with patch(
        "Blender_to_Spine2D_Mesh_Exporter.json_export.get_default_output_dir",
        return_value=json_path_in,
    ):
        # Act
        json_data, _ = json_export.construct_spine_json(
            attachment_name="MyObject",
            processed_vertices=[0, 0, 1, 1],
            uvs=[0, 0, 1, 1],
            triangles=[0, 1, 2],
            hull=3,
            edges=[],
            texture_width=128,
            texture_height=128,
            uv3d_pairs=[],
            textured_uv3d_pairs=[],
            z_groups=[],
            original_z_groups=[],
            center_x=0,
            center_y=0,
            pixels_per_blender_unit_width=1,
            pixels_per_blender_unit_length=1,
            original_object_name="MyObject_copy_for_uv",
            is_segment=False,
            segment_obj_name=None,
            original_uv3d_pairs=None,
            z_groups_info={},
            world_location=None,
        )

    # Assert
    skin_attachments = json_data["skins"][0]["attachments"]
    mesh_data = skin_attachments["MyObject"]["MyObject"]
    image_path = mesh_data["path"]
    expected_filename = "MyObject_texturing_Baked"
    assert image_path == f"{expected_prefix}{expected_filename}"
