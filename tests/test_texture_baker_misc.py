from Blender_to_Spine2D_Mesh_Exporter import texture_baker
import os
from unittest.mock import MagicMock, patch
import bpy


class Image:
    def __init__(self, pixels):
        self.pixels = pixels
        self.size = (2, 1)
        self.name = "img"


class Node:
    def __init__(self, typ, name):
        self.type = typ
        self.name = name


class NodeTree:
    def __init__(self, nodes):
        self.nodes = nodes
        self.links = []


class Material:
    def __init__(self, nodes):
        self.node_tree = NodeTree(nodes)


def test_check_bake_image():
    img = Image([1, 0, 0, 1, 0, 0, 0, 0])
    assert texture_baker.check_bake_image(img, texture_baker.logger)


def test_is_multi_image_material():
    nodes = [Node("TEX_IMAGE", "A"), Node("TEX_IMAGE", "B")]
    mat = Material(nodes)
    assert texture_baker.is_multi_image_material(mat)
    mat.node_tree.nodes = [Node("TEX_IMAGE", "TEMP_BAKE_X")]
    assert not texture_baker.is_multi_image_material(mat)


def test_bake_textures_for_object_uses_custom_images_path(tmp_path):
    """
    Checks that bake_textures_for_object correctly constructs the absolute path
    for saving textures, based on the JSON path, not the disk root.
    """
    # Arrange

    # 1. Simulate a realistic folder structure using the pytest fixture `tmp_path`
    json_output_dir = tmp_path / "json_output"
    images_relative_path = "my_custom_images"  # Relative path, as in the UI

    # Expected result: the full, absolute path that our code should generate
    expected_full_path = os.path.join(str(json_output_dir), images_relative_path)

    # 2. Set up the mock scene to return our paths
    mock_scene = MagicMock()
    mock_scene.spine2d_images_path = images_relative_path
    mock_scene.spine2d_json_path = str(json_output_dir)
    bpy.context.scene = mock_scene

    # 3. Mock dependencies to isolate the function under test
    mock_obj = MagicMock()
    mock_obj.name = "TestObject"

    # Use patch.object for get_default_output_dir so it doesn't interfere
    with patch(
        "Blender_to_Spine2D_Mesh_Exporter.texture_baker_integration.TextureBaker"
    ) as mock_baker_class, patch(
        "Blender_to_Spine2D_Mesh_Exporter.texture_baker_integration.get_texture_size",
        return_value=512,
    ), patch(
        "Blender_to_Spine2D_Mesh_Exporter.texture_baker_integration.sync_bake_sequence_params"
    ), patch(
        "Blender_to_Spine2D_Mesh_Exporter.main.get_default_output_dir",
        return_value=str(json_output_dir),
    ), patch("os.makedirs") as mock_makedirs:
        # Set up the TextureBaker mock so it doesn't perform actual baking
        mock_baker_instance = MagicMock()
        mock_baker_instance.setup_object_for_baking.return_value = (
            MagicMock(),
            MagicMock(),
            [MagicMock()],
        )
        mock_baker_instance.execute_baking.return_value = True
        mock_baker_instance.create_combined_material.return_value = MagicMock()
        mock_baker_instance.apply_material_to_object.return_value = True
        mock_baker_class.return_value = mock_baker_instance

        # Act
        from Blender_to_Spine2D_Mesh_Exporter.texture_baker_integration import (
            bake_textures_for_object,
        )

        bake_textures_for_object(mock_obj, "UVMap")

        # Assert

        # 1. Check that the folder is created at the correct, full path
        mock_makedirs.assert_called_once_with(expected_full_path, exist_ok=True)

        # 2. Check that TextureBaker was initialized with the same correct path
        mock_baker_class.assert_called_once()
        _, called_kwargs = mock_baker_class.call_args
        assert called_kwargs["output_dir"] == expected_full_path
