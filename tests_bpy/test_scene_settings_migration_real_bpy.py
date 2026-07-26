import bpy

import Blender_to_Spine2D_Mesh_Exporter as extension
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    migrate_scene_settings,
    spine2d_scene_settings_load_post,
)


def test_old_scene_migrates_to_auto_once_and_preserves_later_custom_choice():
    extension.register()
    try:
        scene = bpy.context.scene
        scene.spine2d_settings_schema_version = 0
        scene.spine2d_seam_maker_mode = "CUSTOM"

        assert migrate_scene_settings(scene)
        assert scene.spine2d_seam_maker_mode == "AUTO"
        assert (
            scene.spine2d_settings_schema_version
            == CURRENT_SETTINGS_SCHEMA_VERSION
        )

        scene.spine2d_seam_maker_mode = "CUSTOM"
        assert not migrate_scene_settings(scene)
        assert scene.spine2d_seam_maker_mode == "CUSTOM"
        assert spine2d_scene_settings_load_post in bpy.app.handlers.load_post
    finally:
        extension.unregister()

    assert spine2d_scene_settings_load_post not in bpy.app.handlers.load_post
