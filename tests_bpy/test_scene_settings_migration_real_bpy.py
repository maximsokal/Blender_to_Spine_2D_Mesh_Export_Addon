import bpy

import Blender_to_Spine2D_Mesh_Exporter as extension
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    migrate_scene_settings,
    spine2d_scene_settings_load_post,
    spine2d_scene_settings_load_pre,
)


def test_schema_one_scene_migrates_to_auto_and_preserves_later_custom_choice():
    extension.register()
    try:
        scene = bpy.context.scene

        # Reproduce the affected 0.38 state: the first migration marker was already
        # persisted, but the Scene still contains CUSTOM. Set the marker last because a
        # normal post-load user edit intentionally marks the Scene current.
        scene.spine2d_seam_maker_mode = "CUSTOM"
        scene.spine2d_settings_schema_version = 1

        assert CURRENT_SETTINGS_SCHEMA_VERSION == 2
        assert migrate_scene_settings(scene)
        assert scene.spine2d_seam_maker_mode == "AUTO"
        assert (
            scene.spine2d_settings_schema_version
            == CURRENT_SETTINGS_SCHEMA_VERSION
        )

        # A deliberate choice made after schema 2 must remain stable.
        scene.spine2d_seam_maker_mode = "CUSTOM"
        assert (
            scene.spine2d_settings_schema_version
            == CURRENT_SETTINGS_SCHEMA_VERSION
        )
        assert not migrate_scene_settings(scene)
        assert scene.spine2d_seam_maker_mode == "CUSTOM"
        assert spine2d_scene_settings_load_pre in bpy.app.handlers.load_pre
        assert spine2d_scene_settings_load_post in bpy.app.handlers.load_post
    finally:
        extension.unregister()

    assert spine2d_scene_settings_load_pre not in bpy.app.handlers.load_pre
    assert spine2d_scene_settings_load_post not in bpy.app.handlers.load_post
