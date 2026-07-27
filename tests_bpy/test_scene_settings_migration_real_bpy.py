import bpy

import Blender_to_Spine2D_Mesh_Exporter as extension
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    migrate_scene_settings,
    spine2d_scene_settings_load_post,
    spine2d_scene_settings_load_pre,
)


_SEAM_MODE_PROPERTY = "spine2d_seam_maker_mode"
_SCHEMA_PROPERTY = "spine2d_settings_schema_version"


def _remove_persisted_test_values(scene) -> None:
    for property_name in (_SEAM_MODE_PROPERTY, _SCHEMA_PROPERTY):
        try:
            if property_name in scene:
                del scene[property_name]
        except Exception:
            # Cleanup should not hide the registration assertion that failed first.
            pass


def test_schema_two_custom_scene_is_repaired_during_extension_registration():
    # Reproduce the actual user path: the .blend already contains values written by 0.39,
    # while the extension RNA surface is not registered yet. Registering EnumProperty over
    # these ID properties may invoke its update callback, which must not advance schema 3
    # before the migration owner resets CUSTOM to AUTO.
    extension.unregister()
    scene = bpy.context.scene
    _remove_persisted_test_values(scene)
    scene[_SEAM_MODE_PROPERTY] = "CUSTOM"
    scene[_SCHEMA_PROPERTY] = 2

    extension.register()
    try:
        assert CURRENT_SETTINGS_SCHEMA_VERSION == 3
        assert scene.spine2d_settings_schema_version == 3
        assert scene.spine2d_seam_maker_mode == "AUTO"

        # A deliberate choice made after schema 3 remains stable.
        scene.spine2d_seam_maker_mode = "CUSTOM"
        assert scene.spine2d_settings_schema_version == 3
        assert not migrate_scene_settings(scene)
        assert scene.spine2d_seam_maker_mode == "CUSTOM"

        assert spine2d_scene_settings_load_pre in bpy.app.handlers.load_pre
        assert spine2d_scene_settings_load_post in bpy.app.handlers.load_post
    finally:
        extension.unregister()
        _remove_persisted_test_values(scene)

    assert spine2d_scene_settings_load_pre not in bpy.app.handlers.load_pre
    assert spine2d_scene_settings_load_post not in bpy.app.handlers.load_post
