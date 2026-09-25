from math import isclose, radians

import bpy

import Blender_to_Spine2D_Mesh_Exporter as extension
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.scene_settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    migrate_scene_settings,
    spine2d_scene_settings_load_post,
    spine2d_scene_settings_load_pre,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    A1RigProfile,
    SpineJsonTarget,
)


_SEAM_MODE_PROPERTY = "spine2d_seam_maker_mode"
_RIG_PROFILE_PROPERTY = "spine2d_rig_profile"
_TARGET_VERSION_PROPERTY = "spine2d_target_spine_version"
_SCHEMA_PROPERTY = "spine2d_settings_schema_version"
_PARALLAX_HORIZON_PROPERTY = "spine2d_depth_parallax_horizon_angle"


def _remove_persisted_test_values(scene) -> None:
    """Remove every Scene ID property owned by this migration fixture."""

    for property_name in (
        _SEAM_MODE_PROPERTY,
        _RIG_PROFILE_PROPERTY,
        _TARGET_VERSION_PROPERTY,
        _SCHEMA_PROPERTY,
        _PARALLAX_HORIZON_PROPERTY,
    ):
        try:
            if property_name in scene:
                del scene[property_name]
        except Exception:
            # Cleanup should not hide the registration assertion that failed first.
            pass


def _assert_extension_unregistered() -> None:
    """Fail on a leaked previous lifecycle instead of mutating it away."""

    assert not hasattr(bpy.types.Scene, "spine2d_texture_export_mode")
    assert spine2d_scene_settings_load_pre not in bpy.app.handlers.load_pre
    assert spine2d_scene_settings_load_post not in bpy.app.handlers.load_post


def test_schema_two_custom_scene_is_repaired_during_extension_registration():
    """Real registration migrates legacy schema 2 through current schema 8 once."""

    # Reproduce the actual user path: the .blend already contains values written by 0.39,
    # while the extension RNA surface is not registered yet. Registering EnumProperty over
    # these ID properties may invoke its update callback, which must not advance the current
    # schema before migration resets CUSTOM and assigns compatibility defaults.
    _assert_extension_unregistered()
    scene = bpy.context.scene
    _remove_persisted_test_values(scene)
    scene[_SEAM_MODE_PROPERTY] = "CUSTOM"
    scene[_SCHEMA_PROPERTY] = 2

    extension.register()
    try:
        assert CURRENT_SETTINGS_SCHEMA_VERSION == 8
        assert (
            scene.spine2d_settings_schema_version
            == CURRENT_SETTINGS_SCHEMA_VERSION
        )
        assert scene.spine2d_seam_maker_mode == "AUTO"
        assert (
            scene.spine2d_rig_profile
            == A1RigProfile.THREE_AXIS_ROTATION.value
        )
        assert (
            scene.spine2d_target_spine_version
            == SpineJsonTarget.SPINE_4_2.value
        )
        assert isclose(
            float(scene.spine2d_depth_parallax_horizon_angle),
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )

        # Deliberate choices made after the current schema migration remain stable.
        scene.spine2d_seam_maker_mode = "CUSTOM"
        scene.spine2d_rig_profile = A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
        scene.spine2d_target_spine_version = SpineJsonTarget.SPINE_3_8.value
        scene.spine2d_depth_parallax_horizon_angle = radians(30.0)
        assert (
            scene.spine2d_settings_schema_version
            == CURRENT_SETTINGS_SCHEMA_VERSION
        )
        assert not migrate_scene_settings(scene)
        assert scene.spine2d_seam_maker_mode == "CUSTOM"
        assert (
            scene.spine2d_rig_profile
            == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
        )
        assert (
            scene.spine2d_target_spine_version
            == SpineJsonTarget.SPINE_3_8.value
        )
        assert isclose(
            float(scene.spine2d_depth_parallax_horizon_angle),
            radians(30.0),
            rel_tol=1.0e-7,
            abs_tol=1.0e-7,
        )

        assert spine2d_scene_settings_load_pre in bpy.app.handlers.load_pre
        assert spine2d_scene_settings_load_post in bpy.app.handlers.load_post
    finally:
        extension.unregister()
        _remove_persisted_test_values(scene)

    _assert_extension_unregistered()


def test_genuinely_fresh_scene_gets_current_defaults_in_real_bpy():
    """A genuinely new Scene receives every current schema-8 compatibility default."""

    _assert_extension_unregistered()
    scene = bpy.data.scenes.new("Spine2D Fresh Rig Default")
    _remove_persisted_test_values(scene)
    try:
        extension.register()
        assert (
            scene.spine2d_settings_schema_version
            == CURRENT_SETTINGS_SCHEMA_VERSION
        )
        assert (
            scene.spine2d_rig_profile
            == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
        )
        assert (
            scene.spine2d_target_spine_version
            == SpineJsonTarget.SPINE_4_2.value
        )
        assert isclose(
            float(scene.spine2d_depth_parallax_horizon_angle),
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    finally:
        extension.unregister()
        if scene.name in bpy.data.scenes:
            bpy.data.scenes.remove(scene)

    _assert_extension_unregistered()
