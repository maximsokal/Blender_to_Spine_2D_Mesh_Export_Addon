"""Register and unregister the public Rewrite extension against Blender 5.2 RNA."""

from __future__ import annotations

import bpy

import Blender_to_Spine2D_Mesh_Exporter as extension
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import A1RigProfile


RNA_REGISTRATIONS = (
    *extension.ui.RNA_PROPERTIES,
    *extension.generated_material_ui.RNA_PROPERTIES,
    *extension.ui_layout.RNA_PROPERTIES,
    *extension.single_object_operator.RNA_PROPERTIES,
)


def _panel_class(bl_idname: str):
    return bpy.types.Panel.bl_rna_get_subclass_py(bl_idname)


def _operator_class(bl_idname: str):
    category, operator_name = bl_idname.split(".", 1)
    rna_identifier = f"{category.upper()}_OT_{operator_name}"
    return bpy.types.Operator.bl_rna_get_subclass_py(rna_identifier)


def _replace_current_scene_with_fresh() -> None:
    """Use a new Scene because registration defaults do not overwrite user values."""

    previous_scenes = tuple(bpy.data.scenes)
    fresh_scene = bpy.data.scenes.new("Spine2D_RegistrationDefaults")
    windows = tuple(bpy.context.window_manager.windows)
    if not windows:
        bpy.data.scenes.remove(fresh_scene)
        raise RuntimeError("real bpy registration test requires a Blender Window")
    for window in windows:
        window.scene = fresh_scene
    for scene in previous_scenes:
        if scene != fresh_scene:
            bpy.data.scenes.remove(scene)


def _assert_registered() -> None:
    for name, _value in extension.CONFIG_RNA_PROPERTIES:
        assert hasattr(bpy.types.Scene, name), name
    for registration in RNA_REGISTRATIONS:
        assert hasattr(registration.owner, registration.name), registration.name

    assert bpy.context.scene.spine2d_texture_size == 1024
    assert bpy.context.scene.spine2d_seam_maker_mode == "AUTO"
    assert (
        bpy.context.scene.spine2d_rig_profile
        == A1RigProfile.TWO_AXIS_ROTATION_SCALE.value
    )
    assert bpy.context.scene.spine2d_material_source_policy == "REQUIRE_SOURCE"
    assert hasattr(bpy.types.Object, "spine2d_bake_settings")
    assert hasattr(bpy.types.Object, "spine2d_connect_settings")

    assert _panel_class("OBJECT_PT_spine2d_mesh") is extension.ui.OBJECT_PT_Spine2DMeshPanel
    assert _panel_class("OBJECT_PT_spine2d_rig") is extension.ui_layout.OBJECT_PT_Spine2DRigPanel
    assert _panel_class("OBJECT_PT_spine2d_generated_materials") is (
        extension.ui_layout.OBJECT_PT_Spine2DGeneratedMaterialsPanel
    )
    assert _panel_class("OBJECT_PT_spine2d_depth_parallax") is (
        extension.ui_layout.OBJECT_PT_Spine2DDepthParallaxPanel
    )
    assert _panel_class("OBJECT_PT_spine2d_repolish") is None

    assert _operator_class("spine2d.reset_rig_profile") is not None
    assert _operator_class("object.spine2d_single_export") is not None
    assert _operator_class("object.spine2d_multi_export") is not None
    assert _operator_class("object.save_uv_as_json") is not None


def _assert_unregistered() -> None:
    for name, _value in extension.CONFIG_RNA_PROPERTIES:
        assert not hasattr(bpy.types.Scene, name), name
    for registration in RNA_REGISTRATIONS:
        assert not hasattr(registration.owner, registration.name), registration.name

    for panel_id in (
        "OBJECT_PT_spine2d_mesh",
        "OBJECT_PT_spine2d_rig",
        "OBJECT_PT_spine2d_generated_materials",
        "OBJECT_PT_spine2d_depth_parallax",
        "OBJECT_PT_spine2d_repolish",
    ):
        assert _panel_class(panel_id) is None, panel_id

    assert _operator_class("spine2d.reset_rig_profile") is None
    assert _operator_class("object.spine2d_single_export") is None
    assert _operator_class("object.spine2d_multi_export") is None
    assert _operator_class("object.save_uv_as_json") is None

    is_registered = getattr(bpy.app.timers, "is_registered", None)
    if callable(is_registered):
        assert not is_registered(extension.addon_preferences._deferred_view3d_redraw)


def test_every_registration_owner_survives_two_real_rna_cycles(clean_blender_data):
    # Blender preserves existing IDProperty values across RNA re-registration.
    # Defaults must therefore be asserted on a genuinely fresh Scene.
    _replace_current_scene_with_fresh()

    # A second complete cycle catches stale class/RNA/timer ownership left by
    # unregister(). Use the public extension lifecycle rather than private test steps.
    for _cycle in range(2):
        extension.register()
        try:
            _assert_registered()
        finally:
            extension.unregister()
        _assert_unregistered()
