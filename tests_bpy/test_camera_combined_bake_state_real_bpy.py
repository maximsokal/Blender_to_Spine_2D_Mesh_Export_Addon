"""Real Blender 5.2 regression for camera-scoped semantic bake state."""

from __future__ import annotations

import bpy

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    BakeSceneState,
    configure_scene_for_bake,
    preserve_bake_scene_state,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeEvaluationScope,
    BakeExecutionSettings,
    BakeMode,
    BakeSettings,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
    build_bake_plan,
)


_EXPECTED_BLENDER_52_COMBINED_FILTER = frozenset(
    {
        "DIRECT",
        "INDIRECT",
        "COLOR",
        "DIFFUSE",
        "GLOSSY",
        "TRANSMISSION",
        "EMIT",
    }
)


def _plan(tmp_path):
    analysis = ObjectMaterialAnalysis(
        "CameraCombinedStateProbe",
        (
            MaterialAnalysis(
                slot_index=0,
                material_name="ProbeMaterial",
                kind=MaterialKind.PROCEDURAL,
            ),
        ),
    )
    return build_bake_plan(
        analysis,
        BakeSettings(
            width=32,
            height=32,
            output_directory=tmp_path,
            output_stem="camera_combined_state_probe",
        ),
    )


def test_camera_combined_uses_active_camera_transmission_and_restores_state(tmp_path):
    scene = bpy.context.scene
    original_camera = scene.camera
    camera_data = bpy.data.cameras.new("Spine2D_BakeStateProbeCamera")
    camera_object = bpy.data.objects.new(
        "Spine2D_BakeStateProbeCamera",
        camera_data,
    )
    scene.collection.objects.link(camera_object)
    scene.camera = camera_object

    try:
        # Assert the exact Blender 5.2 RNA boundary before testing our configuration.
        bake_properties = {
            property_rna.identifier
            for property_rna in scene.render.bake.bl_rna.properties
        }
        assert "view_from" in bake_properties
        assert "use_pass_diffuse" in bake_properties
        assert "use_pass_glossy" in bake_properties
        assert "use_pass_transmission" in bake_properties
        assert "use_pass_emit" in bake_properties
        assert "use_pass_ambient_occlusion" not in bake_properties
        assert "use_pass_subsurface" not in bake_properties

        state_before = BakeSceneState.capture(scene)

        with preserve_bake_scene_state(scene):
            configure_scene_for_bake(
                scene,
                _plan(tmp_path),
                BakeExecutionSettings(samples=1),
                bake_mode=BakeMode.COMBINED,
                evaluation_scope=BakeEvaluationScope.CAMERA,
            )

            assert scene.render.engine == "CYCLES"
            assert scene.render.bake.view_from == "ACTIVE_CAMERA"
            assert scene.render.bake.use_pass_direct
            assert scene.render.bake.use_pass_indirect
            assert scene.render.bake.use_pass_color
            assert scene.render.bake.use_pass_diffuse
            assert scene.render.bake.use_pass_glossy
            assert scene.render.bake.use_pass_transmission
            assert scene.render.bake.use_pass_emit
            assert scene.cycles.bake_type == "COMBINED"

            pass_filter = frozenset(scene.render.bake.pass_filter)
            assert _EXPECTED_BLENDER_52_COMBINED_FILTER.issubset(pass_filter)

        assert BakeSceneState.capture(scene) == state_before
    finally:
        scene.camera = original_camera
        if camera_object.name in bpy.data.objects:
            bpy.data.objects.remove(camera_object, do_unlink=True)
        if camera_data.name in bpy.data.cameras:
            bpy.data.cameras.remove(camera_data)
