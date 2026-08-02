"""Real-bpy regression for rendered Camera Projection Object Origin placement."""

from __future__ import annotations

from types import SimpleNamespace

import bpy
import pytest
from mathutils import Vector

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_projection_finalization import (
    _rendered_camera_main_position,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.active_camera_projection import (
    resolve_a1_active_camera_projection_frame,
)


def test_rendered_projection_uses_evaluated_blender_object_origin() -> None:
    scene = bpy.context.scene

    mesh = bpy.data.meshes.new("Spine2D_RenderedPivotMesh")
    source = bpy.data.objects.new("Spine2D_RenderedPivotSource", mesh)
    scene.collection.objects.link(source)
    source.location = (0.4, 0.1, 0.2)
    source.rotation_euler = (0.24, -0.18, 0.33)
    source.scale = (1.0, 0.8, 1.15)

    camera_data = bpy.data.cameras.new("Spine2D_RenderedPivotCameraData")
    camera_data.type = "PERSP"
    camera_data.clip_start = 0.1
    camera_data.clip_end = 100.0
    camera_data.lens = 48.0
    camera_data.sensor_width = 36.0
    camera_data.shift_x = 0.06
    camera_data.shift_y = -0.03

    camera = bpy.data.objects.new(
        "Spine2D_RenderedPivotCamera",
        camera_data,
    )
    scene.collection.objects.link(camera)
    camera.location = (5.5, -7.5, 4.5)
    camera.rotation_euler = (
        Vector((0.2, 0.0, 0.2)) - camera.location
    ).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    bpy.context.view_layer.update()

    prepared = SimpleNamespace(source_object=source)
    plan = SimpleNamespace(settings=SimpleNamespace(width=128, height=128))
    main_position, projected_depth = _rendered_camera_main_position(
        prepared,
        plan,
        context=bpy.context,
        scene=scene,
    )

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_source = source.evaluated_get(depsgraph)
    evaluated_origin = tuple(
        float(evaluated_source.matrix_world[index][3])
        for index in range(3)
    )
    frame = resolve_a1_active_camera_projection_frame(
        scene,
        texture_width=128,
        texture_height=128,
        depsgraph=depsgraph,
    )
    expected = frame.project_world_point(evaluated_origin)

    assert main_position == pytest.approx(
        (expected.u, -expected.v),
        abs=1.0e-6,
    )
    assert projected_depth == pytest.approx(expected.depth, abs=1.0e-6)
    assert abs(main_position[0]) > 1.0e-3 or abs(main_position[1]) > 1.0e-3
