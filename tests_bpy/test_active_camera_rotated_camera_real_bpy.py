"""Real-bpy regression for ordinary rotated active cameras."""

from __future__ import annotations

from math import isfinite

import bpy
from mathutils import Vector

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.active_camera_projection import (
    resolve_a1_active_camera_projection_frame,
)


def test_rotated_active_camera_accepts_blender_float32_matrix() -> None:
    scene = bpy.context.scene

    camera_data = bpy.data.cameras.new(
        "Spine2D_RotatedCameraRegressionData"
    )
    camera_data.type = "PERSP"
    camera_data.clip_start = 0.1
    camera_data.clip_end = 100.0
    camera_data.lens = 48.0
    camera_data.sensor_width = 36.0
    camera_data.shift_x = 0.06
    camera_data.shift_y = -0.03

    camera = bpy.data.objects.new(
        "Spine2D_RotatedCameraRegression",
        camera_data,
    )
    scene.collection.objects.link(camera)

    target = Vector((0.2, 0.0, 0.2))
    camera.location = (5.5, -7.5, 4.5)
    camera.rotation_euler = (
        target - camera.location
    ).to_track_quat("-Z", "Y").to_euler()

    # Camera object scale must not enter the rotation-only view frame.
    camera.scale = (1.8, 0.7, 1.3)

    scene.camera = camera
    bpy.context.view_layer.update()

    frame = resolve_a1_active_camera_projection_frame(
        scene,
        texture_width=128,
        texture_height=128,
        depsgraph=bpy.context.evaluated_depsgraph_get(),
    )

    projected = frame.project_world_point(tuple(target))

    assert frame.camera_id == camera.name
    assert all(
        isfinite(value)
        for value in (
            projected.u,
            projected.v,
            projected.depth,
        )
    )
    assert projected.depth < -camera_data.clip_start
