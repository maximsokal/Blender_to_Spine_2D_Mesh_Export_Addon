"""Convert camera-local projection Z into positive camera-distance rig coordinates.

The projection domain follows Blender camera-local convention: visible points in front
of the camera have negative Z. Spine depth controls are easier and less error-prone when
the shared camera is the global zero and every visible point stores a positive distance.
This module performs that representation conversion without changing projected X/Y, UV,
lineage, triangles, or the diagnostic camera-local values retained by the result.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from .depth_camera_projection import (
    DepthCameraProjectionError,
    DepthCameraProjectionResult,
)
from .model import MeshSnapshot, MeshVertex
from .validator import MeshSnapshotValidator


_CAMERA_PLANE_EPSILON = 1.0e-9


def convert_depth_result_to_camera_distance(
    result: DepthCameraProjectionResult,
) -> DepthCameraProjectionResult:
    """Return ``result`` with snapshot Z expressed as positive camera distance.

    ``DepthCameraProjectionResult`` keeps its original ``farthest_visible_depth``,
    ``nearest_visible_depth``, and ``base_depth`` diagnostics in Blender camera-local Z.
    Only the generated snapshot consumed by Z-group and rig construction is converted.
    This preserves one global camera zero across every object in a multi-object export.
    """

    if not isinstance(result, DepthCameraProjectionResult):
        raise TypeError("result must be DepthCameraProjectionResult")

    source = result.snapshot
    if not isinstance(source, MeshSnapshot):
        raise TypeError("result.snapshot must be MeshSnapshot")
    MeshSnapshotValidator().validate_or_raise(source)

    converted_vertices: list[MeshVertex] = []
    for vertex_index, vertex in enumerate(source.vertices):
        camera_z = float(vertex.position[2])
        if not isfinite(camera_z):
            raise DepthCameraProjectionError(
                f"depth snapshot vertex {vertex_index} contains non-finite camera Z"
            )
        distance = -camera_z
        if distance <= _CAMERA_PLANE_EPSILON:
            raise DepthCameraProjectionError(
                "Depth Camera Projection contains a point on or behind the camera "
                f"plane; vertex={vertex_index}, camera_z={camera_z}, "
                f"distance={distance}"
            )
        converted_vertices.append(
            replace(
                vertex,
                position=(
                    float(vertex.position[0]),
                    float(vertex.position[1]),
                    float(distance),
                ),
            )
        )

    converted_snapshot = replace(
        source,
        snapshot_id=f"{source.snapshot_id}:camera-distance",
        vertices=tuple(converted_vertices),
    )
    MeshSnapshotValidator().validate_or_raise(converted_snapshot)

    source_x_y = tuple(
        (float(vertex.position[0]), float(vertex.position[1]))
        for vertex in source.vertices
    )
    converted_x_y = tuple(
        (float(vertex.position[0]), float(vertex.position[1]))
        for vertex in converted_snapshot.vertices
    )
    if converted_x_y != source_x_y:
        raise DepthCameraProjectionError(
            "camera-distance conversion changed projected X/Y coordinates"
        )
    if converted_snapshot.faces != source.faces:
        raise DepthCameraProjectionError(
            "camera-distance conversion changed depth surface faces"
        )
    if converted_snapshot.loops != source.loops:
        raise DepthCameraProjectionError(
            "camera-distance conversion changed depth surface loops or UVs"
        )
    if converted_snapshot.edges != source.edges:
        raise DepthCameraProjectionError(
            "camera-distance conversion changed depth surface edges"
        )

    return replace(result, snapshot=converted_snapshot)


__all__ = ["convert_depth_result_to_camera_distance"]
