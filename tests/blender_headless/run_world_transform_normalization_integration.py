"""Blender 5.2 headless integration for object transform normalization.

Run from the repository root with Blender 5.2 or newer::

    blender --background --factory-startup --python \
        tests/blender_headless/run_world_transform_normalization_integration.py
"""

from __future__ import annotations

from math import radians
from pathlib import Path
import sys

import bpy
from mathutils import Euler, Matrix, Vector


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_reader import (  # noqa: E402
    read_source_mesh_snapshot,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_writer import (  # noqa: E402
    temporary_mesh_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (  # noqa: E402
    normalize_mesh_snapshot_world_transform,
)


_EPSILON = 1.0e-6


def _assert_close(first: Vector, second: Vector, *, label: str) -> None:
    if (first - second).length > _EPSILON:
        raise AssertionError(
            f"{label} differs: first={tuple(first)}, second={tuple(second)}"
        )


def _world_vertex_positions(obj: bpy.types.Object) -> tuple[Vector, ...]:
    return tuple(obj.matrix_world @ vertex.co for vertex in obj.data.vertices)


def _world_face_normals(obj: bpy.types.Object) -> tuple[Vector, ...]:
    linear = obj.matrix_world.to_3x3()
    determinant = linear.determinant()
    cofactor = linear.inverted().transposed() * determinant
    return tuple(
        (cofactor @ polygon.normal).normalized()
        for polygon in obj.data.polygons
    )


def _cleanup_object(obj: bpy.types.Object | None) -> None:
    if obj is None:
        return
    mesh = obj.data if isinstance(obj.data, bpy.types.Mesh) else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def main() -> None:
    if tuple(bpy.app.version) < (5, 2, 0):
        raise AssertionError(
            f"Blender 5.2+ is required, running {tuple(bpy.app.version)}"
        )

    source_mesh_count = len(bpy.data.meshes)
    source_object_count = len(bpy.data.objects)
    mesh = bpy.data.meshes.new("Spine2D_Transform_Source_Mesh")
    source = bpy.data.objects.new("Spine2D_Transform_Source", mesh)
    bpy.context.scene.collection.objects.link(source)
    try:
        mesh.from_pydata(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.5),
                (0.0, 1.0, 0.5),
            ),
            (),
            ((0, 1, 2, 3),),
        )
        mesh.update()

        rotation = Euler(
            (radians(23.0), radians(-17.0), radians(41.0)),
            "XYZ",
        ).to_matrix().to_4x4()
        scale = Matrix.Diagonal((2.0, 0.75, 1.5, 1.0))
        translation = Matrix.Translation((7.5, -3.25, 11.0))
        source.matrix_world = translation @ rotation @ scale
        bpy.context.view_layer.update()

        expected_positions = _world_vertex_positions(source)
        expected_normals = _world_face_normals(source)
        snapshot = read_source_mesh_snapshot(
            source,
            source_object_id=source.name,
            snapshot_id="transform-normalization-source",
        )
        normalized = normalize_mesh_snapshot_world_transform(snapshot)
        if not normalized.changed:
            raise AssertionError("Non-identity source transform was not normalized")
        if normalized.mirrored:
            raise AssertionError("Positive determinant transform was marked mirrored")

        with temporary_mesh_object(
            normalized.snapshot,
            scene=bpy.context.scene,
            name_prefix="__Spine2D_Transform_Target",
        ) as temporary:
            target = temporary.object
            actual_positions = _world_vertex_positions(target)
            actual_normals = _world_face_normals(target)
            if len(actual_positions) != len(expected_positions):
                raise AssertionError("Target vertex count changed during normalization")
            if len(actual_normals) != len(expected_normals):
                raise AssertionError("Target face count changed during normalization")
            for index, (expected, actual) in enumerate(
                zip(expected_positions, actual_positions, strict=True)
            ):
                _assert_close(expected, actual, label=f"world vertex {index}")
            for index, (expected, actual) in enumerate(
                zip(expected_normals, actual_normals, strict=True)
            ):
                _assert_close(expected, actual, label=f"world face normal {index}")

        if any(
            obj.name.startswith("__Spine2D_Transform_Target")
            for obj in bpy.data.objects
        ):
            raise AssertionError("Temporary transform target object leaked")
        if any(
            mesh_item.name.startswith("__Spine2D_Transform_Target")
            for mesh_item in bpy.data.meshes
        ):
            raise AssertionError("Temporary transform target mesh leaked")
    finally:
        _cleanup_object(source)

    if len(bpy.data.objects) != source_object_count:
        raise AssertionError("Source object cleanup did not restore object count")
    if len(bpy.data.meshes) != source_mesh_count:
        raise AssertionError("Source mesh cleanup did not restore mesh count")
    print("Blender 5.2 world transform normalization integration passed")


if __name__ == "__main__":
    main()
