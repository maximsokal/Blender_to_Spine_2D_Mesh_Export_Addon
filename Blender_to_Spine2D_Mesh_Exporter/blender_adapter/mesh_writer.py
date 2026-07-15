"""Materialize immutable mesh snapshots as temporary Blender datablocks.

This adapter uses the direct data API only. It never changes the user's active
object, selection, or mode. The returned object exists only inside the context
manager and all created datablocks are removed in ``finally``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Any, Iterator
from uuid import uuid4

from ..domain.geometry import MeshSnapshot, MeshSnapshotValidator

logger = logging.getLogger(__name__)


class MeshWriteError(RuntimeError):
    """Raised when a MeshSnapshot cannot be materialized safely in Blender."""


@dataclass(frozen=True, slots=True)
class TemporaryMeshObject:
    object: Any
    mesh: Any
    collection: Any


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise MeshWriteError("Blender bpy module is unavailable") from exc
    return bpy


def _matrix_rows(matrix: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    if len(matrix) != 16:
        raise MeshWriteError("world_matrix must contain 16 values")
    return tuple(tuple(matrix[row * 4 + column] for column in range(4)) for row in range(4))


def _set_world_matrix(obj: Any, matrix: tuple[float, ...]) -> None:
    try:
        from mathutils import Matrix

        obj.matrix_world = Matrix(_matrix_rows(matrix))
    except Exception as exc:
        raise MeshWriteError("Unable to assign temporary object world matrix") from exc


def _face_vertex_indices(snapshot: MeshSnapshot) -> tuple[tuple[int, ...], ...]:
    loop_map = snapshot.loop_by_id()
    return tuple(
        tuple(loop_map[loop_id].vertex_id.index for loop_id in face.loop_ids)
        for face in snapshot.faces
    )


def _edge_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _verify_generated_topology(snapshot: MeshSnapshot, mesh: Any) -> None:
    expected = {
        "vertices": len(snapshot.vertices),
        "edges": len(snapshot.edges),
        "loops": len(snapshot.loops),
        "polygons": len(snapshot.faces),
    }
    actual = {
        "vertices": len(mesh.vertices),
        "edges": len(mesh.edges),
        "loops": len(mesh.loops),
        "polygons": len(mesh.polygons),
    }
    if actual != expected:
        raise MeshWriteError(
            f"Temporary mesh topology mismatch; expected={expected}, actual={actual}"
        )

    for face, polygon in zip(snapshot.faces, mesh.polygons):
        if int(polygon.loop_total) != len(face.loop_ids):
            raise MeshWriteError(
                f"Face {face.id.index} loop count changed from {len(face.loop_ids)} "
                f"to {polygon.loop_total} during materialization"
            )


def _write_edge_flags(snapshot: MeshSnapshot, mesh: Any) -> None:
    source_by_vertices = {
        _edge_key(edge.vertex_ids[0].index, edge.vertex_ids[1].index): edge
        for edge in snapshot.edges
    }
    for mesh_edge in mesh.edges:
        key = _edge_key(int(mesh_edge.vertices[0]), int(mesh_edge.vertices[1]))
        source_edge = source_by_vertices.get(key)
        if source_edge is None:
            raise MeshWriteError(
                f"Generated mesh contains unexpected edge between vertices {key}"
            )
        if hasattr(mesh_edge, "use_seam"):
            mesh_edge.use_seam = source_edge.seam
        if hasattr(mesh_edge, "use_edge_sharp"):
            mesh_edge.use_edge_sharp = source_edge.sharp


def _write_face_properties(snapshot: MeshSnapshot, mesh: Any) -> None:
    for face, polygon in zip(snapshot.faces, mesh.polygons):
        polygon.material_index = face.material_index
        polygon.use_smooth = face.smooth


def _write_uv_layers(snapshot: MeshSnapshot, mesh: Any) -> None:
    loop_map = snapshot.loop_by_id()
    for layer_name in snapshot.uv_layer_names:
        layer = mesh.uv_layers.get(layer_name)
        if layer is None:
            layer = mesh.uv_layers.new(name=layer_name)
        for face, polygon in zip(snapshot.faces, mesh.polygons):
            if int(polygon.loop_total) != len(face.loop_ids):
                raise MeshWriteError(
                    f"Cannot write UV layer '{layer_name}': face {face.id.index} loop "
                    "count changed"
                )
            for corner_index, source_loop_id in enumerate(face.loop_ids):
                mesh_loop_index = int(polygon.loop_start) + corner_index
                coordinate = loop_map[source_loop_id].uv(layer_name)
                if coordinate is None:
                    raise MeshWriteError(
                        f"Loop {source_loop_id.index} is missing UV layer '{layer_name}'"
                    )
                layer.data[mesh_loop_index].uv = coordinate

    if snapshot.active_uv_layer is not None:
        active = mesh.uv_layers.get(snapshot.active_uv_layer)
        if active is None:
            raise MeshWriteError(
                f"Active UV layer '{snapshot.active_uv_layer}' was not materialized"
            )
        mesh.uv_layers.active = active


def _remove_collection(bpy_module: Any, collection: Any | None) -> None:
    if collection is None:
        return
    try:
        bpy_module.data.collections.remove(collection, do_unlink=True)
    except TypeError:
        try:
            bpy_module.data.collections.remove(collection)
        except Exception:
            logger.exception("Failed to remove temporary collection")
    except Exception:
        logger.exception("Failed to remove temporary collection")


def _remove_object_and_mesh(
    bpy_module: Any,
    obj: Any | None,
    mesh: Any | None,
) -> None:
    if obj is not None:
        try:
            bpy_module.data.objects.remove(obj, do_unlink=True)
        except Exception:
            logger.exception("Failed to remove temporary mesh object")
    if mesh is not None:
        try:
            if getattr(mesh, "users", 0) == 0:
                bpy_module.data.meshes.remove(mesh)
        except Exception:
            logger.exception("Failed to remove temporary Mesh datablock")


@contextmanager
def temporary_mesh_object(
    snapshot: MeshSnapshot,
    *,
    scene: Any | None = None,
    name_prefix: str = "__Spine2D_UV",
) -> Iterator[TemporaryMeshObject]:
    """Create and clean an isolated Blender Object for one MeshSnapshot."""

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    MeshSnapshotValidator().validate_or_raise(snapshot)
    if not isinstance(name_prefix, str) or not name_prefix.strip():
        raise ValueError("name_prefix must be a non-empty string")

    bpy_module = _load_bpy()
    resolved_scene = scene or getattr(bpy_module.context, "scene", None)
    if resolved_scene is None:
        raise MeshWriteError("A Blender Scene is required")

    token = uuid4().hex
    collection = None
    mesh = None
    obj = None
    try:
        collection = bpy_module.data.collections.new(f"{name_prefix}_{token}_Collection")
        resolved_scene.collection.children.link(collection)

        mesh = bpy_module.data.meshes.new(f"{name_prefix}_{token}_Mesh")
        obj = bpy_module.data.objects.new(f"{name_prefix}_{token}", mesh)
        collection.objects.link(obj)

        mesh.from_pydata(
            [vertex.position for vertex in snapshot.vertices],
            [tuple(vertex_id.index for vertex_id in edge.vertex_ids) for edge in snapshot.edges],
            _face_vertex_indices(snapshot),
        )
        mesh.update(calc_edges=True)
        _verify_generated_topology(snapshot, mesh)
        _write_edge_flags(snapshot, mesh)
        _write_face_properties(snapshot, mesh)
        _write_uv_layers(snapshot, mesh)
        _set_world_matrix(obj, snapshot.world_matrix)

        yield TemporaryMeshObject(object=obj, mesh=mesh, collection=collection)
    except MeshWriteError:
        raise
    except Exception as exc:
        logger.exception("Failed to materialize snapshot '%s'", snapshot.snapshot_id)
        raise MeshWriteError(
            f"Failed to materialize snapshot '{snapshot.snapshot_id}': {exc}"
        ) from exc
    finally:
        _remove_object_and_mesh(bpy_module, obj, mesh)
        _remove_collection(bpy_module, collection)
