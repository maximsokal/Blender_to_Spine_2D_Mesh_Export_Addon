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

from ..domain.geometry import FaceId, LoopId, MeshSnapshot, MeshSnapshotValidator

logger = logging.getLogger(__name__)


class MeshWriteError(RuntimeError):
    """Raised when a MeshSnapshot cannot be materialized safely in Blender."""


@dataclass(frozen=True, slots=True)
class TemporaryMeshObject:
    object: Any
    mesh: Any
    collection: Any


@dataclass(frozen=True, slots=True)
class MeshTopologyCorrespondence:
    """Exact snapshot-face/loop mapping for one materialized Blender mesh.

    Blender is allowed to reorder polygons or cyclically rotate polygon corners
    while preserving the same oriented face. Every later property/UV write must
    therefore use this explicit mapping instead of assuming ``zip`` or
    ``polygon.loop_start + source_corner`` remains identical to the snapshot.
    Reversed winding is rejected because it changes the exported triangle
    orientation and loop-edge semantics.
    """

    snapshot_id: str
    face_to_polygon_index: tuple[tuple[FaceId, int], ...]
    loop_to_mesh_index: tuple[tuple[LoopId, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-empty string")
        for field_name, values, key_type in (
            ("face_to_polygon_index", self.face_to_polygon_index, FaceId),
            ("loop_to_mesh_index", self.loop_to_mesh_index, LoopId),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be tuple")
            keys: list[FaceId | LoopId] = []
            indices: list[int] = []
            for item_index, item in enumerate(values):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise TypeError(
                        f"{field_name}[{item_index}] must be a two-item tuple"
                    )
                key, index = item
                if type(key) is not key_type:
                    raise TypeError(
                        f"{field_name}[{item_index}][0] must be {key_type.__name__}"
                    )
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError(f"{field_name}[{item_index}][1] must be int")
                if index < 0:
                    raise ValueError(
                        f"{field_name}[{item_index}][1] must be non-negative"
                    )
                keys.append(key)
                indices.append(index)
            if len(keys) != len(set(keys)):
                raise ValueError(f"{field_name} contains duplicate snapshot IDs")
            if len(indices) != len(set(indices)):
                raise ValueError(f"{field_name} contains duplicate Blender indices")

    def polygon_index_for(self, face_id: FaceId) -> int:
        if type(face_id) is not FaceId:
            raise TypeError("face_id must be FaceId")
        mapping = dict(self.face_to_polygon_index)
        try:
            return mapping[face_id]
        except KeyError as exc:
            raise KeyError(f"No Blender polygon mapped for face {face_id.index}") from exc

    def mesh_loop_index_for(self, loop_id: LoopId) -> int:
        if type(loop_id) is not LoopId:
            raise TypeError("loop_id must be LoopId")
        mapping = dict(self.loop_to_mesh_index)
        try:
            return mapping[loop_id]
        except KeyError as exc:
            raise KeyError(f"No Blender loop mapped for loop {loop_id.index}") from exc


def _load_bpy() -> Any:
    try:
        import bpy
    except Exception as exc:
        raise MeshWriteError("Blender bpy module is unavailable") from exc
    return bpy


def _matrix_rows(matrix: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    if len(matrix) != 16:
        raise MeshWriteError("world_matrix must contain 16 values")
    return tuple(
        tuple(matrix[row * 4 + column] for column in range(4))
        for row in range(4)
    )


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


def _rotation_offsets(
    expected: tuple[int, ...],
    actual: tuple[int, ...],
) -> tuple[int, ...]:
    """Return every oriented cyclic offset that maps ``expected`` to ``actual``."""

    if len(expected) != len(actual) or not expected:
        return ()
    return tuple(
        offset
        for offset in range(len(expected))
        if all(
            actual[actual_index] == expected[(actual_index + offset) % len(expected)]
            for actual_index in range(len(expected))
        )
    )


def _canonical_oriented_cycle(values: tuple[int, ...]) -> tuple[int, ...]:
    """Return a deterministic key invariant to cyclic rotation, not winding."""

    if not values:
        raise MeshWriteError("A generated polygon cannot contain zero vertices")
    return min(values[offset:] + values[:offset] for offset in range(len(values)))


def _polygon_vertex_indices(mesh: Any, polygon: Any) -> tuple[int, ...]:
    try:
        polygon_vertices = tuple(int(value) for value in polygon.vertices)
    except Exception as exc:
        raise MeshWriteError("Unable to read generated polygon vertices") from exc

    loop_start = int(polygon.loop_start)
    loop_total = int(polygon.loop_total)
    try:
        loop_vertices = tuple(
            int(mesh.loops[loop_start + corner_index].vertex_index)
            for corner_index in range(loop_total)
        )
    except Exception as exc:
        raise MeshWriteError("Unable to read generated polygon loops") from exc
    if loop_vertices != polygon_vertices:
        raise MeshWriteError(
            "Generated polygon vertex order disagrees with its mesh loop order; "
            f"polygon={polygon_vertices}, loops={loop_vertices}"
        )
    return polygon_vertices


def build_mesh_topology_correspondence(
    snapshot: MeshSnapshot,
    mesh: Any,
    *,
    stage: str = "materialization",
) -> MeshTopologyCorrespondence:
    """Map snapshot faces and loops to a generated Blender mesh exactly.

    Polygon collection reordering and oriented cyclic corner rotations are
    supported. Missing, ambiguous, reversed, or edge-inconsistent faces are
    rejected before UV or material data can be attached to the wrong corner.
    """

    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    if mesh is None:
        raise ValueError("mesh cannot be None")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")

    expected_counts = {
        "vertices": len(snapshot.vertices),
        "edges": len(snapshot.edges),
        "loops": len(snapshot.loops),
        "polygons": len(snapshot.faces),
    }
    try:
        actual_counts = {
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "loops": len(mesh.loops),
            "polygons": len(mesh.polygons),
        }
    except Exception as exc:
        raise MeshWriteError(
            f"Unable to inspect generated topology during {stage}"
        ) from exc
    if actual_counts != expected_counts:
        raise MeshWriteError(
            f"Temporary mesh topology mismatch during {stage}; "
            f"expected={expected_counts}, actual={actual_counts}"
        )

    loop_map = snapshot.loop_by_id()
    edge_map = snapshot.edge_by_id()
    polygon_vertices = tuple(
        _polygon_vertex_indices(mesh, polygon) for polygon in mesh.polygons
    )
    polygon_indices_by_cycle: dict[tuple[int, ...], list[int]] = {}
    for polygon_index, vertices in enumerate(polygon_vertices):
        polygon_indices_by_cycle.setdefault(
            _canonical_oriented_cycle(vertices),
            [],
        ).append(polygon_index)

    used_polygon_indices: set[int] = set()
    face_pairs: list[tuple[FaceId, int]] = []
    loop_pairs: list[tuple[LoopId, int]] = []

    for face in snapshot.faces:
        expected_vertices = tuple(
            loop_map[loop_id].vertex_id.index for loop_id in face.loop_ids
        )
        cycle_key = _canonical_oriented_cycle(expected_vertices)
        polygon_candidates = tuple(
            polygon_index
            for polygon_index in polygon_indices_by_cycle.get(cycle_key, ())
            if polygon_index not in used_polygon_indices
        )
        if not polygon_candidates:
            raise MeshWriteError(
                f"Face {face.id.index} has no oriented Blender polygon match "
                f"during {stage}; expected vertices={expected_vertices}"
            )
        if len(polygon_candidates) != 1:
            raise MeshWriteError(
                f"Face {face.id.index} has ambiguous Blender polygon matches "
                f"during {stage}: {polygon_candidates}"
            )

        polygon_index = polygon_candidates[0]
        offsets = _rotation_offsets(
            expected_vertices,
            polygon_vertices[polygon_index],
        )
        if len(offsets) != 1:
            raise MeshWriteError(
                f"Face {face.id.index} has ambiguous cyclic corner offsets "
                f"during {stage}: {offsets}"
            )
        offset = offsets[0]
        used_polygon_indices.add(polygon_index)
        polygon = mesh.polygons[polygon_index]
        loop_start = int(polygon.loop_start)
        face_pairs.append((face.id, polygon_index))

        for source_corner_index, loop_id in enumerate(face.loop_ids):
            actual_corner_index = (source_corner_index - offset) % len(face.loop_ids)
            mesh_loop_index = loop_start + actual_corner_index
            mesh_loop = mesh.loops[mesh_loop_index]
            source_loop = loop_map[loop_id]
            actual_vertex_index = int(mesh_loop.vertex_index)
            if actual_vertex_index != source_loop.vertex_id.index:
                raise MeshWriteError(
                    f"Loop {loop_id.index} mapped to Blender loop {mesh_loop_index} "
                    f"with vertex {actual_vertex_index}, expected "
                    f"{source_loop.vertex_id.index} during {stage}"
                )

            try:
                actual_edge = mesh.edges[int(mesh_loop.edge_index)]
                actual_edge_vertices = _edge_key(
                    int(actual_edge.vertices[0]),
                    int(actual_edge.vertices[1]),
                )
            except Exception as exc:
                raise MeshWriteError(
                    f"Unable to inspect Blender edge for loop {mesh_loop_index} "
                    f"during {stage}"
                ) from exc
            source_edge = edge_map[source_loop.edge_id]
            expected_edge_vertices = _edge_key(
                source_edge.vertex_ids[0].index,
                source_edge.vertex_ids[1].index,
            )
            if actual_edge_vertices != expected_edge_vertices:
                raise MeshWriteError(
                    f"Loop {loop_id.index} edge mismatch during {stage}; "
                    f"expected={expected_edge_vertices}, actual={actual_edge_vertices}"
                )
            loop_pairs.append((loop_id, mesh_loop_index))

    unmatched_polygon_indices = set(range(len(mesh.polygons))) - used_polygon_indices
    if unmatched_polygon_indices:
        raise MeshWriteError(
            f"Generated mesh contains unmatched polygons during {stage}: "
            f"{tuple(sorted(unmatched_polygon_indices))}"
        )
    if len(loop_pairs) != len(snapshot.loops):
        raise MeshWriteError(
            f"Loop correspondence is incomplete during {stage}; "
            f"mapped={len(loop_pairs)}, expected={len(snapshot.loops)}"
        )

    return MeshTopologyCorrespondence(
        snapshot_id=snapshot.snapshot_id,
        face_to_polygon_index=tuple(
            sorted(face_pairs, key=lambda item: item[0].index)
        ),
        loop_to_mesh_index=tuple(
            sorted(loop_pairs, key=lambda item: item[0].index)
        ),
    )


def _verify_generated_topology(snapshot: MeshSnapshot, mesh: Any) -> None:
    """Backward-compatible validation wrapper for existing callers/tests."""

    build_mesh_topology_correspondence(snapshot, mesh)


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


def _write_face_properties(
    snapshot: MeshSnapshot,
    mesh: Any,
    correspondence: MeshTopologyCorrespondence | None = None,
) -> None:
    resolved = correspondence or build_mesh_topology_correspondence(
        snapshot,
        mesh,
        stage="face-property-write",
    )
    polygon_index_by_face = dict(resolved.face_to_polygon_index)
    for face in snapshot.faces:
        polygon = mesh.polygons[polygon_index_by_face[face.id]]
        polygon.material_index = face.material_index
        polygon.use_smooth = face.smooth


def _write_uv_roles(snapshot: MeshSnapshot, mesh: Any) -> None:
    layers = getattr(mesh, "uv_layers", None)
    if layers is None:
        if snapshot.uv_layer_names:
            raise MeshWriteError("Generated mesh has no UV layer collection")
        return

    if snapshot.active_uv_layer is not None:
        active = layers.get(snapshot.active_uv_layer)
        if active is None:
            raise MeshWriteError(
                f"Active UV layer '{snapshot.active_uv_layer}' was not materialized"
            )
        try:
            layers.active = active
        except Exception as exc:
            raise MeshWriteError(
                f"Unable to activate bake UV layer '{snapshot.active_uv_layer}'"
            ) from exc

    render_name = snapshot.render_uv_layer or snapshot.active_uv_layer
    if render_name is None:
        return
    render_layer = layers.get(render_name)
    if render_layer is None:
        raise MeshWriteError(
            f"Render UV layer '{render_name}' was not materialized"
        )
    try:
        for layer in layers:
            layer.active_render = layer is render_layer or layer.name == render_name
    except Exception as exc:
        raise MeshWriteError(
            f"Unable to activate shader render UV layer '{render_name}'"
        ) from exc


def _write_uv_layers(
    snapshot: MeshSnapshot,
    mesh: Any,
    correspondence: MeshTopologyCorrespondence | None = None,
) -> None:
    resolved = correspondence or build_mesh_topology_correspondence(
        snapshot,
        mesh,
        stage="UV-layer-write",
    )
    loop_map = snapshot.loop_by_id()
    for layer_name in snapshot.uv_layer_names:
        layer = mesh.uv_layers.get(layer_name)
        if layer is None:
            layer = mesh.uv_layers.new(name=layer_name)
        for source_loop_id, mesh_loop_index in resolved.loop_to_mesh_index:
            coordinate = loop_map[source_loop_id].uv(layer_name)
            if coordinate is None:
                raise MeshWriteError(
                    f"Loop {source_loop_id.index} is missing UV layer '{layer_name}'"
                )
            layer.data[mesh_loop_index].uv = coordinate

    _write_uv_roles(snapshot, mesh)


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
    """Create and clean an isolated Blender Object for one MeshSnapshot.

    Only failures raised while creating or populating temporary Blender datablocks
    are converted to :class:`MeshWriteError`. Exceptions raised by the caller inside
    the ``with`` block retain their original type and traceback while cleanup still
    runs in ``finally``.
    """

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
        try:
            collection = bpy_module.data.collections.new(
                f"{name_prefix}_{token}_Collection"
            )
            resolved_scene.collection.children.link(collection)

            mesh = bpy_module.data.meshes.new(f"{name_prefix}_{token}_Mesh")
            obj = bpy_module.data.objects.new(f"{name_prefix}_{token}", mesh)
            collection.objects.link(obj)

            mesh.from_pydata(
                [vertex.position for vertex in snapshot.vertices],
                [
                    tuple(vertex_id.index for vertex_id in edge.vertex_ids)
                    for edge in snapshot.edges
                ],
                _face_vertex_indices(snapshot),
            )
            mesh.update(calc_edges=True)
            correspondence = build_mesh_topology_correspondence(
                snapshot,
                mesh,
                stage="materialization",
            )
            _write_edge_flags(snapshot, mesh)
            _write_face_properties(snapshot, mesh, correspondence)
            _write_uv_layers(snapshot, mesh, correspondence)
            _set_world_matrix(obj, snapshot.world_matrix)
        except MeshWriteError:
            raise
        except Exception as exc:
            logger.exception(
                "Failed to materialize snapshot '%s'",
                snapshot.snapshot_id,
            )
            raise MeshWriteError(
                f"Failed to materialize snapshot '{snapshot.snapshot_id}': {exc}"
            ) from exc

        yield TemporaryMeshObject(object=obj, mesh=mesh, collection=collection)
    finally:
        _remove_object_and_mesh(bpy_module, obj, mesh)
        _remove_collection(bpy_module, collection)
