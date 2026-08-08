"""Build immutable triangle topology from Blender's authoritative mesh tessellation.

Blender Mesh polygons may be valid for Blender rendering while a separate generic
polygon triangulator chooses a different 2D projection or rejects a non-simple boundary.
For live Blender source geometry the authoritative interpretation is ``Mesh.loop_triangles``.
This adapter converts that tessellation back into the rewrite ``MeshSnapshot`` contract
without mutating the source Mesh, using operators, or allocating a BMesh.

Blender may emit zero-area loop triangles for artist-authored polygons containing
coincident or collinear corners. Those triangles carry no renderable surface and cannot
form a valid rewrite face normal. They are therefore filtered before topology creation,
but only when every source polygon still retains at least one non-degenerate triangle.
A wholly degenerate polygon remains a hard error.
"""

from __future__ import annotations

import logging
from math import sqrt
from typing import Any

from ..domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshSnapshotValidator,
    VertexId,
)


logger = logging.getLogger(__name__)

_ZERO_AREA_EPSILON = 1.0e-15


class BlenderLoopTriangulationError(RuntimeError):
    """Raised when Blender loop-triangle tessellation cannot preserve snapshot lineage."""


def _edge_key(first: VertexId, second: VertexId) -> tuple[int, int]:
    first_index = int(first.index)
    second_index = int(second.index)
    return (
        (first_index, second_index)
        if first_index < second_index
        else (second_index, first_index)
    )


def _triangle_polygon_index(mesh: Any, triangle: Any, triangle_index: int) -> int:
    """Resolve one Blender loop triangle back to its owning polygon."""

    direct = getattr(triangle, "polygon_index", None)
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BlenderLoopTriangulationError(
                f"loop triangle {triangle_index} has invalid polygon_index {direct!r}"
            ) from exc

    polygons = getattr(mesh, "loop_triangle_polygons", None)
    if polygons is None or triangle_index < 0 or triangle_index >= len(polygons):
        raise BlenderLoopTriangulationError(
            f"loop triangle {triangle_index} has no resolvable source polygon"
        )
    raw = polygons[triangle_index]
    value = getattr(raw, "value", raw)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BlenderLoopTriangulationError(
            f"loop triangle {triangle_index} has invalid polygon mapping {value!r}"
        ) from exc


def _triangle_loop_indices(triangle: Any, triangle_index: int) -> tuple[int, int, int]:
    raw = getattr(triangle, "loops", None)
    if raw is None:
        raise BlenderLoopTriangulationError(
            f"loop triangle {triangle_index} exposes no loop indices"
        )
    try:
        values = tuple(int(value) for value in raw)
    except Exception as exc:
        raise BlenderLoopTriangulationError(
            f"loop triangle {triangle_index} has unreadable loop indices"
        ) from exc
    if len(values) != 3 or len(set(values)) != 3:
        raise BlenderLoopTriangulationError(
            f"loop triangle {triangle_index} must reference three distinct loops: {values!r}"
        )
    return values  # type: ignore[return-value]


def _triangle_normal_or_none(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    """Return a normalized triangle normal or ``None`` for zero-area geometry."""

    ab = tuple(second[index] - first[index] for index in range(3))
    ac = tuple(third[index] - first[index] for index in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = sqrt(sum(component * component for component in cross))
    if length <= _ZERO_AREA_EPSILON:
        return None
    return tuple(component / length for component in cross)  # type: ignore[return-value]


def triangulate_snapshot_with_blender_loop_triangles(
    mesh: Any,
    snapshot: MeshSnapshot,
    *,
    snapshot_id: str | None = None,
) -> MeshSnapshot:
    """Return ``snapshot`` with face topology replaced by Blender loop triangles.

    Preconditions deliberately require one-to-one loop/face indexing between the live
    Blender Mesh and ``snapshot``. This is true for snapshots read directly from that Mesh
    and for evaluated snapshots built while ``Object.to_mesh()`` remains alive.

    Original Blender edges retain source lineage/seam/sharp flags. Tessellation diagonals
    are generated rewrite edges with ``source_id=None``. Triangle loops copy the exact
    source loop lineage and UV payload from the originating Blender corner.

    Zero-area Blender loop triangles are omitted because they have no renderable area and
    cannot provide a valid face normal. The omission is accepted only when Blender still
    provides at least one non-degenerate triangle for every source polygon and the raw
    Blender tessellation itself contains exactly ``N-2`` triangles per polygon.
    """

    if mesh is None:
        raise TypeError("mesh cannot be None")
    if not isinstance(snapshot, MeshSnapshot):
        raise TypeError("snapshot must be MeshSnapshot")
    MeshSnapshotValidator().validate_or_raise(snapshot)

    calc_loop_triangles = getattr(mesh, "calc_loop_triangles", None)
    if not callable(calc_loop_triangles):
        raise BlenderLoopTriangulationError(
            "Blender Mesh.calc_loop_triangles() is unavailable"
        )

    try:
        mesh_loop_count = len(mesh.loops)
        mesh_polygon_count = len(mesh.polygons)
    except Exception as exc:
        raise BlenderLoopTriangulationError(
            "Unable to inspect Blender Mesh loop/polygon domains"
        ) from exc

    snapshot_loop_map = snapshot.loop_by_id()
    snapshot_face_map = snapshot.face_by_id()
    if mesh_loop_count != len(snapshot.loops):
        raise BlenderLoopTriangulationError(
            "Snapshot loop domain does not match Blender Mesh: "
            f"snapshot={len(snapshot.loops)}, mesh={mesh_loop_count}"
        )
    if mesh_polygon_count != len(snapshot.faces):
        raise BlenderLoopTriangulationError(
            "Snapshot face domain does not match Blender Mesh: "
            f"snapshot={len(snapshot.faces)}, mesh={mesh_polygon_count}"
        )

    try:
        calc_loop_triangles()
        loop_triangles = tuple(mesh.loop_triangles)
    except Exception as exc:
        raise BlenderLoopTriangulationError(
            "Blender failed to calculate Mesh.loop_triangles"
        ) from exc
    if snapshot.faces and not loop_triangles:
        raise BlenderLoopTriangulationError(
            "Blender produced no loop triangles for a non-empty Mesh"
        )

    vertex_map = snapshot.vertex_by_id()
    original_edge_by_key: dict[tuple[int, int], MeshEdge] = {}
    for edge in snapshot.edges:
        key = _edge_key(*edge.vertex_ids)
        if key in original_edge_by_key:
            raise BlenderLoopTriangulationError(
                f"Snapshot contains duplicate edge vertex pair {key!r}"
            )
        original_edge_by_key[key] = edge

    triangle_records: list[
        tuple[
            int,
            MeshFace,
            tuple[MeshLoop, MeshLoop, MeshLoop],
            tuple[float, float, float],
        ]
    ] = []
    used_edge_keys: set[tuple[int, int]] = set()
    raw_count_by_polygon: dict[int, int] = {}
    kept_count_by_polygon: dict[int, int] = {}
    degenerate_triangle_indices: list[int] = []

    for triangle_index, triangle in enumerate(loop_triangles):
        polygon_index = _triangle_polygon_index(mesh, triangle, triangle_index)
        face_id = FaceId(polygon_index)
        source_face = snapshot_face_map.get(face_id)
        if source_face is None:
            raise BlenderLoopTriangulationError(
                f"loop triangle {triangle_index} references missing face {polygon_index}"
            )
        raw_count_by_polygon[polygon_index] = raw_count_by_polygon.get(polygon_index, 0) + 1

        loop_indices = _triangle_loop_indices(triangle, triangle_index)
        source_loops: list[MeshLoop] = []
        allowed_face_loops = frozenset(source_face.loop_ids)
        for mesh_loop_index in loop_indices:
            if mesh_loop_index < 0 or mesh_loop_index >= mesh_loop_count:
                raise BlenderLoopTriangulationError(
                    f"loop triangle {triangle_index} references loop {mesh_loop_index} "
                    f"outside [0, {mesh_loop_count})"
                )
            source_loop = snapshot_loop_map.get(LoopId(mesh_loop_index))
            if source_loop is None:
                raise BlenderLoopTriangulationError(
                    f"loop triangle {triangle_index} references missing snapshot loop "
                    f"{mesh_loop_index}"
                )
            if source_loop.id not in allowed_face_loops:
                raise BlenderLoopTriangulationError(
                    f"loop triangle {triangle_index} loop {mesh_loop_index} does not "
                    f"belong to polygon {polygon_index}"
                )
            source_loops.append(source_loop)

        resolved_loops = tuple(source_loops)
        positions = tuple(
            vertex_map[source_loop.vertex_id].position
            for source_loop in resolved_loops
        )
        normal = _triangle_normal_or_none(
            positions[0],
            positions[1],
            positions[2],
        )
        if normal is None:
            degenerate_triangle_indices.append(triangle_index)
            continue

        kept_count_by_polygon[polygon_index] = kept_count_by_polygon.get(polygon_index, 0) + 1
        for corner_index, source_loop in enumerate(resolved_loops):
            following = resolved_loops[(corner_index + 1) % 3]
            used_edge_keys.add(_edge_key(source_loop.vertex_id, following.vertex_id))

        triangle_records.append(
            (
                triangle_index,
                source_face,
                resolved_loops,  # type: ignore[arg-type]
                normal,
            )
        )

    for source_face in snapshot.faces:
        polygon_index = source_face.id.index
        expected_raw_count = len(source_face.loop_ids) - 2
        raw_count = raw_count_by_polygon.get(polygon_index, 0)
        if raw_count != expected_raw_count:
            raise BlenderLoopTriangulationError(
                "Blender loop-triangle count does not match source polygon: "
                f"polygon={polygon_index}, corners={len(source_face.loop_ids)}, "
                f"expected={expected_raw_count}, actual={raw_count}"
            )
        kept_count = kept_count_by_polygon.get(polygon_index, 0)
        if kept_count <= 0:
            raise BlenderLoopTriangulationError(
                "Blender source polygon has no non-degenerate tessellation triangles: "
                f"polygon={polygon_index}, raw_triangles={raw_count}"
            )
        if kept_count > expected_raw_count:
            raise BlenderLoopTriangulationError(
                "Blender retained more non-degenerate triangles than source polygon "
                f"permits: polygon={polygon_index}, expected_max={expected_raw_count}, "
                f"actual={kept_count}"
            )

    existing_keys = tuple(
        sorted(
            (key for key in used_edge_keys if key in original_edge_by_key),
            key=lambda key: original_edge_by_key[key].id.index,
        )
    )
    generated_keys = tuple(sorted(used_edge_keys - set(existing_keys)))
    ordered_edge_keys = existing_keys + generated_keys
    edge_id_by_key = {
        key: EdgeId(index) for index, key in enumerate(ordered_edge_keys)
    }

    edges = tuple(
        MeshEdge(
            id=edge_id_by_key[key],
            source_id=(
                original_edge_by_key[key].source_id
                if key in original_edge_by_key
                else None
            ),
            vertex_ids=(VertexId(key[0]), VertexId(key[1])),
            seam=(
                original_edge_by_key[key].seam
                if key in original_edge_by_key
                else False
            ),
            sharp=(
                original_edge_by_key[key].sharp
                if key in original_edge_by_key
                else False
            ),
        )
        for key in ordered_edge_keys
    )

    loops: list[MeshLoop] = []
    faces: list[MeshFace] = []
    next_loop_index = 0

    for triangle_index, source_face, source_loops, normal in triangle_records:
        triangle_loop_ids: list[LoopId] = []
        for corner_index, source_loop in enumerate(source_loops):
            following = source_loops[(corner_index + 1) % 3]
            key = _edge_key(source_loop.vertex_id, following.vertex_id)
            loop_id = LoopId(next_loop_index)
            next_loop_index += 1
            triangle_loop_ids.append(loop_id)
            loops.append(
                MeshLoop(
                    id=loop_id,
                    source_id=source_loop.source_id,
                    vertex_id=source_loop.vertex_id,
                    edge_id=edge_id_by_key[key],
                    uvs=source_loop.uvs,
                )
            )

        faces.append(
            MeshFace(
                id=FaceId(len(faces)),
                source_id=source_face.source_id,
                loop_ids=tuple(triangle_loop_ids),
                material_index=source_face.material_index,
                normal=normal,
                smooth=source_face.smooth,
            )
        )

    output = MeshSnapshot(
        snapshot_id=snapshot_id or f"{snapshot.snapshot_id}:blender-triangles",
        source_object_id=snapshot.source_object_id,
        object_name=snapshot.object_name,
        vertices=snapshot.vertices,
        edges=edges,
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=snapshot.uv_layer_names,
        active_uv_layer=snapshot.active_uv_layer,
        world_matrix=snapshot.world_matrix,
        render_uv_layer=snapshot.render_uv_layer,
    )
    MeshSnapshotValidator().validate_or_raise(output)

    if degenerate_triangle_indices:
        logger.warning(
            "Filtered %d zero-area Blender loop triangles for '%s': triangle_indices=%s",
            len(degenerate_triangle_indices),
            snapshot.source_object_id,
            tuple(degenerate_triangle_indices),
        )
    logger.debug(
        "Converted Blender loop triangles for '%s': polygons=%d raw_triangles=%d "
        "retained_triangles=%d generated_edges=%d",
        snapshot.source_object_id,
        len(snapshot.faces),
        len(loop_triangles),
        len(output.faces),
        len(generated_keys),
    )
    return output


__all__ = [
    "BlenderLoopTriangulationError",
    "triangulate_snapshot_with_blender_loop_triangles",
]
