"""Build deterministic temporary materials from final Rewrite regions."""

from __future__ import annotations

from colorsys import hsv_to_rgb
from math import fmod
from typing import Tuple

from ..domain.baking.generated_materials import (
    A1GeneratedMaterialPattern,
    A1MaterialSourcePolicy,
    ColorRGBA,
    GeneratedMaterialPlan,
)
from ..domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshSnapshotValidator,
    MeshVertex,
    VertexId,
)
from .a1_texturing_layout import A1UvPropagationResult


_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_DEFAULT_SATURATION = 0.68
_DEFAULT_VALUE = 0.92


def generated_palette_color(index: int) -> ColorRGBA:
    """Return a deterministic high-contrast color for one non-negative index."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be int")
    if index < 0:
        raise ValueError("index must be non-negative")
    hue = fmod(0.07 + index * _GOLDEN_RATIO_CONJUGATE, 1.0)
    red, green, blue = hsv_to_rgb(hue, _DEFAULT_SATURATION, _DEFAULT_VALUE)
    return float(red), float(green), float(blue), 1.0


def _combine_final_regions(
    uv_regions: A1UvPropagationResult,
) -> tuple[MeshSnapshot, Tuple[int, ...]]:
    """Combine final triangulated regions into one disconnected dense snapshot."""

    if not isinstance(uv_regions, A1UvPropagationResult):
        raise TypeError("uv_regions must be A1UvPropagationResult")
    if not uv_regions.regions:
        raise ValueError("uv_regions cannot be empty")

    first = uv_regions.regions[0].snapshot
    vertices: list[MeshVertex] = []
    edges: list[MeshEdge] = []
    loops: list[MeshLoop] = []
    faces: list[MeshFace] = []
    face_region_indices: list[int] = []

    for ready_region in uv_regions.regions:
        snapshot = ready_region.snapshot
        if snapshot.source_object_id != first.source_object_id:
            raise ValueError("all regions must share source_object_id")
        if snapshot.uv_layer_names != first.uv_layer_names:
            raise ValueError("all regions must expose identical UV layers")
        if snapshot.active_uv_layer != first.active_uv_layer:
            raise ValueError("all regions must expose the same active UV layer")
        if snapshot.render_uv_layer != first.render_uv_layer:
            raise ValueError("all regions must expose the same render UV layer")
        if snapshot.world_matrix != first.world_matrix:
            raise ValueError("all regions must expose the same world matrix")

        vertex_map = {
            vertex.id: VertexId(len(vertices) + local_index)
            for local_index, vertex in enumerate(snapshot.vertices)
        }
        edge_map = {
            edge.id: EdgeId(len(edges) + local_index)
            for local_index, edge in enumerate(snapshot.edges)
        }
        loop_map = {
            loop.id: LoopId(len(loops) + local_index)
            for local_index, loop in enumerate(snapshot.loops)
        }

        vertices.extend(
            MeshVertex(
                id=vertex_map[vertex.id],
                source_id=vertex.source_id,
                position=vertex.position,
                normal=vertex.normal,
            )
            for vertex in snapshot.vertices
        )
        edges.extend(
            MeshEdge(
                id=edge_map[edge.id],
                source_id=edge.source_id,
                vertex_ids=(
                    vertex_map[edge.vertex_ids[0]],
                    vertex_map[edge.vertex_ids[1]],
                ),
                seam=edge.seam,
                sharp=edge.sharp,
            )
            for edge in snapshot.edges
        )
        loops.extend(
            MeshLoop(
                id=loop_map[loop.id],
                source_id=loop.source_id,
                vertex_id=vertex_map[loop.vertex_id],
                edge_id=edge_map[loop.edge_id],
                uvs=loop.uvs,
            )
            for loop in snapshot.loops
        )

        for face in snapshot.faces:
            faces.append(
                MeshFace(
                    id=FaceId(len(faces)),
                    source_id=face.source_id,
                    loop_ids=tuple(loop_map[loop_id] for loop_id in face.loop_ids),
                    material_index=0,
                    normal=face.normal,
                    smooth=face.smooth,
                )
            )
            face_region_indices.append(ready_region.prepared_region.region_index)

    combined = MeshSnapshot(
        snapshot_id=f"{uv_regions.source_snapshot_id}:generated-material",
        source_object_id=first.source_object_id,
        object_name=f"{first.object_name}_GeneratedMaterial",
        vertices=tuple(vertices),
        edges=tuple(edges),
        loops=tuple(loops),
        faces=tuple(faces),
        uv_layer_names=first.uv_layer_names,
        active_uv_layer=first.active_uv_layer,
        world_matrix=first.world_matrix,
        render_uv_layer=first.render_uv_layer,
    )
    MeshSnapshotValidator().validate_or_raise(combined)
    return combined, tuple(face_region_indices)


def build_generated_material_plan(
    uv_regions: A1UvPropagationResult,
    *,
    source_policy: A1MaterialSourcePolicy,
    pattern: A1GeneratedMaterialPattern,
    gray_color: ColorRGBA = (0.5, 0.5, 0.5, 1.0),
) -> GeneratedMaterialPlan:
    """Build one generated-color plan from final triangulated export regions."""

    if not isinstance(source_policy, A1MaterialSourcePolicy):
        raise TypeError("source_policy must be A1MaterialSourcePolicy")
    if not isinstance(pattern, A1GeneratedMaterialPattern):
        raise TypeError("pattern must be A1GeneratedMaterialPattern")
    combined, region_indices = _combine_final_regions(uv_regions)

    if pattern is A1GeneratedMaterialPattern.SOLID_GRAY:
        colors = tuple(gray_color for _face in combined.faces)
    elif pattern is A1GeneratedMaterialPattern.REGION_COLORS:
        colors = tuple(generated_palette_color(index) for index in region_indices)
    else:
        colors = tuple(
            generated_palette_color(face_index)
            for face_index, _face in enumerate(combined.faces)
        )
    return GeneratedMaterialPlan(
        source_policy=source_policy,
        pattern=pattern,
        target_snapshot=combined,
        face_colors=colors,
    )


__all__ = [
    "build_generated_material_plan",
    "generated_palette_color",
]
