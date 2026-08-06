from __future__ import annotations

from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application.a1_attachment_projection import (
    project_triangulated_disk_attachment as project_raw_attachment,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_attachment_projection_service import (
    A1AttachmentProjectionError,
    normalize_a1_attachment_projection_hull,
    project_triangulated_disk_attachment,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    SpineValidator,
    build_legacy_mesh_attachment,
)

from test_a1_attachment_projection import make_rig, make_settings
from test_geometry_domain import build_square_snapshot


def _edge_on_square_snapshot():
    """Rotate the square into X/Z while preserving its source topology and UVs."""

    snapshot = build_square_snapshot(snapshot_id="edge-on-square")
    vertices = tuple(
        replace(
            vertex,
            position=(
                float(vertex.position[0]),
                0.0,
                float(vertex.position[1]),
            ),
            normal=(0.0, -1.0, 0.0),
        )
        for vertex in snapshot.vertices
    )
    faces = tuple(
        replace(face, normal=(0.0, -1.0, 0.0))
        for face in snapshot.faces
    )
    return replace(snapshot, vertices=vertices, faces=faces)


def _partially_edge_on_square_snapshot():
    """Create one collapsed setup triangle beside one visible triangle."""

    snapshot = build_square_snapshot(snapshot_id="partially-edge-on-square")
    positions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    vertices = tuple(
        replace(vertex, position=positions[index])
        for index, vertex in enumerate(snapshot.vertices)
    )
    return replace(snapshot, vertices=vertices)


def _twice_area(first, second, third) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _collapsed_triangle_indices(projection) -> tuple[int, ...]:
    positions = tuple(
        vertex.bone_position_pixels
        for vertex in projection.request.vertices
    )
    collapsed = []
    for offset in range(0, len(projection.request.triangles), 3):
        indices = projection.request.triangles[offset : offset + 3]
        first, second, third = tuple(positions[index] for index in indices)
        if abs(_twice_area(first, second, third)) <= 1.0e-9:
            collapsed.append(offset // 3)
    return tuple(collapsed)


def test_model_space_normal_preserves_fully_edge_on_disk_attachment() -> None:
    rig = make_rig()
    snapshot = _edge_on_square_snapshot()

    result = project_triangulated_disk_attachment(
        snapshot,
        rig,
        make_settings(),
    )

    assert result.request.triangles == (0, 1, 2, 0, 2, 3)
    assert result.request.hull == 4
    assert _collapsed_triangle_indices(result) == (0, 1)
    assert len({vertex.bone_position_pixels[1] for vertex in result.request.vertices}) == 1

    built = build_legacy_mesh_attachment(rig, result.request)
    assert built.attachment.triangles == result.request.triangles
    assert built.attachment.hull == result.request.hull
    assert SpineValidator().validate(built.document) == ()


def test_strict_physical_hull_normalization_still_rejects_edge_on_disk() -> None:
    raw = project_raw_attachment(
        _edge_on_square_snapshot(),
        make_rig(),
        make_settings(),
    )

    with pytest.raises(
        A1AttachmentProjectionError,
        match="collapses within Spine pixel-space",
    ):
        normalize_a1_attachment_projection_hull(raw)


def test_model_space_normal_retains_collapsed_triangles_inside_visible_region() -> None:
    result = project_triangulated_disk_attachment(
        _partially_edge_on_square_snapshot(),
        make_rig(),
        make_settings(),
    )

    assert len(result.request.triangles) == 6
    assert _collapsed_triangle_indices(result) == (0,)
    assert set(result.request.triangles) == set(range(len(result.request.vertices)))

    built = build_legacy_mesh_attachment(make_rig(), result.request)
    assert SpineValidator().validate(built.document) == ()
