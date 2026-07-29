from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1DocumentAssemblyError,
    assemble_a1_document,
)
from Blender_to_Spine2D_Mesh_Exporter.application.a1_projected_region_filter import (
    split_xy_visible_region_snapshots,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import SourceFaceId

from test_a1_document_assembly import build_inputs
from test_geometry_domain import build_square_snapshot


def _replace_vertex_positions(snapshot, positions_by_index):
    return replace(
        snapshot,
        vertices=tuple(
            replace(vertex, position=positions_by_index.get(vertex.id.index, vertex.position))
            for vertex in snapshot.vertices
        ),
    )


def _partially_edge_on_square():
    # Face 0 remains visible in XY. Face 1 remains a valid 3D triangle, but source
    # vertices 0 and 3 share XY and differ only along Z, so that face is edge-on in
    # the two-dimensional Spine projection.
    return _replace_vertex_positions(
        build_square_snapshot(snapshot_id="partial-edge-on"),
        {3: (0.0, 0.0, 1.0)},
    )


def _fully_edge_on_square():
    # Both source triangles retain three-dimensional area while every XY point lies
    # on the same horizontal line.
    return _replace_vertex_positions(
        build_square_snapshot(snapshot_id="fully-edge-on"),
        {
            2: (1.0, 0.0, 1.0),
            3: (0.0, 0.0, 1.0),
        },
    )


def test_partial_edge_on_region_keeps_only_visible_source_face():
    source = _partially_edge_on_square()

    result = split_xy_visible_region_snapshots(
        source,
        uniform_scale=128.0,
        center_x=0.0,
        center_y=0.0,
    )

    assert len(result) == 1
    visible = result[0]
    assert tuple(face.source_id for face in visible.faces) == (
        SourceFaceId("Cube", 0),
    )
    assert len(visible.vertices) == 3
    assert len(visible.loops) == 3
    assert len(visible.faces) == 1
    assert len(source.faces) == 2


def test_completely_edge_on_region_is_omitted_without_mutating_source():
    source = _fully_edge_on_square()
    before = source

    result = split_xy_visible_region_snapshots(
        source,
        uniform_scale=128.0,
        center_x=0.0,
        center_y=0.0,
    )

    assert result == ()
    assert source == before


def test_document_assembly_skips_edge_on_region_and_keeps_dense_segment_names():
    _, z_plan, rig, regions, settings = build_inputs()
    visible = regions[0]
    edge_on = _replace_vertex_positions(
        regions[1],
        {2: (0.0, 0.0, 1.0)},
    )

    result = assemble_a1_document(
        rig,
        z_plan,
        (visible, edge_on),
        settings,
    )

    assert tuple(slot.name for slot in result.document.slots) == (
        "Cube_Segment_0",
    )
    assert len(result.projections) == 1
    assert len(result.document_build.components) == 1
    assert result.projections[0].request.triangles == (0, 1, 2)


def test_document_assembly_rejects_object_with_no_visible_xy_faces():
    source, z_plan, rig, _regions, settings = build_inputs()
    edge_on_source = _fully_edge_on_square()
    # Preserve the Source* identity expected by the Z assignment built from the square.
    edge_on_source = replace(
        edge_on_source,
        source_object_id=source.source_object_id,
        object_name=source.object_name,
    )

    with pytest.raises(A1DocumentAssemblyError, match="All prepared regions"):
        assemble_a1_document(
            rig,
            z_plan,
            (edge_on_source,),
            settings,
        )


def test_projected_filter_rejects_invalid_runtime_contracts():
    source = build_square_snapshot()

    with pytest.raises(TypeError, match="snapshot"):
        split_xy_visible_region_snapshots(
            object(),
            uniform_scale=1.0,
            center_x=0.0,
            center_y=0.0,
        )
    with pytest.raises(ValueError, match="greater than zero"):
        split_xy_visible_region_snapshots(
            source,
            uniform_scale=0.0,
            center_x=0.0,
            center_y=0.0,
        )
    with pytest.raises(TypeError, match="center_x"):
        split_xy_visible_region_snapshots(
            source,
            uniform_scale=1.0,
            center_x=True,
            center_y=0.0,
        )
