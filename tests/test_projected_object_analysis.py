from __future__ import annotations

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    EdgeId,
    FaceId,
    LoopId,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceVertexId,
    VertexId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import A1ProjectionDirection
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    ConnectedGroupSettings,
    ConnectedObjectDocument,
    LegacyRigProfile,
    SpineDocument,
    analyse_projected_object,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.connected_group_assembly import (
    _place_projected_group_main_at_anchor,
)


def _snapshot() -> MeshSnapshot:
    source = "Projected"
    positions = (
        (-2.0, 1.0, -2.0),
        (4.0, -3.0, 3.0),
        (1.0, 5.0, 1.0),
    )
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(positions)
    )
    edges = tuple(
        MeshEdge(
            id=EdgeId(index),
            source_id=SourceEdgeId(source, index),
            vertex_ids=(VertexId(first), VertexId(second)),
        )
        for index, (first, second) in enumerate(((0, 1), (1, 2), (2, 0)))
    )
    loops = tuple(
        MeshLoop(
            id=LoopId(index),
            source_id=SourceLoopId(source, 0, index),
            vertex_id=VertexId(vertex_index),
            edge_id=EdgeId(edge_index),
        )
        for index, (vertex_index, edge_index) in enumerate(((0, 0), (1, 1), (2, 2)))
    )
    return MeshSnapshot(
        snapshot_id="projected-analysis",
        source_object_id=source,
        object_name=source,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(
            MeshFace(
                id=FaceId(0),
                source_id=SourceFaceId(source, 0),
                loop_ids=(LoopId(0), LoopId(1), LoopId(2)),
                material_index=0,
                normal=(0.0, 0.0, 1.0),
            ),
        ),
        world_matrix=(
            1.0, 0.0, 0.0, 10.0,
            0.0, 1.0, 0.0, 20.0,
            0.0, 0.0, 1.0, 30.0,
            0.0, 0.0, 0.0, 1.0,
        ),
    )


def _document(*bones: Bone) -> SpineDocument:
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=bones,
        slots=(),
        skins=(),
        animations={},
    )


def test_projected_object_analysis_keeps_origin_geometry_bounds_and_slots_separate():
    analysis = analyse_projected_object(
        component_id="component",
        prefix="Projected",
        source_input_index=2,
        projection_direction=A1ProjectionDirection.ACTIVE_CAMERA,
        snapshot=_snapshot(),
        owned_slot_names=("Projected_0", "Projected_1"),
    )

    assert (
        analysis.projected_origin_u,
        analysis.projected_origin_v,
        analysis.projected_origin_depth,
    ) == (10.0, 20.0, 30.0)
    assert analysis.nearest_vertex_index == 1
    # Active Camera stores local Mesh Y reflected for the downstream attachment
    # projector. The analysis restores the true projected world V: 20 - (-3) = 23.
    assert analysis.nearest_vertex_world_position == (14.0, 23.0, 33.0)
    assert analysis.nearest_vertex_depth == 33.0
    assert analysis.farthest_vertex_index == 0
    assert analysis.farthest_vertex_depth == 28.0
    assert (
        analysis.projected_bounds.minimum_u,
        analysis.projected_bounds.maximum_u,
        analysis.projected_bounds.minimum_v,
        analysis.projected_bounds.maximum_v,
        analysis.projected_bounds.minimum_depth,
        analysis.projected_bounds.maximum_depth,
    ) == (8.0, 14.0, 15.0, 23.0, 28.0, 33.0)
    assert analysis.owned_slot_names == ("Projected_0", "Projected_1")
    assert analysis.block_depth.component_id == "component"
    assert analysis.block_depth.source_input_index == 2
    assert analysis.block_depth.nearest_vertex_depth == 33.0


def test_projected_connected_group_main_uses_selected_anchor_absolute_origin():
    profile = LegacyRigProfile()
    settings = ConnectedGroupSettings(
        texture_width=200,
        texture_height=100,
        group_prefix="all_objects",
        anchor_component_id="beta",
    )
    objects = (
        ConnectedObjectDocument(
            component_id="alpha",
            prefix="Alpha",
            document=_document(Bone("root")),
            world_position=(1.0, 2.0, 3.0),
        ),
        ConnectedObjectDocument(
            component_id="beta",
            prefix="Beta",
            document=_document(Bone("root")),
            world_position=(4.5, -2.25, 8.0),
        ),
    )
    group_main_name = profile.main_bone(settings.group_prefix)
    document = _document(
        Bone("root"),
        Bone(group_main_name, parent="root", x=0.0, y=0.0),
    )

    result = _place_projected_group_main_at_anchor(
        document,
        objects,
        settings,
        profile,
        10.0,
    )

    group_main = next(bone for bone in result.bones if bone.name == group_main_name)
    assert group_main.x == 45.0
    assert group_main.y == -22.5
    assert document.bones[1].x == 0.0
    assert document.bones[1].y == 0.0
