"""Pure regressions for projected standalone object-block setup draw order."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt

import pytest

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
    calculate_a1_projected_snapshot_depth_range,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    Slot,
    SpineDocument,
    SpineDocumentComponent,
    compose_spine_documents,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.object_block_draw_order import (
    SpineObjectBlockDepth,
    SpineObjectBlockDrawOrderError,
    apply_object_block_setup_draw_order,
    object_block_draw_order_component_ids,
)


_OBJECT_ID = "DepthFixture"
_NORMAL = (-1.0 / sqrt(6.0), -2.0 / sqrt(6.0), 1.0 / sqrt(6.0))


def _translation_matrix(x: float, y: float, z: float) -> tuple[float, ...]:
    return (
        1.0,
        0.0,
        0.0,
        x,
        0.0,
        1.0,
        0.0,
        y,
        0.0,
        0.0,
        1.0,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _snapshot() -> MeshSnapshot:
    positions = (
        (0.0, 0.0, -2.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 3.0),
    )
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(_OBJECT_ID, index),
            position=position,
            normal=_NORMAL,
        )
        for index, position in enumerate(positions)
    )
    edges = (
        MeshEdge(
            id=EdgeId(0),
            source_id=SourceEdgeId(_OBJECT_ID, 0),
            vertex_ids=(VertexId(0), VertexId(1)),
        ),
        MeshEdge(
            id=EdgeId(1),
            source_id=SourceEdgeId(_OBJECT_ID, 1),
            vertex_ids=(VertexId(1), VertexId(2)),
        ),
        MeshEdge(
            id=EdgeId(2),
            source_id=SourceEdgeId(_OBJECT_ID, 2),
            vertex_ids=(VertexId(2), VertexId(0)),
        ),
    )
    loops = tuple(
        MeshLoop(
            id=LoopId(index),
            source_id=SourceLoopId(_OBJECT_ID, 0, index),
            vertex_id=VertexId(index),
            edge_id=EdgeId(index),
        )
        for index in range(3)
    )
    face = MeshFace(
        id=FaceId(0),
        source_id=SourceFaceId(_OBJECT_ID, 0),
        loop_ids=(LoopId(0), LoopId(1), LoopId(2)),
        material_index=0,
        normal=_NORMAL,
    )
    return MeshSnapshot(
        snapshot_id="projected-depth-fixture",
        source_object_id=_OBJECT_ID,
        object_name=_OBJECT_ID,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(face,),
        world_matrix=_translation_matrix(5.0, -7.0, 11.0),
    )


def _depth(
    component_id: str,
    *,
    source_input_index: int,
    nearest: float,
    farthest: float | None = None,
) -> SpineObjectBlockDepth:
    return SpineObjectBlockDepth(
        component_id=component_id,
        source_input_index=source_input_index,
        nearest_vertex_index=source_input_index,
        nearest_vertex_depth=nearest,
        farthest_vertex_index=source_input_index,
        farthest_vertex_depth=(nearest - 1.0 if farthest is None else farthest),
    )


def _component(
    component_id: str,
    slot_names: tuple[str, ...],
    *,
    animations: dict[str, object] | None = None,
) -> SpineDocumentComponent:
    bone_name = f"{component_id}_bone"
    return SpineDocumentComponent(
        component_id=component_id,
        document=SpineDocument(
            skeleton={"spine": "4.2.43"},
            bones=(Bone(name="root"), Bone(name=bone_name, parent="root")),
            slots=tuple(Slot(name=name, bone=bone_name) for name in slot_names),
            skins=(),
            animations={} if animations is None else animations,
        ),
    )


def test_projected_snapshot_depth_range_uses_origin_plus_local_depth() -> None:
    result = calculate_a1_projected_snapshot_depth_range(_snapshot())

    assert result.origin_depth == 11.0
    assert result.nearest_vertex_id == VertexId(2)
    assert result.nearest_vertex_depth == 14.0
    assert result.farthest_vertex_id == VertexId(0)
    assert result.farthest_vertex_depth == 9.0
    assert result.depth_span == 5.0


def test_projected_snapshot_depth_range_uses_lowest_vertex_id_for_ties() -> None:
    source = _snapshot()
    tied_vertices = tuple(
        replace(
            vertex,
            position=(
                vertex.position[0],
                vertex.position[1],
                3.0 if vertex.id.index in {1, 2} else vertex.position[2],
            ),
        )
        for vertex in source.vertices
    )

    result = calculate_a1_projected_snapshot_depth_range(
        replace(source, vertices=tied_vertices)
    )

    assert result.nearest_vertex_depth == 14.0
    assert result.nearest_vertex_id == VertexId(1)


def test_object_block_order_is_far_to_near_by_nearest_vertex() -> None:
    entries = (
        _depth("near", source_input_index=0, nearest=5.0),
        _depth("far", source_input_index=1, nearest=-2.0),
        _depth("middle", source_input_index=2, nearest=1.0),
    )

    assert object_block_draw_order_component_ids(entries) == (
        "far",
        "middle",
        "near",
    )


def test_depth_ties_preserve_source_order_then_component_id() -> None:
    entries = (
        _depth("third", source_input_index=2, nearest=1.0),
        _depth("first_b", source_input_index=0, nearest=1.00005),
        _depth("first_a", source_input_index=0, nearest=1.00009),
    )

    assert object_block_draw_order_component_ids(
        entries,
        depth_tolerance=1.0e-4,
    ) == (
        "first_a",
        "first_b",
        "third",
    )


def test_tolerance_clusters_use_a_stable_anchor_instead_of_chain_merging() -> None:
    entries = (
        _depth("anchor", source_input_index=2, nearest=1.0),
        _depth("inside", source_input_index=0, nearest=1.00009),
        _depth("outside", source_input_index=1, nearest=1.00018),
    )

    assert object_block_draw_order_component_ids(
        entries,
        depth_tolerance=1.0e-4,
    ) == (
        "inside",
        "anchor",
        "outside",
    )


def test_apply_object_block_order_preserves_each_component_slot_sequence() -> None:
    components = (
        _component("near", ("near_0", "near_1")),
        _component("far", ("far_0", "far_1", "far_2")),
        _component("middle", ("middle_0", "middle_1")),
    )
    composition = compose_spine_documents(components)
    original_slots = composition.document.slots

    reordered = apply_object_block_setup_draw_order(
        composition.document,
        components,
        (
            _depth("near", source_input_index=0, nearest=5.0),
            _depth("far", source_input_index=1, nearest=-2.0),
            _depth("middle", source_input_index=2, nearest=1.0),
        ),
    )

    assert tuple(slot.name for slot in reordered.slots) == (
        "far_0",
        "far_1",
        "far_2",
        "middle_0",
        "middle_1",
        "near_0",
        "near_1",
    )
    assert composition.document.slots == original_slots


def test_apply_object_block_order_fails_on_existing_draworder_timeline() -> None:
    components = (
        _component(
            "animated",
            ("animated_0", "animated_1"),
            animations={"preview": {"drawOrder": [{"time": 0.0}]}},
        ),
        _component("other", ("other_0",)),
    )
    composition = compose_spine_documents(components)

    with pytest.raises(
        SpineObjectBlockDrawOrderError,
        match="cannot preserve an existing draw-order timeline",
    ):
        apply_object_block_setup_draw_order(
            composition.document,
            components,
            (
                _depth("animated", source_input_index=0, nearest=1.0),
                _depth("other", source_input_index=1, nearest=2.0),
            ),
        )


def test_apply_object_block_order_fails_on_component_ownership_mismatch() -> None:
    components = (
        _component("first", ("first_0",)),
        _component("second", ("second_0",)),
    )
    composition = compose_spine_documents(components)

    with pytest.raises(
        SpineObjectBlockDrawOrderError,
        match="component/depth ownership mismatch",
    ):
        apply_object_block_setup_draw_order(
            composition.document,
            components,
            (_depth("first", source_input_index=0, nearest=1.0),),
        )


def test_object_block_depth_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _depth(
            "invalid",
            source_input_index=0,
            nearest=1.0,
            farthest=2.0,
        )
