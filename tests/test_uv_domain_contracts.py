from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    ConflictingSourceLoopUVError,
    EdgeId,
    FaceId,
    LoopId,
    LoopUV,
    MeshEdge,
    MeshFace,
    MeshLoop,
    MeshSnapshot,
    MeshVertex,
    SourceEdgeId,
    SourceFaceId,
    SourceLoopId,
    SourceLoopUV,
    SourceVertexId,
    UvCorrespondenceMap,
    UvTransferReport,
    VertexId,
    build_uv_correspondence,
    transfer_uv_by_source_loop,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (
    UvLayout,
    UvLayoutError,
    UvLoopCoordinate,
    UvUnwrapResult,
    UvUnwrapSettings,
    UvUnwrapStatistics,
    apply_uv_layout,
    build_uv_layout,
    calculate_uv_statistics,
)


def build_triangle_snapshot(
    *,
    snapshot_id: str = "triangle",
    layer_coordinates=None,
    active_uv_layer: str | None = "UVMap",
    source_loop_ids=None,
) -> MeshSnapshot:
    source = "Object"
    resolved_layers = layer_coordinates or {
        "UVMap": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    }
    resolved_source_loop_ids = source_loop_ids or tuple(
        SourceLoopId(source, 0, index) for index in range(3)
    )
    vertices = tuple(
        MeshVertex(
            id=VertexId(index),
            source_id=SourceVertexId(source, index),
            position=position,
            normal=(0.0, 0.0, 1.0),
        )
        for index, position in enumerate(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        )
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
            source_id=resolved_source_loop_ids[index],
            vertex_id=VertexId(vertex_index),
            edge_id=EdgeId(edge_index),
            uvs=tuple(
                LoopUV(layer_name, coordinates[index])
                for layer_name, coordinates in sorted(resolved_layers.items())
            ),
        )
        for index, (vertex_index, edge_index) in enumerate(((0, 0), (1, 1), (2, 2)))
    )
    face = MeshFace(
        id=FaceId(0),
        source_id=SourceFaceId(source, 0),
        loop_ids=(LoopId(0), LoopId(1), LoopId(2)),
        material_index=0,
        normal=(0.0, 0.0, 1.0),
    )
    return MeshSnapshot(
        snapshot_id=snapshot_id,
        source_object_id=source,
        object_name=source,
        vertices=vertices,
        edges=edges,
        loops=loops,
        faces=(face,),
        uv_layer_names=tuple(sorted(resolved_layers)),
        active_uv_layer=active_uv_layer,
        render_uv_layer=active_uv_layer,
    )


def test_uv_loop_coordinate_rejects_wrong_ids_and_boolean_components():
    source_loop_id = SourceLoopId("Object", 0, 0)

    with pytest.raises(TypeError, match="loop_id must be LoopId"):
        UvLoopCoordinate(
            loop_id=source_loop_id,
            source_loop_id=source_loop_id,
            coordinate=(0.0, 0.0),
        )
    with pytest.raises(TypeError, match=r"coordinate\[0\]"):
        UvLoopCoordinate(
            loop_id=LoopId(0),
            source_loop_id=source_loop_id,
            coordinate=(True, 0.0),
        )


def test_uv_layout_rejects_wrong_items_and_duplicate_local_loops():
    coordinate = UvLoopCoordinate(
        loop_id=LoopId(0),
        source_loop_id=SourceLoopId("Object", 0, 0),
        coordinate=(0.0, 0.0),
    )

    with pytest.raises(TypeError, match=r"coordinates\[0\]"):
        UvLayout("snapshot", "UVMap", (object(),))
    with pytest.raises(ValueError, match="duplicate local LoopId"):
        UvLayout("snapshot", "UVMap", (coordinate, coordinate))


def test_build_uv_layout_captures_exact_loop_and_source_identity():
    snapshot = build_triangle_snapshot()

    layout = build_uv_layout(snapshot, "UVMap")

    assert layout.snapshot_id == snapshot.snapshot_id
    assert tuple(item.loop_id for item in layout.coordinates) == tuple(
        loop.id for loop in snapshot.loops
    )
    assert tuple(item.source_loop_id for item in layout.coordinates) == tuple(
        loop.source_id for loop in snapshot.loops
    )
    with pytest.raises(ValueError, match="non-empty"):
        build_uv_layout(snapshot, "")


def test_partial_layout_cannot_introduce_layer_on_only_some_loops():
    snapshot = build_triangle_snapshot()
    layout = UvLayout(
        snapshot_id=snapshot.snapshot_id,
        layer_name="NewUV",
        coordinates=(
            UvLoopCoordinate(
                loop_id=snapshot.loops[0].id,
                source_loop_id=snapshot.loops[0].source_id,
                coordinate=(0.25, 0.25),
            ),
        ),
    )

    with pytest.raises(UvLayoutError, match="cannot introduce a new UV layer"):
        apply_uv_layout(snapshot, layout, require_complete=False)


def test_partial_layout_preserves_existing_target_layer_on_omitted_loops():
    snapshot = build_triangle_snapshot(
        layer_coordinates={
            "ExistingUV": ((0.1, 0.1), (0.2, 0.2), (0.3, 0.3)),
            "SourceUV": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        },
        active_uv_layer="SourceUV",
    )
    layout = UvLayout(
        snapshot_id=snapshot.snapshot_id,
        layer_name="ExistingUV",
        coordinates=(
            UvLoopCoordinate(
                loop_id=snapshot.loops[0].id,
                source_loop_id=snapshot.loops[0].source_id,
                coordinate=(0.9, 0.9),
            ),
        ),
    )

    updated = apply_uv_layout(snapshot, layout, require_complete=False)

    assert updated.active_uv_layer == "ExistingUV"
    assert tuple(loop.uv("ExistingUV") for loop in updated.loops) == (
        (0.9, 0.9),
        (0.2, 0.2),
        (0.3, 0.3),
    )
    with pytest.raises(TypeError, match="require_complete must be bool"):
        apply_uv_layout(snapshot, layout, require_complete="no")


@pytest.mark.parametrize(
    "changes, expected_message",
    (
        ({"smart_angle_limit_degrees": True}, "finite number"),
        ({"island_margin": False}, "finite number"),
        ({"weight_factor": True}, "finite number"),
        ({"iterations": True}, "iterations must be int"),
        ({"weight_group": ""}, "non-empty"),
    ),
)
def test_uv_unwrap_settings_reject_boolean_numerics_and_empty_group(
    changes,
    expected_message,
):
    with pytest.raises((TypeError, ValueError), match=expected_message):
        UvUnwrapSettings(**changes)


def test_uv_unwrap_statistics_validate_counts_and_bounds():
    with pytest.raises(TypeError, match="loop_count must be int"):
        UvUnwrapStatistics(True, 0.0, 1.0, 0.0, 1.0, 0)
    with pytest.raises(ValueError, match="at most 2"):
        UvUnwrapStatistics(2, 0.0, 1.0, 0.0, 1.0, 3)
    with pytest.raises(ValueError, match="minimum_u"):
        UvUnwrapStatistics(2, 1.0, 0.0, 0.0, 1.0, 0)
    with pytest.raises(ValueError, match="minimum_v"):
        UvUnwrapStatistics(2, 0.0, 1.0, 1.0, 0.0, 0)


def test_uv_unwrap_result_requires_matching_active_layer_and_statistics():
    snapshot = build_triangle_snapshot()
    settings = UvUnwrapSettings(layer_name="UVMap")
    statistics = calculate_uv_statistics(snapshot, "UVMap")

    result = UvUnwrapResult(snapshot, settings, statistics)
    assert result.statistics == statistics

    wrong_statistics = replace(statistics, maximum_u=2.0)
    with pytest.raises(ValueError, match="statistics do not match"):
        UvUnwrapResult(snapshot, settings, wrong_statistics)

    multi_layer_snapshot = build_triangle_snapshot(
        layer_coordinates={
            "SpineBakeUV": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            "SourceUV": ((0.1, 0.1), (0.9, 0.1), (0.1, 0.9)),
        },
        active_uv_layer="SourceUV",
    )
    bake_statistics = calculate_uv_statistics(multi_layer_snapshot, "SpineBakeUV")
    with pytest.raises(ValueError, match="active_uv_layer"):
        UvUnwrapResult(
            multi_layer_snapshot,
            UvUnwrapSettings(layer_name="SpineBakeUV"),
            bake_statistics,
        )


def test_source_loop_uv_and_correspondence_map_reject_malformed_entries():
    source_id = SourceLoopId("Object", 0, 0)
    entry = SourceLoopUV(source_id, (0.0, 0.0))

    with pytest.raises(TypeError, match=r"coordinate\[0\]"):
        SourceLoopUV(source_id, (True, 0.0))
    with pytest.raises(TypeError, match=r"entries\[0\]"):
        UvCorrespondenceMap("UVMap", (object(),))
    with pytest.raises(ValueError, match="duplicate SourceLoopId"):
        UvCorrespondenceMap("UVMap", (entry, entry))


def test_transfer_report_rejects_boolean_count_duplicates_and_overlap():
    first = SourceLoopId("Object", 0, 0)
    second = SourceLoopId("Object", 0, 1)

    with pytest.raises(TypeError, match="updated_loop_count must be int"):
        UvTransferReport("SourceUV", "TargetUV", True, (), ())
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        UvTransferReport("SourceUV", "TargetUV", 0, (first, first), ())
    with pytest.raises(ValueError, match="cannot overlap"):
        UvTransferReport("SourceUV", "TargetUV", 0, (first,), (first, second))


def test_build_correspondence_rejects_boolean_tolerance_and_conflicting_duplicates():
    snapshot = build_triangle_snapshot()
    with pytest.raises(TypeError, match="duplicate_tolerance"):
        build_uv_correspondence(snapshot, "UVMap", duplicate_tolerance=True)

    duplicate_source_ids = (
        SourceLoopId("Object", 0, 0),
        SourceLoopId("Object", 0, 0),
        SourceLoopId("Object", 0, 2),
    )
    conflicting = build_triangle_snapshot(source_loop_ids=duplicate_source_ids)
    with pytest.raises(ConflictingSourceLoopUVError):
        build_uv_correspondence(conflicting, "UVMap")


def test_transfer_rejects_non_boolean_completion_policy():
    snapshot = build_triangle_snapshot()

    with pytest.raises(TypeError, match="require_complete must be bool"):
        transfer_uv_by_source_loop(
            snapshot,
            snapshot,
            source_layer_name="UVMap",
            target_layer_name="UVMap",
            require_complete="yes",
        )


def test_incomplete_transfer_uses_explicit_fallback_and_reports_both_sides():
    source = build_triangle_snapshot(
        snapshot_id="source",
        layer_coordinates={
            "SourceUV": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        },
        active_uv_layer="SourceUV",
    )
    target_source_ids = (
        source.loops[0].source_id,
        SourceLoopId("Object", 0, 99),
        source.loops[2].source_id,
    )
    target = build_triangle_snapshot(
        snapshot_id="target",
        layer_coordinates={
            "TargetUV": ((0.2, 0.2), (0.3, 0.3), (0.4, 0.4)),
        },
        active_uv_layer="TargetUV",
        source_loop_ids=target_source_ids,
    )

    updated, report = transfer_uv_by_source_loop(
        source,
        target,
        source_layer_name="SourceUV",
        target_layer_name="TargetUV",
        require_complete=False,
    )

    assert report.updated_loop_count == 2
    assert report.missing_source_loop_ids == (target_source_ids[1],)
    assert report.unused_source_loop_ids == (source.loops[1].source_id,)
    assert tuple(loop.uv("TargetUV") for loop in updated.loops) == (
        (0.0, 0.0),
        (0.3, 0.3),
        (0.0, 1.0),
    )
