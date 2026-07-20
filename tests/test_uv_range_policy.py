from dataclasses import fields
from math import inf, nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.baking.projection_layout import (
    build_full_frame_layout,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
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
    SourceVertexId,
    VertexId,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (
    UvRangeError,
    UvRangePolicy,
    UvRangeReport,
    UvRangeViolation,
    UvUnwrapSettings,
    enforce_uv_range,
    inspect_uv_range,
)


def build_triangle_snapshot(coordinates) -> MeshSnapshot:
    source = "RangeObject"
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
            source_id=SourceLoopId(source, 0, index),
            vertex_id=VertexId(vertex_index),
            edge_id=EdgeId(edge_index),
            uvs=(LoopUV("SpineBakeUV", coordinates[index]),),
        )
        for index, (vertex_index, edge_index) in enumerate(((0, 0), (1, 1), (2, 2)))
    )
    return MeshSnapshot(
        snapshot_id="range-triangle",
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
        uv_layer_names=("SpineBakeUV",),
        active_uv_layer="SpineBakeUV",
        render_uv_layer="SpineBakeUV",
    )


def test_default_range_policy_is_strict_and_appended_for_compatibility():
    settings = UvUnwrapSettings()
    names = tuple(field.name for field in fields(UvUnwrapSettings))

    assert settings.range_policy is UvRangePolicy.REQUIRE_UNIT_SQUARE
    assert settings.range_epsilon == pytest.approx(1.0e-6)
    assert names[-2:] == ("range_policy", "range_epsilon")


def test_epsilon_accepts_boundary_noise_without_clamping_coordinates():
    coordinates = (
        (-1.0e-6, 0.5),
        (1.0 + 1.0e-6, 0.5),
        (0.5, 1.0),
    )
    snapshot = build_triangle_snapshot(coordinates)

    report = inspect_uv_range(snapshot, "SpineBakeUV", epsilon=1.0e-6)

    assert report.inside_unit_square is True
    assert report.outside_loop_count == 0
    assert tuple(loop.uv("SpineBakeUV") for loop in snapshot.loops) == coordinates


def test_epsilon_rejects_values_beyond_the_inclusive_tolerance():
    snapshot = build_triangle_snapshot(
        ((-1.1e-6, 0.5), (1.0 + 1.1e-6, 0.5), (0.5, 0.5))
    )

    report = inspect_uv_range(snapshot, "SpineBakeUV", epsilon=1.0e-6)

    assert tuple(item.loop_id for item in report.violations) == (LoopId(0), LoopId(1))
    assert tuple(item.coordinate for item in report.violations) == (
        (-1.1e-6, 0.5),
        (1.0 + 1.1e-6, 0.5),
    )


def test_strict_policy_raises_with_the_exact_range_report():
    snapshot = build_triangle_snapshot(((0.0, 0.0), (1.01, 0.5), (0.0, 1.0)))

    with pytest.raises(UvRangeError) as error:
        enforce_uv_range(
            snapshot,
            "SpineBakeUV",
            policy=UvRangePolicy.REQUIRE_UNIT_SQUARE,
            epsilon=1.0e-6,
        )

    assert error.value.report.outside_loop_count == 1
    assert error.value.report.violations[0].loop_id == LoopId(1)
    assert error.value.report.violations[0].source_loop_id == SourceLoopId(
        "RangeObject", 0, 1
    )


def test_warn_only_returns_violations_and_never_clamps():
    coordinates = ((0.0, 0.0), (1.25, 0.5), (0.0, 1.0))
    snapshot = build_triangle_snapshot(coordinates)

    report = enforce_uv_range(
        snapshot,
        "SpineBakeUV",
        policy=UvRangePolicy.WARN_ONLY,
        epsilon=0.0,
    )

    assert report.outside_loop_count == 1
    assert report.violations[0].coordinate == (1.25, 0.5)
    assert tuple(loop.uv("SpineBakeUV") for loop in snapshot.loops) == coordinates


@pytest.mark.parametrize("invalid", (True, nan, inf, -inf, -1.0e-6))
def test_range_epsilon_rejects_boolean_non_finite_and_negative_values(invalid):
    snapshot = build_triangle_snapshot(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

    with pytest.raises((TypeError, ValueError)):
        inspect_uv_range(snapshot, "SpineBakeUV", epsilon=invalid)


@pytest.mark.parametrize(
    "changes, expected",
    (
        ({"range_policy": "WARN_ONLY"}, "range_policy"),
        ({"range_epsilon": True}, "range_epsilon"),
        ({"range_epsilon": nan}, "range_epsilon"),
        ({"range_epsilon": -0.001}, "range_epsilon"),
    ),
)
def test_unwrap_settings_reject_malformed_range_configuration(changes, expected):
    with pytest.raises((TypeError, ValueError), match=expected):
        UvUnwrapSettings(**changes)


def test_range_report_rejects_duplicate_loop_violations():
    violation = UvRangeViolation(
        loop_id=LoopId(0),
        source_loop_id=SourceLoopId("RangeObject", 0, 0),
        coordinate=(1.1, 0.5),
    )

    with pytest.raises(ValueError, match="duplicate LoopId"):
        UvRangeReport(
            snapshot_id="snapshot",
            layer_name="SpineBakeUV",
            epsilon=0.0,
            loop_count=2,
            violations=(violation, violation),
        )


def test_full_frame_camera_layout_generates_exact_normalized_corner_uvs():
    layout = build_full_frame_layout(8, 4)

    assert tuple(layout.spine_uv(point) for point in layout.hull) == (
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
    )
