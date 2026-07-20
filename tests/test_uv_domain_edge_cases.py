from dataclasses import replace
from math import inf, nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import (
    LoopId,
    MissingSourceLoopError,
    SourceLoopId,
    build_uv_correspondence,
    transfer_uv_by_source_loop,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (
    UvLayout,
    UvLayoutError,
    UvLoopCoordinate,
    UvUnwrapSettings,
    UvUnwrapStatistics,
    apply_uv_layout,
    calculate_uv_statistics,
)

from test_uv_domain_contracts import build_triangle_snapshot


@pytest.mark.parametrize("invalid_value", (nan, inf, -inf))
def test_uv_coordinates_reject_all_non_finite_values(invalid_value):
    with pytest.raises(ValueError, match="finite"):
        UvLoopCoordinate(
            loop_id=LoopId(0),
            source_loop_id=SourceLoopId("Object", 0, 0),
            coordinate=(invalid_value, 0.0),
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"smart_angle_limit_degrees": nan},
        {"island_margin": inf},
        {"area_weight": -inf},
        {"pack_margin": nan},
        {"weight_factor": inf},
    ),
)
def test_uv_unwrap_settings_reject_non_finite_operator_values(changes):
    with pytest.raises(ValueError, match="finite"):
        UvUnwrapSettings(**changes)


@pytest.mark.parametrize(
    "field_name",
    ("minimum_u", "maximum_u", "minimum_v", "maximum_v"),
)
def test_uv_unwrap_statistics_reject_non_finite_bounds(field_name):
    values = {
        "loop_count": 3,
        "minimum_u": 0.0,
        "maximum_u": 1.0,
        "minimum_v": 0.0,
        "maximum_v": 1.0,
        "outside_unit_square_count": 0,
    }
    values[field_name] = nan

    with pytest.raises(ValueError, match="finite"):
        UvUnwrapStatistics(**values)


def test_layout_rejects_unknown_loop_before_snapshot_replacement():
    snapshot = build_triangle_snapshot()
    layout = UvLayout(
        snapshot_id=snapshot.snapshot_id,
        layer_name="UVMap",
        coordinates=(
            UvLoopCoordinate(
                loop_id=LoopId(99),
                source_loop_id=SourceLoopId("Object", 0, 99),
                coordinate=(0.5, 0.5),
            ),
        ),
    )

    with pytest.raises(UvLayoutError, match="unknown loops"):
        apply_uv_layout(snapshot, layout, require_complete=False)


def test_layout_rejects_changed_source_lineage_for_known_loop():
    snapshot = build_triangle_snapshot()
    coordinates = tuple(
        UvLoopCoordinate(
            loop_id=loop.id,
            source_loop_id=(
                SourceLoopId("Object", 0, 99)
                if loop.id == LoopId(1)
                else loop.source_id
            ),
            coordinate=loop.uv("UVMap"),
        )
        for loop in snapshot.loops
    )

    with pytest.raises(UvLayoutError, match="source lineage changed"):
        apply_uv_layout(
            snapshot,
            UvLayout(snapshot.snapshot_id, "UVMap", coordinates),
        )


def test_repeated_source_loop_with_equal_uv_is_deduplicated_deterministically():
    repeated_source_ids = (
        SourceLoopId("Object", 0, 0),
        SourceLoopId("Object", 0, 0),
        SourceLoopId("Object", 0, 2),
    )
    snapshot = build_triangle_snapshot(
        layer_coordinates={
            "UVMap": ((0.25, 0.25), (0.25, 0.25), (0.0, 1.0)),
        },
        source_loop_ids=repeated_source_ids,
    )

    correspondence = build_uv_correspondence(snapshot, "UVMap")

    assert tuple(entry.source_loop_id for entry in correspondence.entries) == (
        repeated_source_ids[0],
        repeated_source_ids[2],
    )
    assert correspondence.as_dict()[repeated_source_ids[0]] == (0.25, 0.25)


def test_duplicate_tolerance_must_be_finite_and_non_negative():
    snapshot = build_triangle_snapshot()

    with pytest.raises(ValueError, match="finite"):
        build_uv_correspondence(snapshot, "UVMap", duplicate_tolerance=nan)
    with pytest.raises(ValueError, match="cannot be negative"):
        build_uv_correspondence(snapshot, "UVMap", duplicate_tolerance=-0.001)


def test_complete_transfer_reports_exact_missing_source_loop_ids():
    source = build_triangle_snapshot(
        snapshot_id="source",
        layer_coordinates={
            "SourceUV": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        },
        active_uv_layer="SourceUV",
    )
    missing_source_id = SourceLoopId("Object", 0, 99)
    target = build_triangle_snapshot(
        snapshot_id="target",
        layer_coordinates={
            "TargetUV": ((0.2, 0.2), (0.3, 0.3), (0.4, 0.4)),
        },
        active_uv_layer="TargetUV",
        source_loop_ids=(
            source.loops[0].source_id,
            missing_source_id,
            source.loops[2].source_id,
        ),
    )

    with pytest.raises(MissingSourceLoopError) as error:
        transfer_uv_by_source_loop(
            source,
            target,
            source_layer_name="SourceUV",
            target_layer_name="TargetUV",
            require_complete=True,
        )

    assert error.value.source_loop_ids == (missing_source_id,)


def test_uv_statistics_count_every_outside_loop_once():
    snapshot = build_triangle_snapshot(
        layer_coordinates={
            "UVMap": ((-0.01, 0.5), (1.01, 0.5), (0.5, 1.01)),
        },
    )

    statistics = calculate_uv_statistics(snapshot, "UVMap")

    assert statistics.outside_unit_square_count == 3
    assert statistics.minimum_u == -0.01
    assert statistics.maximum_u == 1.01
