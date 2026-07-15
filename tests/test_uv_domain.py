from dataclasses import replace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import LoopId
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import (
    UvLayout,
    UvLayoutError,
    UvLoopCoordinate,
    UvUnwrapMethod,
    UvUnwrapSettings,
    apply_uv_layout,
    build_uv_layout,
    calculate_uv_statistics,
)

from test_geometry_domain import build_square_snapshot


def test_build_and_apply_uv_layout_preserves_exact_loop_lineage():
    snapshot = build_square_snapshot()
    layout = build_uv_layout(snapshot, "UVMap")
    moved = UvLayout(
        snapshot_id=layout.snapshot_id,
        layer_name="BakedUV",
        coordinates=tuple(
            replace(
                entry,
                coordinate=(entry.coordinate[0] * 0.5, entry.coordinate[1] * 0.5),
            )
            for entry in layout.coordinates
        ),
    )

    updated = apply_uv_layout(snapshot, moved)

    assert updated.active_uv_layer == "BakedUV"
    assert "BakedUV" in updated.uv_layer_names
    for source, loop in zip(moved.coordinates, updated.loops):
        assert loop.id == source.loop_id
        assert loop.source_id == source.source_loop_id
        assert loop.uv("BakedUV") == source.coordinate


def test_layout_rejects_wrong_source_loop_identity():
    snapshot = build_square_snapshot()
    layout = build_uv_layout(snapshot, "UVMap")
    wrong_entry = replace(
        layout.coordinates[0],
        source_loop_id=layout.coordinates[1].source_loop_id,
    )
    broken = replace(
        layout,
        coordinates=(wrong_entry,) + layout.coordinates[1:],
    )
    with pytest.raises(UvLayoutError):
        apply_uv_layout(snapshot, broken)


def test_layout_can_report_incomplete_data_without_inventing_matches():
    snapshot = build_square_snapshot()
    layout = build_uv_layout(snapshot, "UVMap")
    incomplete = replace(layout, coordinates=layout.coordinates[:-1])

    with pytest.raises(UvLayoutError):
        apply_uv_layout(snapshot, incomplete)

    updated = apply_uv_layout(snapshot, incomplete, require_complete=False)
    assert updated.loops[-1].uv("UVMap") == snapshot.loops[-1].uv("UVMap")


def test_layout_rejects_unknown_local_loop():
    snapshot = build_square_snapshot()
    layout = build_uv_layout(snapshot, "UVMap")
    unknown = UvLoopCoordinate(
        loop_id=LoopId(999),
        source_loop_id=layout.coordinates[0].source_loop_id,
        coordinate=(0.0, 0.0),
    )
    with pytest.raises(UvLayoutError):
        apply_uv_layout(
            snapshot,
            replace(layout, coordinates=layout.coordinates + (unknown,)),
        )


def test_uv_statistics_detect_coordinates_outside_unit_square():
    snapshot = build_square_snapshot()
    layout = build_uv_layout(snapshot, "UVMap")
    changed = replace(
        layout,
        coordinates=(
            replace(layout.coordinates[0], coordinate=(-0.1, 0.0)),
            replace(layout.coordinates[1], coordinate=(1.1, 0.5)),
        )
        + layout.coordinates[2:],
    )
    updated = apply_uv_layout(snapshot, changed)
    statistics = calculate_uv_statistics(updated, "UVMap")
    assert statistics.loop_count == len(snapshot.loops)
    assert statistics.minimum_u == -0.1
    assert statistics.maximum_u == 1.1
    assert statistics.outside_unit_square_count == 2


def test_uv_unwrap_settings_are_typed_and_bounded():
    settings = UvUnwrapSettings(method=UvUnwrapMethod.CONFORMAL)
    assert settings.layer_name == "SpineBakeUV"
    with pytest.raises(ValueError):
        UvUnwrapSettings(island_margin=-0.1)
    with pytest.raises(ValueError):
        UvUnwrapSettings(area_weight=2.0)
